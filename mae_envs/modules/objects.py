import numpy as np
from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.sim_funcs import (qpos_idxs_from_joint_prefix,
                                            qvel_idxs_from_joint_prefix)
from mujoco_worldgen import Geom, Material, ObjFromXML
from mujoco_worldgen.transforms import set_geom_attr_transform
from mujoco_worldgen.util.rotation import normalize_angles
from mae_envs.util.transforms import remove_hinge_axis_transform
from mae_envs.modules import EnvModule, rejection_placement, get_size_from_xml


class Boxes(EnvModule):
    '''
    Add moveable boxes to the environment.
        Args:
            n_boxes (int or (int, int)): number of boxes. If tuple of ints, every episode the
                number of boxes is drawn uniformly from range(n_boxes[0], n_boxes[1] + 1)
            n_elongated_boxes (int or (int, int)): Number of elongated boxes. If tuple of ints,
                every episode the number of elongated boxes is drawn uniformly from
                range(n_elongated_boxes[0], min(curr_n_boxes, n_elongated_boxes[1]) + 1)
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per box
            box_size (float): box size
            box_mass (float): box mass
            friction (float): box friction
            box_only_z_rot (bool): If true, boxes can only be rotated around the z-axis
            boxid_obs (bool): If true, the id of boxes is observed
            boxsize_obs (bool): If true, the size of the boxes is observed (note that the size
                is still observed if boxsize_obs is False but there are elongated boxes)
            polar_obs (bool): Give observations about rotation in polar coordinates
            mark_box_corners (bool): If true, puts a site in the middle of each of the 4 vertical
                box edges for each box (these sites are used for calculating distances in the
                blueprint construction task).
    '''
    @store_args
    def __init__(self, n_boxes, n_elongated_boxes=0, placement_fn=None,
                 box_size=0.5, box_mass=1.0, friction=None, box_only_z_rot=False,
                 boxid_obs=True, boxsize_obs=False, polar_obs=True, mark_box_corners=False):
        if type(n_boxes) not in [tuple, list, np.ndarray]:
            self.n_boxes = [n_boxes, n_boxes]
        if type(n_elongated_boxes) not in [tuple, list, np.ndarray]:
            self.n_elongated_boxes = [n_elongated_boxes, n_elongated_boxes]

    def build_world_step(self, env, floor, floor_size):
        env.metadata['box_size'] = self.box_size

        self.curr_n_boxes = env._random_state.randint(self.n_boxes[0], self.n_boxes[1] + 1)

        env.metadata['curr_n_boxes'] = np.zeros((self.n_boxes[1]))
        env.metadata['curr_n_boxes'][:self.curr_n_boxes] = 1
        env.metadata['curr_n_boxes'] = env.metadata['curr_n_boxes'].astype(np.bool)

        self.curr_n_elongated_boxes = env._random_state.randint(
            self.n_elongated_boxes[0], min(self.n_elongated_boxes[1], self.curr_n_boxes) + 1)

        self.box_size_array = self.box_size * np.ones((self.curr_n_boxes, 3))
        if self.curr_n_elongated_boxes > 0:
            # sample number of x-aligned boxes
            n_xaligned = env._random_state.randint(self.curr_n_elongated_boxes + 1)
            self.box_size_array[:n_xaligned, :] = self.box_size * np.array([3.3, 0.3, 1.0])
            self.box_size_array[n_xaligned:self.curr_n_elongated_boxes, :] = (self.box_size * np.array([0.3, 3.3, 1.0]))
        env.metadata['box_size_array'] = self.box_size_array

        successful_placement = True
        for i in range(self.curr_n_boxes):
            char = chr(ord('A') + i % 26)
            geom = Geom("box", self.box_size_array[i, :], name=f'moveable_box{i}')
            geom.set_material(Material(texture="chars/" + char + ".png"))
            geom.add_transform(set_geom_attr_transform('mass', self.box_mass))
            if self.mark_box_corners:
                for j, (x, y) in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
                    geom.mark(f'moveable_box{i}_corner{j}', relative_xyz=(x, y, 0.5),
                              rgba=[1., 1., 1., 0.])
            if self.friction is not None:
                geom.add_transform(set_geom_attr_transform('friction', self.friction))
            if self.box_only_z_rot:
                geom.add_transform(remove_hinge_axis_transform(np.array([1.0, 0.0, 0.0])))
                geom.add_transform(remove_hinge_axis_transform(np.array([0.0, 1.0, 0.0])))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, _ = rejection_placement(env, _placement_fn, floor_size,
                                             self.box_size_array[i, :2])
                if pos is not None:
                    floor.append(geom, placement_xy=pos)
                else:
                    successful_placement = False
            else:
                floor.append(geom)
        return successful_placement

    def modify_sim_step(self, env, sim):
        # Cache qpos, qvel idxs
        self.box_geom_idxs = np.array([sim.model.geom_name2id(f'moveable_box{i}')
                                       for i in range(self.curr_n_boxes)])
        self.box_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'moveable_box{i}:')
                                       for i in range(self.curr_n_boxes)])
        self.box_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'moveable_box{i}:')
                                       for i in range(self.curr_n_boxes)])
        if self.mark_box_corners:
            self.box_corner_idxs = np.array([sim.model.site_name2id(f'moveable_box{i}_corner{j}')
                                             for i in range(self.curr_n_boxes)
                                             for j in range(4)])

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        box_inds = np.expand_dims(np.arange(self.curr_n_boxes), -1)
        box_geom_idxs = np.expand_dims(self.box_geom_idxs, -1)
        box_qpos = qpos[self.box_qpos_idxs]
        box_qvel = qvel[self.box_qvel_idxs]
        box_angle = normalize_angles(box_qpos[:, 3:])
        polar_angle = np.concatenate([np.cos(box_angle), np.sin(box_angle)], -1)
        if self.polar_obs:
            box_qpos = np.concatenate([box_qpos[:, :3], polar_angle], -1)
        box_obs = np.concatenate([box_qpos, box_qvel], -1)

        if self.boxid_obs:
            box_obs = np.concatenate([box_obs, box_inds], -1)
        if self.n_elongated_boxes[1] > 0 or self.boxsize_obs:
            box_obs = np.concatenate([box_obs, self.box_size_array], -1)

        obs = {'box_obs': box_obs,
               'box_angle': box_angle,
               'box_geom_idxs': box_geom_idxs,
               'box_pos': box_qpos[:, :3],
               'box_xpos': sim.data.geom_xpos[self.box_geom_idxs]}

        if self.mark_box_corners:
            obs.update({'box_corner_pos': sim.data.site_xpos[self.box_corner_idxs],
                        'box_corner_idxs': np.expand_dims(self.box_corner_idxs, -1)})

        return obs


class Ramps(EnvModule):
    '''
    Add moveable ramps to the environment.
        Args:
            n_ramps (int): number of ramps
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per ramp
            friction (float): ramp friction
            polar_obs (bool): Give observations about rotation in polar coordinates
            pad_ramp_size (bool): pads 3 rows of zeros to the ramp observation. This makes
                ramp observations match the dimensions of elongated box observations.
    '''
    @store_args
    def __init__(self, n_ramps, placement_fn=None, friction=None, polar_obs=True,
                 pad_ramp_size=False):
        pass

    def build_world_step(self, env, floor, floor_size):
        successful_placement = True

        env.metadata['curr_n_ramps'] = np.ones((self.n_ramps)).astype(np.bool)

        for i in range(self.n_ramps):
            char = chr(ord('A') + i % 26)
            geom = geom = ObjFromXML('ramp', name=f"ramp{i}")
            geom.set_material(Material(texture="chars/" + char + ".png"))
            if self.friction is not None:
                geom.add_transform(set_geom_attr_transform('friction', self.friction))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, _ = rejection_placement(env, _placement_fn, floor_size, get_size_from_xml(geom))
                if pos is not None:
                    floor.append(geom, placement_xy=pos)
                else:
                    successful_placement = False
            else:
                floor.append(geom)
        return successful_placement

    def modify_sim_step(self, env, sim):
        # Cache qpos, qvel idxs
        self.ramp_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'ramp{i}')
                                        for i in range(self.n_ramps)])
        self.ramp_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'ramp{i}')
                                        for i in range(self.n_ramps)])
        self.ramp_geom_idxs = np.array([sim.model.geom_name2id(f'ramp{i}:ramp')
                                        for i in range(self.n_ramps)])

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        ramp_geom_idxs = np.expand_dims(self.ramp_geom_idxs, -1)
        ramp_qpos = qpos[self.ramp_qpos_idxs]
        ramp_qvel = qvel[self.ramp_qvel_idxs]
        ramp_angle = normalize_angles(ramp_qpos[:, 3:])
        polar_angle = np.concatenate([np.cos(ramp_angle), np.sin(ramp_angle)], -1)
        if self.polar_obs:
            ramp_qpos = np.concatenate([ramp_qpos[:, :3], polar_angle], -1)

        ramp_obs = np.concatenate([ramp_qpos, ramp_qvel], -1)
        if self.pad_ramp_size:
            ramp_obs = np.concatenate([ramp_obs, np.zeros((ramp_obs.shape[0], 3))], -1)

        obs = {'ramp_obs': ramp_obs,
               'ramp_angle': ramp_angle,
               'ramp_geom_idxs': ramp_geom_idxs,
               'ramp_pos': ramp_qpos[:, :3]}

        return obs


class Cylinders(EnvModule):
    '''
        Add cylinders to the environment.
        Args:
            n_objects (int): Number of cylinders
            diameter (float or (float, float)): Diameter of cylinders. If tuple of floats, every
                episode the diameter is drawn uniformly from (diameter[0], diameter[1]).
                (Note that all cylinders within an episode still share the same diameter)
            height (float or (float, float)): Height of cylinders. If tuple of floats, every
                episode the height is drawn uniformly from (height[0], height[1]).
                (Note that all cylinders within an episode still share the same height)
            make_static (bool): Makes the cylinders static, preventing them from moving. Note that
                the observations (and observation keys) are different when make_static=True
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per cylinder
            rgba ([float, float, float, float]): Determines cylinder color.
    '''
    @store_args
    def __init__(self, n_objects, diameter, height, make_static=False,
                 placement_fn=None, rgba=[1., 1., 1., 1.]):
        if type(diameter) not in [list, np.ndarray]:
            self.diameter = [diameter, diameter]
        if type(height) not in [list, np.ndarray]:
            self.height = [height, height]

    def build_world_step(self, env, floor, floor_size):
        default_name = 'static_cylinder' if self.make_static else 'moveable_cylinder'
        diameter = env._random_state.uniform(self.diameter[0], self.diameter[1])
        height = env._random_state.uniform(self.height[0], self.height[1])
        obj_size = (diameter, height, 0)
        successful_placement = True
        for i in range(self.n_objects):
            geom = Geom('cylinder', obj_size, name=f'{default_name}{i}', rgba=self.rgba)
            if self.make_static:
                geom.mark_static()

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, _ = rejection_placement(env, _placement_fn, floor_size, diameter * np.ones(2))
                if pos is not None:
                    floor.append(geom, placement_xy=pos)
                else:
                    successful_placement = False
            else:
                floor.append(geom)

        return successful_placement

    def modify_sim_step(self, env, sim):
        if self.make_static:
            self.s_cylinder_geom_idxs = np.array([sim.model.geom_name2id(f'static_cylinder{i}')
                                                  for i in range(self.n_objects)])
        else:
            self.m_cylinder_geom_idxs = np.array([sim.model.geom_name2id(f'moveable_cylinder{i}')
                                                  for i in range(self.n_objects)])
            qpos_idxs = [qpos_idxs_from_joint_prefix(sim, f'moveable_cylinder{i}')
                         for i in range(self.n_objects)]
            qvel_idxs = [qvel_idxs_from_joint_prefix(sim, f'moveable_cylinder{i}')
                         for i in range(self.n_objects)]
            self.m_cylinder_qpos_idxs = np.array(qpos_idxs)
            self.m_cylinder_qvel_idxs = np.array(qvel_idxs)

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        if self.make_static:
            s_cylinder_geom_idxs = np.expand_dims(self.s_cylinder_geom_idxs, -1)
            s_cylinder_xpos = sim.data.geom_xpos[self.s_cylinder_geom_idxs]
            obs = {'static_cylinder_geom_idxs': s_cylinder_geom_idxs,
                   'static_cylinder_xpos': s_cylinder_xpos}
        else:
            m_cylinder_geom_idxs = np.expand_dims(self.m_cylinder_geom_idxs, -1)
            m_cylinder_xpos = sim.data.geom_xpos[self.m_cylinder_geom_idxs]
            m_cylinder_qpos = qpos[self.m_cylinder_qpos_idxs]
            m_cylinder_qvel = qvel[self.m_cylinder_qvel_idxs]
            mc_angle = normalize_angles(m_cylinder_qpos[:, 3:])
            polar_angle = np.concatenate([np.cos(mc_angle), np.sin(mc_angle)], -1)
            m_cylinder_qpos = np.concatenate([m_cylinder_qpos[:, :3], polar_angle], -1)
            m_cylinder_obs = np.concatenate([m_cylinder_qpos, m_cylinder_qvel], -1)
            obs = {'moveable_cylinder_geom_idxs': m_cylinder_geom_idxs,
                   'moveable_cylinder_xpos': m_cylinder_xpos,
                   'moveable_cylinder_obs': m_cylinder_obs}

        return obs


class LidarSites(EnvModule):
    '''
    Adds sites to visualize Lidar rays
        Args:
            n_agents (int): number of agents
            n_lidar_per_agent (int): number of lidar sites per agent
    '''
    @store_args
    def __init__(self, n_agents, n_lidar_per_agent):
        pass

    def build_world_step(self, env, floor, floor_size):
        for i in range(self.n_agents):
            for j in range(self.n_lidar_per_agent):
                floor.mark(f"agent{i}:lidar{j}", (0.0, 0.0, 0.0), rgba=np.zeros((4,)))
        return True

    def modify_sim_step(self, env, sim):
        # set lidar size and shape
        self.lidar_ids = np.array([[sim.model.site_name2id(f"agent{i}:lidar{j}")
                                    for j in range(self.n_lidar_per_agent)]
                                   for i in range(self.n_agents)])
        # set lidar site shape to cylinder
        sim.model.site_type[self.lidar_ids] = 5
        sim.model.site_size[self.lidar_ids, 0] = 0.02
