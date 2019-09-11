import numpy as np
from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.sim_funcs import (qpos_idxs_from_joint_prefix,
                                            qvel_idxs_from_joint_prefix)
from mujoco_worldgen.transforms import set_geom_attr_transform
from mujoco_worldgen.util.rotation import normalize_angles
from mae_envs.util.transforms import (add_weld_equality_constraint_transform,
                                      set_joint_damping_transform)
from mae_envs.modules import EnvModule, rejection_placement, get_size_from_xml
from mujoco_worldgen import ObjFromXML


class Agents(EnvModule):
    '''
        Add Agents to the environment.
        Args:
            n_agents (int): number of agents
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for
                spec. If list of functions, then it is assumed there is one function given
                per agent
            color (tuple or list of tuples): rgba for agent. If list of tuples, then it is
                assumed there is one color given per agent
            friction (float): agent friction
            damp_z (bool): if False, reduce z damping to 1
            polar_obs (bool): Give observations about rotation in polar coordinates
    '''
    @store_args
    def __init__(self, n_agents, placement_fn=None, color=None, friction=None,
                 damp_z=False, polar_obs=True):
        pass

    def build_world_step(self, env, floor, floor_size):
        env.metadata['n_agents'] = self.n_agents
        successful_placement = True

        for i in range(self.n_agents):
            env.metadata.pop(f"agent{i}_initpos", None)

        for i in range(self.n_agents):
            obj = ObjFromXML("particle_hinge", name=f"agent{i}")
            if self.friction is not None:
                obj.add_transform(set_geom_attr_transform('friction', self.friction))
            if self.color is not None:
                _color = (self.color[i]
                          if isinstance(self.color[0], (list, tuple, np.ndarray))
                          else self.color)
                obj.add_transform(set_geom_attr_transform('rgba', _color))
            if not self.damp_z:
                obj.add_transform(set_joint_damping_transform(1, 'tz'))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                obj_size = get_size_from_xml(obj)
                pos, pos_grid = rejection_placement(env, _placement_fn, floor_size, obj_size)
                if pos is not None:
                    floor.append(obj, placement_xy=pos)
                    # store spawn position in metadata. This allows sampling subsequent agents
                    # close to previous agents
                    env.metadata[f"agent{i}_initpos"] = pos_grid
                else:
                    successful_placement = False
            else:
                floor.append(obj)
        return successful_placement

    def modify_sim_step(self, env, sim):
        # Cache qpos, qvel idxs
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        self.agent_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'agent{i}')
                                        for i in range(self.n_agents)])
        env.metadata['agent_geom_idxs'] = [sim.model.geom_name2id(f'agent{i}:agent')
                                           for i in range(self.n_agents)]

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        agent_qpos = qpos[self.agent_qpos_idxs]
        agent_qvel = qvel[self.agent_qvel_idxs]
        agent_angle = agent_qpos[:, [-1]] - np.pi / 2  # Rotate the angle to match visual front
        agent_qpos_qvel = np.concatenate([agent_qpos, agent_qvel], -1)
        polar_angle = np.concatenate([np.cos(agent_angle), np.sin(agent_angle)], -1)
        if self.polar_obs:
            agent_qpos = np.concatenate([agent_qpos[:, :-1], polar_angle], -1)
        agent_angle = normalize_angles(agent_angle)
        obs = {
            'agent_qpos_qvel': agent_qpos_qvel,
            'agent_angle': agent_angle,
            'agent_pos': agent_qpos[:, :3]}

        return obs


class AgentManipulation(EnvModule):
    '''
        Adding this module is necessary for the grabbing mechanic implemented in GrabObjWrapper
        (found in mae_envs/wrappers/manipulation.py) to work correctly.
    '''
    @store_args
    def __init__(self):
        pass

    def build_world_step(self, env, floor, floor_size):
        for i in range(env.n_agents):
            floor.add_transform(add_weld_equality_constraint_transform(
                f'agent{i}:gripper', f'agent{i}:particle', 'floor0'))
        return True

    def modify_sim_step(self, env, sim):
        sim.model.eq_active[:] = 0
