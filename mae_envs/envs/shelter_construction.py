import numpy as np
import gym
from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.geometry import raycast
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions, SplitObservations,
                                           SelectKeysWrapper)
from mae_envs.wrappers.util import (DiscretizeActionWrapper, MaskActionWrapper,
                                    DiscardMujocoExceptionEpisodes, SpoofEntityWrapper,
                                    AddConstantObservationsWrapper,
                                    ConcatenateObsWrapper)
from mae_envs.wrappers.manipulation import (GrabObjWrapper, GrabClosestWrapper,
                                            LockObjWrapper, LockAllWrapper)
from mae_envs.wrappers.lidar import Lidar
from mae_envs.wrappers.team import TeamMembership
from mae_envs.wrappers.line_of_sight import AgentAgentObsMask2D, AgentGeomObsMask2D
from mae_envs.envs.base import Base
from mae_envs.modules.agents import Agents, AgentManipulation
from mae_envs.modules.walls import WallScenarios
from mae_envs.modules.objects import Boxes, Cylinders, LidarSites
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.util import (uniform_placement, center_placement,
                                   uniform_placement_middle)


class ShelterRewardWrapper(gym.Wrapper):
    '''
        Reward wrapper for the shelter construction task. There are invisible rays
        going from the edge of the playing area to the cylinder that needs to be 
        guarded; at each timestep the agent receives negative reward proportional
        to the number of rays that make contact with the cylinder.
        Args:
            num_rays_per_side (int): Number of rays that shoot out of each side of the
                square playing area. The ray starting points are spaced out evenly.
            reward_scale (float): scales the reward by this factor
    '''
    @store_args
    def __init__(self, env, num_rays_per_side=30, reward_scale=1):
        super().__init__(env)
        self.ray_start_points = []

        grid_cell_size = self.unwrapped.floor_size / self.unwrapped.grid_size
        # start points for the rays should not be exactly on the edge of the floor,
        # so that they do not hit the outside walls
        sp_min_xy = 1.01 * grid_cell_size
        sp_max_xy = self.unwrapped.floor_size - (1.01 * grid_cell_size)
        for i in range(num_rays_per_side):
            sp_offset = i / num_rays_per_side * (sp_max_xy - sp_min_xy)
            new_start_points = [(sp_min_xy + sp_offset, sp_min_xy, 0),
                                (sp_max_xy, sp_min_xy + sp_offset, 0),
                                (sp_max_xy - sp_offset, sp_max_xy, 0),
                                (sp_min_xy, sp_max_xy - sp_offset, 0)]
            self.ray_start_points.extend(new_start_points)

        self.ray_start_points = np.array(self.ray_start_points)

    def reset(self):
        obs = self.env.reset()
        self.sim = self.unwrapped.sim
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        target_geom = obs['static_cylinder_geom_idxs'][0, 0]
        rew = rew + np.zeros((self.unwrapped.n_agents, 1))
        for pt in self.ray_start_points:
            _, collision_geom = raycast(self.sim, pt1=pt, geom2_id=target_geom)
            if collision_geom == target_geom:
                rew -= 1

        rew *= self.reward_scale
        return obs, rew, done, info


def make_env(n_substeps=15, horizon=80, deterministic_mode=False,
             floor_size=6.0, grid_size=30,
             n_agents=1,
             objective_diameter=[1, 1], objective_placement='center',
             num_rays_per_side=25, shelter_reward_scale=1,
             n_boxes=2, n_elongated_boxes=0,
             box_size=0.5, box_only_z_rot=False,
             lock_box=True, grab_box=True, grab_selective=False, lock_grab_radius=0.25,
             lock_type='any_lock_specific', grab_exclusive=False,
             grab_out_of_vision=False, lock_out_of_vision=True,
             box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
             action_lims=(-0.9, 0.9), polar_obs=True,
             n_lidar_per_agent=0, visualize_lidar=False, compress_lidar_scale=None,
             boxid_obs=True, boxsize_obs=True, team_size_obs=False, additional_obs={}):

    grab_radius_multiplier = lock_grab_radius / box_size
    lock_radius_multiplier = lock_grab_radius / box_size

    env = Base(n_agents=n_agents, n_substeps=n_substeps, horizon=horizon,
               floor_size=floor_size, grid_size=grid_size,
               action_lims=action_lims, deterministic_mode=deterministic_mode)

    env.add_module(WallScenarios(grid_size=grid_size, door_size=2, scenario='empty',
                                 friction=other_friction))

    if objective_placement == 'center':
        objective_placement_fn = center_placement
    elif objective_placement == 'uniform_away_from_walls':
        objective_placement_fn = uniform_placement_middle(0.7)

    env.add_module(Cylinders(1, diameter=objective_diameter, height=box_size,
                             make_static=True, placement_fn=objective_placement_fn))

    env.add_module(Agents(n_agents,
                          placement_fn=uniform_placement,
                          color=[np.array((66., 235., 244., 255.)) / 255] * n_agents,
                          friction=other_friction,
                          polar_obs=polar_obs))
    if np.max(n_boxes) > 0:
        env.add_module(Boxes(n_boxes=n_boxes, placement_fn=uniform_placement,
                             friction=box_floor_friction, polar_obs=polar_obs,
                             n_elongated_boxes=n_elongated_boxes,
                             boxid_obs=boxid_obs, boxsize_obs=boxsize_obs,
                             box_size=box_size,
                             box_only_z_rot=box_only_z_rot))
    if n_lidar_per_agent > 0 and visualize_lidar:
        env.add_module(LidarSites(n_agents=n_agents, n_lidar_per_agent=n_lidar_per_agent))

    env.add_module(AgentManipulation())
    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))
    env.add_module(WorldConstants(gravity=gravity))
    env.reset()
    keys_self = ['agent_qpos_qvel', 'hider', 'prep_obs']
    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel']
    keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']
    keys_mask_external = []

    env = AddConstantObservationsWrapper(env, new_obs=additional_obs)
    keys_external += list(additional_obs)
    keys_mask_external += [ob for ob in additional_obs if 'mask' in ob]

    env = ShelterRewardWrapper(env, num_rays_per_side=num_rays_per_side,
                               reward_scale=shelter_reward_scale)
    env = SplitMultiAgentActions(env)

    if team_size_obs:
        keys_self += ['team_size']
    env = TeamMembership(env, np.zeros((n_agents,)))
    env = AgentAgentObsMask2D(env)
    env = DiscretizeActionWrapper(env, 'action_movement')
    if np.max(n_boxes) > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='box_pos', mask_obs_key='mask_ab_obs',
                                 geom_idxs_obs_key='box_geom_idxs')
        keys_external += ['mask_ab_obs', 'box_obs']
        keys_mask_external.append('mask_ab_obs')
    if lock_box and np.max(n_boxes) > 0:
        env = LockObjWrapper(env, body_names=[f'moveable_box{i}' for i in range(n_boxes)],
                             agent_idx_allowed_to_lock=np.arange(n_agents),
                             lock_type=lock_type,
                             radius_multiplier=lock_radius_multiplier,
                             obj_in_game_metadata_keys=["curr_n_boxes"],
                             agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ab_obs"])

    if grab_box and np.max(n_boxes) > 0:
        env = GrabObjWrapper(env, [f'moveable_box{i}' for i in range(n_boxes)],
                             radius_multiplier=grab_radius_multiplier,
                             grab_exclusive=grab_exclusive,
                             obj_in_game_metadata_keys=['curr_n_boxes'])

    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=visualize_lidar,
                    compress_lidar_scale=compress_lidar_scale)
        keys_copy += ['lidar']
        keys_external += ['lidar']

    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy)
    if n_agents == 1:
        env = SpoofEntityWrapper(env, 2, ['agent_qpos_qvel', 'hider', 'prep_obs'], ['mask_aa_obs'])
    env = SpoofEntityWrapper(env, n_boxes, ['box_obs', 'you_lock', 'team_lock', 'obj_lock'], ['mask_ab_obs'])
    keys_mask_external += ['mask_ab_obs_spoof']
    env = LockAllWrapper(env, remove_object_specific_lock=True)
    if not grab_out_of_vision and grab_box:
        env = MaskActionWrapper(env, 'action_pull', ['mask_ab_obs'])  # Can only pull if in vision
    if not grab_selective and grab_box:
        env = GrabClosestWrapper(env)
    env = DiscardMujocoExceptionEpisodes(env)
    env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel', 'hider', 'prep_obs'],
                                      'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock']})
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external + keys_mask_self + keys_mask_external)
    return env
