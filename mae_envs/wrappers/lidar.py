import gym
import numpy as np
from mujoco_worldgen.util.rotation import quat_from_angle_and_axis
from mujoco_worldgen.util.geometry import raycast
from mae_envs.wrappers.util import update_obs_space


class Lidar(gym.ObservationWrapper):
    '''
        Creates LIDAR-type observations based on Mujoco raycast

        Args:
            n_lidar_per_agent (int): Number of concentric lidar rays per agent
            lidar_range (float): Maximum range of lidar
            compress_lidar_scale (float): Scale for non-linear compression of
                lidar range
            visualize_lidar (bool): If true, visualize lidar using thin cylinders
                representing lidar rays (requires environment to create corresponding
                sites)
    '''
    def __init__(self, env, n_lidar_per_agent=30, lidar_range=6.0,
                 compress_lidar_scale=None, visualize_lidar=False):
        super().__init__(env)
        self.n_lidar_per_agent = n_lidar_per_agent
        self.lidar_range = lidar_range
        self.compress_lidar_scale = compress_lidar_scale
        self.visualize_lidar = visualize_lidar
        self.n_agents = self.unwrapped.n_agents

        self.observation_space = update_obs_space(
            env, {'lidar': (self.n_agents, self.n_lidar_per_agent, 1)})

        # generate concentric lidar rays centered at origin
        self.lidar_angles = np.linspace(0, 2*np.pi, num=self.n_lidar_per_agent, endpoint=False)
        self.lidar_rays = self.lidar_range * np.array([np.cos(self.lidar_angles),
                                                       np.sin(self.lidar_angles),
                                                       np.zeros_like(self.lidar_angles)]).T
        self.lidar_rays = self.lidar_rays[None, :]

    def reset(self):
        obs = self.env.reset()

        sim = self.unwrapped.sim

        # Cache ids
        self.agent_body_ids = np.array([sim.model.body_name2id(f"agent{i}:particle")
                                        for i in range(self.n_agents)])
        self.agent_geom_ids = np.array([sim.model.geom_name2id(f'agent{i}:agent')
                                        for i in range(self.n_agents)])

        if self.visualize_lidar:
            self.lidar_ids = np.array([[sim.model.site_name2id(f"agent{i}:lidar{j}")
                                        for j in range(self.n_lidar_per_agent)]
                                       for i in range(self.n_agents)])

        return self.observation(obs)

    def place_lidar_ray_markers(self, agent_pos, lidar_endpoints):
        sim = self.unwrapped.sim

        site_offset = sim.data.site_xpos[self.lidar_ids, :] - sim.model.site_pos[self.lidar_ids, :]

        # compute location of lidar rays
        sim.model.site_pos[self.lidar_ids, :] = .5 * (agent_pos[:, None, :] + lidar_endpoints) - site_offset

        # compute length of lidar rays
        rel_vec = lidar_endpoints - agent_pos[:, None, :]
        rel_vec_length = np.linalg.norm(rel_vec, axis=-1)
        sim.model.site_size[self.lidar_ids, 1] = rel_vec_length / 2

        # compute rotation of lidar rays
        # normalize relative vector
        rel_vec_norm = rel_vec / rel_vec_length[:, :, None]
        # set small relative vectors to zero instead
        rel_vec_norm[rel_vec_length <= 1e-8, :] = 0.0
        # start vector
        start_vec = np.array([0.0, 0.0, 1.0])
        # calculate rotation axis: cross product between start and goal vector
        rot_axis = np.cross(start_vec, rel_vec_norm)
        norm_rot_axis = np.linalg.norm(rot_axis, axis=-1)

        # calculate rotation angle and quaternion
        rot_angle = np.arctan2(norm_rot_axis, np.dot(rel_vec_norm, start_vec))
        quat = quat_from_angle_and_axis(rot_angle, rot_axis)

        # if norm of cross product is very small, set rotation to identity
        eps = 1e-3
        quat[norm_rot_axis <= eps, :] = np.array([1.0, 0.0, 0.0, 0.0])

        sim.model.site_quat[self.lidar_ids, :] = quat

    def observation(self, obs):
        sim = self.unwrapped.sim
        agent_pos = sim.data.body_xpos[self.agent_body_ids]

        lidar_endpoints = agent_pos[:, None, :] + self.lidar_rays

        # Would be nice to vectorize in the future with better mujoco-py interface
        lidar = np.zeros((self.n_agents, self.n_lidar_per_agent))
        for i in range(self.n_agents):
            for j in range(self.n_lidar_per_agent):
                lidar[i, j] = raycast(sim, geom1_id=self.agent_geom_ids[i],
                                      pt2=lidar_endpoints[i, j], geom_group=None)[0]

        lidar[lidar < 0.0] = self.lidar_range

        if self.compress_lidar_scale is not None:
            obs['lidar'] = (self.compress_lidar_scale *
                            np.tanh(lidar[..., None] / self.compress_lidar_scale))
        else:
            obs['lidar'] = lidar[..., None]

        if self.visualize_lidar:
            # recalculate lidar endpoints
            lidar_endpoints = agent_pos[:, None, :] + \
                    lidar[:, :, None] / self.lidar_range * self.lidar_rays
            self.place_lidar_ray_markers(agent_pos, lidar_endpoints)
            sim.model.site_rgba[self.lidar_ids, :] = np.array([0.0, 0.0, 1.0, 0.2])
            sim.forward()

        return obs
