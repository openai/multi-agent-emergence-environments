import numpy as np
from mujoco_worldgen.util.types import store_args
from mae_envs.modules import EnvModule, rejection_placement


class ConstructionSites(EnvModule):
    '''
        Adds construction sites to the environment. A construction site consists of 5
        regular mujoco sites, with four of them (the 'corner' sites) forming a rectangle
        and the last site being placed in the center of the rectangle.
        Args:
            n_sites (int or (int, int)): Number of construction sites. If tuple of ints, every
                episode the number of sites is drawn uniformly from
                range(n_sites[0], n_sites[1] + 1)
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per agent
            site_name (str): Name for the sites.
            site_size (float): Site size
            site_height (float): Site height
            n_elongated_sites (int or (int, int)): Number of elongated sites. If tuple of ints,
                every episode the number of elongated sites is drawn uniformly from
                range(n_elongated_sites[0], n_elongated_sited[1] + 1)
    '''
    @store_args
    def __init__(self, n_sites, placement_fn=None, site_name='construction_site',
                 site_size=0.5, site_height=0.25, n_elongated_sites=0):
        if type(n_sites) not in [tuple, list, np.ndarray]:
            self.n_sites = [n_sites, n_sites]
        if type(n_elongated_sites) not in [tuple, list, np.ndarray]:
            self.n_elongated_sites = [n_elongated_sites, n_elongated_sites]

    def _mark_site_square(self, floor, floor_size, site_name,
                          site_relative_xyz, site_dims):
        x, y, z = site_relative_xyz
        floor.mark(site_name, relative_xyz=(x, y, z),
                   rgba=[1., 1., 1., 1.], size=0.1)

        corner_rel_offset_x, corner_rel_offset_y = (site_dims / floor_size) / 2
        corner_rel_xy = [[x - corner_rel_offset_x, y - corner_rel_offset_y],
                         [x - corner_rel_offset_x, y + corner_rel_offset_y],
                         [x + corner_rel_offset_x, y - corner_rel_offset_y],
                         [x + corner_rel_offset_x, y + corner_rel_offset_y]]
        for i, (x_corner, y_corner) in enumerate(corner_rel_xy):
            floor.mark(f'{site_name}_corner{i}',
                       relative_xyz=(x_corner, y_corner, z),
                       size=0.05, rgba=[0.8, 0.8, 0.8, 1.])

    def build_world_step(self, env, floor, floor_size):
        self.curr_n_sites = env._random_state.randint(self.n_sites[0], self.n_sites[1] + 1)
        self.curr_n_elongated_sites = env._random_state.randint(
            self.n_elongated_sites[0], self.n_elongated_sites[1] + 1)

        env.metadata['curr_n_sites'] = self.curr_n_sites
        env.metadata['curr_n_elongated_sites'] = self.curr_n_elongated_sites

        self.site_size_array = self.site_size * np.ones((self.curr_n_sites, 2))
        if self.curr_n_elongated_sites > 0:
            n_xaligned = env._random_state.randint(self.curr_n_elongated_sites + 1)
            self.site_size_array[:n_xaligned, :] = self.site_size * np.array([3.3, 0.3])
            self.site_size_array[n_xaligned:self.curr_n_elongated_sites, :] = (
                self.site_size * np.array([0.3, 3.3]))

        successful_placement = True
        for i in range(self.curr_n_sites):
            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, _ = rejection_placement(env, _placement_fn, floor_size,
                                             self.site_size_array[i])
                if pos is not None:
                    self._mark_site_square(floor, floor_size, f'{self.site_name}{i}',
                                           (pos[0], pos[1], self.site_height),
                                           self.site_size_array[i])
                else:
                    successful_placement = False
            else:
                # place the site so that all the corners are still within the play area
                pos_min = self.site_size_array[i].max() / (floor_size * 1.1) / 2
                pos = env._random_state.uniform(pos_min, 1 - pos_min, 2)
                self._mark_site_square(floor, floor_size, f'{self.site_name}{i}',
                                       (pos[0], pos[1], self.site_height),
                                       self.site_size_array[i])

        return successful_placement

    def modify_sim_step(self, env, sim):
        self.construction_site_idxs = np.array(
            [sim.model.site_name2id(f'{self.site_name}{i}')
             for i in range(self.curr_n_sites)]
            )
        self.construction_site_corner_idxs = np.array(
            [sim.model.site_name2id(f'{self.site_name}{i}_corner{j}')
             for i in range(self.curr_n_sites) for j in range(4)]
            )

    def observation_step(self, env, sim):
        site_pos = sim.data.site_xpos[self.construction_site_idxs]
        site_corner_pos = sim.data.site_xpos[self.construction_site_corner_idxs]
        site_obs = np.concatenate((site_pos,
                                  site_corner_pos.reshape((self.curr_n_sites, 12))),
                                  axis=-1)

        mask_site_obs = np.ones((env.n_agents, self.curr_n_sites))

        obs = {'construction_site_pos': site_pos,
               'construction_site_corner_pos': site_corner_pos,
               'construction_site_obs': site_obs,
               'mask_acs_obs': mask_site_obs}

        return obs
