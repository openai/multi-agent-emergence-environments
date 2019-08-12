import numpy as np
import logging
from mujoco_worldgen import Floor, WorldBuilder, WorldParams, Env
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions, SplitObservations,
                                           SelectKeysWrapper)
from mae_envs.wrappers.util import DiscretizeActionWrapper, DiscardMujocoExceptionEpisodes
from mae_envs.wrappers.line_of_sight import AgentAgentObsMask2D
from mae_envs.modules.agents import Agents
from mae_envs.modules.walls import RandomWalls
from mae_envs.modules.objects import Boxes, Ramps


class Base(Env):
    '''
        Multi-agent Base Environment.
        Args:
            horizon (int): Number of steps agent gets to act
            n_substeps (int): Number of internal mujoco steps per outer environment step;
                essentially this is action repeat.
            n_agents (int): number of agents in the environment
            floor_size (float): size of the floor
            grid_size (int): size of the grid that we'll use to place objects on the floor
            action_lims (float tuple): lower and upper limit of mujoco actions
            deterministic_mode (bool): if True, seeds are incremented rather than randomly sampled.
    '''
    def __init__(self, horizon=250, n_substeps=5, n_agents=2,
                 floor_size=6., grid_size=30,
                 action_lims=(-1.0, 1.0), deterministic_mode=False,
                 **kwargs):
        super().__init__(get_sim=self._get_sim,
                         get_obs=self._get_obs,
                         action_space=tuple(action_lims),
                         horizon=horizon,
                         deterministic_mode=deterministic_mode)
        self.n_agents = n_agents
        self.metadata = {}
        self.metadata['n_actors'] = n_agents
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.floor_size = floor_size
        self.grid_size = grid_size
        self.kwargs = kwargs
        self.placement_grid = np.zeros((grid_size, grid_size))
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)

    def _get_obs(self, sim):
        '''
            Loops through modules, calls their observation_step functions, and
                adds the result to the observation dictionary.
        '''
        obs = {}
        for module in self.modules:
            obs.update(module.observation_step(self, self.sim))
        return obs

    def _get_sim(self, seed):
        '''
            Calls build_world_step and then modify_sim_step for each module. If
            a build_world_step failed, then restarts.
        '''
        world_params = WorldParams(size=(self.floor_size, self.floor_size, 2.5),
                                   num_substeps=self.n_substeps)
        successful_placement = False
        failures = 0
        while not successful_placement:
            if (failures + 1) % 10 == 0:
                logging.warning(f"Failed {failures} times in creating environment")
            builder = WorldBuilder(world_params, seed)
            floor = Floor()

            builder.append(floor)

            self.placement_grid = np.zeros((self.grid_size, self.grid_size))

            successful_placement = np.all([module.build_world_step(self, floor, self.floor_size)
                                           for module in self.modules])
            failures += 1

        sim = builder.get_sim()

        for module in self.modules:
            module.modify_sim_step(self, sim)

        return sim


def make_env(n_substeps=5, horizon=250, deterministic_mode=False, n_agents=2,
             n_boxes=2, n_ramps=1):
    '''
        This make_env function is not used anywhere; it exists to provide a simple, bare-bones
        example of how to construct a multi-agent environment using the modules framework.
    '''
    env = Base(n_agents=n_agents, n_substeps=n_substeps, horizon=horizon,
               deterministic_mode=deterministic_mode)
    env.add_module(RandomWalls(grid_size=30, num_rooms=4, min_room_size=6, door_size=2))
    if n_boxes > 0:
        env.add_module(Boxes(n_boxes=n_boxes))
    if n_ramps > 0:
        env.add_module(Ramps(n_ramps=n_ramps))
    env.add_module(Agents(n_agents))
    env.reset()
    keys_self = ['agent_qpos_qvel']
    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel']
    keys_mask_external = []
    env = SplitMultiAgentActions(env)
    env = DiscretizeActionWrapper(env, 'action_movement')
    env = AgentAgentObsMask2D(env)
    env = SplitObservations(env, keys_self + keys_mask_self)
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_external=keys_external,
                            keys_mask=keys_mask_self + keys_mask_external,
                            flatten=False)
    env = DiscardMujocoExceptionEpisodes(env)
    return env
