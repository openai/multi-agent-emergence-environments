import gym
import numpy as np
from gym.spaces import Dict
from mujoco_worldgen.util.types import store_args


class AbstractBaseEnv(gym.Env):
    '''
    Barebones Gym Env that allows game to be constructed soley in wrappers.
    '''
    @store_args
    def __init__(self, n_agents):
        self.metadata = {}
        self.metadata['n_agents'] = n_agents
        self.metadata['n_actors'] = n_agents
        self.observation_space = Dict({})
        self.action_space = Dict({})

    def step(self, action):
        return {}, np.zeros(self.n_agents), False, {}

    def reset(self, **kwargs):
        return {}
