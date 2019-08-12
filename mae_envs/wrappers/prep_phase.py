import gym
import numpy as np
from copy import deepcopy
from mae_envs.wrappers.util import update_obs_space


class PreparationPhase(gym.Wrapper):
    '''
        Rewards are switched off during preparation.

        Args: prep_fraction (float): Fraction of total time that is preparation time
    '''
    def __init__(self, env, prep_fraction=.2):
        super().__init__(env)
        self.prep_fraction = prep_fraction
        self.prep_time = self.prep_fraction * self.unwrapped.horizon
        self.n_agents = self.metadata['n_agents']
        self.step_counter = 0
        self.observation_space = update_obs_space(self, {'prep_obs': [self.n_agents, 1]})

    def reset(self):
        self.step_counter = 0
        self.in_prep_phase = True
        return self.observation(self.env.reset())

    def reward(self, reward):
        if self.in_prep_phase:
            reward = np.zeros_like(reward)

        return reward

    def observation(self, obs):
        obs['prep_obs'] = (np.ones((self.n_agents, 1)) *
                           np.minimum(1.0, self.step_counter / (self.prep_time + 1e-5)))

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = self.reward(rew)
        self.step_counter += 1
        self.in_prep_phase = self.step_counter < self.prep_time
        info['in_prep_phase'] = self.in_prep_phase

        return self.observation(obs), rew, done, info


class NoActionsInPrepPhase(gym.Wrapper):
    '''Agents have all actions turned off during preparation phase.
        For MultiDiscrete and Discrete, assumes zero action is the rounded down middle action'''

    def __init__(self, env, agent_idxs):
        super().__init__(env)
        self.agent_idxs = np.array(agent_idxs)

    def reset(self):
        obs = self.env.reset()
        self.in_prep_phase = True
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(self.action(action))
        self.in_prep_phase = info['in_prep_phase']
        return obs, rew, done, info

    def action(self, action):
        ac = deepcopy(action)
        if self.in_prep_phase:
            for k, space in self.action_space.spaces.items():
                _space = space.spaces[0]
                if isinstance(_space, gym.spaces.MultiDiscrete):
                    zero_ac = (_space.nvec - 1) // 2
                elif isinstance(_space, gym.spaces.Discrete):
                    zero_ac = (_space.n - 1) // 2
                else:
                    zero_ac = 0.0
                ac[k][self.agent_idxs] = zero_ac

        return ac


class MaskPrepPhaseAction(gym.Wrapper):
    '''
        Masks a (binary) action during preparation phase
    '''
    def __init__(self, env, action_key):
        super().__init__(env)
        self.action_key = action_key

    def reset(self):
        obs = self.env.reset()
        self.in_prep_phase = True
        return obs

    def step(self, action):
        action[self.action_key] = (action[self.action_key] * (1 - self.in_prep_phase)).astype(bool)

        obs, rew, done, info = self.env.step(action)
        self.in_prep_phase = info['in_prep_phase']

        return obs, rew, done, info
