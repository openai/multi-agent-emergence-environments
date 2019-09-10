import gym
import numpy as np
from scipy.linalg import circulant
from gym.spaces import Tuple, Box, Dict
from copy import deepcopy


class SplitMultiAgentActions(gym.ActionWrapper):
    '''
        Splits mujoco generated actions into a dict of tuple actions.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.metadata['n_actors']
        lows = np.split(self.action_space.low, self.n_agents)
        highs = np.split(self.action_space.high, self.n_agents)
        self.action_space = Dict({
            'action_movement': Tuple([Box(low=low, high=high, dtype=self.action_space.dtype)
                                      for low, high in zip(lows, highs)])
        })

    def action(self, action):
        return action['action_movement'].flatten()


class JoinMultiAgentActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.metadata['n_actors']
        low = np.concatenate([space.low for space in self.action_space.spaces])
        high = np.concatenate([space.high for space in self.action_space.spaces])
        self.action_space = Box(low=low, high=high, dtype=self.action_space.spaces[0].dtype)

    def action(self, action):
        # action should be a tuple of different agent actions
        return np.split(action, self.n_agents)


class SplitObservations(gym.ObservationWrapper):
    """
        Split observations for each agent. All non-mask observations with names not in 'keys_self'
        or 'keys_copy' are transposed so that their first dimension is n_agents.
        Args:
            keys_self: list of observation names which are agent specific. E.g. this will
                    permute qpos such that each agent sees its own qpos as the first numbers
            keys_copy: list of observation names that are just passed down as is
    """
    def __init__(self, env, keys_self, keys_copy=[]):
        super().__init__(env)
        self.keys_self = sorted(keys_self)
        self.keys_copy = sorted(keys_copy)
        self.n_agents = self.metadata['n_actors']
        new_spaces = {}
        for k, v in self.observation_space.spaces.items():
            # If obs is a self obs, then we only want to include other agents obs,
            # as we will pass the self obs separately.
            assert len(v.shape) > 1, f'Obs {k} has shape {v.shape}'
            if 'mask' in k:
                if k in self.keys_self:
                    new_spaces[k] = Box(low=v.low[:, 1:], high=v.high[:, 1:], dtype=v.dtype)
                else:
                    new_spaces[k] = v
            elif k in self.keys_self:
                assert v.shape[0] == self.n_agents, (
                    f"For self obs, obs dim 0 must equal number of agents. {k} has shape {v.shape}")
                obs_shape = (v.shape[0], self.n_agents - 1, v.shape[1])
                lows = np.tile(v.low, self.n_agents - 1).reshape(obs_shape)
                highs = np.tile(v.high, self.n_agents - 1).reshape(obs_shape)
                new_spaces[k] = Box(low=lows, high=highs, dtype=v.dtype)
            elif k in self.keys_copy:
                new_spaces[k] = deepcopy(v)
            else:
                obs_shape = (v.shape[0], self.n_agents, v.shape[1])
                lows = np.tile(v.low, self.n_agents).reshape(obs_shape).transpose((1, 0, 2))
                highs = np.tile(v.high, self.n_agents).reshape(obs_shape).transpose((1, 0, 2))
                new_spaces[k] = Box(low=lows, high=highs, dtype=v.dtype)

        for k in self.keys_self:
            new_spaces[k + '_self'] = self.observation_space.spaces[k]

        self.observation_space = Dict(new_spaces)

    def observation(self, obs):
        new_obs = {}
        for k, v in obs.items():
            if 'mask' in k:
                new_obs[k] = self._process_masks(obs[k], self_mask=(k in self.keys_self))
            elif k in self.keys_self:
                new_obs[k + '_self'] = obs[k]
                new_obs[k] = obs[k][circulant(np.arange(self.n_agents))]
                new_obs[k] = new_obs[k][:, 1:, :]  # Remove self observation
            elif k in self.keys_copy:
                new_obs[k] = obs[k]
            else:
                new_obs[k] = np.tile(v, self.n_agents).reshape([v.shape[0], self.n_agents, v.shape[1]]).transpose((1, 0, 2))

        return new_obs

    def _process_masks(self, mask_obs, self_mask=False):
        '''
            mask_obs will be a (n_agent, n_object) boolean matrix. If the mask is over non-agent
                objects then we do nothing. If the mask is over other agents (self_mask is True),
                then we permute each row such that the mask is consistent with the circulant
                permutation used for self observations
        '''
        new_mask = mask_obs.copy()
        if self_mask:
            assert np.all(new_mask.shape == np.array((self.n_agents, self.n_agents)))
            # Permute each row to the right by one more than the previous
            # E.g., [[1,2],[3,4]] -> [[1,2],[4,3]]
            idx = circulant(np.arange(self.n_agents))
            new_mask = new_mask[np.arange(self.n_agents)[:, None], idx]
            new_mask = new_mask[:, 1:]  # Remove self observation
        return new_mask


class SelectKeysWrapper(gym.ObservationWrapper):
    """
        Select keys for final observations.
        Expects that all observations come in shape (n_agents, n_objects, n_dims)
        Args:
            keys_self (list): observation names that are specific to an agent
                These will be concatenated into 'observation_self' observation
            keys_external (list): observation names that are external to agent
            keys_mask (list): observation names coresponding to agent observation masks.
                These will be split in the same way as keys self, but not concatenated into
                observation self. This argument will be ignored if flatten is true
            flatten (bool): if true, internal and external observations
    """

    def __init__(self, env, keys_self, keys_external, keys_mask=[], flatten=True):
        super().__init__(env)
        self.keys_self = sorted([k + '_self' for k in keys_self])
        self.keys_external = sorted(keys_external)
        self.keys_mask = sorted(keys_mask)
        self.flatten = flatten

        # Change observation space to look like a single agent observation space.
        # This makes constructing policies much easier
        if flatten:
            size = sum([np.prod(self.env.observation_space.spaces[k].shape[1:])
                        for k in self.keys_self + self.keys_external])
            self.observation_space = Dict(
                {'observation_self': Box(-np.inf, np.inf, (size,), np.float32)})
        else:
            size_self = sum([self.env.observation_space.spaces[k].shape[1]
                             for k in self.keys_self])
            obs_self = {'observation_self': Box(-np.inf, np.inf, (size_self,), np.float32)}
            obs_extern = {k: Box(-np.inf, np.inf, v.shape[1:], np.float32)
                          for k, v in self.observation_space.spaces.items()
                          if k in self.keys_external + self.keys_mask}
            obs_self.update(obs_extern)
            self.observation_space = Dict(obs_self)

    def observation(self, observation):
        if self.flatten:
            extern_obs = [observation[k].reshape((observation[k].shape[0], -1))
                          for k in self.keys_external]
            obs = np.concatenate([observation[k] for k in self.keys_self] + extern_obs, axis=-1)
            return {'observation_self': obs}
        else:
            obs = np.concatenate([observation[k] for k in self.keys_self], -1)
            obs = {'observation_self': obs}
            obs_extern = {k: v for k, v in observation.items() if k in self.keys_external + self.keys_mask}
            obs.update(obs_extern)
            return obs
