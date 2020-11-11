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
        Split observations for each agent.
        Args:
            keys_self: list of observation names which are agent specific. E.g. this will
                    permute qpos such that each agent sees its own qpos as the first numbers
            keys_copy: list of observation names that are just passed down as is
            keys_self_matrices: list of observation names that should be (n_agent, n_agent, dim) where
                each agent has a custom observation of another agent. This is different from self_keys
                in that self_keys we assume that observations are symmetric, whereas these can represent
                unique pairwise interactions/observations
    """
    def __init__(self, env, keys_self, keys_copy=[], keys_self_matrices=[]):
        super().__init__(env)
        self.keys_self = sorted(keys_self)
        self.keys_copy = sorted(keys_copy)
        self.keys_self_matrices = sorted(keys_self_matrices)
        self.n_agents = self.metadata['n_agents']
        new_spaces = {}
        for k, v in self.observation_space.spaces.items():
            # If obs is a self obs, then we only want to include other agents obs,
            # as we will pass the self obs separately.
            assert len(v.shape) > 1, f'Obs {k} has shape {v.shape}'
            if 'mask' in k and k not in self.keys_self_matrices:
                new_spaces[k] = v
            elif k in self.keys_self_matrices:
                new_spaces[k] = Box(low=v.low[:, 1:], high=v.high[:, 1:], dtype=v.dtype)
            elif k in self.keys_self:
                assert v.shape[0] == self.n_agents, \
                    f"For self obs, obs dim 0 should equal number of agents. {k} has shape {v.shape}"
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
            # Masks that aren't self matrices should just be copied
            if 'mask' in k and k not in self.keys_self_matrices:
                new_obs[k] = obs[k]
            # Circulant self matrices
            elif k in self.keys_self_matrices:
                new_obs[k] = self._process_self_matrix(obs[k])
            # Circulant self keys
            elif k in self.keys_self:
                new_obs[k + '_self'] = obs[k]
                new_obs[k] = obs[k][circulant(np.arange(self.n_agents))]
                new_obs[k] = new_obs[k][:, 1:, :]  # Remove self observation
            elif k in self.keys_copy:
                new_obs[k] = obs[k]
            # Everything else should just get copied for each agent (e.g. external obs)
            else:
                new_obs[k] = np.tile(v, self.n_agents).reshape([v.shape[0], self.n_agents, v.shape[1]]).transpose((1, 0, 2))

        return new_obs

    def _process_self_matrix(self, self_matrix):
        '''
            self_matrix will be a (n_agent, n_agent) boolean matrix. Permute each row such that the matrix is consistent with
                the circulant permutation used for self observations. E.g. this should be used for agent agent masks
        '''
        assert np.all(self_matrix.shape[:2] == np.array((self.n_agents, self.n_agents))), \
            f"The first two dimensions of {self_matrix} were not (n_agents, n_agents)"

        new_mat = self_matrix.copy()
        # Permute each row to the right by one more than the previous
        # E.g., [[1,2],[3,4]] -> [[1,2],[4,3]]
        idx = circulant(np.arange(self.n_agents))
        new_mat = new_mat[np.arange(self.n_agents)[:, None], idx]
        new_mat = new_mat[:, 1:]  # Remove self observation
        return new_mat


class SelectKeysWrapper(gym.ObservationWrapper):
    """
        Select keys for final observations.
        Expects that all observations come in shape (n_agents, n_objects, n_dims)
        Args:
            keys_self (list): observation names that are specific to an agent
                These will be concatenated into 'observation_self' observation
            keys_other (list): observation names that should be passed through
            flatten (bool): if true, internal and external observations
    """

    def __init__(self, env, keys_self, keys_other, flatten=False):
        super().__init__(env)
        self.keys_self = sorted([k + '_self' for k in keys_self])
        self.keys_other = sorted(keys_other)
        self.flatten = flatten

        # Change observation space to look like a single agent observation space.
        # This makes constructing policies much easier
        if flatten:
            size_self = sum([np.prod(self.env.observation_space.spaces[k].shape[1:])
                             for k in self.keys_self + self.keys_other])
            self.observation_space = Dict(
                {'observation_self': Box(-np.inf, np.inf, (size_self,), np.float32)})
        else:
            size_self = sum([self.env.observation_space.spaces[k].shape[1]
                             for k in self.keys_self])
            obs_self = {'observation_self': Box(-np.inf, np.inf, (size_self,), np.float32)}
            obs_extern = {k: Box(-np.inf, np.inf, v.shape[1:], np.float32)
                          for k, v in self.observation_space.spaces.items()
                          if k in self.keys_other}
            obs_self.update(obs_extern)
            self.observation_space = Dict(obs_self)

    def observation(self, observation):
        if self.flatten:
            other_obs = [observation[k].reshape((observation[k].shape[0], -1))
                         for k in self.keys_other]
            obs = np.concatenate([observation[k] for k in self.keys_self] + other_obs, axis=-1)
            return {'observation_self': obs}
        else:
            obs = np.concatenate([observation[k] for k in self.keys_self], -1)
            obs = {'observation_self': obs}
            other_obs = {k: v for k, v in observation.items() if k in self.keys_other}
            obs.update(other_obs)
            return obs
