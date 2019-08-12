import gym
from mujoco_py import MujocoException
from gym.spaces import Dict, Box
import numpy as np
from copy import deepcopy
import logging


def update_obs_space(env, delta):
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, shape, np.float32)
    return Dict(spaces)


class NumpyArrayRewardWrapper(gym.RewardWrapper):
    """
        Convenience wrapper that casts rewards to the multiagent format
        (numpy array of shape (n_agents,))
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        return np.zeros((self.unwrapped.n_agents,)) + rew


class DiscretizeActionWrapper(gym.ActionWrapper):
    '''
        Take a Box action and convert it to a MultiDiscrete Action through quantization
        Args:
            action_key: (string) action to discretize
            nbuckets: (int) number of discrete actions per dimension. It should be odd such
                        that actions centered around 0 will have the middle action be 0.
    '''
    def __init__(self, env, action_key, nbuckets=11):
        super().__init__(env)
        self.action_key = action_key
        self.discrete_to_continuous_act_map = []
        for i, ac_space in enumerate(self.action_space.spaces[action_key].spaces):
            assert isinstance(ac_space, Box)
            action_map = np.array([np.linspace(low, high, nbuckets)
                                   for low, high in zip(ac_space.low, ac_space.high)])
            _nbuckets = np.ones((len(action_map))) * nbuckets
            self.action_space.spaces[action_key].spaces[i] = gym.spaces.MultiDiscrete(_nbuckets)
            self.discrete_to_continuous_act_map.append(action_map)
        self.discrete_to_continuous_act_map = np.array(self.discrete_to_continuous_act_map)

    def action(self, action):
        action = deepcopy(action)
        ac = action[self.action_key]

        # helper variables for indexing the discrete-to-continuous action map
        agent_idxs = np.tile(np.arange(ac.shape[0])[:, None], ac.shape[1])
        ac_idxs = np.tile(np.arange(ac.shape[1]), ac.shape[0]).reshape(ac.shape)

        action[self.action_key] = self.discrete_to_continuous_act_map[agent_idxs, ac_idxs, ac]
        return action


class DiscardMujocoExceptionEpisodes(gym.Wrapper):
    '''
        Catches Mujoco Exceptions. Sends signal to discard Episode.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.episode_error = False

    def step(self, action):
        assert not self.episode_error, "Won't Continue Episode After Mujoco Exception -- \
            Please discard episode and reset. If info['discard_episode'] is True the episode\
            should be discarded"
        try:
            obs, rew, done, info = self.env.step(action)
            info['discard_episode'] = False
        except MujocoException as e:
            self.episode_error = True
            # Done is set to False such that rollout workers do not accidently send data in
            # the event that timelimit is up in the same step as an error occured.
            obs, rew, done, info = {}, 0.0, False, {'discard_episode': True}
            logging.info(str(e))
            logging.info("Encountered Mujoco Exception During Environment Step.\
                          Reset Episode Required")

        return obs, rew, done, info

    def reset(self):
        try:
            obs = self.env.reset()
        except MujocoException:
            logging.info("Encountered Mujoco Exception During Environment Reset.\
                          Trying Reset Again")
            obs = self.reset()
        self.episode_error = False
        return obs


class MaskActionWrapper(gym.Wrapper):
    '''
        For a boolean action, sets it to zero given a mask from the previous step.
            For example you could mask the grab action based on whether you can see the box
        Args:
            action_key (string): key in action dictionary to be masked
            mask_keys (string): keys in observation dictionary with which to mask. The shape
                of the concatenation of the masks (along the 1st dimension) should exactly
                match that of action_key
    '''
    def __init__(self, env, action_key, mask_keys):
        super().__init__(env)
        self.action_key = action_key
        self.mask_keys = mask_keys

    def reset(self):
        self.prev_obs = self.env.reset()
        return deepcopy(self.prev_obs)

    def step(self, action):
        mask = np.concatenate([self.prev_obs[k] for k in self.mask_keys], -1)
        action[self.action_key] = np.logical_and(action[self.action_key], mask)
        self.prev_obs, rew, done, info = self.env.step(action)
        return deepcopy(self.prev_obs), rew, done, info


class AddConstantObservationsWrapper(gym.ObservationWrapper):
    '''
        Adds new constant observations to the environment.
        Args:
            new_obs: Dictionary with the new observations.
    '''
    def __init__(self, env, new_obs):
        super().__init__(env)
        self.new_obs = new_obs
        for obs_key in self.new_obs:
            assert obs_key not in self.observation_space.spaces, (
                f'Observation key {obs_key} exists in original observation space')
            if type(self.new_obs[obs_key]) in [list, tuple]:
                self.new_obs[obs_key] = np.array(self.new_obs[obs_key])
            shape = self.new_obs[obs_key].shape
            self.observation_space = update_obs_space(self, {obs_key: shape})

    def observation(self, obs):
        for key, val in self.new_obs.items():
            obs[key] = val
        return obs


class SpoofEntityWrapper(gym.ObservationWrapper):
    '''
        Add extra entities along entity dimension such that shapes can match between
            environments with differing number of entities. This is meant to be used
            after SplitObservations and SelectKeysWrapper. This will also add masks that are
            1 except along the new columns (which could be used by fully observed value function)
        Args:
            total_n_entities (int): total number of entities after spoofing (including spoofed ones)
            keys (list): observation keys with which to add entities along the second dimension
            mask_keys (list): mask keys with which to add columns.
    '''
    def __init__(self, env, total_n_entities, keys, mask_keys):
        super().__init__(env)
        self.total_n_entities = total_n_entities
        self.keys = keys
        self.mask_keys = mask_keys
        for key in self.keys + self.mask_keys:
            shape = list(self.observation_space.spaces[key].shape)
            shape[1] = total_n_entities
            self.observation_space = update_obs_space(self, {key: shape})
        for key in self.mask_keys:
            shape = list(self.observation_space.spaces[key].shape)
            self.observation_space = update_obs_space(self, {key + '_spoof': shape})

    def observation(self, obs):
        for key in self.keys:
            n_to_spoof = self.total_n_entities - obs[key].shape[1]
            if n_to_spoof > 0:
                obs[key] = np.concatenate([obs[key], np.zeros((obs[key].shape[0], n_to_spoof, obs[key].shape[-1]))], 1)
        for key in self.mask_keys:
            n_to_spoof = self.total_n_entities - obs[key].shape[1]
            obs[key + '_spoof'] = np.concatenate([np.ones_like(obs[key]), np.zeros((obs[key].shape[0], n_to_spoof))], -1)
            if n_to_spoof > 0:
                obs[key] = np.concatenate([obs[key], np.zeros((obs[key].shape[0], n_to_spoof))], -1)

        return obs


class ConcatenateObsWrapper(gym.ObservationWrapper):
    '''
        Group multiple observations under the same key in the observation dictionary.
        Args:
            obs_groups: dict of {key_to_save: [keys to concat]}
    '''
    def __init__(self, env, obs_groups):
        super().__init__(env)
        self.obs_groups = obs_groups
        for key_to_save, keys_to_concat in obs_groups.items():
            assert np.all([np.array(self.observation_space.spaces[keys_to_concat[0]].shape[:-1]) ==
                           np.array(self.observation_space.spaces[k].shape[:-1])
                           for k in keys_to_concat]), \
                f"Spaces were {[(k, v) for k, v in self.observation_space.spaces.items() if k in keys_to_concat]}"
            new_last_dim = sum([self.observation_space.spaces[k].shape[-1] for k in keys_to_concat])
            new_shape = list(self.observation_space.spaces[keys_to_concat[0]].shape[:-1]) + [new_last_dim]
            self.observation_space = update_obs_space(self, {key_to_save: new_shape})

    def observation(self, obs):
        for key_to_save, keys_to_concat in self.obs_groups.items():
            obs[key_to_save] = np.concatenate([obs[k] for k in keys_to_concat], -1)
        return obs
