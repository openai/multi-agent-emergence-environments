import gym
import numpy as np
from typing import Tuple, List
from mae_envs.wrappers.util import update_obs_space
from mujoco_worldgen.util.types import store_args


def get_all_integer_partitions(n, min_team_size=1, max_team_size=np.inf):
    '''
        Return a list of all integer partitions of n.
        Args:
            n (int): number of entities.
            min_team_size (int): minimum number of entities in a partition
            max_team_size (int): maximum number of entities in a partition
    '''
    if n <= max_team_size:
        yield (n,)
    for i in range(min_team_size, n // 2 + 1):
        for p in get_all_integer_partitions(n - i, i, max_team_size):
            yield (i,) + p


class RUSPGenerator:
    '''
        Helper class to generate the randomized uncertain relationship graph. Agents are first
            partitioned into groups. Within each group we randomize the amount each agent shares
            reward with everyone else in the group. We then sample independent noise such that each
            agent observes an inependent noisy observation of the relationship graph.

        Reward sharing values are sampled from a beta distribution with parameters alpha and beta. For
            all results in the paper except where we experiment with team hardness, we set both
            alpha and beta to 1.

        To compute noise added to the relationship graphs, we first sample the noise level (standard devation
            of a gaussian) from a uniform distribution independently per relationship, per agent.
            We then sample a single value from this Gaussian with sampled standard deviation centered around the true value

        Args:
            min_team_size (int): minimum size of a group of agents with non-zero reward sharing amounts
            max_team_size (int): maximum size of a group of agents with non-zero reward sharing amounts
            alpha (float): reward sharing beta distribution parameter
            beta (float): reward sharing beta distribution parameter
            allow_diagonal_non_1 (bool): if True then diagonal elements of the reward sharing matrix (an agents
                weight over its own reward) can be less than 1 (sampled from the same beta distribution as for other
                relationships)
            obs_noise_std_range (tuple of float): Range (maximum and minimum) that noise standard deviation can be sampled
                from.
    '''
    @store_args
    def __init__(self, *,
                 # Prosociality Graph
                 min_team_size: int = 1,
                 max_team_size: int = 1,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 allow_diagonal_non_1: bool = True,
                 # Uncertainty
                 obs_noise_std_range: Tuple[float] = [0.0, 1.0],
                 **kwargs):
        assert min_team_size >= 1
        assert max_team_size >= 1
        assert max_team_size >= min_team_size
        assert alpha > 0
        assert beta > 0
        assert np.all(np.array(obs_noise_std_range) >= 0)
        self.cached_partitions = {}  # Keys are (n_agents, min_team_size, max_team_size)

    def _partition_agents(self, n_agents, min_team_size, max_team_size):
        '''
            Return a random partition from the set of all integer partitions
        '''
        settings = (n_agents, min_team_size, max_team_size)
        if settings not in self.cached_partitions:
            self.cached_partitions[settings] = list(get_all_integer_partitions(n_agents, min_team_size, max_team_size))
        all_partitions = self.cached_partitions[settings]
        random_partitions = all_partitions[np.random.randint(len(all_partitions))]

        return random_partitions

    def _generate_social_preferences(self, n_agents):
        '''
            Generate the relationship graph (without uncertainty)
        '''
        # Generate random partitions
        if self.max_team_size != self.min_team_size:
            random_partitions = self._partition_agents(n_agents, self.min_team_size, self.max_team_size)
        else:
            random_partitions = np.random.randint(self.min_team_size, self.max_team_size + 1, (n_agents))
        random_partitions = np.cumsum(random_partitions)
        random_partitions = random_partitions[random_partitions <= n_agents]
        random_partitions = np.concatenate([[0], random_partitions, [n_agents]])

        # Convert random partitions into a block diagonal matrix
        self.reward_xform_mat = np.zeros((n_agents, n_agents))
        for i in range(len(random_partitions) - 1):
            block = slice(random_partitions[i], random_partitions[i + 1])
            self.reward_xform_mat[block, block] = 1

        # Randomize reward sharing values in block diagonal matrix
        self.reward_xform_mat *= np.random.beta(a=self.alpha, b=self.beta, size=(n_agents, n_agents))

        # Make sure off-diagonal is symmetric
        self.reward_xform_mat = np.tril(self.reward_xform_mat, -1) + np.tril(self.reward_xform_mat).T

        if not self.allow_diagonal_non_1:
            np.fill_diagonal(self.reward_xform_mat, 1.0)

        # Randomly shuffle agents so that agent indicies do not matter
        random_shuffle_mat = np.eye(n_agents)
        np.random.shuffle(random_shuffle_mat)
        # We rotate, sum over teams, then unrotate
        self.reward_xform_mat = np.matmul(np.matmul(random_shuffle_mat.T, self.reward_xform_mat), random_shuffle_mat)

        # Normalize rows
        self.unnormalized_reward_xform_mat = self.reward_xform_mat.copy()
        self.reward_xform_mat /= np.sum(self.reward_xform_mat, axis=1, keepdims=True)

    def _generate_uncertainty(self, n_agents):
        '''
            Generate uncertainty levels and noise to be applied to the matrices
        '''
        self.noise_std = np.random.uniform(low=self.obs_noise_std_range[0],
                                           high=self.obs_noise_std_range[1],
                                           size=(n_agents, n_agents, n_agents))
        self.noise = np.random.normal(scale=self.noise_std)

    def _precompute_observations(self, n_agents):
        '''
            Precompute observations since they are static per episode.
        '''
        # We have independent noisy observations per agents, so we copy the reward matrix n_agents times and
        #   then add the noise matrices
        rew_mats = np.repeat(np.expand_dims(self.unnormalized_reward_xform_mat, 0), n_agents, axis=0)
        noisy_rew_mats = rew_mats + self.noise
        self.precomputed_obs = {}

        def _index_into_mats(key, *indices):
            '''
                Helper function to create 3 observation types with the same indices
            '''
            self.precomputed_obs[key] = rew_mats[indices]  # Non-noisy version of the reward matrix
            self.precomputed_obs[key + "_noisy"] = noisy_rew_mats[indices]  # Noisy version of the reward matrix
            self.precomputed_obs[key + '_noise_level'] = self.noise_std[indices]  # Noise level associated with each entry in the noisy reward matrices

        def _transpose_existing(new_key, existing_key):
            '''
                Helper function to transpose all 3 observations for an key. This is useful if an agent policy
                    or value function needs to observe what other agents observe about it.
            '''
            self.precomputed_obs[new_key] = self.precomputed_obs[existing_key].T
            self.precomputed_obs[new_key + "_noisy"] = self.precomputed_obs[existing_key + "_noisy"].T
            self.precomputed_obs[new_key + '_noise_level'] = self.precomputed_obs[existing_key + '_noise_level'].T

        # Relationship variable of myself (What is the weight over my own reward) with my own noise variable.
        #   This is in effect the 3D diagonal, so the output shape will be (n_agents,)
        _index_into_mats('self_rew_value', np.arange(n_agents), np.arange(n_agents), np.arange(n_agents))

        # Relationship variable of other agents weight over their own reward with my own noise variable (s)
        #   Row i is the diagonal of the ith matrix
        _index_into_mats('other_rew_value_s', slice(None), np.arange(n_agents), np.arange(n_agents))

        # My relationship variable with other agents (so) with my noise (s)
        #   Row i is row i of the ith matrix
        _index_into_mats('rew_share_so_s', np.arange(n_agents), np.arange(n_agents), slice(None))

        # Others relationship variable with me (os) with their noise (o)
        #   Should only be used in the value function
        _transpose_existing('rew_share_os_o', 'rew_share_so_s')


class RUSPWrapper(RUSPGenerator, gym.Wrapper):
    '''
        Gym wrapper for generating relationship graphs. Generates a new relationship graph and uncertainties on reset.
            Provides all observations necessary to agents and transforms reward according to the relationship graph.

        Observations:
            Each observation has the true value, the noisy value "_noisy" and the uncertainty level "_noise_level"

            self_rew_value: Relationship variable of myself (What is the weight over my own reward) with my own noise variable.
            other_rew_value_s: Relationship variable of other agents weight over their own reward with my own noise variable (s)
            rew_share_so_s: My relationship variable with other agents (so) with my noise (s)
            rew_share_os_o: Others relationship variable with me (os) with their noise (o). Should only be used in the value function
    '''
    @store_args
    def __init__(self, env, **graph_kwargs):
        RUSPGenerator.__init__(self, **graph_kwargs)
        gym.Wrapper.__init__(self, env)
        n_a = self.metadata['n_agents']
        self.obs_keys_with_shapes = {
            'self_rew_value': [n_a, 1],
            'self_rew_value_noisy': [n_a, 1],
            'self_rew_value_noise_level': [n_a, 1],
            'other_rew_value_s': [n_a, n_a, 1],
            'other_rew_value_s_noisy': [n_a, n_a, 1],
            'other_rew_value_s_noise_level': [n_a, n_a, 1],
            'rew_share_so_s': [n_a, n_a, 1],
            'rew_share_so_s_noisy': [n_a, n_a, 1],
            'rew_share_so_s_noise_level': [n_a, n_a, 1],
            'rew_share_os_o': [n_a, n_a, 1],
            'rew_share_os_o_noisy': [n_a, n_a, 1],
            'rew_share_os_o_noise_level': [n_a, n_a, 1],
        }
        self.observation_space = update_obs_space(self, self.obs_keys_with_shapes)

    def reset(self):
        self._generate_social_preferences(self.metadata['n_agents'])
        self._generate_uncertainty(self.metadata['n_agents'])
        self._precompute_observations(self.metadata['n_agents'])
        return self.observation(self.env.reset())

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = np.matmul(self.reward_xform_mat, rew)
        return self.observation(obs), rew, done, info

    def observation(self, obs):
        for k in self.obs_keys_with_shapes:
            obs[k] = np.expand_dims(self.precomputed_obs[k], -1)
        return obs


def add_rew_share_observation_keys(*, keys_self: List[str],
                                   keys_additional_self_vf: List[str],
                                   keys_other_agents: List[str],
                                   keys_additional_other_agents_vf: List[str],
                                   keys_self_matrices: List[str],
                                   **kwargs):
    '''
        Determines how keys about the relationship graph should be observed.
        Args:
            keys_self: keys that the agent should observe about itself
            keys_additional_self_vf: keys about an agent but only that the value function should observe
            keys_other_agents: keys about other agents
            keys_additional_other_agents_vf: keys about other agents but only that the value function should observe
            keys_self_matrices: keys that are shaped (n_agents, n_agents, X). These need to be dealth with differently
    '''

    keys_self += [
        'self_rew_value_noisy',
        'self_rew_value_noise_level',
    ]
    keys_additional_self_vf.append('self_rew_value')

    keys_other_agents += [
        'rew_share_so_s_noisy',
        'rew_share_so_s_noise_level',
        'other_rew_value_s_noisy',
        'other_rew_value_s_noise_level'
    ]

    other_rew_value_keys = [
        'other_rew_value_s_noisy',
        'other_rew_value_s_noise_level',
    ]

    keys_additional_other_agents_vf += [
        'rew_share_so_s',
        'other_rew_value_s',
        'rew_share_os_o_noisy',
        'rew_share_os_o_noise_level',
    ]

    keys_self_matrices += [
        'other_rew_value_s',
        'other_rew_value_s_noisy',
        'other_rew_value_s_noise_level',
        'rew_share_so_s',
        'rew_share_so_s_noisy',
        'rew_share_so_s_noise_level',
        'rew_share_os_o',
        'rew_share_os_o_noisy',
        'rew_share_os_o_noise_level',
    ]

    return keys_self, keys_additional_self_vf, keys_other_agents, keys_additional_other_agents_vf, keys_self_matrices
