import gym
import numpy as np
from gym.spaces import Tuple, Discrete
from scipy.linalg import circulant
from mae_envs.wrappers.util import update_obs_space
from mujoco_worldgen.util.types import store_args


class RandomizedHorizonWrapper(gym.Wrapper):
    '''
        Randomize the horizon for a game.
            A fixed horizon can be set by setting lower_lim = upper_lim.
            A randomized but finite horizon can be set by setting upper_lim > lower_lim (randomized uniformly between these bounds)
            A discounted infinite horizon can be set by setting prob_per_step_to_stop, which is the probability that
                the episode will end on any given timestep. This is implemented by sampling the horizon from a geometric distribution.
        Args:
            lower_lim (int): Lower limit of the horizon
            upper_lim (int): Upper limit of the horizon
            prob_per_step_to_stop (float): probability that episode will end on any given timestep.

        Either lower and upper lim must both be set or prob_per_step_to_stop must be set.

        Observations:
            horizon (n_agents, 1): Episode horizon. Intended for value function
            fraction_episode_done (n_agents, 1): Fraction of the episode complete. Intended for value function
            timestep (n_agents, 1): raw timestep. Intended for policy
    '''
    @store_args
    def __init__(self, env, lower_lim=None, upper_lim=None, prob_per_step_to_stop=None):
        super().__init__(env)
        assert (lower_lim is not None and upper_lim is not None) or prob_per_step_to_stop is not None
        if prob_per_step_to_stop is not None:
            assert prob_per_step_to_stop > 0 and prob_per_step_to_stop < 1
        self.observation_space = update_obs_space(self, {
            'fraction_episode_done': [self.metadata['n_agents'], 1],
            'horizon': [self.metadata['n_agents'], 1],
            'timestep': [self.metadata['n_agents'], 1]
        })
        self.observation_space = update_obs_space(self, {})

    def reset(self, **kwargs):
        self._t = 0
        if self.prob_per_step_to_stop is not None:
            self._horizon = np.random.geometric(p=self.prob_per_step_to_stop)
        else:
            self._horizon = np.random.randint(self.lower_lim, self.upper_lim + 1) if self.lower_lim < self.upper_lim else self.lower_lim
        return self.observation(self.env.reset())

    def step(self, action):
        self._t += 1
        obs, rew, done, info = self.env.step(action)
        if self._t >= self._horizon:
            done = True
        return self.observation(obs), rew, done, info

    def observation(self, obs):
        obs['timestep'] = np.ones((self.metadata['n_agents'], 1), dtype=int) * self._t
        obs['fraction_episode_done'] = np.ones((self.metadata['n_agents'], 1)) * self._t / self._horizon
        obs['horizon'] = np.ones((self.metadata['n_agents'], 1)) * self._horizon
        return obs


class RandomIdentityVector(gym.Wrapper):
    '''
        Give agents a vector_dim dimension random identity sampled uniformly between 0 and 1.

        Args:
            vector_dim (int): Dimension of the identity vector

        Observations:
            agent_identity (n_agents, vector_dim): identity for each agent
    '''
    @store_args
    def __init__(self, env, vector_dim=16):
        super().__init__(env)
        self.observation_space = update_obs_space(self, {'agent_identity': [self.metadata['n_agents'], self.vector_dim]})

    def reset(self):
        self.agent_identities = np.random.uniform(0, 1, (self.metadata['n_agents'], self.vector_dim))
        return self.observation(self.env.reset())

    def observation(self, obs):
        obs['agent_identity'] = self.agent_identities
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.observation(obs), rew, done, info


class OtherActorAttentionAction(gym.Wrapper):
    '''
        Utility class to make actions that attend over other agents possible. Agents will likely recieve an entity
            based observation of others. The order of these entities is defined by a circulant matrix (see
            mae_envs.wrappers.multi_agent:SplitObservations). If a policy constructions an attention action head
            based on these observations we need to properly process its choice as the ordering will be different
            for every agent.

        This class defines a Discrete action head with number of options n_agents - 1 (attention over all other agents)
            and it defines a function _get_target_actor that given the choice of a particular agent maps this back
            to the true ordering of agents.
        Args:
            action_name (string): name of the action to create
    '''
    @store_args
    def __init__(self, env, action_name):
        super().__init__(env)
        self.action_name = action_name
        self.n_agents = self.metadata['n_agents']
        self.action_space.spaces[action_name] = Tuple([Discrete(n=self.metadata['n_agents'] - 1)
                                                       for _ in range(self.n_agents)])

        # This matches the circulant ordering used for "Others" Observations (see mae_envs.wrappers.multi_agent:SplitObservations)
        self.other_actors = np.arange(self.n_agents)[circulant(np.arange(self.n_agents))[:, 1:]]
        self.other_actors = dict(zip(np.arange(self.n_agents), self.other_actors))

    def _get_target_actor(self, actor, action):
        '''
            Return the true index of the targeted agent. Indicies given by the action will be in a rotated space defined
                based on how entities are presented to the policy, so we must map back to the underlying ordering.

            If the index is -1, this means no other agent was chosen.
        '''
        if action[self.action_name][actor] == -1:
            return np.array([])
        else:
            return np.array([self.other_actors[actor][action[self.action_name][actor]]])


class ActionOptionsWrapper(gym.Wrapper):
    '''
        Allows one to define a hierarchical action space by defining a meta action that chooses which
            sub action head to execute. E.g. you want agents to be only able to attack OR eat.

        Args:
            action_keys (list): list of action head names that will be options
            defaults (dict): mapping from action_key to the value that should be passed if that action is NOT chosen.
                Downstream wrappers for this action_key will handle cases when the default is passed.
            do_nothing_option (bool): If true adds the option to pick none of the available actions and pass the default
                value for all of them.

        Observations:
            previous_choice (n_agents, number options): One hot observation of each agent's previous action choice.
    '''
    @store_args
    def __init__(self, env, action_keys, defaults, do_nothing_option=True):
        super().__init__(env)
        if self.do_nothing_option:
            self.action_keys.append('do_nothing')
        self.n_agents = self.metadata['n_agents']
        self.action_space.spaces['action_choose_option'] = Tuple([Discrete(n=len(self.action_keys))
                                                                  for _ in range(self.metadata['n_agents'])])
        self.observation_space = update_obs_space(self, {'previous_choice': [self.metadata['n_agents'], len(self.action_keys)]})

    def reset(self):
        self.previous_choice = np.zeros((self.metadata['n_agents'], len(self.action_keys)))
        return self.observation(self.env.reset())

    def step(self, action):
        for i in range(self.n_agents):
            for ac_ind, ac_name in enumerate(self.action_keys):
                if ac_ind != action['action_choose_option'][i] and ac_name != 'do_nothing':
                    action[ac_name][i] = self.defaults[ac_name]

        self.previous_choice = np.eye(len(self.action_keys))[action['action_choose_option']]
        obs, rew, done, info = self.env.step(action)
        return self.observation(obs), rew, done, info

    def observation(self, obs):
        obs['previous_choice'] = self.previous_choice
        return obs
