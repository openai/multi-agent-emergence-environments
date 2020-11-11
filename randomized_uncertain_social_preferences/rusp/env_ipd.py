import numpy as np
import gym
from gym.spaces import Tuple, Discrete
from mae_envs.wrappers.util import update_obs_space
from mujoco_worldgen.util.types import store_args
from rusp.abstract_base_env import AbstractBaseEnv
from rusp.wrappers_util import RandomizedHorizonWrapper
from rusp.wrappers_rusp import RUSPWrapper, add_rew_share_observation_keys
from mae_envs.wrappers.util import ConcatenateObsWrapper
from mae_envs.wrappers.multi_agent import (SplitObservations, SelectKeysWrapper)


class IteratedMatrixGameWrapper(gym.Wrapper):
    '''
        2 player matrix game. Agents get a single binary action "action_defect". Agents
            get to observe the last action each agent took. Agents are rewarded according to
            payoff_matrix.
        Args:
            payoff_matrix (2x2x2 np.ndarray): the payoff payoff_matrix. We index into this payoff_matrix
                according to agent actions.
        Observations:
            prev_ac (n_agents, 1): previous action each agent took.
    '''
    @store_args
    def __init__(self, env, payoff_matrix):
        super().__init__(env)
        self.n_agents = self.metadata['n_agents']
        self.action_space.spaces['action_defect'] = Tuple([Discrete(n=2) for _ in range(self.n_agents)])
        self.observation_space = update_obs_space(self, {'prev_ac': [self.n_agents, 1]})

    def reset(self):
        self.previous_action = -1 * np.ones(self.n_agents)
        self.num_defects = np.zeros(self.n_agents)
        self.num_coops = np.zeros(self.n_agents)
        return self.observation(self.env.reset())

    def step(self, action):
        self.previous_action = action['action_defect'].copy()
        obs, _, done, info = self.env.step(action)

        self.num_defects += action['action_defect']
        self.num_coops += (1 - action['action_defect'])
        rew = self.payoff_matrix[action['action_defect'][0], action['action_defect'][1]]

        if done:
            info.update({f'actor{i}_n_defects': n_defects for i, n_defects in enumerate(self.num_defects)})
            info.update({f'actor{i}_n_coops': n_coops for i, n_coops in enumerate(self.num_coops)})
        return self.observation(obs), rew, done, info

    def observation(self, obs):
        obs['prev_ac'] = self.previous_action[:, None]
        return obs


class LastAgentScripted(gym.Wrapper):
    '''
    Replace the last agent with either a all-cooperate, all-defect, or tit-for-tat scripted policy.
        The last agent is considered part of the environment, so we remove them from the observation
        and action space.
    Args:
        policy_to_play (string): One of "allc", "alld", or "tft"
    '''
    def __init__(self, env, policy_to_play):
        super().__init__(env)
        assert policy_to_play in ['allc', 'alld', 'tft']
        self.policy_to_play = policy_to_play
        self.metadata['n_actors'] -= 1
        for k, v in self.action_space.spaces.items():
            self.action_space.spaces[k] = Tuple(v.spaces[:-1])

    def reset(self):
        self.previous_action = 0
        return self.observation(self.env.reset())

    def step(self, action):
        if self.policy_to_play == 'allc':
            ac_to_play = 0
        elif self.policy_to_play == 'alld':
            ac_to_play = 1
        elif self.policy_to_play == 'tft':
            ac_to_play = self.previous_action

        self.previous_action = action['action_defect'][0]
        action['action_defect'] = np.concatenate([action['action_defect'], [ac_to_play]])

        obs, rew, done, info = self.env.step(action)
        return self.observation(obs), rew[:-1], done, info

    def observation(self, obs):
        obs = {k: v[:-1] for k, v in obs.items()}
        return obs


def make_env(horizon=10, horizon_lower=None, horizon_upper=None,
             prob_per_step_to_stop=0.1,  # If set then we play the infinite game,
             mutual_cooperate=2, defected_against=-2, successful_defect=4, mutual_defect=0,
             # Evals
             against_all_c=False, against_all_d=False, against_tft=False,
             # Random Teams
             rusp_args={}):
    env = AbstractBaseEnv(2)

    env = RandomizedHorizonWrapper(env, lower_lim=horizon_lower or horizon, upper_lim=horizon_upper or horizon,
                                   prob_per_step_to_stop=prob_per_step_to_stop)
    # Construct Payoff Matrix
    cc = [mutual_cooperate, mutual_cooperate]
    cd = [defected_against, successful_defect]
    dc = list(reversed(cd))
    dd = [mutual_defect, mutual_defect]
    payoff_matrix = np.array([[cc, cd],
                              [dc, dd]])
    env = IteratedMatrixGameWrapper(env, payoff_matrix=payoff_matrix)

    env = RUSPWrapper(env, **rusp_args)

    keys_self = ['prev_ac', 'timestep']
    keys_additional_self_vf = ['fraction_episode_done', 'horizon']

    keys_other_agents = ['prev_ac']
    keys_additional_other_agents_vf = []
    keys_self_matrices = []
    add_rew_share_observation_keys(keys_self=keys_self,
                                   keys_additional_self_vf=keys_additional_self_vf,
                                   keys_other_agents=keys_other_agents,
                                   keys_additional_other_agents_vf=keys_additional_other_agents_vf,
                                   keys_self_matrices=keys_self_matrices,
                                   **rusp_args)
    keys_external = ['other_agents',
                     'other_agents_vf',
                     'additional_self_vf_obs']

    env = SplitObservations(env, keys_self + keys_additional_self_vf,
                            keys_copy=[], keys_self_matrices=keys_self_matrices)
    env = ConcatenateObsWrapper(env, {'other_agents': keys_other_agents,
                                      'other_agents_vf': ['other_agents'] + keys_additional_other_agents_vf,
                                      'additional_self_vf_obs': [k + '_self' for k in keys_additional_self_vf]})
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external)

    if against_all_c or against_all_d or against_tft:
        if against_all_c:
            policy_to_play = 'allc'
        elif against_all_d:
            policy_to_play = 'alld'
        elif against_tft:
            policy_to_play = 'tft'
        env = LastAgentScripted(env, policy_to_play)
    return env
