import numpy as np
import gym
from copy import deepcopy
from gym.spaces import Tuple, Discrete
from mae_envs.wrappers.util import update_obs_space
from mujoco_worldgen.util.types import store_args
from mae_envs.wrappers.util import ConcatenateObsWrapper
from mae_envs.wrappers.multi_agent import (SplitObservations, SelectKeysWrapper)
from rusp.wrappers_rusp import RUSPWrapper, add_rew_share_observation_keys
from rusp.wrappers_util import RandomIdentityVector, RandomizedHorizonWrapper
from rusp.abstract_base_env import AbstractBaseEnv


class MaskYourePlaying(gym.ObservationWrapper):
    '''
        Construct a binary mask depending on who's playig. This can be used to mask the policy gradient
        on steps an agent isn't playing.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.spaces['mask'] = gym.spaces.Dict({
            'action_defect': gym.spaces.Box(-np.inf, np.inf, (self.metadata['n_actors'], 2), float),
        })

    def observation(self, obs):
        obs['mask'] = {'action_defect': np.zeros((self.metadata['n_actors'], 2), dtype=bool)}
        obs['mask']['action_defect'][np.squeeze(obs['youre_playing_self'], -1)] = 1
        return obs


class LastAgentScripted(gym.Wrapper):
    '''
    Replace the last agent with either a all-cooperate, all-defect, or tit-for-tat scripted policy.
        The last agent is considered part of the environment, so we remove them from the action space
        and do not return an observation for them. In this setting tit-for-tat remembers the action
        for each agent separately.
    Args:
        policy_to_play (string): One of "allc", "alld", or "tft"
    '''
    def __init__(self, env, policy_to_play):
        super().__init__(env)
        assert policy_to_play in ['allc', 'alld', 'tft']
        self.n_agents = self.unwrapped.n_agents
        self.policy_to_play = policy_to_play
        self.metadata['n_actors'] -= 1
        for k, v in self.action_space.spaces.items():
            self.action_space.spaces[k] = Tuple(v.spaces[:-1])

    def reset(self):
        self.previous_action_against_me = np.zeros(self.n_agents - 1, dtype=int)

        obs = self.env.reset()
        self._youre_playing = np.squeeze(obs['youre_playing_self'].copy(), -1)
        return self.observation(obs)

    def step(self, action):
        action = deepcopy(action)
        if self.policy_to_play == 'allc':
            ac_to_play = 0
        elif self.policy_to_play == 'alld':
            ac_to_play = 1
        elif self.policy_to_play == 'tft':
            # Take the zeroeth index incase this agent isn't currently playing
            ac_to_play = self.previous_action_against_me[self._youre_playing[:-1]][0]

        # Only update previous action against me if scripted agent is playing
        if self._youre_playing[-1]:
            self.previous_action_against_me[self._youre_playing[:-1]] = action['action_defect'][self._youre_playing[:-1]]

        action['action_defect'] = np.concatenate([action['action_defect'], [ac_to_play]])

        obs, rew, done, info = self.env.step(action)
        self._youre_playing = np.squeeze(obs['youre_playing_self'].copy(), -1)
        return self.observation(obs), rew[:-1], done, info

    def observation(self, obs):
        obs = {k: v[:-1] for k, v in obs.items()}
        return obs


class MultiPlayerIteratedMatrixGame(gym.Wrapper):
    '''
        N player matrix game. Agents get a single binary action "action_defect". Agents
            get to observe the last action each agent took. Agents are rewarded according to
            payoff_matrix.
        Args:
            payoff_matrix (2x2x2 np.ndarray): the payoff payoff_matrix. We index into this payoff_matrix
                according to agent actions.

        Observations:
            prev_ac (n_agents, 1): previous action each agent took. If an agent wasn't playing that timestep
                we return -1 for this observation.
            prev_ac_while_playing (n_agents, 1): previous action each action took when they were playing.
    '''
    @store_args
    def __init__(self, env, payoff_matrix):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        # 0 means to cooperate, 1 means to defect
        self.action_space.spaces['action_defect'] = Tuple([Discrete(n=2) for _ in range(self.n_agents)])
        self.observation_space = update_obs_space(self, {
            'prev_ac': [self.n_agents, 1],
            'prev_ac_while_playing': [self.n_agents, 1]
        })

    def reset(self):
        self.previous_action = -1 * np.ones(self.n_agents)
        self.previous_action_while_playing = -1 * np.ones(self.n_agents)
        self.num_defects = np.zeros((self.n_agents, self.n_agents))
        self.num_coops = np.zeros((self.n_agents, self.n_agents))

        # sls stands for "since last started". This is useful for evaluation settings
        #   where we want agents to gain rapport. Since last started means since the
        #   last agent (index n-1) took its first action
        self.num_defects_sls = np.zeros((self.n_agents, self.n_agents))
        self.num_coops_sls = np.zeros((self.n_agents, self.n_agents))
        self.last_started = False
        obs = self.env.reset()

        # This comes from ChooseAgentsToPlay wrapper
        self._youre_playing = np.squeeze(obs['youre_playing'].copy(), -1)
        return self.observation(obs)

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        # Update statistics for agents that are playing (p1 and p2)
        p1, p2 = np.where(self._youre_playing)[0]
        self.num_defects[p1, p2] += action['action_defect'][p1]
        self.num_defects[p2, p1] += action['action_defect'][p2]
        self.num_coops[p1, p2] += 1 - action['action_defect'][p1]
        self.num_coops[p2, p1] += 1 - action['action_defect'][p2]
        if p1 == self.n_agents - 1 or p2 == self.n_agents - 1 or self.last_started:
            self.last_started = True
            self.num_defects_sls[p1, p2] += action['action_defect'][p1]
            self.num_defects_sls[p2, p1] += action['action_defect'][p2]
            self.num_coops_sls[p1, p2] += 1 - action['action_defect'][p1]
            self.num_coops_sls[p2, p1] += 1 - action['action_defect'][p2]

        self.previous_action = action['action_defect'].copy()
        self.previous_action[~self._youre_playing] = -1  # if you weren't playing don't give info on what you chose
        self.previous_action_while_playing[self._youre_playing] = action['action_defect'][self._youre_playing].copy()

        rew = np.zeros(self.n_agents)
        rew[[p1, p2]] = self.payoff_matrix[action['action_defect'][p1], action['action_defect'][p2]]
        assert np.all(rew[~self._youre_playing] == 0)

        # Calling step will update the next players, so update this after computing reward.
        self._youre_playing = np.squeeze(obs['youre_playing'].copy(), -1)

        if done:
            info.update({f'actor{i}_n_coops': n_coops for i, n_coops in enumerate(np.sum(self.num_coops, 1))})
            info.update({f'actor{i}_n_defects': n_defects for i, n_defects in enumerate(np.sum(self.num_defects, 1))})

            # Compute fraction of actions that were defects against
            #    (a) all agents as compared to the last agent, i.e. the difference in these fractions
            #    (b) the last agent
            # We compute these statistics because for evaluation the last agent may be scripted, and so they are useful
            #   in comparing e.g. the difference in fraction of defects against an all-defect versus all-cooperate policy
            num_actions = self.num_coops + self.num_defects
            frac_defects_against_each_other = np.sum(self.num_defects[:-1, :-1]) / np.sum(num_actions[:-1, :-1])
            frac_defects_against_last = np.sum(self.num_defects[:-1, -1]) / np.sum(num_actions[:-1, -1])
            info.update({'frac_defects_all_minus_last': frac_defects_against_each_other})
            info.update({'frac_defects_against_last': frac_defects_against_last})

            # In the Prior Rapport setting (see paper), we want to measure the fraction of defects against the last agent
            #   AFTER the last agent has started acting, which is after the period in which the other agents have been able to
            #   gain rapport.
            num_actions_sls = self.num_coops_sls + self.num_defects_sls
            frac_defects_against_each_other_sls = np.sum(self.num_defects_sls[:-1, :-1]) / np.sum(num_actions_sls[:-1, :-1])
            frac_defects_against_last_sls = np.sum(self.num_defects_sls[:-1, -1]) / np.sum(num_actions_sls[:-1, -1])
            if not (np.isnan(frac_defects_against_each_other_sls) or np.isnan(frac_defects_against_last_sls)):
                info.update({'frac_defects_all_minus_last_sls': frac_defects_against_each_other_sls})
                info.update({'frac_defects_against_last_sls': frac_defects_against_last_sls})
        return self.observation(obs), rew, done, info

    def observation(self, obs):
        obs['prev_ac'] = self.previous_action[:, None]
        obs['prev_ac_while_playing'] = self.previous_action_while_playing[:, None]
        return obs


class ChooseAgentsToPlay(gym.Wrapper):
    '''
        Pick which 2 agents will play on each timestep.
        Args:
            last_step_first_agent_vs_last_agent (bool): On the last step of the game, the first and last agent will play
            last_agent_always_plays (bool): Last agent will play every round
            last_doesnt_play_until_t (int): The last agent will only get to play after this amount of rounds
            last_must_play_at_t (int): Only active when last_doesnt_play_until_t is not None,
                but makes the last agent play at that round instead of just including them as an option to play at that round
    '''
    @store_args
    def __init__(self, env,
                 last_step_first_agent_vs_last_agent: bool,
                 last_agent_always_plays: bool,
                 last_doesnt_play_until_t: int = None,
                 last_must_play_at_t: bool = False):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.observation_space = update_obs_space(self, {'you_played': [self.n_agents, 1],
                                                         'youre_playing': [self.n_agents, 1]})

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._t = obs['timestep'].copy()[0, 0]  # Comes from RandomizedHorizonWrapper

        self._you_played = self._youre_playing.copy()
        self._sample_new_players()
        return self.observation(obs), rew, done, info

    def reset(self):
        obs = self.env.reset()
        self._t = obs['timestep'].copy()[0, 0]
        self._horizon = obs['horizon'].copy()[0, 0]
        self._you_played = np.zeros(self.n_agents).astype(bool)
        self._sample_new_players()
        return self.observation(obs)

    def observation(self, obs):
        # We call observation after reseting players, so the mask should be for the current teams.
        obs['you_played'] = self._you_played[:, None]
        obs['youre_playing'] = self._youre_playing[:, None]
        return obs

    def _sample_new_players(self):
        exclude_first = self.last_step_first_agent_vs_last_agent and self._t < self._horizon - 1
        must_include_first_last = (self.last_step_first_agent_vs_last_agent and self._t == self._horizon - 1)
        exclude_last = (self.last_doesnt_play_until_t is not None and self._t < self.last_doesnt_play_until_t)

        p1_options = np.arange(self.n_agents)
        p2_options = np.arange(self.n_agents)

        if self.last_agent_always_plays and not exclude_last:
            p2_options = np.array([self.n_agents - 1])
            p1_options = p1_options[p1_options != p2_options]
        if exclude_last:
            p1_options = p1_options[p1_options != self.n_agents - 1]
            p2_options = p2_options[p2_options != self.n_agents - 1]
        if must_include_first_last:
            p1_options = np.array([0])
            p2_options = np.array([self.n_agents - 1])
        if exclude_first:
            p1_options = p1_options[p1_options != 0]
            p2_options = p2_options[p2_options != 0]
        if self.last_doesnt_play_until_t is not None and self.last_doesnt_play_until_t == self._t and self.last_must_play_at_t:
            p1_options = p1_options[p1_options != self.n_agents - 1]
            p2_options = np.array([self.n_agents - 1])

        p1 = np.random.choice(p1_options)
        p2_options = p2_options[p2_options != p1]
        p2 = np.random.choice(p2_options)

        self._youre_playing = np.zeros((self.n_agents,), dtype=bool)
        self._youre_playing[[p1, p2]] = 1


def make_env(n_agents=3,
             # Horizon
             horizon=20, horizon_lower=None, horizon_upper=None,
             prob_per_step_to_stop=0.05,
             # Matrix Payouts
             mutual_cooperate=2, defected_against=-2, successful_defect=4, mutual_defect=0,
             # Agent IDs
             agent_identity_dim=16,
             # Evals
             against_all_c=False, against_all_d=False, against_tft=False,
             last_step_first_agent_vs_last_agent=False, last_agent_always_plays=False,
             last_doesnt_play_until_t=None,
             last_must_play_at_t=False,
             # RUSP
             rusp_args={}):
    env = AbstractBaseEnv(n_agents)

    env = RandomizedHorizonWrapper(env, lower_lim=horizon_lower or horizon, upper_lim=horizon_upper or horizon,
                                   prob_per_step_to_stop=prob_per_step_to_stop)

    env = RandomIdentityVector(env, vector_dim=agent_identity_dim)

    env = ChooseAgentsToPlay(env, last_step_first_agent_vs_last_agent=last_step_first_agent_vs_last_agent,
                             last_agent_always_plays=last_agent_always_plays,
                             last_doesnt_play_until_t=last_doesnt_play_until_t,
                             last_must_play_at_t=last_must_play_at_t)

    # Construct Payoff Matrix
    cc = [mutual_cooperate, mutual_cooperate]
    cd = [defected_against, successful_defect]
    dc = list(reversed(cd))
    dd = [mutual_defect, mutual_defect]
    payoff_matrix = np.array([[cc, cd],
                              [dc, dd]])
    env = MultiPlayerIteratedMatrixGame(env, payoff_matrix=payoff_matrix)

    env = RUSPWrapper(env, **rusp_args)

    keys_self = ['prev_ac',
                 'you_played',
                 'youre_playing',
                 'agent_identity',
                 'timestep']
    keys_additional_self_vf = ['fraction_episode_done', 'horizon']

    keys_other_agents = ['prev_ac', 'youre_playing', 'agent_identity']

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

    keys_copy = []

    env = SplitObservations(env, keys_self + keys_additional_self_vf,
                            keys_copy=keys_copy, keys_self_matrices=keys_self_matrices)

    env = ConcatenateObsWrapper(env, {'other_agents': keys_other_agents,
                                      'other_agents_vf': ['other_agents'] + keys_additional_other_agents_vf,
                                      'additional_self_vf_obs': [k + '_self' for k in keys_additional_self_vf]})

    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external + keys_copy + ['youre_playing_self'])  # need to copy youre_playing_self through for the LastAgentScripted wrapper

    if against_all_c or against_all_d or against_tft:
        if against_all_c:
            policy_to_play = 'allc'
        elif against_all_d:
            policy_to_play = 'alld'
        elif against_tft:
            policy_to_play = 'tft'
        env = LastAgentScripted(env, policy_to_play)
    env = MaskYourePlaying(env)

    return env
