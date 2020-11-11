import numpy as np
from collections import defaultdict
from mujoco_worldgen.util.types import store_args
from mae_envs.wrappers.util import update_obs_space
from mae_envs.wrappers.util import ConcatenateObsWrapper
from mae_envs.wrappers.multi_agent import (SplitObservations, SelectKeysWrapper)
from rusp.wrappers_rusp import RUSPWrapper, add_rew_share_observation_keys
from rusp.wrappers_util import RandomIdentityVector, RandomizedHorizonWrapper, OtherActorAttentionAction, ActionOptionsWrapper
from rusp.abstract_base_env import AbstractBaseEnv


class PrisonersBuddy(OtherActorAttentionAction):
    '''
        Agents must mutually choose others to get reward (mutual_cooperate_rew). They can choose to sitout
            and receive zero reward. If they make an unreciprocated choice, the targeted agent will recieve a defection
            reward (successful_defect_rew). We call it a defection reward since they could reciprocated the choice.
            The agent making the unreciprocated choice receives a penalty (defected_against_rew).

            Agents get a chance to "communicate" in that their choices only elicit rewards every few timesteps (choosing_period).
            This gives them time to break symmetry.

        Observations:
            chose_me (n_agents, n_agents, 1): which other agents (column) chose me (row agent) last step
            i_chose (n_agents, n_agents, 1): which other agents (column) did I choose (row agent) last step
            chose_me_rew (n_agents, n_agents, 1): which other agents (column) chose me (row agent) last step reward was given
            i_chose_rew (n_agents, n_agents, 1): which other agents (column) did I choose (row agent) last step reward was given
            i_chose_any_rew (n_agents, 1): Did I choose to sitout or choose someone last timestep reward was given
            previous_choice_identity (n_agents, agent_identity_dim): ID of agent I previously chose
            next_choice_is_real (n_agents, 1): is the next timestep one in which reward will be given
    '''
    @store_args
    def __init__(self, env, choosing_period,
                 agent_identity_dim=4,
                 mutual_cooperate_rew=2,
                 defected_against_rew=-1,
                 successful_defect_rew=1):
        super().__init__(env, 'action_choose_agent')
        self.observation_space = update_obs_space(self, {
            'chose_me': [self.n_agents, self.n_agents, 1],
            'i_chose': [self.n_agents, self.n_agents, 1],
            'chose_me_rew': [self.n_agents, self.n_agents, 1],
            'i_chose_rew': [self.n_agents, self.n_agents, 1],
            'i_chose_any_rew': [self.n_agents, 1],
            'previous_choice_identity': [self.n_agents, agent_identity_dim],
            'next_choice_is_real': [self.n_agents, 1],
        })

    def reset(self):
        self._t = 1  # Start t at 1 such that first round is not a reward round
        self._chose_me = np.zeros((self.n_agents, self.n_agents))
        self._chose_me_rew = np.zeros((self.n_agents, self.n_agents))
        self._n_times_not_chosen = np.zeros((self.n_agents))
        self._n_times_team_changed = np.zeros((self.n_agents))
        self._n_agents_on_team = []
        self._previous_choice_identity = np.zeros((self.n_agents, self.agent_identity_dim))
        self._i_chose_any_rew_obs = np.zeros((self.n_agents, 1))
        self._team_lengths = []
        self._n_successful_defections = 0
        self._current_team_lengths = defaultdict(lambda: 0)
        self._previous_teams = np.ones(self.n_agents, dtype=int) * -1
        self._both_chose = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        self._perfect_game = True
        self._first_choice = True

        return self.observation(self.env.reset())

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._chose_me = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        targets = np.ones(self.n_agents, dtype=int) * -1
        for i in range(self.n_agents):
            target = self._get_target_actor(i, action)
            if len(target):
                targets[i] = target[0]
                self._chose_me[target[0], i] = 1

        self._previous_choice_identity = obs['agent_identity'][targets]
        self._previous_choice_identity[targets == -1] = 0

        # Reward rounds
        if self._t % self.choosing_period == 0:
            self._both_chose = self._chose_me * self._chose_me.T
            self._chose_me_rew = self._chose_me.copy()

            self._teams = np.argmax(self._both_chose, axis=1)  # Indicies of teamate
            self._teams[np.all(self._both_chose == 0, axis=1)] = -1  # Make sure those without team are set to -1 instead of 0

            rew = self._prisoners_buddy_reward_update(rew)

            # Track stats
            self._n_times_not_chosen[np.sum(self._chose_me, 1) == 0] += 1
            # Since both_chose is symmetric, just get the index of nonzero entry in upper triangle
            current_team_indices = np.c_[np.nonzero(np.triu(self._both_chose))]
            current_team_tuples = list(map(tuple, current_team_indices))
            teams_done = [k for k in self._current_team_lengths.keys() if k not in current_team_tuples]

            for team_done in teams_done:
                self._team_lengths.append(self._current_team_lengths[team_done])
                del self._current_team_lengths[team_done]
            for current_team_tuple in current_team_tuples:
                self._current_team_lengths[current_team_tuple] += 1

            self._i_chose_any_rew_obs = np.any(self._chose_me_rew, 0)[:, None]

            if self._first_choice:
                self._first_choice = False
            else:
                all_teams_didnt_change = np.all(self._previous_teams == self._teams)
                max_number_of_teams_filled = np.sum(self._teams != -1) == ((self.n_agents // 2) * 2)
                self._perfect_game = self._perfect_game and all_teams_didnt_change and max_number_of_teams_filled

            self._previous_teams = self._teams

        self._t += 1

        if done:
            self._team_lengths += list(self._current_team_lengths.values())
            info['average_team_length'] = np.mean(self._team_lengths) if len(self._team_lengths) else 0
            info['n_times_team_changed'] = np.sum(self._n_times_team_changed)
            info['n_agents_on_team_per_step'] = np.mean(self._n_agents_on_team)
            info['number_decisions'] = self._t / self.choosing_period
            info['n_unique_not_chosen'] = np.sum(self._n_times_not_chosen > 0)
            info['n_successful_defections'] = self._n_successful_defections
            info['perfect_game'] = self._perfect_game

        return self.observation(obs), rew, done, info

    def observation(self, obs):
        obs['chose_me'] = self._chose_me[:, :, None]
        obs['i_chose'] = self._chose_me.T[:, :, None]
        obs['chose_me_rew'] = self._chose_me_rew[:, :, None]
        obs['i_chose_rew'] = self._chose_me_rew.T[:, :, None]
        obs['i_chose_any_rew'] = self._i_chose_any_rew_obs
        obs['previous_choice_identity'] = self._previous_choice_identity
        # assumes this is called after t is increased
        obs['next_choice_is_real'] = np.ones((self.n_agents, 1)) if self._t % self.choosing_period == 0 else np.zeros((self.n_agents, 1))
        return obs

    def _prisoners_buddy_reward_update(self, rew):
        on_team = np.any(self._both_chose, axis=1)
        chose_me_oneway = (self._chose_me & ~self._both_chose)
        num_chose_me_oneway = np.sum(chose_me_oneway, axis=1)
        i_chose_one_way = np.any(chose_me_oneway, axis=0)

        assert np.all(np.sum(chose_me_oneway, axis=0) <= 1)
        assert np.all((i_chose_one_way & on_team) == 0)

        previous_has_team = (self._previous_teams != -1)
        your_team_changed = (self._teams != self._previous_teams)

        rew[on_team] += self.mutual_cooperate_rew
        rew[i_chose_one_way] += self.defected_against_rew
        rew += num_chose_me_oneway * self.successful_defect_rew

        # Stats
        self._n_successful_defections += np.sum(i_chose_one_way)
        self._n_times_team_changed += (previous_has_team & your_team_changed)
        self._n_agents_on_team.append(np.sum(on_team))

        return rew


def make_env(n_agents=5, horizon=50, horizon_lower=None, horizon_upper=None,
             prob_per_step_to_stop=0.02,
             choosing_period=5,
             mutual_cooperate_rew=2, defected_against_rew=-2, successful_defect_rew=1,
             agent_identity_dim=16,
             rusp_args={}):
    env = AbstractBaseEnv(n_agents)
    env = RandomizedHorizonWrapper(env, lower_lim=horizon_lower or horizon, upper_lim=horizon_upper or horizon,
                                   prob_per_step_to_stop=prob_per_step_to_stop)
    env = RandomIdentityVector(env, vector_dim=agent_identity_dim)

    env = PrisonersBuddy(env, choosing_period=choosing_period,
                         agent_identity_dim=agent_identity_dim,
                         mutual_cooperate_rew=mutual_cooperate_rew, defected_against_rew=defected_against_rew,
                         successful_defect_rew=successful_defect_rew)

    env = ActionOptionsWrapper(env, ['action_choose_agent'], {'action_choose_agent': -1})

    env = RUSPWrapper(env, **rusp_args)

    keys_self = ['previous_choice',
                 'next_choice_is_real',
                 'i_chose_any_rew',
                 'agent_identity',
                 'previous_choice_identity',
                 'timestep']
    keys_additional_self_vf = ['fraction_episode_done', 'horizon']

    keys_other_agents = [
        'previous_choice',
        'chose_me',
        'i_chose',
        'chose_me_rew',
        'i_chose_rew',
        'i_chose_any_rew',
        'agent_identity',
        'previous_choice_identity'
    ]
    keys_additional_other_agents_vf = []
    keys_self_matrices = ['chose_me',
                          'i_chose',
                          'chose_me_rew',
                          'i_chose_rew']

    keys_external = ['other_agents',
                     'other_agents_vf',
                     'additional_self_vf_obs']

    add_rew_share_observation_keys(keys_self=keys_self,
                                   keys_additional_self_vf=keys_additional_self_vf,
                                   keys_other_agents=keys_other_agents,
                                   keys_additional_other_agents_vf=keys_additional_other_agents_vf,
                                   keys_self_matrices=keys_self_matrices,
                                   **rusp_args)

    env = SplitObservations(env, keys_self + keys_additional_self_vf,
                            keys_copy=[], keys_self_matrices=keys_self_matrices)
    env = ConcatenateObsWrapper(env, {'other_agents': keys_other_agents,
                                      'other_agents_vf': ['other_agents'] + keys_additional_other_agents_vf,
                                      'additional_self_vf_obs': [k + '_self' for k in keys_additional_self_vf]})
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external)

    return env
