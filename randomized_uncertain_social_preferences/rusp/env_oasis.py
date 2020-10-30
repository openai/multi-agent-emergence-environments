import gym
import logging
import numpy as np
from collections import OrderedDict
from mae_envs.envs.base import Base
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions,
                                           SplitObservations, SelectKeysWrapper)
from mae_envs.wrappers.util import (DiscretizeActionWrapper, DiscardMujocoExceptionEpisodes,
                                    ConcatenateObsWrapper)
from mae_envs.wrappers.food import FoodHealthWrapper, AlwaysEatWrapper
from mae_envs.modules.agents import Agents
from mae_envs.modules.food import Food
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.util import uniform_placement, close_to_other_object_placement
from mae_envs.wrappers.line_of_sight import (AgentAgentObsMask2D)
from mae_envs.wrappers.util import update_obs_space
from mujoco_worldgen.util.types import store_args
from rusp.wrappers_rusp import RUSPWrapper, add_rew_share_observation_keys
from rusp.wrappers_util import RandomIdentityVector, RandomizedHorizonWrapper, OtherActorAttentionAction, ActionOptionsWrapper


def zero_action(ac_space):
    '''
        Define default zero action for when an agent dies such that it stays in place and doesn't do anything.
    '''
    ac = OrderedDict()
    for ac_key, s in ac_space.spaces.items():
        assert isinstance(s, gym.spaces.Tuple), f"space {s} is not a Tuple"
        single_agent_space = s.spaces[0]
        if isinstance(single_agent_space, gym.spaces.Box):
            ac[ac_key] = np.zeros_like(s.sample())
        elif isinstance(single_agent_space, gym.spaces.Discrete):
            ac[ac_key] = np.ones_like(s.sample()) * (single_agent_space.n // 2)
        elif isinstance(single_agent_space, gym.spaces.MultiDiscrete):
            ac[ac_key] = np.ones_like(s.sample(), dtype=int) * (single_agent_space.nvec // 2)
        else:
            raise NotImplementedError("MultiDiscrete not NotImplementedError")
    return ac


class ZeroRews(gym.Wrapper):
    '''
        Change reward to a vector such that downstream wrappers do not need to check if it
            is already a vector.
    '''
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, np.zeros((self.metadata['n_agents'],)), done, info


class OasisActionMasks(gym.ObservationWrapper):
    '''
        Construct masks for all actions in Oasis such that the policy gradient can be masked when
            an agent dies. Unfortunately training code is not being released, but you can implement this
            simply by setting the logprob of your policy to a large negative number.
    '''
    @store_args
    def __init__(self, env, mask_all_when_dead=True):
        super().__init__(env)
        self.observation_space.spaces['mask'] = gym.spaces.Dict({
            'action_movement': gym.spaces.Box(-np.inf, np.inf, (self.metadata['n_actors'], 11), float),
            'action_attack_agent': gym.spaces.Box(-np.inf, np.inf, (self.metadata['n_actors'], self.metadata['n_actors'] - 1), float),
            'action_choose_option': gym.spaces.Box(-np.inf, np.inf, (self.metadata['n_actors'], 3), float),
        })

    def observation(self, obs):
        obs['mask'] = {
            'action_movement': np.ones((self.metadata['n_actors'], 3, 11), dtype=bool),
            'action_attack_agent': np.ones((self.metadata['n_actors'], self.metadata['n_actors'] - 1), dtype=bool),
            'action_choose_option': np.ones((self.metadata['n_actors'], 3), dtype=bool),
        }
        if self.mask_all_when_dead:
            obs['mask']['action_movement'] *= (1 - obs['mask_is_dead'])[..., None].astype(bool)
            obs['mask']['action_attack_agent'] *= (1 - obs['mask_is_dead']).astype(bool)
            obs['mask']['action_choose_option'] *= (1 - obs['mask_is_dead']).astype(bool)
        return obs


class AgentHealthWrapper(gym.Wrapper):
    '''
        Manages agent health and death. This wrapper keeps track of each agent's health. Upstream wrappers
            can update agent health by assigning "health_delta" in the info dict. health_delta will be a (n_agents) shaped
            array. If an agent dies (health <= 0) it will be sent to the edge of the play area and held in place for a number
            of steps at which point it may re-enter play.

        Args:
            max_health (float): maximum health an agent can reach. Agents are spawned with maximum health.
            death_rew (float): reward for dying (this should be negative)
            steps_freeze_on_death (int): number of timesteps
            life_rew (float): reward given to an agent on each step for being alive. What other reward is there than life itself!

        Observations:
            agent_health (n_agents, 1): current health
            is_dead (n_agents, 1): boolean indicating if an agent is dead or alive
            time_to_alive (n_agents, 1): number of timesteps until each agent is alive again (0 if they are currently alive)
    '''
    @store_args
    def __init__(self, env, max_health=20, death_rew=-100, steps_freeze_on_death=100, life_rew=1):
        super().__init__(env)
        assert death_rew <= 0
        assert life_rew >= 0
        assert steps_freeze_on_death >= 0
        self.observation_space = update_obs_space(self, {
            'agent_health': [self.metadata['n_agents'], 1],
            'is_dead': [self.metadata['n_agents'], 1],
            'time_to_alive': [self.metadata['n_agents'], 1]
        })
        self.zero_action = zero_action(self.action_space)
        logging.info(f"Only {self.zero_action.keys()} will be zerod during death")

    def reset(self):
        self.healths = np.ones(self.metadata['n_agents']) * self.max_health
        self.time_since_death = np.ones(self.metadata['n_agents']) * np.inf
        self.is_dead = np.zeros(self.metadata['n_agents'])
        self.agent_died_count = np.zeros(self.metadata['n_agents'])
        return self.observation(self.env.reset())

    def step(self, action):
        self.is_dead = np.logical_or(self.healths <= 0, self.time_since_death < self.steps_freeze_on_death)
        # If an agent just died, its health will be <= 0, update position and health
        for i in np.where(self.healths <= 0)[0]:
            x_ind = self.unwrapped.sim.model.joint_name2id(f'agent{i}:tx')
            y_ind = self.unwrapped.sim.model.joint_name2id(f'agent{i}:ty')
            fs = self.unwrapped.floor_size
            self.unwrapped.sim.data.qpos[x_ind] = np.random.choice([np.random.uniform(-1, 0), np.random.uniform(fs, fs + 1)])
            self.unwrapped.sim.data.qpos[y_ind] = np.random.choice([np.random.uniform(-1, 0), np.random.uniform(fs, fs + 1)])
            self.healths[i] = self.max_health
            self.time_since_death[i] = 0
            self.agent_died_count[i] += 1
        self.unwrapped.sim.forward()  # Forward the sim so their position gets updated sooner

        # Zero out actions for all dead agents
        if np.any(self.is_dead):
            for ac_key, ac in action.items():
                ac[self.is_dead] = self.zero_action[ac_key][self.is_dead]

        obs, rew, done, info = self.env.step(action)

        # Update healths
        self.healths[~self.is_dead] += info['health_delta'][~self.is_dead]  # only change health of alive agents
        self.healths = np.minimum(self.healths, self.max_health)
        self.time_since_death += 1

        rew[self.healths <= 0] += self.death_rew

        # Reward for living
        rew[~self.is_dead] += self.life_rew

        # Done stats
        if done:
            info['n_unique_died'] = np.sum(self.agent_died_count > 0)
            info['only_one_died'] = (np.sum(self.agent_died_count > 0) == 1)
            info['n_died'] = np.sum(self.agent_died_count)
            info['n_died_min'] = np.min(self.agent_died_count)
            info['n_died_max'] = np.max(self.agent_died_count)
            info['n_died_std'] = np.std(self.agent_died_count)
            info['n_died_total_minus_max'] = np.sum(self.agent_died_count) - np.max(self.agent_died_count)

        return self.observation(obs), rew, done, info

    def observation(self, obs):
        obs['agent_health'] = self.healths[:, None]
        obs['is_dead'] = self.is_dead[:, None]
        obs['mask_is_dead'] = self.is_dead[:, None].astype(bool)
        obs['time_to_alive'] = np.minimum(1, self.time_since_death / self.steps_freeze_on_death)[:, None]
        return obs


class FoodIncreaseHealth(gym.Wrapper):
    '''
        Adds a positive health_delta if the agent ate food (eating logic found in mae_envs.wrappers.food).

        Args:
            health_per_food_bounds ([float, float]): health gained per food eaten is randomized per episode, sampled
                uniformly within this bound.
    '''
    def __init__(self, env, health_per_food_bounds):
        super().__init__(env)
        self.health_per_food_bounds = health_per_food_bounds

    def reset(self):
        self.health_per_food = np.random.uniform(self.health_per_food_bounds[0], self.health_per_food_bounds[1])
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if 'health_delta' not in info:
            info['health_delta'] = np.zeros((self.metadata['n_agents']))

        info['health_delta'] += np.sum(info['agents_eat'], 1) * self.health_per_food
        info['total_health_gained_from_food'] = np.sum(info['agents_eat']) * self.health_per_food

        return obs, rew, done, info


class TimeDecreaseHealth(gym.Wrapper):
    '''
        Decrease agent health by a constant amount every timestep.

        Args:
            health_per_step (float):amount to decrease health every step.
    '''
    @store_args
    def __init__(self, env, health_per_step=-1):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if 'health_delta' not in info:
            info['health_delta'] = np.zeros((self.metadata['n_agents']))

        info['health_delta'] += self.health_per_step

        return obs, rew, done, info


class AttackAction(OtherActorAttentionAction):
    '''
        Attack action wrapper. Agents may attack other agents that are within range. Attacking causes
            the targeted agent to lose health (attack damage). If an agent is attacked, we stop them from
            eating that round.
        Args:
            attack_damage (float): amount of health gained from an attack -- must be a negative value.
            attack_range (float): maximum distance a target can be for an agent to attack them.
            mask_eat_if_attacked (bool): if True, an attacked agent will be disallowed to eat on the timestep
                they are attacked
            only_attack_in_front (bool): if True, agents can only attack when another agent is in their "field of view"

        Observations:
            attacked_me (n_agents, n_agents, 1): each row is an agent's binary observation of which other agent attacked them
            n_attacked_me (n_agents, 1): this is 'attacked_me' summed over the second dimension -- the total number of agents
                that attacked each agent.
    '''
    @store_args
    def __init__(self, env, attack_damage=-5, attack_range=0.7, mask_eat_if_attacked=True,
                 only_attack_in_front=True):
        assert attack_damage <= 0
        assert attack_range >= 0
        super().__init__(env, 'action_attack_agent')
        self.observation_space = update_obs_space(self, {
            'attacked_me': [self.n_agents, self.n_agents, 1],
            'n_attacked_me': [self.n_agents, 1]
        })

    def reset(self):
        self.attack_counts = np.zeros(self.n_agents)
        self.attacked_me = np.zeros((self.n_agents, self.n_agents))
        self.previous_obs = self.observation(self.env.reset())
        return self.previous_obs

    def step(self, action):
        attack_matrix = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        for i in range(self.n_agents):
            target_actors = self._get_target_actor(i, action)
            if len(target_actors):
                # See if the targeted agent can be attacked (in range and in front)
                aa_ranges = np.linalg.norm(self.previous_obs['agent_pos'][i] - self.previous_obs['agent_pos'][target_actors], axis=1)
                in_range = aa_ranges < self.attack_range
                in_front = self.previous_obs['mask_aa_obs'][i, target_actors]
                able_to_attack = np.logical_and(in_range, in_front) if self.only_attack_in_front else in_range
                if np.any(able_to_attack):
                    # Filter down to those that are in range and in front
                    target_actors = target_actors[able_to_attack]
                    aa_ranges = aa_ranges[able_to_attack]
                    # Only attack the closest agent to you
                    target_actor = target_actors[np.argsort(aa_ranges)[0]]
                    attack_matrix[i, target_actor] = 1
        self.attacked_me = attack_matrix.T
        self.attack_counts += np.sum(attack_matrix, 1)

        # Compute health updates
        health_deltas = np.zeros((self.n_agents, self.n_agents))
        health_deltas[self.attacked_me] += self.attack_damage

        health_deltas = np.sum(health_deltas, 1)

        # Turn off the eat action if you were attacked
        if self.mask_eat_if_attacked:
            action['action_eat_food'] *= ~np.any(self.attacked_me, 1, keepdims=True)

        obs, rew, done, info = self.env.step(action)
        info['health_delta'] += health_deltas
        self.previous_obs = self.observation(obs)

        if done:
            info['n_attacks'] = np.sum(self.attack_counts)
            info['n_attacks_per_agent'] = np.mean(self.attack_counts)

        return self.previous_obs, rew, done, info

    def observation(self, obs):
        obs['attacked_me'] = self.attacked_me[:, :, None]
        obs['n_attacked_me'] = np.sum(self.attacked_me, 1, keepdims=True)
        return obs


class ColorAgentsByOption(gym.Wrapper):
    '''
        Purely for visualization purposes. Colors agents red if they are attacking,
            green if they are eating, and blue if they are doing neither.
    '''
    @store_args
    def __init__(self, env, action_key, options_list):
        super().__init__(env)
        self.colors = {
            'action_attack_agent': np.array([1., 0, 0, 1.0]),
            'action_eat_food': np.array([0, 1., 0, 1.0]),
            'do_nothing': np.array([0, 1, 1, 1]),
        }

    def step(self, action):
        for i in range(self.unwrapped.n_agents):
            self._color_agent(self.options_list[action[self.action_key][i]], i)
        return self.env.step(action)

    def _color_agent(self, ac_name, agent_ind):
        sim = self.unwrapped.sim
        geom_ind = sim.model.geom_name2id(f'agent{agent_ind}:agent')
        sim.model.geom_rgba[geom_ind] = self.colors[ac_name]


def make_env(n_substeps=15, n_agents=3,
             floor_size=[1.5, 6], action_lims=(-0.9, 0.9),
             grid_size=60, other_friction=0.01, box_floor_friction=0.2, gravity=[0, 0, -50],
             horizon=1000, horizon_lower=None, horizon_upper=None, prob_per_step_to_stop=0.001,
             # Food
             n_food=1, n_food_cluster=1, food_radius=0.4,
             food_respawn_time=0, max_food_health=5, food_together_radius=0.4,
             food_rew_type='selfish', food_reward_scale=0.0,
             # Health
             max_agent_health=20, health_per_food_bounds=[2.1, 2.7], health_per_step=-1.0,
             # Attacking
             attack_range=0.7, attack_damage=-5.0, only_attack_in_front=True,
             # Death
             life_rew=1, death_rew=-100, steps_freeze_on_death=100,
             # Random Teams
             rusp_args={},
             # ID
             id_dim=16,
             # Action Masking
             mask_all_when_dead=True):
    env = Base(n_agents=n_agents,
               n_substeps=n_substeps,
               floor_size=floor_size,
               horizon=99999999999999,  # Just a big number so actual horizon is done by RandomizedHorizonWrapper
               action_lims=action_lims,
               deterministic_mode=False,
               grid_size=grid_size)
    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))
    env.add_module(WorldConstants(gravity=gravity))

    env.add_module(Agents(n_agents,
                          placement_fn=uniform_placement,
                          friction=other_friction))

    # Food
    env.metadata['food_together_radius'] = food_together_radius

    assert n_food % n_food_cluster == 0
    cluster_assignments = np.repeat(np.arange(0, n_food, n_food // n_food_cluster), n_food // n_food_cluster)
    food_placement = [close_to_other_object_placement(
        "food", i, "food_together_radius") for i in cluster_assignments]
    food_placement[::n_food // n_food_cluster] = [uniform_placement] * n_food_cluster

    env.add_module(Food(n_food, placement_fn=food_placement))

    env.reset()

    keys_self = [
        'agent_qpos_qvel',
        'agent_identity',
        'agent_health',
        'is_dead',
        'time_to_alive',
        'timestep'
    ]
    keys_additional_self_vf = ['fraction_episode_done', 'horizon']
    keys_copy = ['mask_is_dead']
    keys_other_agents = [
        'agent_qpos_qvel',
        'agent_identity',
        'agent_health',
        'is_dead',
        'time_to_alive',
    ]
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

    keys_self_masks = ['mask_aa_obs']

    env = SplitMultiAgentActions(env)
    env = DiscretizeActionWrapper(env, 'action_movement')
    env = AgentAgentObsMask2D(env)

    env = ZeroRews(env)

    env = RandomizedHorizonWrapper(env, lower_lim=horizon_lower or horizon, upper_lim=horizon_upper or horizon,
                                   prob_per_step_to_stop=prob_per_step_to_stop)

    env = FoodHealthWrapper(env, respawn_time=(np.inf if food_respawn_time is None else food_respawn_time),
                            eat_thresh=(np.inf if food_radius is None else food_radius),
                            max_food_health=max_food_health, food_rew_type=food_rew_type,
                            reward_scale=food_reward_scale, split_eat_between_agents=True)
    keys_external += ['mask_af_obs', 'food_obs']
    keys_copy.append('close_enough_to_food')

    env = FoodIncreaseHealth(env, health_per_food_bounds=health_per_food_bounds)
    env = TimeDecreaseHealth(env, health_per_step=health_per_step)

    # Attack action should go before Food Health wrapper, since it masks eat action
    env = AttackAction(env, attack_damage=attack_damage, attack_range=attack_range,
                       only_attack_in_front=only_attack_in_front)
    env = ActionOptionsWrapper(env, ['action_attack_agent', 'action_eat_food'], {'action_attack_agent': -1, 'action_eat_food': 0})
    env = ColorAgentsByOption(env, 'action_choose_option', ['action_attack_agent', 'action_eat_food', 'do_nothing'])
    keys_self.append('previous_choice')
    keys_other_agents.append('previous_choice')
    keys_self_matrices.append('attacked_me')
    keys_self.append('n_attacked_me')
    keys_other_agents += ['attacked_me', 'n_attacked_me']

    env = AgentHealthWrapper(env, max_health=max_agent_health, death_rew=death_rew,
                             steps_freeze_on_death=steps_freeze_on_death, life_rew=life_rew)

    # This needs to come before options wrapper, so we can't group it above
    env = AlwaysEatWrapper(env, agent_idx_allowed=np.arange(n_agents))

    env = RUSPWrapper(env, **rusp_args)

    env = RandomIdentityVector(env, vector_dim=id_dim)

    env = SplitObservations(env, keys_self + keys_additional_self_vf,
                            keys_copy=keys_copy, keys_self_matrices=keys_self_matrices + keys_self_masks)
    env = ConcatenateObsWrapper(env, {'other_agents': keys_other_agents,
                                      'other_agents_vf': ['other_agents'] + keys_additional_other_agents_vf,
                                      'additional_self_vf_obs': [k + '_self' for k in keys_additional_self_vf]})
    env = DiscardMujocoExceptionEpisodes(env)
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external + keys_copy + keys_self_masks)
    env = OasisActionMasks(env, mask_all_when_dead=mask_all_when_dead)
    return env
