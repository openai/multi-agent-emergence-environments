import gym
import numpy as np
from mae_envs.wrappers.util import update_obs_space
from gym.spaces import Tuple, MultiDiscrete


class FoodHealthWrapper(gym.Wrapper):
    '''
        Adds food health to underlying env.
        Manages food levels.

        Args:
            eat_thresh (float): radius within which food items can be eaten
            max_food_health (int): number of times a food item can be eaten
                                   before it disappears
            respawn_time (int): Number of time steps after which food items
                                that have been eaten reappear
            food_rew_type (string): can be
                'selfish': each agent gets an inividual reward for the food they eat
                'joint_mean': food rewards are averaged over teams
            reward_scale (float or (float, float)): scales the reward by this amount. If tuple of
                floats, the exact reward scaling is uniformly sampled from
                (reward_scale[0], reward_scale[1]) at the beginning of every episode.
            reward_scale_obs (bool): If true, adds the reward scale for the current
                episode to food_obs
    '''
    def __init__(self, env, eat_thresh=0.5, max_food_health=10, respawn_time=np.inf,
                 food_rew_type='selfish', reward_scale=1.0, reward_scale_obs=False):
        super().__init__(env)
        self.eat_thresh = eat_thresh
        self.max_food_health = max_food_health
        self.respawn_time = respawn_time
        self.food_rew_type = food_rew_type
        self.n_agents = self.metadata['n_agents']

        if type(reward_scale) not in [list, tuple, np.ndarray]:
            reward_scale = [reward_scale, reward_scale]
        self.reward_scale = reward_scale
        self.reward_scale_obs = reward_scale_obs

        # Reset obs/action space to match
        self.max_n_food = self.metadata['max_n_food']
        self.curr_n_food = self.metadata['curr_n_food']
        self.max_food_size = self.metadata['food_size']
        food_dim = 5 if self.reward_scale_obs else 4
        self.observation_space = update_obs_space(self.env, {'food_obs': (self.max_n_food, food_dim),
                                                             'food_health': (self.max_n_food, 1),
                                                             'food_eat': (self.max_n_food, 1)})
        self.action_space.spaces['action_eat_food'] = Tuple([MultiDiscrete([2] * self.max_n_food)
                                                             for _ in range(self.n_agents)])

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        # Reset obs/action space to match
        self.curr_n_food = self.metadata['curr_n_food']

        self.food_site_ids = np.array([sim.model.site_name2id(f'food{i}')
                                       for i in range(self.curr_n_food)])
        # Reset food healths
        self.food_healths = np.ones((self.curr_n_food, 1)) * self.max_food_health
        self.eat_per_food = np.zeros((self.curr_n_food, 1))

        # Reset food size
        self.respawn_counters = np.zeros((self.curr_n_food,))

        self.curr_reward_scale = np.random.uniform(self.reward_scale[0], self.reward_scale[1])

        return self.observation(obs)

    def observation(self, obs):
        # Add food position and healths to obersvations
        food_pos = obs['food_pos']
        obs['food_health'] = self.food_healths
        obs['food_obs'] = np.concatenate([food_pos, self.food_healths], 1)
        if self.reward_scale_obs:
            obs['food_obs'] = np.concatenate([obs['food_obs'], np.ones((self.curr_n_food, 1)) * self.curr_reward_scale], 1)
        obs['food_eat'] = self.eat_per_food
        return obs

    def step(self, action):
        action_eat_food = action.pop('action_eat_food')
        obs, rew, done, info = self.env.step(action)

        if self.curr_n_food > 0:
            # Eat food that is close enough
            dist_to_food = np.linalg.norm(obs['agent_pos'][:, None] - obs['food_pos'][None], axis=-1)
            eat = np.logical_and(dist_to_food < self.eat_thresh, self.food_healths.T > 0)
            eat = np.logical_and(eat, action_eat_food).astype(np.float32)
            eat_per_food = np.sum(eat, 0)

            # Make sure that all agents can't have the last bite of food.
            # At that point, food is split evenly
            over_eat = self.food_healths[:, 0] < eat_per_food
            eat[:, over_eat] *= (self.food_healths[over_eat, 0] / eat_per_food[over_eat])
            eat_per_food = np.sum(eat, 0)
            self.eat_per_food = eat_per_food[:, None]

            # Update food healths and sizes
            self.food_healths -= eat_per_food[:, None]
            health_diff = eat_per_food[:, None]
            size_diff = health_diff * (self.max_food_size / self.max_food_health)
            size = self.unwrapped.sim.model.site_size[self.food_site_ids] - size_diff
            size = np.maximum(0, size)
            self.unwrapped.sim.model.site_size[self.food_site_ids] = size

            self.food_healths[self.respawn_counters == self.respawn_time] = self.max_food_health
            self.unwrapped.sim.model.site_size[self.food_site_ids[self.respawn_counters == self.respawn_time]] = self.max_food_size
            self.respawn_counters[self.food_healths[:, 0] == 0] += 1
            self.respawn_counters[self.food_healths[:, 0] != 0] = 0

            assert np.all(self.food_healths >= 0), \
                f"There is a food health below 0: {self.food_healths}"

            # calculate food reward
            if self.food_rew_type == 'selfish':
                food_rew = np.sum(eat, axis=1)
            elif self.food_rew_type == 'joint_mean':
                food_rew = np.sum(eat, axis=1)
                team_index = self.metadata['team_index']
                for team_index_number in np.unique(team_index):
                    food_rew[team_index == team_index_number] = np.mean(food_rew[team_index == team_index_number])
            else:
                raise ValueError(f"Food reward type {self.food_rew_type} unknown.")
        else:
            food_rew = 0.0

        rew += food_rew * self.curr_reward_scale
        return self.observation(obs), rew, done, info


class AlwaysEatWrapper(gym.ActionWrapper):
    '''
        Remove eat action and replace it with always eating.
        Args:
            agent_idx_allowed (ndarray): indicies of agents allowed to eat.
    '''
    def __init__(self, env, agent_idx_allowed):
        super().__init__(env)
        self.action_space.spaces.pop('action_eat_food')
        self.agent_idx_allowed = agent_idx_allowed

    def action(self, action):
        action['action_eat_food'] = np.zeros((self.metadata['n_agents'], self.metadata['curr_n_food']))
        action['action_eat_food'][self.agent_idx_allowed] = 1.
        return action
