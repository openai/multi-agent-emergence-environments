import gym
import numpy as np


class RestrictAgentsRect(gym.RewardWrapper):
    '''
        Give subset of agents negative reward if they leave a given area
        Args:
            restrict_rect (list of four floats): coordinates of rectangle
                defined as [x_min, y_min, x_max, y_max]
            reward_scale (float): reward for going out of bounds is -reward_scale at each
                timestep
            penalize_objects_out (bool): If true, penalizes all agents whenever an object is
                outside the specified area.
    '''
    def __init__(self, env, restrict_rect, reward_scale=10., penalize_objects_out=False):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.restrict_rect = np.array(restrict_rect)
        self.reward_scale = reward_scale
        self.penalize_objects_out = penalize_objects_out

        assert len(self.restrict_rect) == 4, \
            "Restriction rectangle must be of format [x_min, y_min, x_max, y_max]"

        self.rect_middle = 0.5 * np.array([restrict_rect[0] + restrict_rect[2],
                                           restrict_rect[1] + restrict_rect[3]])

        self.rect_size = np.array([restrict_rect[2] - restrict_rect[0],
                                   restrict_rect[3] - restrict_rect[1]])

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        self.agent_body_idxs = np.array([sim.model.body_name2id(f"agent{i}:particle")
                                         for i in range(self.n_agents)])
        if self.penalize_objects_out:
            obj_body_idxs = ([sim.model.body_name2id(f'moveable_box{i}') for i in np.where(self.metadata['curr_n_boxes'])[0]] +
                             [sim.model.body_name2id(f'ramp{i}:ramp') for i in np.where(self.metadata['curr_n_ramps'])[0]])
            self.obj_body_idxs = np.array(obj_body_idxs)

        return obs

    def reward(self, reward):
        sim = self.unwrapped.sim
        agent_pos = sim.data.body_xpos[self.agent_body_idxs, :2]
        outside_rect = np.any(np.abs(agent_pos - self.rect_middle) > (self.rect_size / 2), axis=1)
        if self.penalize_objects_out:
            obj_pos = sim.data.body_xpos[self.obj_body_idxs, :2]
            any_obj_outside_rect = np.any(np.abs(obj_pos - self.rect_middle) > (self.rect_size / 2))
            if any_obj_outside_rect:
                reward[:] = - self.reward_scale
        reward[outside_rect] = - self.reward_scale

        return reward
