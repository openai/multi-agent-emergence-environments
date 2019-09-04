import numpy as np
import time
from mujoco_py import const, MjViewer
import glfw
from gym.spaces import Box, MultiDiscrete, Discrete


class EnvViewer(MjViewer):

    def __init__(self, env):
        self.env = env
        self.elapsed = [0]
        self.seed = self.env.seed()
        super().__init__(self.env.unwrapped.sim)
        self.n_agents = self.env.metadata['n_actors']
        self.action_types = list(self.env.action_space.spaces.keys())
        self.num_action_types = len(self.env.action_space.spaces)
        self.num_action = self.num_actions(self.env.action_space)
        self.agent_mod_index = 0
        self.action_mod_index = 0
        self.action_type_mod_index = 0
        self.action = self.zero_action(self.env.action_space)
        self.env_reset()

    def num_actions(self, ac_space):
        n_actions = []
        for k, tuple_space in ac_space.spaces.items():
            s = tuple_space.spaces[0]
            if isinstance(s, Box):
                n_actions.append(s.shape[0])
            elif isinstance(s, Discrete):
                n_actions.append(1)
            elif isinstance(s, MultiDiscrete):
                n_actions.append(s.nvec.shape[0])
            else:
                raise NotImplementedError(f"not NotImplementedError")

        return n_actions

    def zero_action(self, ac_space):
        ac = {}
        for k, space in ac_space.spaces.items():
            if isinstance(space.spaces[0], Box):
                ac[k] = np.zeros_like(space.sample())
            elif isinstance(space.spaces[0], Discrete):
                ac[k] = np.ones_like(space.sample()) * (space.spaces[0].n // 2)
            elif isinstance(space.spaces[0], MultiDiscrete):
                ac[k] = np.ones_like(space.sample(), dtype=int) * (space.spaces[0].nvec // 2)
            else:
                raise NotImplementedError("MultiDiscrete not NotImplementedError")
                # return action_space.nvec // 2  # assume middle element is "no action" action
        return ac

    def env_reset(self):
        start = time.time()
        # get the seed before calling env.reset(), so we display the one
        # that was used for the reset.
        self.seed = self.env.seed()
        self.env.reset()
        self.elapsed.append(time.time() - start)
        self.update_sim(self.env.unwrapped.sim)

    def key_callback(self, window, key, scancode, action, mods):
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        if key == glfw.KEY_ESCAPE:
            self.env.close()

        # Increment experiment seed
        elif key == glfw.KEY_N:
            self.seed[0] += 1
            self.env.seed(self.seed)
            self.env_reset()
            self.action = self.zero_action(self.env.action_space)
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            self.seed = [max(self.seed[0] - 1, 0)]
            self.env.seed(self.seed)
            self.env_reset()
            self.action = self.zero_action(self.env.action_space)
        current_action_space = self.env.action_space.spaces[self.action_types[self.action_type_mod_index]].spaces[0]
        if key == glfw.KEY_A:
            if isinstance(current_action_space, Box):
                self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index][self.action_mod_index] -= 0.05
            elif isinstance(current_action_space, Discrete):
                self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index] = \
                    (self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index] - 1) % current_action_space.n
            elif isinstance(current_action_space, MultiDiscrete):
                self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index][self.action_mod_index] = \
                    (self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index][self.action_mod_index] - 1) \
                    % current_action_space.nvec[self.action_mod_index]
        elif key == glfw.KEY_Z:
            if isinstance(current_action_space, Box):
                self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index][self.action_mod_index] += 0.05
            elif isinstance(current_action_space, Discrete):
                self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index] = \
                    (self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index] + 1) % current_action_space.n
            elif isinstance(current_action_space, MultiDiscrete):
                self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index][self.action_mod_index] = \
                    (self.action[self.action_types[self.action_type_mod_index]][self.agent_mod_index][self.action_mod_index] + 1) \
                    % current_action_space.nvec[self.action_mod_index]
        elif key == glfw.KEY_K:
            self.action_mod_index = (self.action_mod_index + 1) % self.num_action[self.action_type_mod_index]
        elif key == glfw.KEY_J:
            self.action_mod_index = (self.action_mod_index - 1) % self.num_action[self.action_type_mod_index]
        elif key == glfw.KEY_Y:
            self.agent_mod_index = (self.agent_mod_index + 1) % self.n_agents
        elif key == glfw.KEY_U:
            self.agent_mod_index = (self.agent_mod_index - 1) % self.n_agents
        elif key == glfw.KEY_G:
            self.action_type_mod_index = (self.action_type_mod_index + 1) % self.num_action_types
            self.action_mod_index = 0
        elif key == glfw.KEY_B:
            self.action_type_mod_index = (self.action_type_mod_index - 1) % self.num_action_types
            self.action_mod_index = 0

        super().key_callback(window, key, scancode, action, mods)

    def run(self, once=False):
        while True:
            _, _, _, env_info = self.env.step(self.action)
            if env_info.get('discard_episode', False):
                self.env.reset()
            self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
            self.add_overlay(const.GRID_TOPRIGHT, "Apply action", "A (-0.05) / Z (+0.05)")
            self.add_overlay(const.GRID_TOPRIGHT, "on agent index %d out %d" % (self.agent_mod_index, self.n_agents), "Y / U")
            self.add_overlay(const.GRID_TOPRIGHT, f"on action type {self.action_types[self.action_type_mod_index]}", "G / B")
            self.add_overlay(const.GRID_TOPRIGHT, "on action index %d out %d" % (self.action_mod_index, self.num_action[self.action_type_mod_index]), "J / K")
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Reset took", "%.2f sec." % (sum(self.elapsed) / len(self.elapsed)))
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Action", str(self.action))
            self.render()
            if once:
                return
