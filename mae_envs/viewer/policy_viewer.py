#!/usr/bin/env python
import time
import glfw
import numpy as np
from operator import itemgetter
from mujoco_py import const, MjViewer
from mujoco_worldgen.util.types import store_args
from ma_policy.util import listdict2dictnp


def splitobs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]


class PolicyViewer(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.total_rew = 0.0
        self.ob = env.reset()
        for policy in self.policies:
            policy.reset()
        assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        while self.duration is None or time.time() < self.end_time:
            if len(self.policies) == 1:
                action, _ = self.policies[0].act(self.ob)
            else:
                self.ob = splitobs(self.ob, keepdims=False)
                ob_policy_idx = np.split(np.arange(len(self.ob)), len(self.policies))
                actions = []
                for i, policy in enumerate(self.policies):
                    inp = itemgetter(*ob_policy_idx[i])(self.ob)
                    inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
                    ac, info = policy.act(inp)
                    actions.append(ac)
                action = listdict2dictnp(actions, keepdims=True)

            self.ob, rew, done, env_info = self.env.step(action)
            self.total_rew += rew

            if done or env_info.get('discard_episode', False):
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.ob = self.env.reset()
        for policy in self.policies:
            policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)
