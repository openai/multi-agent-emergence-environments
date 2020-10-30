import numpy as np
from rusp.env_prisoners_buddy import make_env


def test_env_runs():
    env = make_env()
    env.reset()

    action = {'action_choose_agent': [0, 0, 3, 0, 1], 'action_choose_option': [1, 0, 0, 0, 0]}
    for i in range(5):
        obs, rew, done, info = env.step(action)

    assert np.all(rew == [1, -2, 3, 2, -2])
