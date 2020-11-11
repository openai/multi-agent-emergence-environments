import numpy as np
from rusp.env_ipd import make_env


def test_env_runs():
    env = make_env()
    env.reset()

    action = {'action_defect': np.array([0, 0])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([2, 2]))

    action = {'action_defect': np.array([1, 0])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([4, -2]))

    action = {'action_defect': np.array([0, 1])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([-2, 4]))

    action = {'action_defect': np.array([1, 1])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([0, 0]))


def test_env_against_all_c():
    env = make_env(against_all_c=True)
    env.reset()

    action = {'action_defect': np.array([0])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([2]))

    action = {'action_defect': np.array([1])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([4]))


def test_env_against_all_d():
    env = make_env(against_all_d=True)
    env.reset()

    action = {'action_defect': np.array([0])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([-2]))

    action = {'action_defect': np.array([1])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([0]))


def test_env_against_tft():
    env = make_env(against_tft=True)
    env.reset()

    action = {'action_defect': np.array([0])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([2]))

    action = {'action_defect': np.array([1])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([4]))

    action = {'action_defect': np.array([1])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([0]))

    action = {'action_defect': np.array([0])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([-2]))

    action = {'action_defect': np.array([0])}
    obs, rew, done, info = env.step(action)
    assert np.all(rew == np.array([2]))
