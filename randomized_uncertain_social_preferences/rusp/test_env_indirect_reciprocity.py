from rusp.env_indirect_reciprocity import make_env
import numpy as np
from copy import deepcopy


def _test_fixed_policy(against_all_d=False, against_all_c=False):
    env = make_env(against_all_d=against_all_d, against_all_c=against_all_c,
                   last_agent_always_plays=True)
    prev_obs = env.reset()
    for i in range(1000):
        currently_playing = np.squeeze(prev_obs['youre_playing_self'])
        ac = {'action_defect': np.random.randint(0, 2, size=(env.metadata['n_actors']))}

        obs, rew, done, info = env.step(ac)

        if against_all_d:
            assert np.all(rew[currently_playing & (ac['action_defect'] == 0)] == -2)
            assert np.all(rew[currently_playing & (ac['action_defect'] == 1)] == 0)
        elif against_all_c:
            assert np.all(rew[currently_playing & (ac['action_defect'] == 0)] == 2)
            assert np.all(rew[currently_playing & (ac['action_defect'] == 1)] == 4)
        else:
            assert False
        assert np.all(rew[~currently_playing] == 0)

        prev_obs = obs

        if done:
            prev_obs = env.reset()


def test_all_d():
    _test_fixed_policy(against_all_d=True)


def test_all_c():
    _test_fixed_policy(against_all_c=True)


# Tests for play orderings
def test_last_always_plays():
    env = make_env(last_agent_always_plays=True)
    obs = env.reset()
    assert obs['youre_playing_self'][-1, 0]
    ac = {'action_defect': np.random.randint(0, 2, size=(env.metadata['n_actors']))}
    for i in range(1000):
        obs, _, done, _ = env.step(ac)
        assert obs['youre_playing_self'][-1, 0]

        if done:
            obs = env.reset()
            assert obs['youre_playing_self'][-1, 0]


def test_last_first_versus_last():
    env = make_env(last_step_first_agent_vs_last_agent=True)
    prev_obs = env.reset()
    ac = {'action_defect': np.random.randint(0, 2, size=(env.metadata['n_actors']))}
    for i in range(1000):
        obs, _, done, _ = env.step(ac)

        if done:
            assert prev_obs['youre_playing_self'][-1, 0]
            assert prev_obs['youre_playing_self'][0, 0]
            obs = env.reset()

        prev_obs = deepcopy(obs)


def test_last_doesnt_play_until():
    env = make_env(last_doesnt_play_until_t=5)
    ac = {'action_defect': np.random.randint(0, 2, size=(env.metadata['n_actors']))}
    obs = env.reset()
    done = False
    t = 0
    for i in range(1000):
        if t < 5:
            assert not obs['youre_playing_self'][-1, 0]
        obs, rew, done, info = env.step(ac)
        t += 1

        if done:
            obs = env.reset()
            done = False
            t = 0


def test_last_doesnt_play_until_and_last_must_play_at_t():
    env = make_env(last_doesnt_play_until_t=5, last_must_play_at_t=True)
    ac = {'action_defect': np.random.randint(0, 2, size=(env.metadata['n_actors']))}
    obs = env.reset()
    done = False
    t = 0
    for i in range(1000):
        if t < 5:
            assert not obs['youre_playing_self'][-1, 0]
        if t == 5:
            assert obs['youre_playing_self'][-1, 0]
        obs, rew, done, info = env.step(ac)
        t += 1

        if done:
            obs = env.reset()
            done = False
            t = 0
