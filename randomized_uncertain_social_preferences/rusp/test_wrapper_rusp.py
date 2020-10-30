from rusp.wrappers_rusp import RUSPGenerator
import _jsonnet
import json
import os
import numpy as np


def test_compute_observations():
    N_AGENTS = 2
    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    graph_generator = RUSPGenerator()

    graph_generator._generate_social_preferences(N_AGENTS)
    graph_generator._generate_uncertainty(N_AGENTS)

    graph_generator.noise_std = np.arange(1, N_AGENTS ** 3 + 1).reshape((N_AGENTS, N_AGENTS, N_AGENTS))
    graph_generator.noise = np.arange(1, N_AGENTS ** 3 + 1).reshape((N_AGENTS, N_AGENTS, N_AGENTS)) * 10
    graph_generator.unnormalized_reward_xform_mat = graph_generator.reward_xform_mat = np.arange(1, N_AGENTS ** 2 + 1).reshape((N_AGENTS, N_AGENTS))

    graph_generator._precompute_observations(N_AGENTS)

    assert np.all(graph_generator.precomputed_obs['self_rew_value'] == np.array([1, 4]))
    assert np.all(graph_generator.precomputed_obs['self_rew_value_noisy'] == np.array([1, 4]) + np.array([10, 80]))
    assert np.all(graph_generator.precomputed_obs['self_rew_value_noise_level'] == np.array([1, 8]))

    assert np.all(graph_generator.precomputed_obs['other_rew_value_s'] == np.array(
        [[1, 4],
         [1, 4]]))
    assert np.all(graph_generator.precomputed_obs['other_rew_value_s_noisy'] == np.array(
        [[1 + 10, 4 + 40],
         [1 + 50, 4 + 80]]))
    assert np.all(graph_generator.precomputed_obs['other_rew_value_s_noise_level'] == np.array(
        [[1, 4],
         [5, 8]]))

    assert np.all(graph_generator.precomputed_obs['rew_share_so_s'] == np.array(
        [[1, 2],
         [3, 4]]))
    assert np.all(graph_generator.precomputed_obs['rew_share_so_s_noisy'] == np.array(
        [[1 + 10, 2 + 20],
         [3 + 70, 4 + 80]]))
    assert np.all(graph_generator.precomputed_obs['rew_share_so_s_noise_level'] == np.array(
        [[1, 2],
         [7, 8]]))

    assert np.all(graph_generator.precomputed_obs['rew_share_os_o'] == np.array(
        [[1, 3],
         [2, 4]]))
    assert np.all(graph_generator.precomputed_obs['rew_share_os_o_noisy'] == np.array(
        [[1 + 10, 3 + 70],
         [2 + 20, 4 + 80]]))
    assert np.all(graph_generator.precomputed_obs['rew_share_os_o_noise_level'] == np.array(
        [[1, 7],
         [2, 8]]))
