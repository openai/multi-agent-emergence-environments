#!/usr/bin/env python3
import logging
import click
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from mae_envs.viewer.env_viewer import EnvViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments


logger = logging.getLogger(__name__)


@click.command()
@click.argument('argv', nargs=-1, required=False)
def main(argv):
    '''
    examine.py is used to display environments and run policies.

    For an example environment jsonnet, see
        mujoco-worldgen/examples/example_env_examine.jsonnet
    You can find saved policies and the in the 'examples' together with the environment they were
    trained in and the hyperparameters used. The naming used is 'examples/<env_name>.jsonnet' for
    the environment jsonnet file and 'examples/<env_name>.npz' for the policy weights file.
    Example uses:
        bin/examine.py hide_and_seek
        bin/examine.py mae_envs/envs/base.py
        bin/examine.py base n_boxes=6 n_ramps=2 n_agents=3
        bin/examine.py my_env_jsonnet.jsonnet
        bin/examine.py my_env_jsonnet.jsonnet my_policy.npz
        bin/examine.py hide_and_seek my_policy.npz n_hiders=3 n_seekers=2 n_boxes=8 n_ramps=1
    '''
    names, kwargs = parse_arguments(argv)

    env_name = names[0]
    core_dir = abspath(join(dirname(__file__), '..'))
    envs_dir = 'mae_envs/envs'
    xmls_dir = 'xmls'

    if len(names) == 1:  # examine the environment
        examine_env(env_name, kwargs,
                    core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                    env_viewer=EnvViewer)

    if len(names) >= 2:  # run policies on the environment
        # importing PolicyViewer and load_policy here because they depend on several
        # packages which are only needed for playing policies, not for any of the
        # environments code.
        from mae_envs.viewer.policy_viewer import PolicyViewer
        from ma_policy.load_policy import load_policy
        policy_names = names[1:]
        env, args_remaining_env = load_env(env_name, core_dir=core_dir,
                                           envs_dir=envs_dir, xmls_dir=xmls_dir,
                                           return_args_remaining=True, **kwargs)

        if isinstance(env.action_space, Tuple):
            env = JoinMultiAgentActions(env)
        if env is None:
            raise Exception(f'Could not find environment based on pattern {env_name}')

        env.reset()  # generate action and observation spaces
        assert np.all([name.endswith('.npz') for name in policy_names])
        policies = [load_policy(name, env=env, scope=f'policy_{i}')
                    for i, name in enumerate(policy_names)]


        args_remaining_policy = args_remaining_env

        if env is not None and policies is not None:
            args_to_pass, args_remaining_viewer = extract_matching_arguments(PolicyViewer, kwargs)
            args_remaining = set(args_remaining_env)
            args_remaining = args_remaining.intersection(set(args_remaining_policy))
            args_remaining = args_remaining.intersection(set(args_remaining_viewer))
            assert len(args_remaining) == 0, (
                f"There left unused arguments: {args_remaining}. There shouldn't be any.")
            viewer = PolicyViewer(env, policies, **args_to_pass)
            viewer.run()


    print(main.__doc__)


if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
