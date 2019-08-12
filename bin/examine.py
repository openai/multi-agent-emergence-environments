#!/usr/bin/env python3
import logging
import click
from os.path import abspath, dirname, join

from mae_envs.util.env_viewer import EnvViewer
from mujoco_worldgen.util.envs import examine_env
from mujoco_worldgen.util.parse_arguments import parse_arguments


logger = logging.getLogger(__name__)


@click.command()
@click.argument('argv', nargs=-1, required=False)
def main(argv):
    '''
    examine.py is used to display environments
    Argument to this program is matched against each of this options.
    For an example environment jsonnet, see
        mujoco-worldgen/examples/example_env_examine.jsonnet
    Example uses:
        bin/examine.py hide_and_seek
        bin/examine.py mae_envs/envs/base.py
        bin/examine.py base n_boxes=6 n_ramps=2 n_agents=3
        bin/examine.py my_env_jsonnet.jsonnet
    '''
    env_names, env_kwargs = parse_arguments(argv)
    assert len(env_names) == 1, 'You must provide exactly 1 environment to examine.'
    env_name = env_names[0]
    core_dir = abspath(join(dirname(__file__), '..'))
    examine_env(env_name, env_kwargs,
                core_dir=core_dir, envs_dir='mae_envs/envs', xmls_dir='xmls',
                env_viewer=EnvViewer)

    print(main.__doc__)


if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
