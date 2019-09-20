**Status:** Archive (code is provided as-is, no updates expected)

# Multiagent emergence environments
Environment generation code for [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528) ([blog](https://openai.com/blog/emergent-tool-use/))

### Installation
This repository depends on the [mujoco-worldgen](https://github.com/openai/mujoco-worldgen) package. You will need to clone the mujoco-worldgen repository and install it and its dependencies:
```
pip install -r mujoco-worldgen/requirements.txt
pip install -e mujoco-worldgen/
pip install -e multi-agent-emergence-environments/
```

This repository has been tested only on Mac OS X and Ubuntu 16.04 with Python 3.6

### Use

Environment construction works in the following way: You start from the `Base` environment (defined in `mae_envs/envs/base.py`) and then you add environment modules (e.g. `Boxes`, `Ramps`, `RandomWalls`, etc.) and then wrappers on top. You can see examples in the `mae_envs/envs` folder.

If you want to construct a new environment, we highly recommend using the above paradigm in order to minimize code duplication. If you need new objects or game dynamics that don't already exist in this codebase, add them in via a new `EnvModule` class or a `gym.Wrapper` class rather than subclassing `Base` (or mujoco-worldgen's `Env` class). In general, `EnvModules` should be used for adding objects or sites to the environment, or otherwise modifying the mujoco simulator; wrappers should be used for everything else (e.g. adding rewards, additional observations, or implementing game mechanics like Lock and Grab).

The environments defined in this repository are: \
*Hide and seek* - `mae_envs/envs/hide_and_seek.py` - The Hide and Seek environment described in the paper. This encompasses the *random rooms*, *quadrant* and *food* versions of the game (you can switch between them by changing the arguments given to the `make_env` function in the file) \
*Box locking* - `mae_envs/envs/box_locking.py` - Encompasses the *Lock and Return* and *Sequential Lock* transfer tasks described in the paper. \
*Blueprint Construction* - `mae_envs/envs/blueprint_construction.py` \
*Shelter Construction* - `mae_envs/envs/shelter_construction.py`

You can test out environments by using the `bin/examine` script. Example usage: `bin/examine.py base`. \
You can also use `bin/examine` to play a saved policy on an environment. There are several environment jsonnets and policies in the `examples` folder. Example usage:

```bin/examine.py examples/hide_and_seek_quadrant.jsonnet examples/hide_and_seek_quadrant.npz``` 

Note that to be able to play saved policies, you will need to install a few additional packages. You can do this via

`pip install -r multi-agent-emergence-environments/requirements_ma_policy.txt`
