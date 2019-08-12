import logging
from mujoco_worldgen.transforms import set_geom_attr_transform
from mae_envs.modules import EnvModule


class FloorAttributes(EnvModule):
    '''
        For each (key, value) in kwargs, sets the floor geom attribute key to value.
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build_world_step(self, env, floor, floor_size):
        for k, v in self.kwargs.items():
            floor.add_transform(set_geom_attr_transform(k, v))
        return True


class WorldConstants(EnvModule):
    '''
        For each (key, value) in kwargs, sets sim.model.opt[key] = value
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def modify_sim_step(self, env, sim):
        for k, v in self.kwargs.items():
            if not hasattr(sim.model.opt, k):
                logging.warning(f"sim.model.opt does not have attribute {k}")
            else:
                getattr(sim.model.opt, k)[:] = v
