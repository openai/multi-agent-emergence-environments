
class EnvModule():
    '''
        Dummy class outline for "Environment Modules".
        NOTE: If in any function you are going to randomly sample a number,
            use env._random_state instead of numpy.random
    '''
    def build_world_step(self, env, floor, floor_size):
        '''
            This function allows you to add objects to worldgen floor object.
                You could also cache variables needed for observations or add
                information to the env.metadata dict
            Args:
                env (gym.Env): the environment
                floor (worldgen.Floor): square worldgen floor object
                floor_size (float): size of the worlgen floor object
            Returns: True if the the build_world_step was successful, False if it failed
                e.g. your build_world_step might fail because no valid object placements
                were found.
        '''
        return True

    def modify_sim_step(self, env, sim):
        '''
            After an MJSim has been created, this function can be used to modify that sim
                and cache any variables you can only get after the sim is created
            Args:
                env (gym.env): the environment
                sim (mujoco_py.MJSim): mujoco simulation object
            Returns: None
        '''
        pass

    def observation_step(self, env, sim):
        '''
            Create any observations specific to this module.
            Args:
                env (gym.env): the environment
                sim (mujoco_py.MJSim): mujoco simulation object
            Returns: dict of observations
        '''
        return {}
