import gym
import numpy as np
from mae_envs.wrappers.util import update_obs_space


class TeamMembership(gym.ObservationWrapper):
    '''
        This wrapper just stores team membership information at initialization.
        The information is stored as a key in the self.metadata property, which ensures
        that it is available even if this wrapper is not on top of the wrapper
        hierarchy.

        Arguments:
            team_index: list/numpy vector of team membership index
                        length must be equal to number of agents
                        e.g. [0,0,0,1,1,1] means first 3 agents are in team 0,
                        second 3 agents in team 1
            n_teams: if team_index is None, agents are split in n_teams number
                     of teams, with as equal team sizes as possible.
                     if team_index is set, this argument is ignored

        One planned use of this wrapper is to evaluate the "TrueSkill" score
        during training, which requires knowing which agent belongs to which team

        Note: This wrapper currently does not align the reward structure with the
              teams, but that could be easily implemented if desired.
    '''
    def __init__(self, env, team_index=None, n_teams=2):
        super().__init__(env)
        self.n_agents = self.metadata['n_actors']

        if team_index is None:
            assert n_teams >= 1, "Number of teams must be at least 1"
            # split teams: 5 agents and 3 teams will result in team_index = [0,0,1,1,2]
            team_index = np.array_split(np.arange(self.n_agents), n_teams)
            team_index = np.concatenate([np.ones_like(ar) * i for i, ar in enumerate(team_index)])

        assert len(team_index) == self.n_agents, (
            "team_index parameter length must be equal to number of agents")
        if isinstance(team_index, np.ndarray):
            assert team_index.ndim == 1, (
                "team_index parameter must be numpy array of dimension 1")

        # store in metadata property that gets automatically inherited
        # make sure we copy value of team_index if it's a numpy array
        self.metadata['team_index'] = np.array(team_index)
        self.team_idx = np.array(team_index)
        self.observation_space = update_obs_space(env, {'team_size': (self.n_agents, 1)})

    def observation(self, obs):
        obs['team_size'] = np.sum(self.team_idx[:, None] == self.team_idx[None, :],
                                  axis=1, keepdims=True)
        return obs
