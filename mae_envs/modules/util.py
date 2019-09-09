import numpy as np
from mujoco_worldgen.parser import parse_file


def get_size_from_xml(obj):
    '''
        Args:
            obj (worldgen.Obj): worldgen object
        Returns: size of object annotation:outerbound if it exists, None if it doesn't
    '''
    outer_bound = None
    for body in parse_file(obj._generate_xml_path())['worldbody']['body']:
        if body.get('@name', '') == 'annotation:outer_bound':
            outer_bound = body
    if outer_bound is None:
        return None
    else:
        return outer_bound['geom'][0]['@size'][:2] * 2


def rejection_placement(env, placement_fn, floor_size, obj_size, num_tries=10):
    '''
        Args:
            env (gym.Env): environment
            placement_fn (function): Function that returns a position on a grid
                Args:
                    grid (np.ndarray): 2D occupancy grid. 1's mean occupied
                    obj_size_in_cells (int np.ndarray): number of cells in [x, y]
                        that this object would occupy on the grid. Currently only supports
                        rectangular object sizes (but so does worldgen)
                    env.metadata (dict): environment metadata
                    random_state (np.random.RandomState): numpy random state
                Returns: x, y placement position on grid
            floor_size (float): size of floor
            obj_size (float np.ndarray): [x, y] size of object
            num_tries (int): number of tries to place object
        Returns: int np.ndarray([x, y]) position on grid or None if no placement was found.
    '''
    grid = env.placement_grid
    grid_size = len(grid)
    cell_size = floor_size / grid_size
    obj_size_in_cells = np.ceil(obj_size / cell_size).astype(int)

    for i in range(num_tries):
        if placement_fn is not None:
            pos = placement_fn(grid, obj_size_in_cells, env.metadata, env._random_state)
        else:
            # Assume that we'll always have boundary walls so don't sample there
            pos = np.array([env._random_state.randint(1, grid_size - obj_size_in_cells[0] - 1),
                            env._random_state.randint(1, grid_size - obj_size_in_cells[1] - 1)])
        if np.any(grid[pos[0]:pos[0] + obj_size_in_cells[0], pos[1]:pos[1] + obj_size_in_cells[1]]):
            continue
        else:
            extra_room = obj_size_in_cells * cell_size - obj_size
            pos_on_floor = pos / grid_size * floor_size
            pos_on_floor += env._random_state.uniform([0, 0], extra_room)
            placement = pos_on_floor / (floor_size - obj_size)
            grid[pos[0]:pos[0] + obj_size_in_cells[0], pos[1]:pos[1] + obj_size_in_cells[1]] = 1
            return placement, pos
    return None, None


def uniform_placement(grid, obj_size, metadata, random_state):
    grid_size = len(grid)
    pos = np.array([random_state.randint(1, grid_size - obj_size[0] - 1),
                    random_state.randint(1, grid_size - obj_size[1] - 1)])

    return pos


def close_to_other_object_placement(object_type, object_index, radius_key):
    def close_placement_fn(grid, obj_size, metadata, random_state):
        init_pos_key = f"{object_type}{object_index}_initpos"

        assert init_pos_key in metadata, \
            f"First object position must be specified in metadata['{init_pos_key}']"
        assert radius_key in metadata, \
            f"metadata['{radius_key}'] mus be specified."

        grid_size = len(grid)

        anchor_obj_pos = metadata[f"{init_pos_key}"]
        rad_in_cells = metadata[radius_key]

        distr_limits_min = np.maximum(1, anchor_obj_pos - rad_in_cells)
        distr_limits_max = np.minimum(grid_size - 1, anchor_obj_pos + rad_in_cells)

        pos = np.array([random_state.randint(distr_limits_min[0], distr_limits_max[0]),
                        random_state.randint(distr_limits_min[1], distr_limits_max[1])])

        return pos

    return close_placement_fn


def uniform_placement_middle(area_side_length_fraction):
    '''
        Creates a sampling function that samples object position uniformly within the
        middle of the playing area. E.g. if the playing area is
           ------
           |AAAA|
           |ABBA|
           |ABBA|
           |AAAA|
           ------
        then uniform_placement_middle(0.5) will returned a function that samples the object position
        from any of the B cells.
        Args:
            area_side_length_fraction (float, between 0 and 1): Length of the sides of the middle
                square being sampled from, as fraction of the overall playing field
    '''
    def uniform_placement_middle_fn(grid, obj_size, metadata, random_state):
        grid_size = len(grid)
        distr_limits_min = ((grid_size - obj_size) * (1 - area_side_length_fraction) / 2 + area_side_length_fraction).astype(int)
        distr_limits_max = ((grid_size - obj_size) * (1 + area_side_length_fraction) / 2 - area_side_length_fraction).astype(int)

        pos = np.array([random_state.randint(distr_limits_min[0], distr_limits_max[0]),
                        random_state.randint(distr_limits_min[1], distr_limits_max[1])])

        return pos

    return uniform_placement_middle_fn


def center_placement(grid, obj_size_in_cells, metadata, random_state):
    half_grid_size = int(len(grid) / 2)
    pos = np.array([half_grid_size - int(obj_size_in_cells[0]/2),
                    half_grid_size - int(obj_size_in_cells[1]/2)])
    return pos
