import numpy as np
from mujoco_worldgen.util.types import store_args
from mujoco_worldgen import Geom
from mujoco_worldgen.transforms import set_geom_attr_transform
from mae_envs.modules import EnvModule


class Wall:
    '''
        Defines a wall object which is essentially a pair of points on a grid
            with some useful helper functions for creating randomized rooms.
        Args:
            pt1, pt2 (float tuple): points defining the wall
            height (float): wall height
            rgba (float tuple): wall rgba
    '''
    def __init__(self, pt1, pt2, height=0.5, rgba=(0, 1, 0, 1)):
        assert pt1[0] == pt2[0] or pt1[1] == pt2[1], (
            "Currently only horizontal and vertical walls are supported")
        self.is_vertical = pt1[0] == pt2[0]
        # Make sure pt2 is top right of pt1
        if np.any(np.array(pt2) - np.array(pt1) < 0):
            self.pt1 = np.array(pt2)
            self.pt2 = np.array(pt1)
        else:
            self.pt1 = np.array(pt1)
            self.pt2 = np.array(pt2)
        self.length = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        self.height = height
        self.rgba = rgba
        # Variables defining where other walls split from this wall on the left and right.
        # For horizontal walls, left means below, right means above
        self.left_edges = [self.pt1, self.pt2]
        self.right_edges = [self.pt1, self.pt2]

    def is_touching(self, pt):
        '''
            Is pt (tuple) touching this wall
        '''
        if self.is_vertical:
            return pt[0] == self.pt1[0] and pt[1] >= self.pt1[1] and pt[1] <= self.pt2[1]
        else:
            return pt[1] == self.pt1[1] and pt[0] >= self.pt1[0] and pt[0] <= self.pt2[0]

    def maybe_add_edge(self, wall):
        '''
            Check if wall is originating from this wall. If so add it to the list of edges.
        '''
        if self.is_vertical == wall.is_vertical:
            return
        if self.is_touching(wall.pt1):
            self.right_edges.append(wall.pt1)
        elif self.is_touching(wall.pt2):
            self.left_edges.append(wall.pt2)

    def intersects(self, wall):
        '''
            Check if intersects with wall.
        '''
        if self.is_vertical == wall.is_vertical:
            return False
        return np.all(np.logical_and(self.pt1 < wall.pt2, wall.pt1 < self.pt2))

    def split_for_doors(self, num_doors=1, door_size=1, all_connect=False,
                        random_state=np.random.RandomState()):
        '''
            Split this wall into many walls with 'doors' in between.
            Args:
                num_doors (int): upper bound of number of doors to create
                door_size (int): door size in grid cells
                all_connect (bool): create a door in every wall segment between pairs of points
                    where other walls connect with this wall
                random_state (np.random.RandomState): random state to use for sampling
        '''
        edges = np.unique(self.left_edges + self.right_edges, axis=0)
        edges = np.array(sorted(edges, key=lambda x: x[1] if self.is_vertical else x[0]))
        rel_axis = edges[:, 1] if self.is_vertical else edges[:, 0]
        diffs = np.diff(rel_axis)
        possible_doors = diffs >= door_size + 1

        # Door regions are stretches on the wall where we could create a door.
        door_regions = np.arange(len(edges) - 1)
        door_regions = door_regions[possible_doors]

        # The number of doors on this wall we want to/can create
        num_doors = len(edges) - 1 if all_connect else num_doors
        num_doors = min(num_doors, len(door_regions))
        if num_doors == 0 or door_size == 0:
            return [self], []

        # Sample num_doors regions to which we will add doors.
        door_regions = np.sort(random_state.choice(door_regions, num_doors, replace=False))
        new_walls = []
        doors = []
        new_wall_start = edges[0]
        for door in door_regions:
            # door_start and door_end are the first and last point on the wall bounding the door
            # (inclusive boundary)
            door_start = random_state.randint(1, diffs[door] - door_size + 1)
            door_end = door_start + door_size - 1

            # Because door boundaries are inclusive, we add 1 to the door_end to get next wall
            # start cell and subtract one from the door_start to get the current wall end cell.
            if self.is_vertical:
                new_wall_end = [edges[door][0], edges[door][1] + door_start - 1]
                next_new_wall_start = [new_wall_start[0], edges[door][1] + door_end + 1]
                door_start_cell = [edges[door][0], edges[door][1] + door_start]
                door_end_cell = [new_wall_start[0], edges[door][1] + door_end]
            else:
                new_wall_end = [edges[door][0] + door_start - 1, edges[door][1]]
                next_new_wall_start = [edges[door][0] + door_end + 1, edges[door][1]]
                door_start_cell = [edges[door][0] + door_start, edges[door][1]]
                door_end_cell = [new_wall_start[0] + door_end, edges[door][1]]

            # Store doors as inclusive boundaries.
            doors.append([door_start_cell, door_end_cell])
            # Check that the new wall isn't size 0
            if np.linalg.norm(np.array(new_wall_start) - np.array(new_wall_end)) > 0:
                new_walls.append(Wall(new_wall_start, new_wall_end))
            new_wall_start = next_new_wall_start
        if np.linalg.norm(np.array(new_wall_start) - np.array(edges[-1])) > 0:
            new_walls.append(Wall(new_wall_start, edges[-1]))
        return new_walls, doors


def connect_walls(wall1, wall2, min_dist_between, random_state=np.random.RandomState()):
    '''
        Draw a random new wall connecting wall1 and wall2. Return None if
        the drawn wall was closer than min_dist_between to another wall
        or the wall wasn't valid.
        NOTE: This DOES NOT check if the created wall overlaps with any existing walls, that
            should be done outside of this function
        Args:
            wall1, wall2 (Wall): walls to draw a new wall between
            min_dist_between (int): closest another parallel wall can be to the new wall in grid cells.
            random_state (np.random.RandomState): random state to use for sampling
    '''
    if wall1.is_vertical != wall2.is_vertical:
        return None
    length = random_state.randint(1, wall1.length)
    if wall1.is_vertical:
        pt1 = [wall1.pt1[0], wall1.pt1[1] + length]
        pt2 = [wall2.pt1[0], wall1.pt1[1] + length]
    else:
        pt1 = [wall1.pt1[0] + length, wall1.pt1[1]]
        pt2 = [wall1.pt1[0] + length, wall2.pt1[1]]

    # Make sure that the new wall actually touches both walls
    # and there is no wall close to this new wall
    wall1_right_of_wall2 = np.any(np.array(pt2) - np.array(pt1) < 0)
    if wall1_right_of_wall2:
        dists = np.array(pt1)[None, :] - np.array(wall1.left_edges)
    else:
        dists = np.array(pt1)[None, :] - np.array(wall1.right_edges)
    min_dist = np.linalg.norm(dists, axis=1).min()

    if wall2.is_touching(pt2) and min_dist > min_dist_between:
        return Wall(pt1, pt2)
    return None


def choose_new_split(walls, min_dist_between, num_tries=10, random_state=np.random.RandomState()):
    '''
        Given a list of walls, choose a random wall and draw a new wall perpendicular to it.
        NOTE: Right now this O(n_walls^2). We could probably get this to linear if we did
            something smarter with the occupancy grid. Until n_walls gets way bigger this
            should be fine though.
        Args:
            walls (Wall list): walls to possibly draw a new wall from
            min_dist_between (int): closest another parallel wall can be to the new wall in grid cells.
            num_tries (int): number of times before we can fail in placing a wall before giving up
            random_state (np.random.RandomState): random state to use for sampling
    '''
    for i in range(num_tries):
        wall1 = random_state.choice(walls)
        proposed_walls = [connect_walls(wall1, wall2, min_dist_between, random_state=random_state)
                          for wall2 in walls if wall2 != wall1]
        proposed_walls = [wall for wall in proposed_walls
                          if wall is not None
                          and not np.any([wall.intersects(_wall) for _wall in walls])]
        if len(proposed_walls):
            new_wall = random_state.choice(proposed_walls)
            for wall in walls:
                wall.maybe_add_edge(new_wall)
                new_wall.maybe_add_edge(wall)
            return new_wall
    return None


def split_walls(walls, door_size, random_state=np.random.RandomState()):
    '''
        Add a door to each wall in walls. Return the new walls and doors.
        Args:
            walls (Wall list): walls
            door_size (int): door size in grid cells
            random_state (np.random.RandomState): random state to use for sampling
    '''
    split_walls = []
    doors = []
    for wall in walls:
        new_walls, new_doors = wall.split_for_doors(door_size=door_size, random_state=random_state)
        split_walls += new_walls
        doors += new_doors
    return split_walls, doors


def construct_door_obs(doors, floor_size, grid_size):
    '''
        Construct door observations in mujoco frame from door positions in grid frame.
        Args:
            doors ((n_doors, 2, 2) array): list of pairs of points of door edges.
            floor_size (float): size of floor
            grid_size (int): size of placement grid
    '''
    _doors = doors + 0.5
    scaling = floor_size / grid_size
    _door_sizes = np.array([np.linalg.norm(door[1] - door[0]) * scaling for door in _doors])
    _doors = np.array([(door[0] + (door[1] - door[0]) / 2) * scaling for door in _doors])
    return np.concatenate([_doors, _door_sizes[:, None]], -1)


def add_walls_to_grid(grid, walls):
    '''
        Draw walls onto a grid.
        Args:
            grid (np.ndarray): 2D occupancy grid
            walls (Wall list): walls
    '''
    for wall in walls:
        if wall.is_vertical:
            grid[wall.pt1[0], wall.pt1[1]:wall.pt2[1] + 1] = 1
        else:
            grid[wall.pt1[0]:wall.pt2[0] + 1, wall.pt1[1]] = 1


def walls_to_mujoco(floor, floor_size, grid_size, walls, friction=None):
    '''
        Take a list of walls in grid frame and add them to the floor in the worldgen frame.
        Args:
            floor (worldgen.Floor): floor
            floor_size (float): size of floor
            grid_size (int): size of placement grid
            walls (Wall list): list of walls
            friction (float): wall friction
    '''
    wall_width = floor_size / grid_size / 2
    grid_cell_length = floor_size / grid_size
    for i, wall in enumerate(walls):
        if wall.is_vertical:
            wall_length_grid = (wall.pt2[1] - wall.pt1[1] + 1)
            offset = np.array([-1, 1])
        else:
            wall_length_grid = (wall.pt2[0] - wall.pt1[0] + 1)
            offset = np.array([1, -1])

        # Convert to mujoco frame
        wall_length = wall_length_grid * grid_cell_length
        # Subtract 1 grid_cell_length such that walls originate and end in the center of a grid cell
        # Subtract 1 wall_width such that perpendicular walls do not intersect at the center of a
        # grid cell
        wall_length -= grid_cell_length + wall_width

        if wall.is_vertical:
            size = (wall_width, wall_length, wall.height)
        else:
            size = (wall_length, wall_width, wall.height)

        # Position of object should be in the middle of a grid cell (add 0.5) shifted by
        #     the wall width such that corners don't overlap
        pos = np.array([wall.pt1[0] + 0.5, wall.pt1[1] + 0.5]) / grid_size
        pos += offset * wall_width / floor_size / 2

        # Convert from mujoco to worldgen scale
        scale_x = (floor_size - size[0]) / floor_size
        scale_y = (floor_size - size[1]) / floor_size
        pos = pos / np.array([scale_x, scale_y])
        geom = Geom('box', size, name=f"wall{i}")
        geom.mark_static()
        geom.add_transform(set_geom_attr_transform('rgba', wall.rgba))
        geom.add_transform(set_geom_attr_transform('group', 1))
        if friction is not None:
            geom.add_transform(set_geom_attr_transform('friction', friction))
        floor.append(geom, placement_xy=pos)


def outside_walls(grid_size, rgba=(0, 1, 0, 0.1), use_low_wall_height=False):
    height = 0.5 if use_low_wall_height else 4.0
    return [Wall([0, 0], [0, grid_size - 1], height=height, rgba=rgba),
            Wall([0, 0], [grid_size - 1, 0], height=height, rgba=rgba),
            Wall([grid_size - 1, 0], [grid_size - 1, grid_size - 1], height=height, rgba=rgba),
            Wall([0, grid_size - 1], [grid_size - 1, grid_size - 1], height=height, rgba=rgba)]


class RandomWalls(EnvModule):
    '''
    Add random walls to the environment. This must be the first module added to the environment
        Args:
            grid_size (int): grid size to place walls on
            num_rooms (int): number of rooms to create
            min_room_size (int): minimum size of a room in grid cells
            door_size (int): size of doors in grid cells
            friction (float): wall friction
            outside_walls (bool): If false, don't add outside walls to mujoco
            outside_wall_rgba (array): RGBA color of outside walls
            random_room_number (bool): If true, the actual number of rooms is
                sampled uniformly between 1 and num_rooms
            gen_door_obs (bool): If true, generate door observation (currently does not
                work with random room number)
            prob_outside_walls (float): probability that outside walls are used
            low_outside_walls (bool): If true, outside walls are the same height as inside walls.
                This is just used for pretty rendering
    '''
    @store_args
    def __init__(self, grid_size, num_rooms, min_room_size, door_size, friction=None,
                 num_tries=10, outside_wall_rgba=(0, 1, 0, 0.1),
                 random_room_number=False, gen_door_obs=True, prob_outside_walls=1.0,
                 low_outside_walls=False):
        pass

    def build_world_step(self, env, floor, floor_size):
        # Create rooms
        walls = outside_walls(self.grid_size, rgba=self.outside_wall_rgba,
                              use_low_wall_height=self.low_outside_walls)
        failures = 0

        if self.random_room_number:
            self.num_actual_rooms = env._random_state.randint(self.num_rooms) + 1
        else:
            self.num_actual_rooms = self.num_rooms

        while len(walls) < self.num_actual_rooms + 3:
            new_wall = choose_new_split(walls, self.min_room_size, random_state=env._random_state)
            if new_wall is None:
                walls = outside_walls(self.grid_size, rgba=self.outside_wall_rgba,
                                      use_low_wall_height=self.low_outside_walls)
                failures += 1
            else:
                walls.append(new_wall)
            if failures == self.num_tries:
                return False

        # Add doors
        new_walls, doors = split_walls(walls[4:], self.door_size, random_state=env._random_state)
        if env._random_state.uniform() < self.prob_outside_walls:
            walls = walls[:4] + new_walls
        else:
            walls = new_walls

        # Convert doors into mujoco frame
        if self.gen_door_obs:
            self.door_obs = construct_door_obs(np.array(doors), floor_size, self.grid_size)

        walls_to_mujoco(floor, floor_size, self.grid_size, walls, friction=self.friction)
        add_walls_to_grid(env.placement_grid, walls)
        return True

    def observation_step(self, env, sim):
        if self.gen_door_obs:
            obs = {'door_obs': self.door_obs}
        else:
            obs = {}

        return obs


class WallScenarios(EnvModule):
    '''
    Add a wall scenario to the environment. This must be the first module added to the environment.
        Args:
            grid_size (int): grid size to place walls on
            door_size (int): size of doors in grid cells
            scenario (string): Options:
                'empty': no walls
                'half': one wall in the middle with a random door
                'quadrant': one quadrant is walled off with random door(s)
                'var_quadrant': same as 'quadrant' but the room size is also randomized
                'var_tri': three rooms, one taking about half of the area and the other
                    two taking about a quarter of the area. Random doors
            friction (float): wall friction
            p_door_dropout (float): probability we don't place one of the doors either
                quadrant scenario
            low_outside_walls (bool): If true, outside walls are the same height as inside walls.
                This is just used for pretty rendering
    '''
    @store_args
    def __init__(self, grid_size, door_size, scenario, friction=None, p_door_dropout=0.0,
                 low_outside_walls=False):
        assert scenario in ['var_quadrant', 'quadrant', 'half', 'var_tri', 'empty']

    def build_world_step(self, env, floor, floor_size):
        # Outside walls
        walls = outside_walls(self.grid_size, use_low_wall_height=self.low_outside_walls)
        if self.scenario in ['quadrant', 'var_quadrant']:
            q_size = env._random_state.uniform(0.3, 0.6) if self.scenario == 'var_quadrant' else 0.5
            q_size = int(q_size * self.grid_size)
            env.metadata['quadrant_size'] = q_size
            new_walls = [
                Wall([self.grid_size - q_size, 0], [self.grid_size - q_size, q_size]),
                Wall([self.grid_size - q_size, q_size], [self.grid_size - 1, q_size])]
            if env._random_state.uniform(0, 1) < self.p_door_dropout:
                wall_to_split = env._random_state.randint(0, 2)
                walls += [new_walls[(1 - wall_to_split)]]
                walls_to_split = [new_walls[wall_to_split]]
            else:
                walls_to_split = new_walls
        elif self.scenario == 'half':
            walls_to_split += [Wall([self.grid_size - 1, self.grid_size // 2],
                                    [0, self.grid_size // 2])]
        elif self.scenario == 'var_tri':
            wall1_splitoff_point, wall2_splitoff_point = [
                int(self.grid_size * env._random_state.uniform(0.4, 0.6)) for _ in range(2)
            ]
            wall1_orientation = 'vertical' if env._random_state.uniform() < 0.5 else 'horizontal'
            # if first wall is horizontal, 'left' means below and 'right' means above
            wall2_orientation = 'left' if env._random_state.uniform() < 0.5 else 'right'

            env.metadata['tri_wall_splitoff_points'] = [wall1_splitoff_point, wall2_splitoff_point]
            env.metadata['tri_wall_orientations'] = [wall1_orientation, wall2_orientation]
            if wall1_orientation == 'horizontal':
                walls_to_split = [Wall([self.grid_size - 1, wall1_splitoff_point],
                                       [0, wall1_splitoff_point])]
                if wall2_orientation == 'left':
                    walls_to_split += [Wall([wall2_splitoff_point, wall1_splitoff_point],
                                            [wall2_splitoff_point, 0])]
                    rooms = [[(1, self.grid_size - 1),
                              (wall1_splitoff_point + 1, self.grid_size - 1)],
                             [(1, wall2_splitoff_point - 1),
                              (1, wall1_splitoff_point - 1)],
                             [(wall2_splitoff_point + 1, self.grid_size - 1),
                              (1, wall1_splitoff_point - 1)]]
                elif wall2_orientation == 'right':
                    walls_to_split += [Wall([wall2_splitoff_point, self.grid_size - 1],
                                            [wall2_splitoff_point, wall1_splitoff_point])]
                    rooms = [[(1, self.grid_size - 1),
                              (0, wall1_splitoff_point - 1)],
                             [(1, wall2_splitoff_point - 1),
                              (wall1_splitoff_point + 1, self.grid_size - 1)],
                             [(wall2_splitoff_point + 1, self.grid_size - 1),
                              (wall1_splitoff_point + 1, self.grid_size - 1)]]
            elif wall1_orientation == 'vertical':
                walls_to_split = [Wall([wall1_splitoff_point, self.grid_size - 1],
                                       [wall1_splitoff_point, 0])]
                if wall2_orientation == 'left':
                    walls_to_split += [Wall([wall1_splitoff_point, wall2_splitoff_point],
                                            [0, wall2_splitoff_point])]
                    rooms = [[(wall1_splitoff_point + 1, self.grid_size - 1),
                              (1, self.grid_size - 1)],
                             [(1, wall1_splitoff_point - 1),
                              (1, wall2_splitoff_point - 1)],
                             [(1, wall1_splitoff_point - 1),
                              (wall2_splitoff_point + 1, self.grid_size - 1)]]
                elif wall2_orientation == 'right':
                    walls_to_split += [Wall([self.grid_size - 1, wall2_splitoff_point],
                                            [wall1_splitoff_point, wall2_splitoff_point])]
                    rooms = [[(0, wall1_splitoff_point - 1),
                              (1, self.grid_size - 1)],
                             [(wall1_splitoff_point + 1, self.grid_size - 1),
                              (1, wall2_splitoff_point - 1)],
                             [(wall1_splitoff_point + 1, self.grid_size - 1),
                              (wall2_splitoff_point + 1, self.grid_size - 1)]]
            env.metadata['tri_room_grid_cell_range'] = rooms

            # this is used when we want to consecutively place objects in every room
            # e.g. if we want object i to go in room (i % 3)
            env.metadata['tri_placement_rotation'] = []
        elif self.scenario == 'empty':
            walls_to_split = []

        # Add doors
        new_walls, doors = split_walls(walls_to_split, self.door_size,
                                       random_state=env._random_state)
        walls += new_walls

        env.metadata['doors'] = np.array(doors)

        # Convert doors into mujoco frame
        if len(doors) > 0:
            self.door_obs = construct_door_obs(np.array(doors), floor_size, self.grid_size)
        else:
            self.door_obs = None

        walls_to_mujoco(floor, floor_size, self.grid_size, walls, friction=self.friction)
        add_walls_to_grid(env.placement_grid, walls)
        return True

    def observation_step(self, env, sim):
        if self.door_obs is not None:
            obs = {'door_obs': self.door_obs}
        else:
            obs = {}

        return obs
