import numpy as np
from mujoco_worldgen.util.rotation import normalize_angles
from mujoco_worldgen.util.geometry import raycast


def in_cone2d(origin_pts, origin_angles, cone_angle, target_pts):
    '''
        Computes whether 2D points target_pts are in the cones originating from
            origin_pts at angle origin_angles with cone spread angle cone_angle.
        Args:
            origin_pts (np.ndarray): array with shape (n_points, 2) of origin points
            origin_angles (np.ndarray): array with shape (n_points,) of origin angles
            cone_angle (float): cone angle width
            target_pts (np.ndarray): target points to check whether in cones
        Returns:
            np.ndarray of bools. Each row corresponds to origin cone, and columns to
                target points
    '''
    assert isinstance(origin_pts, np.ndarray)
    assert isinstance(origin_angles, np.ndarray)
    assert isinstance(cone_angle, float)
    assert isinstance(target_pts, np.ndarray)
    assert origin_pts.shape[0] == origin_angles.shape[0]
    assert len(origin_angles.shape) == 1, "Angles should only have 1 dimension"
    np.seterr(divide='ignore', invalid='ignore')
    cone_vec = np.array([np.cos(origin_angles), np.sin(origin_angles)]).T
    # Compute normed vectors between all pairs of agents
    pos_diffs = target_pts[None, ...] - origin_pts[:, None, :]
    norms = np.sqrt(np.sum(np.square(pos_diffs), -1, keepdims=True))
    unit_diffs = pos_diffs / norms
    # Dot product between unit vector in middle of cone and the vector
    dot_cone_diff = np.sum(unit_diffs * cone_vec[:, None, :], -1)
    angle_between = np.arccos(dot_cone_diff)
    # Right now the only thing that should be nan will be targets that are on the origin point
    # This can only happen for the origin looking at itself, so just make this always true
    angle_between[np.isnan(angle_between)] = 0.

    return np.abs(normalize_angles(angle_between)) <= cone_angle


def insight(sim, geom1_id, geom2_id=None, pt2=None, dist_thresh=np.inf, check_body=True):
    '''
        Check if geom2 or pt2 is in line of sight of geom1.
        Args:
            sim: Mujoco sim object
            geom1 (int): geom id
            geom2 (int): geom id
            pt2 (tuple): xy point
            dist_thresh (float): Adds a distance threshold for vision. Objects beyond the threshold
                are considered out of sight.
            check_body (bool): Check whether the raycast hit any geom in the body that geom2 is in
                rather than if it just hit geom2
    '''
    dist, collision_geom = raycast(sim, geom1_id, geom2_id=geom2_id, pt2=pt2)
    if geom2_id is not None:
        if check_body:
            body2_id, collision_body_id = sim.model.geom_bodyid[[geom2_id, collision_geom]]
            return (collision_body_id == body2_id and dist < dist_thresh)
        else:
            return (collision_geom == geom2_id and dist < dist_thresh)
    else:
        pt1 = sim.data.geom_xpos[geom1_id]
        dist_pt2 = np.linalg.norm(pt2 - pt1)
        # if dist == -1 then we're raycasting from a geom to a point within itself,
        #   and all objects have line of sight of themselves.
        return (dist == -1.0 or dist > dist_pt2) and dist_pt2 < dist_thresh
