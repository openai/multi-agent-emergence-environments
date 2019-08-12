import numpy as np
from mujoco_worldgen.util.rotation import quat_mul, quat_conjugate


def dist_pt_to_cuboid(pt1, cuboid_center, cuboid_dims, cuboid_quat):
    '''
        This function calculates the shortest distance between test points
        and cuboids at arbitrary locations, widths and rotations

        Args:
            pt1 (num points x 3): test point positions
            cuboid_center (num cuboids x 3): cuboid centers
            cuboid_dims (num cuboids x 3): cuboid half-width
            cuboid_quat (num cuboids x 4): cuboid quaternion

        Returns:
            Distance array of size num points x num cuboids
    '''
    assert cuboid_center.shape[0] == cuboid_dims.shape[0] == cuboid_quat.shape[0], \
        "First dimension of cuboid_center, cuboid_dims and cuboid_quat need to match, " + \
        f"but were {cuboid_center.shape[0]}, {cuboid_dims.shape[0]} and {cuboid_quat.shape[0]}."
    assert pt1.shape[1] == cuboid_center.shape[1] == cuboid_dims.shape[1] == 3, \
        "Second dimension of pt1, cuboid_center and cuboid_dims needs to be 3, " + \
        f"but were {pt1.shape[1]}, {cuboid_center.shape[1]} and {cuboid_dims.shape[1]}."
    assert cuboid_quat.shape[1] == 4, \
        f"Second dimension of cuboid_quat needs to be 4, but was {cuboid_quat.shape[1]}."

    # calculate relative position of test points
    rel_pos = pt1[:, None, :] - cuboid_center[None, :, :]

    # convert into quaternion (leading dimension is zero)
    q_rel_pos = np.concatenate([np.zeros_like(rel_pos[:, :, [0]]), rel_pos], axis=-1)

    # broadcast cuboid_quat by hand
    cuboid_quat = np.repeat(cuboid_quat[None, :], pt1.shape[0], axis=0)

    # rotate relative position in cuboid frame
    # since cuboid_quat specifies how the cuboid is rotated wrt to the standard coordinate system,
    # we need to rotate the test points using the inverse rotation (i.e. conjugate quaternion)
    #
    # For rotation of vectors using quaternions see
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    q_rel_pos = quat_mul(quat_conjugate(cuboid_quat), quat_mul(q_rel_pos, cuboid_quat))

    # now we can pretend that the cuboid is aligned to x-axis
    # calculate vector to closest point on the cuboid
    # this can be done as described here:
    # https://gamedev.stackexchange.com/questions/44483/how-do-i-calculate-distance-between-a-point-and-an-axis-aligned-rectangle
    dist_vec = np.maximum(0, np.abs(q_rel_pos[:, :, 1:]) - cuboid_dims[None, :, :])

    # distance is length of distance vector
    dist = np.linalg.norm(dist_vec, axis=-1)

    return dist
