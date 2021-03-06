"""
Conversions utils for robotics.
Authors: Mikhail Kurenkov (Mikhail.Kurenkov@skoltech.ru),
        Nikolay Zherdev (Nikolay.Zherdev@skoltech.ru)
"""

import numpy as np
import sys
import cv2
if sys.version_info.major == 2:
    from geometry_msgs.msg import Pose, Transform
    import tf_conversions
    import tf2_ros
    import rospy


def rotate(points, angle):
    s, c = np.sin(angle), np.cos(angle)
    r = np.array([[c, s], [-s, c]])
    return np.dot(points[:, :2], r)
# poses_[:, :2] = np.dot(poses_[:, :2], r)


def convert2xy(scan, fov=260, min_dist=0.02):
    """Converts scan data to list of XY points descarding values with distances
    less than `min_dist`.

    Parameters
    ----------
    scan : array-like
        List of scan measurments in mm
    fov : scalar, optional
        Field Of View of the sensor in degrees
    min_dist : scalar, optional
        Minimal distance of measurment in mm for filtering out values

    Returns
    -------
    points : ndarray
        List of XY points
    """
    angles = np.radians(np.linspace(-fov/2, fov/2, len(scan)))
    points = np.vstack([scan*np.cos(angles), scan*np.sin(angles)]).T
    return points[scan>min_dist]


# def convert2map(pose, points, map_pix, map_size, prob):
#     """Converts list of XY points to 2D array map in which each pixel denotes
#     probability of pixel being occupied.
#
#     Parameters
#     ----------
#     pose : ndarray
#         XY coordinates of the robot in the map reference frame
#     points : ndarray
#         List of XY points measured by sensor in the map reference frame
#     map_pix : int
#         Size of map pixel in m
#     map_size : tuple
#         Size of the map in pixels
#     prob : float
#         Probability
#
#
#     Returns
#     -------
#     map : ndarray
#         2D array representing map with dtype numpy.float32
#     """
#     zero = (pose//map_pix).astype(np.int32)
#     pixels = (points//map_pix).astype(np.int32)
#     mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < map_size[0]) & \
#            (pixels[:, 1] >= 0) & (pixels[:, 1] < map_size[1])
#     pixels = pixels[mask]
#     img = Image.new('L', (map_size[1], map_size[0]))
#     draw = ImageDraw.Draw(img)
#     zero = (zero[1], zero[0])
#     for p in set([(q[1], q[0]) for q in pixels]):
#         draw.line([zero, p], fill=1)
#     data = -np.fromstring(img.tobytes(), np.int8).reshape(map_size)
#     data[pixels[:, 0], pixels[:, 1]] = 1
#     return 0.5 + prob*data.astype(np.float32)


def cvt_local2global(local_point, src_point):
    """
    Convert local_point (defined in robot's local frame) to the frame where src_point is defined (map, for ex).

    For example, we have a robot located in one point (src_point param) and we want it to move half a meter back,
    while keeping the same orientation. The easiest way to do it is to imagine that robot stays at [x=0, y=0, theta=0]
    in his local frame and to substract 0.5 from corresponding axis.
    Considering X axis is looking forward, we'll get [-0.5, 0, 0] - this is a local_point param
    (map -- base)

    :param local_point: A local point or array of local points that must be converted 1-D np.array or 2-D np.array,
                        change in pos and orientation in local robot fame
    :param src_point: Point from which robot starts moving
    :return: coordinates [x, y, theta] where we want robot to be in src_points' frame
    """
    size = local_point.shape[-1]
    x, y, a = 0, 0, 0
    if size == 3:
        x, y, a = local_point.T
    elif size == 2:
        x, y = local_point.T
    X, Y, A = src_point.T
    x1 = x * np.cos(A) - y * np.sin(A) + X
    y1 = x * np.sin(A) + y * np.cos(A) + Y
    a1 = (a + A + np.pi) % (2 * np.pi) - np.pi
    if size == 3:
        return np.array([x1, y1, a1]).T
    elif size == 2:
        return np.array([x1, y1]).T
    else:
        return


def cvt_global2local(global_point, src_point):
    """
    Returns coordinates of a global_point in src_point's frame.

    For example, we have robot's position in map frame (global_point) and we have one of it's
    previous positions in the same global space (src_point). The task is to find a delta between
    these two points in local frame of src_point. (odom -- base)

    Another example: once collision points are calculated in global frame, they need to be converted to robot's local
    frame to check if they are inside "robot's collision ellipse"

    :param global_point:
    :param src_point:
    :return:
    """
    size = global_point.shape[-1]
    x1, y1, a1 = 0, 0, 0
    if size == 3:
        x1, y1, a1 = global_point.T
    elif size == 2:
        x1, y1 = global_point.T
    X, Y, A = src_point.T
    x = x1 * np.cos(A) + y1 * np.sin(A) - X * np.cos(A) - Y * np.sin(A)
    y = -x1 * np.sin(A) + y1 * np.cos(A) + X * np.sin(A) - Y * np.cos(A)
    a = (a1 - A + np.pi) % (2 * np.pi) - np.pi
    if size == 3:
        return np.array([x, y, a]).T
    elif size == 2:
        return np.array([x, y]).T
    else:
        return


def find_src(global_point, local_point):
    """
    Transformation map looks as following: map --> odom --> base

    For example, we have VICON ground truth robot position (global_point) and position by odometry in odom_frame,
    and we need to find an error between them.

    :param global_point:
    :param local_point:
    :return:
    """
    x, y, a = local_point.T
    x1, y1, a1 = global_point.T
    A = (a1 - a) % (2 * np.pi)
    X = x1 - x * np.cos(A) + y * np.sin(A)
    Y = y1 - x * np.sin(A) - y * np.cos(A)
    return np.array([X, Y, A]).T


def cvt_point2ros_pose(point):
    pose = Pose()
    pose.position.x = point[0]
    pose.position.y = point[1]
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, point[2])
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose


def cvt_ros_pose2point(pose):
    x = pose.position.x
    y = pose.position.y
    q = [pose.orientation.x,
         pose.orientation.y,
         pose.orientation.z,
         pose.orientation.w]
    _, _, a = tf_conversions.transformations.euler_from_quaternion(q)
    return np.array([x, y, a])


def cvt_point2ros_transform(point):
    transform = Transform()
    transform.translation.x = point[0]
    transform.translation.y = point[1]
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, point[2])
    transform.rotation.x = q[0]
    transform.rotation.y = q[1]
    transform.rotation.z = q[2]
    transform.rotation.w = q[3]
    return transform


def cvt_ros_transform2point(transform):
    x = transform.translation.x
    y = transform.translation.y
    q = [transform.rotation.x,
         transform.rotation.y,
         transform.rotation.z,
         transform.rotation.w]
    _, _, a = tf_conversions.transformations.euler_from_quaternion(q)
    return np.array([x, y, a])


def cvt_ros_scan2points(scan):
    ranges = np.array(scan.ranges)
    ranges[ranges != ranges] = 0
    ranges[ranges == np.inf] = 0
    n = ranges.shape[0]
    angles = np.arange(scan.angle_min, scan.angle_min + n * scan.angle_increment, scan.angle_increment)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.array([x, y]).T


def get_transform(buffer_, child_frame, parent_frame, stamp):
    try:
        t = buffer_.lookup_transform(parent_frame, child_frame, stamp)
        return cvt_ros_transform2point(t.transform)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as msg:
        rospy.logwarn(str(msg))
        return None


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def cvt_world_points2map(world_points, origin, res):
    map_points = cvt_global2local(world_points, origin)
    return (map_points) / res
    # return map_points / res


def cvt_map_points2world(map_points, origin, res):
    points = map_points * res + np.ones(2) * res / 2.
    return cvt_local2global(points, origin)


def insert_to_map(map_, inds, value):
    shape = map_.shape
    assert len(shape) == 2
    assert len(inds.shape) == 2
    inds = inds[(inds[:, 0] >= 0) & (inds[:, 0] < shape[1]) & (inds[:, 1] >= 0) & (inds[:, 1] < shape[0])]
    map_[inds[:, 1], inds[:, 0]] = value


def combine_maps(img, shape_output, transform, res_output, res_input):
    transform_matrix = np.array([
        [np.cos(transform[2]), -np.sin(transform[2]), transform[0] / res_input],
        [np.sin(transform[2]), np.cos(transform[2]), transform[1] / res_input]
    ]) * res_input / res_output
    return cv2.warpAffine(img, transform_matrix, (shape_output[1], shape_output[0]))

# ----------


def euler_angles_to_rotation_matrix(theta):
    """

    :param theta: ax (roll), ay (pitch), az (yaw)
    :return: (3x3) matrix
    """
    r_x = np.array([[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]),
                     np.cos(theta[0])]])
    r_y = np.array([[np.cos(theta[1]), 0,
                     np.sin(theta[1])], [0, 1, 0],
                    [-np.sin(theta[1]), 0,
                     np.cos(theta[1])]])
    r_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])
    r = np.dot(r_z, np.dot(r_y, r_x))
    return r


def make_pose3(rotation_matrix, translation_vector):
    """

    :param rotation_matrix: shape TODO
    :param translation_vector: shape TODO
    :return:
    """
    return np.concatenate((rotation_matrix, translation_vector[:, None]), axis=1)


def get_rotation_matrix(pose):
    return pose[:, :3]


def get_translation_matrix(pose):
    return pose[:, 3]


def pose_multiplication3(pose1, pose2):
    """
    pose1 * pose2
    :param pose1: np.ndarray shape (3x4)
    :param pose2: np.ndarray shape (3x4)
    :return: np.ndarray shape (3x4)
    """
    assert type(pose1) is np.ndarray
    assert type(pose2) is np.ndarray
    assert pose1.shape == (3, 4)
    assert pose2.shape == (3, 4)
    r1, t1 = pose1[:, :3], pose1[:, 3]
    r2, t2 = pose2[:, :3], pose2[:, 3]
    r = r1.dot(r2)
    t = r1.dot(t2) + t1
    return make_pose3(r, t)


def inverse_pose3(pose):
    assert type(pose) is np.ndarray
    assert pose.shape == (3, 4)
    r, t = pose[:, :3], pose[:, 3]
    return make_pose3(np.linalg.inv(r), (-np.linalg.inv(r)).dot(t))


def cvt_local2global3(local_pose, src_pose):
    """
    Transform pose of a robot from local 3d frame to global 3d frame, global_pose
    pose - concatenation of rotation matrix and translation vector (coordinates of start of frame)
    :param local_pose: np.ndarray shape (3x4)
    :param src_pose: np.ndarray shape (3x4)
    :return:np.ndarray shape (3x4)
    """
    return pose_multiplication3(src_pose, local_pose)


def cvt_global2local3(global_pose, src_pose):
    return pose_multiplication3(inverse_pose3(src_pose), global_pose)


def find_src3(global_pose, local_pose):
    return pose_multiplication3(global_pose, inverse_pose3(local_pose))


def cvt_point2pose3(point):
    return make_pose3(euler_angles_to_rotation_matrix([0, 0, point[2]]),
                      np.array([point[0], point[1], 0]))


def cvt_ros_transform2pose(transform):
    x = transform.translation.x
    y = transform.translation.y
    z = transform.translation.z
    q = [transform.rotation.x,
         transform.rotation.y,
         transform.rotation.z,
         transform.rotation.w]
    ax, ay, az = tf_conversions.transformations.euler_from_quaternion(q)
    return np.array([x, y, z, ax, ay, az])


def transform_pc(point_cloud, transform):
    """
    Transform point cloud from local frame to robot frame
    :param point_cloud: np.array (n x 3)
    :param transform: np.array([x, y, z, ax, ay, az])
    :return: (n x 3)
    """
    theta = transform[3:]
    r = euler_angles_to_rotation_matrix(theta)
    pc_transformed = np.dot(r, point_cloud.T)
    return pc_transformed.T + transform[:3]


# if __name__ == "__main__":
#     point1 = np.array([2, 3, 3.2])
#     point2 = np.array([3, 4, 5.2])
#     pose1 = cvt_point2pose3(point1)
#     pose2 = cvt_point2pose3(point2)
#
#     print(cvt_point2pose3(cvt_global2local2(point1, point2)))
#     print(cvt_global2local3(pose1, pose2))
#     print(cvt_point2pose3(cvt_global2local2(point1, point2)) - cvt_global2local3(pose1, pose2))
#
#     print(cvt_point2pose3(cvt_local2global2(point1, point2)))
#     print(cvt_local2global3(pose1, pose2))
#     print(cvt_point2pose3(cvt_local2global2(point1, point2)) - cvt_local2global3(pose1, pose2))
#
#     print(cvt_point2pose3(find_src2(point1, point2)))
#     print(find_src3(pose1, pose2))
#     print(cvt_point2pose3(find_src2(point1, point2)) - find_src3(pose1, pose2))
