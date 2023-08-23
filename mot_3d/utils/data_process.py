from pyquaternion import Quaternion
import numpy as np


def global_to_lidar(bbox, calib_data, ego_data):
    """
    Convert bounding box from global coordinate system to lidar coordinate system.

    Args:
        bbox: ndarray (7), bounding box in global coordinate system, [x, y, z, w, l, h, heading].
        calib_data: ndarray (7,), calibration information.
        ego_data:   ndarray (7,), ego information.

    Returns:
        ndarray (7, ), bounding box in lidar coordinate system. [x, y, z, w, l, h, heading].
    """
    if not isinstance(calib_data, dict):
        calib_data = {
            'translation': calib_data[:3],
            'rotation': calib_data[3:],
        }
    if not isinstance(ego_data, dict):
        ego_data = {
            'translation': ego_data[:3],
            'rotation': ego_data[3:],
        }

    # global frame
    center, size, heading = bbox[:3], bbox[3:6], bbox[6]
    orientation = Quaternion(axis=[0, 0, 1], radians=heading).elements
    # orientation = Quaternion(axis=[0, 0, 1], radians=heading / 2).elements

    # from global to ego
    quaternion = Quaternion(ego_data['rotation']).inverse
    center -= np.array(ego_data['translation'])
    center = np.dot(quaternion.rotation_matrix, center)
    orientation = quaternion * orientation

    # from ego to lidar
    quaternion = Quaternion(calib_data['rotation']).inverse
    center -= np.array(calib_data['translation'])
    center = np.dot(quaternion.rotation_matrix, center)
    orientation = quaternion * orientation

    lidar_bbox = np.hstack((center, size, orientation.yaw_pitch_roll[0]))
    return lidar_bbox


if __name__ == '__main__':
    box = np.random.random([7])
    box[-1] = np.pi / 2
    calib = np.array([0.985793, 0., 1.84019, 0.70674924, -0.01530099, 0.01739745, -0.70708467])
    ego = np.random.random([7])
    bbox = global_to_lidar(box, calib, ego)
    print(box, bbox)
