import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

nusc = NuScenes(version='v1.0-mini', dataroot='F:/Project/nuscenes-devkit/v1.0-mini', verbose=True)

def calculate_projection_matrix(sample_index=0, cam_channel='CAM_FRONT', print_flag=False):
    """
    Calculate the projection matrix of the camera
    
    Args:
        sample_index: sample index
        cam_channel: camera channel name, default is front camera
    
    Returns:
        K: intrinsic matrix
        R_t: extrinsic matrix (rotation and translation)
        P: projection matrix
    """
    sample = nusc.sample[sample_index]
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    # Intrinsic matrix
    K = np.array(cs_record['camera_intrinsic'])
    
    # Rotation matrix
    cam_rotation = Quaternion(cs_record['rotation'])
    R = cam_rotation.rotation_matrix
    
    # Translation vector
    t = np.array(cs_record['translation']).reshape(3, 1)
    
    # Combine rotation and translation
    R_t = np.hstack((R, t))
    
    # Projection matrix
    P = K @ R_t
    
    # Get ego pose
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    # Get focal length and principal point
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    if print_flag:
        print(f"\n{'='*40}")
        print(f"Camera: {cam_channel}\n")
        print(f"Camera intrinsic matrix K:\n {K}")
        print(f"\nfocal length: fx={fx}, fy={fy}")
        print(f"principal point: cx={cx}, cy={cy}")
        print(f"\nCamera rotation matrix R:\n {R}")
        print(f"\nCamera translation vector t:\n {t}")
        print(f"\nCamera extrinsic matrix [R|t]:\n {R_t}")
        print(f"\nCamera projection matrix P = K[R|t]:\n {P}")
        print(f"\nEgo vehicle position:\n {ego_translation}")
        print(f"\nEgo vehicle rotation quaternion:\n {ego_rotation.elements}")
    
    return K, R_t, P

if __name__ == "__main__":
    K, R_t, P = calculate_projection_matrix(sample_index=0, cam_channel='CAM_FRONT', print_flag=True)
    