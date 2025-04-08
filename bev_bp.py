import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
import concurrent.futures
from backward_projection import create_bev_view, smooth_bev_image
from pyquaternion import Quaternion

nusc = NuScenes(version='v1.0-mini', dataroot='F:/Project/nuscenes-devkit/v1.0-mini', verbose=False)
camera_channels = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
        ]

def get_camera_info(sample_token, cam_channel):
    """
    Get camera position and orientation information in vehicle coordinate system
    
    Args:
    - sample_token: sample token
    - cam_channel: camera channel
    
    Returns:
    - camera_info: dictionary containing camera information
    """
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    # Get camera intrinsic parameters
    intrinsic = np.array(cs_record['camera_intrinsic'])
    
    # Get camera extrinsic parameters
    translation = np.array(cs_record['translation'])
    rotation = Quaternion(cs_record['rotation']).rotation_matrix
    
    # Get ego pose
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation']).rotation_matrix
    
    # Return camera information
    camera_info = {
        'translation': translation,
        'rotation': rotation,
        'intrinsic': intrinsic,
        'ego_translation': ego_translation,
        'ego_rotation': ego_rotation
    }
    
    return camera_info

def process_single_camera(sample_token, cam_channel, bev_width=40, bev_length=40, resolution=0.04):
    """
    Process the BEV projection of a single camera
    
    Args:
    - sample_token: sample token
    - cam_channel: camera channel
    - bev_width: BEV width (meters)
    - bev_length: BEV length (meters)   
    - resolution: resolution (meters/pixel)
    
    Returns:
    - bird_eye_view: processed BEV image
    - projection_mask: projection mask
    - camera_info: camera information   
    """
    camera_info = get_camera_info(sample_token, cam_channel)
    
    bird_eye_view, _ = create_bev_view(
        sample_token=sample_token, cam_channel=cam_channel, 
        bev_width=bev_width, 
        bev_length=bev_length, 
        resolution=resolution
    )
    bird_eye_view = smooth_bev_image(bird_eye_view, cam_channel=cam_channel)
    
    # Create projection mask (Alpha > 0 area)
    projection_mask = (bird_eye_view[:, :, 3] > 0).astype(np.uint8)
    
    return bird_eye_view, projection_mask, camera_info
    
def process_camera_worker(params):
    """
    Create a new thread to process the BEV projection of a single camera
    
    Args:
    - params: dictionary containing processing parameters
    
    Returns:
    - camera channel
    - processing result (bev, mask, camera_info)
    """
    sample_token = params['sample_token']
    cam_channel = params['cam_channel']
    bev_width = params['bev_width']
    bev_length = params['bev_length']
    resolution = params['resolution']
    
    print(f"\nThread starts processing camera: {cam_channel}")
    
    bev, mask, camera_info = process_single_camera(
        sample_token, cam_channel, bev_width, bev_length, resolution
    )
    print(f"Camera {cam_channel} processed")
    return {
        'cam_channel': cam_channel,
        'bev': bev,
        'mask': mask,
        'camera_info': camera_info
    }

def process_all_cameras_parallel(sample_token, bev_width=40, bev_length=40, resolution=0.04, max_workers=6):
    """
    Parallelly process the BEV projection of all cameras
    
    Args:
    - sample_token: sample token
    - bev_width: BEV width (meters)
    - bev_length: BEV length (meters)
    - resolution: resolution (meters/pixel)
    - max_workers: maximum number of threads
    
    Returns:
    - bev_results: dictionary containing all camera BEV results
    """
    bev_results = {}
    
    # Prepare thread task parameters
    tasks = []
    for cam_channel in camera_channels:
        params = {
            'sample_token': sample_token,
            'cam_channel': cam_channel,
            'bev_width': bev_width,
            'bev_length': bev_length,
            'resolution': resolution
        }
        tasks.append(params)
        
    # Use thread pool to parallelly process
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_camera_worker, params) for params in tasks]
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            cam_channel = result['cam_channel']
            bev_results[cam_channel] = {
                'bev': result['bev'],
                'mask': result['mask'],
                'camera_info': result['camera_info']
            }
    
    print(f"Successfully processed {len(bev_results)}/{len(camera_channels)} cameras")
    
    return bev_results

def position_based_stitch_bev_images(bev_results):
    """
    Assemble multiple BEV images based on camera position information
    
    Args:
    - bev_results: dictionary containing all camera BEV results
    - resolution: resolution (meters/pixel)
    
    Returns:
    - stitched_bev: stitched BEV image
    """
    
    # Initialize variables
    bev_height, bev_width_pixels = bev_results[list(bev_results.keys())[0]]['bev'].shape[:2]
    stitched_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
    camera_contributions = np.zeros((bev_height, bev_width_pixels, len(bev_results), 4), dtype=np.float32)
    camera_weights = np.zeros((bev_height, bev_width_pixels, len(bev_results)), dtype=np.float32)
    camera_positions = {}
    
    # Priority order: front camera > back camera > side front camera > side back camera
    camera_priority = {
        'CAM_FRONT': 1,
        'CAM_FRONT_LEFT': 3,
        'CAM_FRONT_RIGHT': 3,
        'CAM_BACK': 2,
        'CAM_BACK_LEFT': 4,
        'CAM_BACK_RIGHT': 4
    }
    
    # Collect contributions from each camera
    for i, (cam_channel, result) in enumerate(bev_results.items()):
        # Extract camera information and BEV image
        bev = result['bev'].astype(np.float32)
        camera_info = result['camera_info']
        translation = camera_info['translation']
        # Position of the camera in the ego coordinate system
        camera_positions[cam_channel] = translation
        
        # Collect valid pixels
        valid_pixels = bev[:, :, 3] > 0    
        if not np.any(valid_pixels):
            continue
            
        base_priority = camera_priority.get(cam_channel)
        
        # Collect coordinates of valid pixels
        y_indices, x_indices = np.where(valid_pixels)
        
        # Valid pixels
        mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
        mask[y_indices, x_indices] = 1
        
        # Distance from each pixel to the edge
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # Normalize
        max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1.0
        dist_transform = dist_transform / max_dist
        
        # Calculate contribution for each pixel
        for j in range(len(y_indices)):
            y, x = y_indices[j], x_indices[j]
            
            # Calculate weight
            edge_weight = dist_transform[y, x]
            pixel_weight = base_priority * (1.0 - 0.3 * edge_weight)
            pixel_weight = 1 / pixel_weight
            
            # Store contribution and weight
            camera_contributions[y, x, i, :] = bev[y, x, :]
            camera_weights[y, x, i] = pixel_weight
    
    # Merge contributions from all cameras
    for y in range(bev_height):
        for x in range(bev_width_pixels):
            pixel_weights = camera_weights[y, x, :]
            
            if np.sum(pixel_weights) == 0:
                continue
            # If only one camera has contribution then use it
            non_zero_indices = np.nonzero(pixel_weights)[0]
            if len(non_zero_indices) == 1:
                i = non_zero_indices[0]
                stitched_bev[y, x, :] = camera_contributions[y, x, i, :].astype(np.uint8)
                continue
            
            pixel_weights = pixel_weights / np.sum(pixel_weights)
            # Merge contributions
            blended_pixel = np.zeros(4, dtype=np.float32)
            for i in range(len(bev_results)):
                if pixel_weights[i] > 0:
                    blended_pixel += pixel_weights[i] * camera_contributions[y, x, i, :]
            if blended_pixel[3] > 0:
                stitched_bev[y, x, :] = blended_pixel.astype(np.uint8)
            else:
                stitched_bev[y, x, :] = blended_pixel.astype(np.uint8)
    
    print("\nCamera positions in the ego coordinate system:")
    for cam, pos in camera_positions.items():
        print(f"{cam}: {pos}")
    
    return stitched_bev

def create_colored_bev_visualization(bev_results):
    """
    Create colorful BEV mask to show the camera coverage
    
    Args:
    - bev_results: dictionary containing all camera BEV results
    
    Returns:
    - colored_bev: colored BEV image
    """
    # Set each camera color
    camera_colors = {
        'CAM_FRONT': [255, 0, 0],          # Red
        'CAM_FRONT_LEFT': [255, 165, 0],   # Orange
        'CAM_FRONT_RIGHT': [255, 255, 0],  # Yellow
        'CAM_BACK': [0, 0, 255],           # Blue
        'CAM_BACK_LEFT': [75, 0, 130],     # Indigo
        'CAM_BACK_RIGHT': [238, 130, 238]  # Purple
    }
    
    bev_height, bev_width = bev_results[list(bev_results.keys())[0]]['bev'].shape[:2]
    colored_bev = np.zeros((bev_height, bev_width, 4), dtype=np.uint8)
    
    # Create mask for each camera
    for cam_channel, result in bev_results.items():
        mask = result['bev'][:, :, 3] > 0
            
        if cam_channel in camera_colors:
            color = np.array(camera_colors[cam_channel], dtype=np.uint8)
            for c in range(3):
                colored_bev[:, :, c][mask] = color[c]
            
            # Set Alpha channel to 255
            colored_bev[:, :, 3][mask] = 255
    
    return colored_bev

def visualize_multicam_bev(sample_token, bev_results, stitched_bev=None, save_flag=False):
    """
    Visualize multicamera BEV results
    
    Args:
    - sample_token: sample token
    - bev_results: dictionary containing all camera BEV results
    - stitched_bev: stitched BEV image (optional)
    - save_flag: whether to save image
    """
    # Get the number of rows and columns for the plot
    n_cols = 3
    n_rows = 4  # For the stitched BEV and the colored visualization
    plt.figure(figsize=(n_cols * 5, n_rows * 5))
    
    # Plot each camera's BEV
    for i, (cam_channel, result) in enumerate(bev_results.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f'BEV: {cam_channel}')
        bev_img = result['bev']
        plt.imshow(bev_img)
        plt.axis('off')
        
        # Get the center coordinates of the image (representing the vehicle position)
        bev_height, bev_width_pixels = bev_img.shape[:2]
        center_x = bev_width_pixels // 2
        center_y = bev_height // 2
        
        # Plot the vehicle position
        plt.plot(center_x, center_y, 'o', color='red', markersize=3)
        
        # Plot the camera position
        translation = result['camera_info']['translation']
        cam_x = translation[0]
        cam_y = translation[1]
        # Calculate the camera position in the image (relative to the center point) - adapted to horizontal mirroring
        cam_pixel_x = center_x - int(cam_y / 0.04)  # Y-axis corresponds to the horizontal axis, and the negative sign becomes a subtraction
        cam_pixel_y = center_y - int(cam_x / 0.04)  # X-axis corresponds to the vertical axis (positive upwards)
        if 0 <= cam_pixel_y < bev_height and 0 <= cam_pixel_x < bev_width_pixels:
            plt.plot(cam_pixel_x, cam_pixel_y, 'o', color='blue', markersize=3)
    
    if stitched_bev is not None:
        plt.subplot(n_rows, n_cols, 7)
        plt.title('Stitched BEV (All Cameras)')
        plt.imshow(stitched_bev)
        plt.axis('off')
        
        # Plot the camera position
        for cam_channel, result in bev_results.items():
            translation = result['camera_info']['translation']
            cam_x = translation[0]
            cam_y = translation[1]
            cam_pixel_x = center_x - int(cam_y / 0.04)  # Y-axis corresponds to the horizontal axis, and the negative sign becomes a subtraction
            cam_pixel_y = center_y - int(cam_x / 0.04)  # X-axis corresponds to the vertical axis (positive upwards)
                
            plt.plot(cam_pixel_x, cam_pixel_y, 'o', color='blue', markersize=3)
        
        # Plot the vehicle center position
        bev_height, bev_width_pixels = stitched_bev.shape[:2]
        center_x = bev_width_pixels // 2
        center_y = bev_height // 2
        plt.plot(center_x, center_y, 'o', color='red', markersize=3)
    
    # Plot the visualization
    colored_viz = create_colored_bev_visualization(bev_results)
    plt.subplot(n_rows, n_cols, 8)
    plt.title('Camera Coverage Visualization')
    plt.imshow(colored_viz)
    plt.axis('off')
    
    plt.tight_layout()
    if save_flag:
        plt.savefig(f'multicam_bev_stitched_{sample_token[:8]}.png', transparent=True)
    plt.show()
    
    # Show the BEV individually
    if stitched_bev is not None:
        plt.figure(figsize=(10, 10))
        plt.title('Stitched Bird\'s Eye View (All Cameras)')
        plt.imshow(stitched_bev)
        plt.axis('off')
        
        # Plot all camera positions
        for cam_channel, result in bev_results.items():
            translation = result['camera_info']['translation']
            cam_x = translation[0]
            cam_y = translation[1]
            cam_pixel_x = center_x - int(cam_y / 0.04)
            cam_pixel_y = center_y - int(cam_x / 0.04)
                
            plt.plot(cam_pixel_x, cam_pixel_y, 'o', color='blue', markersize=3)
        
        # Plot the vehicle center position
        bev_height, bev_width_pixels = stitched_bev.shape[:2]
        center_x = bev_width_pixels // 2
        center_y = bev_height // 2
        plt.plot(center_x, center_y, 'o', color='red', markersize=3)
        
        if save_flag:
            plt.savefig(f'stitched_bev_{sample_token[:8]}.png', transparent=True)
        plt.show()

def detect_camera_edges(bevs, dilate_size=3):
    """
    检测各个相机图像之间的边缘区域
    
    参数:
    - bevs: 包含所有相机BEV结果的字典
    - dilate_size: 扩张边缘的大小（像素）
    
    返回:
    - edge_maps: 每个相机的边缘区域字典
    - overlap_map: 相机重叠区域图
    """
    # 获取BEV尺寸
    first_cam = list(bevs.keys())[0]
    bev_height, bev_width = bevs[first_cam]['bev'].shape[:2]
    
    # 创建存储边缘的字典
    edge_maps = {}
    
    # 创建相机重叠区域图
    overlap_map = np.zeros((bev_height, bev_width), dtype=np.uint8)
    valid_pixel_count = np.zeros((bev_height, bev_width), dtype=np.uint8)
    
    # 首先标记所有有效区域
    for cam_channel, result in bevs.items():
        bev = result['bev']
        
        # 获取有效像素掩码
        if bev.shape[2] == 4:  # RGBA
            valid_mask = bev[:, :, 3] > 0
        else:
            valid_mask = result['mask'] > 0
        
        # 更新有效像素计数
        valid_pixel_count[valid_mask] += 1
    
    # 标记相机重叠区域（至少被两个相机覆盖）
    overlap_map = (valid_pixel_count >= 2).astype(np.uint8)
    
    # 为每个相机找出其边缘区域
    for cam_channel, result in bevs.items():
        bev = result['bev']
        
        # 获取有效像素掩码
        if bev.shape[2] == 4:  # RGBA
            valid_mask = bev[:, :, 3] > 0
        else:
            valid_mask = result['mask'] > 0
            
        # 使用膨胀和腐蚀找到边缘
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        dilated = cv2.dilate(valid_mask.astype(np.uint8), kernel)
        eroded = cv2.erode(valid_mask.astype(np.uint8), kernel)
        edge = dilated - eroded
        
        # 只保留与其他相机重叠的边缘部分
        edge = edge & overlap_map
        
        # 存储边缘区域
        edge_maps[cam_channel] = edge
    
    return edge_maps, overlap_map

def smooth_camera_transitions(bevs, edge_maps, blur_radius=5):
    """
    平滑相机之间的过渡区域
    
    参数:
    - bevs: 包含所有相机BEV结果的字典
    - edge_maps: 通过detect_camera_edges得到的边缘区域
    - blur_radius: 模糊半径
    
    返回:
    - smoothed_bevs: 平滑处理后的BEV结果
    """
    # 创建平滑后的结果字典
    smoothed_bevs = {}
    
    # 对每个相机进行边缘平滑
    for cam_channel, result in bevs.items():
        bev = result['bev'].copy()
        edge = edge_maps[cam_channel]
        
        # 检查是否存在边缘
        if np.any(edge):
            # 创建高斯模糊的版本
            blurred = bev.copy()
            
            # 对RGB通道应用高斯模糊
            for c in range(3):  # 只处理RGB通道
                # 只在边缘区域应用模糊
                channel = bev[:, :, c].copy()
                blurred_channel = cv2.GaussianBlur(channel, (blur_radius*2+1, blur_radius*2+1), 0)
                # 在边缘区域使用模糊版本
                channel[edge == 1] = blurred_channel[edge == 1]
                bev[:, :, c] = channel
        
        # 保存结果
        smoothed_result = result.copy()
        smoothed_result['bev'] = bev
        smoothed_bevs[cam_channel] = smoothed_result
    
    return smoothed_bevs

def create_transition_weights(bevs, edge_maps, transition_width=10):
    """
    创建用于相机之间平滑过渡的权重图
    
    参数:
    - bevs: 包含所有相机BEV结果的字典
    - edge_maps: 边缘区域
    - transition_width: 过渡区域的宽度
    
    返回:
    - weight_maps: 每个相机的权重图
    """
    # 获取BEV尺寸
    first_cam = list(bevs.keys())[0]
    bev_height, bev_width = bevs[first_cam]['bev'].shape[:2]
    
    # 创建权重图字典
    weight_maps = {}
    
    # 为每个相机创建初始权重图（全部为1）
    for cam_channel, result in bevs.items():
        bev = result['bev']
        
        # 获取有效像素掩码
        if bev.shape[2] == 4:  # RGBA
            valid_mask = bev[:, :, 3] > 0
        else:
            valid_mask = result['mask'] > 0
            
        # 初始化权重图
        weight_map = np.ones((bev_height, bev_width), dtype=np.float32)
        
        # 在有效区域外设为0
        weight_map[~valid_mask] = 0
        
        # 存储权重图
        weight_maps[cam_channel] = weight_map
    
    # 对于每个相机的边缘，创建距离场
    for cam_channel, edge in edge_maps.items():
        if np.any(edge):
            # 计算到边缘的距离
            dist = cv2.distanceTransform(1 - edge.astype(np.uint8), cv2.DIST_L2, 3)
            
            # 将距离转换为过渡权重
            edge_weight = np.clip(dist / transition_width, 0, 1)
            
            # 更新该相机的权重图（边缘区域渐变）
            valid_mask = weight_maps[cam_channel] > 0
            weight_maps[cam_channel][valid_mask & (edge_weight < 1)] = edge_weight[valid_mask & (edge_weight < 1)]
    
    return weight_maps

def compute_pixel_distances(sample_token, bev_results, bev_width=40, bev_length=40, resolution=0.04):
    """
    计算每个BEV像素到对应相机的距离（作为深度估计）
    
    参数:
    - sample_token: 样本token
    - bev_results: 包含所有相机BEV结果的字典
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    
    返回:
    - distance_maps: 每个相机BEV像素的距离图
    """
    # 获取第一张图像的尺寸
    first_cam = list(bev_results.keys())[0]
    bev_height, bev_width_pixels = bev_results[first_cam]['bev'].shape[:2]
    
    # 创建字典存储每个相机的距离图
    distance_maps = {}
    
    # 计算每个相机对应的距离图
    for cam_channel, result in bev_results.items():
        # 提取相机数据
        camera_info = result['camera_info']
        
        # 获取相机在车辆坐标系中的位置
        translation = camera_info['translation']
        
        # 初始化距离图（无限大初始值）
        distance_map = np.full((bev_height, bev_width_pixels), np.inf, dtype=np.float32)
        
        # 获取有效像素掩码
        if result['bev'].shape[2] == 4:  # RGBA格式
            valid_mask = result['bev'][:, :, 3] > 0
        else:  # RGB格式
            valid_mask = result['mask'] > 0
        
        # 获取有效像素的坐标
        y_indices, x_indices = np.where(valid_mask)
        
        # 计算图像中心（车辆位置）
        center_x = bev_width_pixels // 2
        center_y = bev_height // 2
        
        # 计算每个有效BEV像素到相机的距离
        for i in range(len(y_indices)):
            y, x = y_indices[i], x_indices[i]
            
            # 将BEV像素坐标转换为车辆坐标系的物理坐标
            physical_x = (center_y - y) * resolution  # 纵向位置（前/后）
            physical_y = (x - center_x) * resolution  # 横向位置（左/右）
            
            # 计算地面点到相机的欧氏距离
            dx = physical_x - translation[0]
            dy = physical_y - translation[1]
            dz = 0 - translation[2]  # 地平面z=0
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # 更新距离图
            distance_map[y, x] = distance
        
        # 存储结果
        distance_maps[cam_channel] = distance_map
    
    return distance_maps

def smooth_blend_stitch_bev_images(bev_results, sample_token=None, bev_width=40, bev_length=40, resolution=0.04, use_depth=True, transition_width=10):
    """
    使用平滑混合策略拼接多个相机的BEV图像
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - sample_token: 样本token (仅当use_depth=True时需要)
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    - use_depth: 是否使用深度信息进行融合
    - transition_width: 过渡区域的宽度
    
    返回:
    - stitched_bev: 拼接后的BEV图像
    """
    if not bev_results:
        raise ValueError("没有有效的BEV结果可拼接")
    
    # 获取BEV尺寸（假设所有BEV图像尺寸相同）
    first_cam = list(bev_results.keys())[0]
    bev_height, bev_width_pixels = bev_results[first_cam]['bev'].shape[:2]
    
    # 检查是否为RGBA图像
    is_rgba = bev_results[first_cam]['bev'].shape[2] == 4
    
    # 初始化拼接结果
    if is_rgba:
        stitched_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
    else:
        stitched_bev = np.ones((bev_height, bev_width_pixels, 3), dtype=np.uint8) * 255
    
    # 相机位置的字典 
    camera_positions = {}
    
    # 优先级顺序：前部相机 > 侧前相机 > 侧后相机 > 后部相机
    camera_priority = {
        'CAM_FRONT': 1,
        'CAM_FRONT_LEFT': 2,
        'CAM_FRONT_RIGHT': 2,
        'CAM_BACK': 4,
        'CAM_BACK_LEFT': 3,
        'CAM_BACK_RIGHT': 3
    }
    
    # 先检测各个相机的边缘区域
    print("检测相机边缘区域...")
    edge_maps, overlap_map = detect_camera_edges(bev_results, dilate_size=5)
    
    # 对边缘区域应用平滑处理
    print("应用边缘平滑处理...")
    smoothed_bevs = smooth_camera_transitions(bev_results, edge_maps, blur_radius=7)
    
    # 计算过渡权重
    print("创建过渡权重...")
    weight_maps = create_transition_weights(smoothed_bevs, edge_maps, transition_width=transition_width)
    
    # 如果需要使用深度信息
    if use_depth and sample_token is not None:
        # 计算每个相机的像素深度
        print("计算像素深度...")
        distance_maps = compute_pixel_distances(sample_token, smoothed_bevs, bev_width, bev_length, resolution)
        
        # 基于最小深度和平滑权重进行融合
        print("基于最小深度和平滑权重融合...")
        
        # 初始化深度图和融合权重图
        min_depth_map = np.full((bev_height, bev_width_pixels), np.inf, dtype=np.float32)
        fusion_weights = np.zeros((bev_height, bev_width_pixels, len(smoothed_bevs)), dtype=np.float32)
        
        # 首先找出每个像素的最小深度
        for i, (cam_channel, result) in enumerate(smoothed_bevs.items()):
            distance_map = distance_maps[cam_channel]
            
            if is_rgba:
                valid_pixels = result['bev'][:, :, 3] > 0
            else:
                valid_pixels = result['mask'] > 0
            
            # 记录相机在车辆坐标系中的位置
            camera_positions[cam_channel] = result['camera_info']['translation']
            
            # 更新最小深度图
            update_pixels = np.logical_and(valid_pixels, distance_map < min_depth_map)
            min_depth_map[update_pixels] = distance_map[update_pixels]
            
            # 存储相机权重
            fusion_weights[:, :, i][valid_pixels] = weight_maps[cam_channel][valid_pixels]
        
        # 归一化融合权重
        weight_sum = np.sum(fusion_weights, axis=2, keepdims=True)
        valid_weights = weight_sum[:, :, 0] > 0
        fusion_weights[valid_weights] = fusion_weights[valid_weights] / weight_sum[valid_weights]
        
        # 使用归一化权重融合所有相机图像
        for i, (cam_channel, result) in enumerate(smoothed_bevs.items()):
            bev = result['bev']
            cam_weight = np.expand_dims(fusion_weights[:, :, i], axis=2)
            
            # 对RGB通道应用权重融合
            for c in range(3):
                stitched_bev[:, :, c] += (bev[:, :, c] * cam_weight[:, :, 0]).astype(np.uint8)
            
            # 更新Alpha通道（如果有）
            if is_rgba:
                stitched_bev[:, :, 3][valid_weights] = 255
    else:
        # 不使用深度，仅基于过渡权重融合
        print("基于平滑权重融合...")
        
        # 归一化所有权重图，使它们的和为1
        weight_sum = np.zeros((bev_height, bev_width_pixels), dtype=np.float32)
        for weight_map in weight_maps.values():
            weight_sum += weight_map
        
        # 防止除以零
        weight_sum[weight_sum == 0] = 1.0
        
        # 使用归一化权重融合所有相机图像
        for cam_channel, result in smoothed_bevs.items():
            bev = result['bev']
            camera_positions[cam_channel] = result['camera_info']['translation']
            normalized_weight = weight_maps[cam_channel] / weight_sum
            weight_3d = np.expand_dims(normalized_weight, axis=2)
            
            # 对RGB通道应用权重融合
            for c in range(3):
                stitched_bev[:, :, c] += (bev[:, :, c] * weight_3d[:, :, 0]).astype(np.uint8)
            
            # 更新Alpha通道（如果有）
            if is_rgba:
                alpha_weight = normalized_weight * (bev[:, :, 3] > 0).astype(np.float32)
                stitched_bev[:, :, 3] += (255 * alpha_weight).astype(np.uint8)
        
        # 确保Alpha通道不超过255
        if is_rgba:
            stitched_bev[:, :, 3] = np.clip(stitched_bev[:, :, 3], 0, 255)
    
    # 应用后处理平滑边缘
    print("应用后处理平滑边缘...")
    if is_rgba:
        # 提取RGB和Alpha通道
        rgb_channels = stitched_bev[:, :, :3]
        alpha_channel = stitched_bev[:, :, 3]
        
        # 应用高斯模糊来平滑边缘
        kernel_size = max(3, int(5 * resolution / 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 创建二值掩码（Alpha > 0的区域）
        mask = (alpha_channel > 0).astype(np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 创建边缘掩码
        edge_mask = closed_mask - mask
        
        # 对边缘区域应用模糊
        if np.any(edge_mask):
            blur_size = max(3, int(3 * resolution / 0.1))
            if blur_size % 2 == 0:
                blur_size += 1
            blurred_rgb = cv2.GaussianBlur(rgb_channels, (blur_size, blur_size), 0)
            
            # 在边缘区域使用模糊结果
            edge_indices = np.where(edge_mask > 0)
            rgb_channels[edge_indices] = blurred_rgb[edge_indices]
        
        # 更新Alpha通道
        alpha_channel = cv2.morphologyEx(alpha_channel, cv2.MORPH_CLOSE, kernel)
        
        # 重新组合RGB和Alpha
        stitched_bev = np.dstack((rgb_channels, alpha_channel))
    else:
        # 对RGB图像应用高斯模糊
        kernel_size = max(5, int(7 * resolution / 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        stitched_bev = cv2.morphologyEx(stitched_bev, cv2.MORPH_CLOSE, kernel)
        
        blur_size = max(3, int(5 * resolution / 0.1))
        if blur_size % 2 == 0:
            blur_size += 1
        stitched_bev = cv2.GaussianBlur(stitched_bev, (blur_size, blur_size), 0)
    
    print("相机在车辆坐标系中的位置:")
    for cam, pos in camera_positions.items():
        print(f"{cam}: {pos}")
    
    return stitched_bev

def create_multicam_bev(sample_token, bev_width=40, bev_length=40, resolution=0.04, save_flag=False, blend_factor=0.3, fusion_strategy=''):
    """
    Create BEV
    
    Args:
    - sample_token: sample token
    - bev_width: bev width (meter)
    - bev_length: bev length (meter)
    - resolution: resolution (meter/pixel)
    - use_parallel: whether to use parallel processing
    - save_flag: whether to save image
    - blend_factor: blend factor (0-1)
    - fusion_strategy: fusion strategy, optional 'position_based' or 'smooth_blend'
    
    Returns:
    - stitched_bev: stitched BEV image
    - bev_results: bev results of each camera
    """    
    bev_results = process_all_cameras_parallel(sample_token, bev_width, bev_length, resolution)
    
    if fusion_strategy == 'smooth_blend':
        transition_width = int(10 * blend_factor)
        stitched_bev = smooth_blend_stitch_bev_images(
            bev_results, 
            sample_token=sample_token,
            bev_width=bev_width, 
            bev_length=bev_length, 
            resolution=resolution,
            use_depth=True,
            transition_width=transition_width
        )
    else:
        stitched_bev = position_based_stitch_bev_images(bev_results)
    
    visualize_multicam_bev(sample_token, bev_results, stitched_bev, save_flag)

    if save_flag and stitched_bev.shape[2] == 4:
        cv2.imwrite(f'stitched_bev_{fusion_strategy}_{sample_token[:8]}.png', 
                   cv2.cvtColor(stitched_bev, cv2.COLOR_RGBA2BGRA))
        print('Successfully saved stitched BEV image')
    
    return stitched_bev, bev_results

if __name__ == "__main__":
    print("=" * 40)
    
    my_sample = nusc.sample[70]
    
    stitched_bev, bev_results = create_multicam_bev(
        my_sample['token'], 
        bev_width=40,
        bev_length=40,
        resolution=0.04,
        save_flag=False,
        blend_factor=0.6,  # Increase blend factor to enhance edge smoothing
        fusion_strategy='smooth_blend_'
    ) 