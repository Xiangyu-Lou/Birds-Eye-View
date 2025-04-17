import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
import concurrent.futures
from backward_projection import create_bev_view, smooth_bev_image
from pyquaternion import Quaternion
from object_detection import run_detection_on_sample

nusc = NuScenes(version='v1.0-mini', dataroot='F:/Project/Birds-Eye-View/v1.0-mini', verbose=False)
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
    - detection_results: object detection results
    """
    camera_info = get_camera_info(sample_token, cam_channel)
    
    # bird_eye_view, _ = create_bev_view(
    #     sample_token=sample_token, cam_channel=cam_channel, 
    #     bev_width=bev_width, 
    #     bev_length=bev_length, 
    #     resolution=resolution
    # )
    bird_eye_view, original_img, detection_img, projected_boxes = run_detection_on_sample(
        sample_token=sample_token, cam_channel=cam_channel, 
        bev_width=bev_width, 
        bev_length=bev_length, 
        resolution=resolution
    )
    
    bird_eye_view = smooth_bev_image(bird_eye_view, cam_channel=cam_channel)
    
    # Create projection mask (Alpha > 0 area)
    projection_mask = (bird_eye_view[:, :, 3] > 0).astype(np.uint8)
    
    # return bird_eye_view, projection_mask, camera_info
    return bird_eye_view, projection_mask, camera_info, projected_boxes
    
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
    
    # bev, mask, camera_info = process_single_camera(
    bev, mask, camera_info, detection_results = process_single_camera(
        sample_token, cam_channel, bev_width, bev_length, resolution
    )
    print(f"Camera {cam_channel} processed")
    
    # return {
    #     'cam_channel': cam_channel,
    #     'bev': bev,
    #     'mask': mask,
    #     'camera_info': camera_info
    # }
    return {
        'cam_channel': cam_channel,
        'bev': bev,
        'mask': mask,
        'camera_info': camera_info,
        'detection_results': detection_results
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
            # bev_results[cam_channel] = {
            #     'bev': result['bev'],
            #     'mask': result['mask'],
            #     'camera_info': result['camera_info']
            # }
            bev_results[cam_channel] = {
                'bev': result['bev'],
                'mask': result['mask'],
                'camera_info': result['camera_info'],
                'detection_results': result['detection_results']
            }
    
    print(f"Successfully processed {len(bev_results)}/{len(camera_channels)} cameras")
    
    return bev_results

def stitch_bev_images(bev_results):
    """
    Assemble multiple BEV images based on camera position information
    
    Args:
    - bev_results: dictionary containing all camera BEV results
    
    Returns:
    - stitched_bev: stitched BEV image
    """
    
    # Initialize variables
    bev_height, bev_width_pixels = bev_results[list(bev_results.keys())[0]]['bev'].shape[:2]
    stitched_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
    camera_positions = {}
    weight = 0.5
    
    # Create a count matrix to track how many cameras contribute to each pixel
    camera_count = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    
    # Count the number of cameras contributing to each pixel
    for cam_channel, result in bev_results.items():
        # Extract camera information and BEV image
        bev = result['bev']
        camera_info = result['camera_info']
        translation = camera_info['translation']
        # Position of the camera in the ego coordinate system
        camera_positions[cam_channel] = translation
        
        # Identify valid pixels from this camera
        valid_pixels = bev[:, :, 3] > 0
        
        # Increment the count for each pixel
        camera_count[valid_pixels] += 1
    
    # Blend images with equal weights in overlap areas
    for cam_channel, result in bev_results.items():
        bev = result['bev'].astype(np.float32)
        
        # Get valid pixels
        valid_pixels = bev[:, :, 3] > 0
        y_indices, x_indices = np.where(valid_pixels)
        
        # Process each valid pixel
        for j in range(len(y_indices)):
            y, x = y_indices[j], x_indices[j]
            
            # Check if this is an overlap area
            count = camera_count[y, x]
            
            if count == 1:
                stitched_bev[y, x, :] += (bev[y, x, :]).astype(np.uint8)
            elif count == 2:
                # Add weighted contribution
                stitched_bev[y, x, :] += (bev[y, x, :] * weight).astype(np.uint8)
    
    print("\nCamera positions in the ego coordinate system:")
    for cam, pos in camera_positions.items():
        print(f"{cam}: {pos}")
    
    return stitched_bev

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
    
    return stitched_bev

def compute_contour_based_weights(bev_results):
    """
    Compute contour-based weights
    
    Args:
    - bev_results: dictionary of bev results
    
    Returns:
    - weight_maps: weight map for each camera
    """
    # Initialize variables
    bev_height, bev_width = bev_results[list(bev_results.keys())[0]]['bev'].shape[:2]
    weight_maps = {}
    for cam_channel in bev_results.keys():
        weight_maps[cam_channel] = np.ones((bev_height, bev_width), dtype=np.float32)
    
    # Get valid region mask for each camera
    cam_masks = {}
    for cam_channel, result in bev_results.items():
        bev = result['bev']
        if bev.shape[2] == 4:
            valid_mask = bev[:, :, 3] > 0
        else:
            valid_mask = result['mask'] > 0
        cam_masks[cam_channel] = valid_mask.astype(np.uint8)
    
    # Process overlap regions between camera pairs
    cam_pairs = []
    for i, cam1 in enumerate(bev_results.keys()):
        for j, cam2 in enumerate(bev_results.keys()):
            if i < j:
                cam_pairs.append((cam1, cam2))
    
    # Compute overlap regions and weights for each camera pair
    for cam1, cam2 in cam_pairs:
        # Compute overlap region
        overlap = cam_masks[cam1] & cam_masks[cam2]
        
        # If there is no overlap region, skip
        if not np.any(overlap):
            continue
        
        # 获取轮廓 - polyA 和 polyB
        # 从有效掩码中减去重叠区域
        mask1 = cam_masks[cam1] - overlap
        mask2 = cam_masks[cam2] - overlap
        
        # 使用形态学闭操作填充小孔
        kernel = np.ones((5, 5), np.uint8)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        
        # 找到轮廓
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果找不到轮廓，跳过
        if not contours1 or not contours2:
            continue
            
        # 获取最大轮廓
        polyA = max(contours1, key=cv2.contourArea)
        polyB = max(contours2, key=cv2.contourArea)
        
        # 简化轮廓，减少计算量
        epsilon1 = 0.01 * cv2.arcLength(polyA, True)
        epsilon2 = 0.01 * cv2.arcLength(polyB, True)
        polyA = cv2.approxPolyDP(polyA, epsilon1, True)
        polyB = cv2.approxPolyDP(polyB, epsilon2, True)
        
        # 在重叠区域中计算权重
        y_indices, x_indices = np.where(overlap > 0)
        
        for i in range(len(y_indices)):
            y, x = y_indices[i], x_indices[i]
            
            # 计算点到两个轮廓的距离
            # 注意：OpenCV 4.x 要求点必须是浮点数元组
            point = (float(x), float(y))
            dist_to_A = cv2.pointPolygonTest(polyA, point, True)
            dist_to_B = cv2.pointPolygonTest(polyB, point, True)
            
            # 将负距离（外部点）转为正距离
            dist_to_A = abs(dist_to_A)
            dist_to_B = abs(dist_to_B)
            
            # 计算权重 w = dB²/(dA²+dB²)
            if dist_to_A + dist_to_B > 0:
                weight = dist_to_B * dist_to_B / (dist_to_A * dist_to_A + dist_to_B * dist_to_B)
                
                # 更新cam1的权重
                weight_maps[cam1][y, x] = weight
                # 更新cam2的权重 (1-weight)
                weight_maps[cam2][y, x] = 1.0 - weight
    
    return weight_maps

def balance_brightness(bev_results, weight_maps):
    """
    调整图像亮度使其更加一致
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - weight_maps: 权重图
    
    返回:
    - adjusted_bevs: 亮度调整后的BEV结果
    """
    # 创建调整后的结果字典
    adjusted_bevs = {}
    
    # 计算每个相机的平均亮度
    cam_brightness = {}
    for cam_channel, result in bev_results.items():
        bev = result['bev']
        if bev.shape[2] == 4:  # RGBA
            valid_mask = bev[:, :, 3] > 0
            # 只计算RGB通道的平均值
            rgb = bev[:, :, :3]
        else:
            valid_mask = result['mask'] > 0
            rgb = bev
            
        # 计算有效区域的平均亮度
        if np.any(valid_mask):
            mean_brightness = np.mean(rgb[valid_mask])
            cam_brightness[cam_channel] = mean_brightness
    
    # 如果没有有效数据，直接返回原始结果
    if not cam_brightness:
        return bev_results.copy()
    
    # 计算总平均亮度
    global_mean = np.mean(list(cam_brightness.values()))
    
    # 计算每个相机的亮度调整因子
    brightness_factors = {}
    for cam_channel, mean_brightness in cam_brightness.items():
        if mean_brightness > 0:
            # 调整因子使每个相机的亮度接近全局平均值
            factor = global_mean / mean_brightness
            # 限制调整范围，避免过度调整
            factor = np.clip(factor, 0.7, 1.3)
            brightness_factors[cam_channel] = factor
        else:
            brightness_factors[cam_channel] = 1.0
    
    # 应用亮度调整
    for cam_channel, result in bev_results.items():
        bev = result['bev'].copy()
        factor = brightness_factors.get(cam_channel, 1.0)
        
        if bev.shape[2] == 4:  # RGBA
            # 只调整RGB通道
            bev[:, :, :3] = np.clip(bev[:, :, :3] * factor, 0, 255).astype(np.uint8)
        else:
            bev = np.clip(bev * factor, 0, 255).astype(np.uint8)
        
        # 保存调整后的结果
        adjusted_result = result.copy()
        adjusted_result['bev'] = bev
        adjusted_bevs[cam_channel] = adjusted_result
    
    return adjusted_bevs

def contour_based_stitch_bev_images(bev_results):
    """
    Fusion BEV images based on contour distance
    
    Args:
    - bev_results: dictionary of bev results
    - balance_brightness_flag: whether to balance brightness
    
    Returns:
    - stitched_bev: stitched BEV image
    """    
    # Initialize variables
    bev_height, bev_width_pixels = bev_results[list(bev_results.keys())[0]]['bev'].shape[:2]
    stitched_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8) 
    camera_positions = {}
    
    # Compute contour-based weights
    weight_maps = compute_contour_based_weights(bev_results)
    
    # Balance brightness
    adjusted_bevs = balance_brightness(bev_results, weight_maps)
    
    # Merge all camera images based on weights
    for cam_channel, result in adjusted_bevs.items():
        bev = result['bev']
        camera_positions[cam_channel] = result['camera_info']['translation']
        weight = np.expand_dims(weight_maps[cam_channel], axis=2)
        
        # 对RGB通道应用权重融合
        # 分离RGB和Alpha通道
        rgb = bev[:, :, :3]
        alpha = bev[:, :, 3]
        
        # 对RGB通道应用权重
        for c in range(3):
            stitched_bev[:, :, c] += (rgb[:, :, c] * weight[:, :, 0]).astype(np.uint8)
        
        # 更新Alpha通道
        valid_mask = alpha > 0
        stitched_bev[:, :, 3][valid_mask] = 255
    
    print("相机在车辆坐标系中的位置:")
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
    
    # 颜色映射 - 用于检测结果的可视化
    color_map = {
        'person': (255, 0, 0, 255),   # 红色
        'car': (0, 255, 0, 255),      # 绿色
        'truck': (0, 0, 255, 255),    # 蓝色
        'bus': (255, 255, 0, 255),    # 黄色
        'motorcycle': (255, 0, 255, 255), # 洋红色
        'bicycle': (0, 255, 255, 255),    # 青色
    }
    default_color = (128, 128, 128, 255)  # 默认灰色
    
    # Plot each camera's BEV
    for i, (cam_channel, result) in enumerate(bev_results.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f'BEV: {cam_channel}')
        bev_img = result['bev']
        
        # 检查结果中是否包含检测结果
        if 'detection_results' in result and len(result['detection_results']) > 0:
            # 创建带有检测框的BEV图像
            bev_with_detections = bev_img.copy()
            projected_boxes = result['detection_results']
            
            # 绘制检测框
            for box in projected_boxes:
                x, y = box['bev_x'], box['bev_y']
                size = box['size']
                cls = box['class']
                
                # 根据类别选择颜色
                color = color_map.get(cls, default_color)
                
                # 在BEV图像上绘制圆形标记
                cv2.circle(bev_with_detections, (x, y), size, color, -1)
                
                # 添加类别标签
                cv2.putText(bev_with_detections, cls, (x, y - size - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1, cv2.LINE_AA)
            
            plt.imshow(bev_with_detections)
        else:
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
        
        # 在拼接图上显示所有相机的检测结果
        stitched_with_detections = stitched_bev.copy()
        all_detections = []
        
        # 收集所有相机的检测结果
        for cam_channel, result in bev_results.items():
            if 'detection_results' in result:
                all_detections.extend(result['detection_results'])
        
        # 在拼接图上绘制所有检测结果
        if all_detections:
            for box in all_detections:
                x, y = box['bev_x'], box['bev_y']
                size = box['size']
                cls = box['class']
                
                # 根据类别选择颜色
                color = color_map.get(cls, default_color)
                
                # 在BEV图像上绘制圆形标记
                cv2.circle(stitched_with_detections, (x, y), size, color, -1)
                
                # 添加类别标签
                cv2.putText(stitched_with_detections, cls, (x, y - size - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1, cv2.LINE_AA)
            
            plt.imshow(stitched_with_detections)
        else:
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
        
        # 在单独的图上显示拼接BEV和检测结果
        stitched_with_detections = stitched_bev.copy()
        all_detections = []
        
        # 收集所有相机的检测结果
        for cam_channel, result in bev_results.items():
            if 'detection_results' in result:
                all_detections.extend(result['detection_results'])
        
        # 在拼接图上绘制所有检测结果
        if all_detections:
            for box in all_detections:
                x, y = box['bev_x'], box['bev_y']
                size = box['size']
                cls = box['class']
                
                # 根据类别选择颜色
                color = color_map.get(cls, default_color)
                
                # 在BEV图像上绘制圆形标记
                cv2.circle(stitched_with_detections, (x, y), size, color, -1)
                
                # 添加类别标签
                cv2.putText(stitched_with_detections, cls, (x, y - size - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1, cv2.LINE_AA)
            
            plt.imshow(stitched_with_detections)
        else:
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
        
        # 如果检测到了物体，单独显示带有检测框的拼接图
        if all_detections:
            plt.figure(figsize=(10, 10))
            plt.title('Stitched BEV with Object Detection')
            plt.imshow(stitched_with_detections)
            plt.axis('off')
            
            # 添加图例
            for i, (cls, color) in enumerate(color_map.items()):
                plt.plot([], [], 'o', color=[c/255 for c in color[:3]], label=cls)
            plt.legend(loc='upper right')
            
            if save_flag:
                plt.savefig(f'stitched_bev_detection_{sample_token[:8]}.png', transparent=True)
            plt.show()

def filter_overlapping_detections(bev_results, bev_width=40, bev_length=40, resolution=0.04):
    """
    过滤重叠区域的重复检测，优先保留CAM_FRONT, CAM_BACK_LEFT和CAM_BACK_RIGHT的检测结果
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - bev_width: BEV宽度(米)
    - bev_length: BEV长度(米)
    - resolution: 分辨率(米/像素)
    
    返回:
    - filtered_results: 过滤后的BEV结果
    """
    # 复制原始结果
    filtered_results = {cam: result.copy() for cam, result in bev_results.items()}
    
    # 优先保留的相机
    priority_cameras = ['CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 需要过滤的相机
    filter_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
    
    # 相机覆盖范围掩码
    camera_masks = {}
    overlap_mask = None
    
    # 生成BEV尺寸
    bev_height = int(bev_length / resolution)
    bev_width_pixels = int(bev_width / resolution)
    
    # 创建各个相机的覆盖范围掩码
    for cam, result in bev_results.items():
        if 'mask' in result:
            camera_masks[cam] = result['mask']
        elif 'bev' in result and result['bev'].shape[2] == 4:
            # 如果没有直接提供mask，则从alpha通道创建
            camera_masks[cam] = (result['bev'][:, :, 3] > 0).astype(np.uint8)
    
    # 找到重叠区域
    # 初始化重叠掩码
    overlap_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    
    # 优先相机覆盖区域
    priority_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    for cam in priority_cameras:
        if cam in camera_masks:
            priority_mask = np.logical_or(priority_mask, camera_masks[cam])
    
    # 需要过滤的相机覆盖区域
    for cam in filter_cameras:
        if cam in camera_masks:
            # 将此相机与优先相机的重叠部分标记为重叠区域
            cam_overlap = np.logical_and(camera_masks[cam], priority_mask)
            overlap_mask = np.logical_or(overlap_mask, cam_overlap)
    
    # 遍历需要过滤的相机，移除在重叠区域内的检测
    for cam in filter_cameras:
        if cam in filtered_results and 'detection_results' in filtered_results[cam]:
            # 获取检测结果
            detections = filtered_results[cam]['detection_results']
            filtered_detections = []
            
            # 检查每个检测结果
            for box in detections:
                x, y = box['bev_x'], box['bev_y']
                
                # 检查坐标是否在图像范围内
                if 0 <= y < overlap_mask.shape[0] and 0 <= x < overlap_mask.shape[1]:
                    # 如果在重叠区域外，保留检测结果
                    if not overlap_mask[y, x]:
                        filtered_detections.append(box)
                else:
                    # 如果坐标超出范围，也保留（不应该发生，但为了安全）
                    filtered_detections.append(box)
            
            # 更新过滤后的检测结果
            filtered_results[cam]['detection_results'] = filtered_detections
            
            # 输出统计信息
            removed_count = len(detections) - len(filtered_detections)
            if removed_count > 0:
                print(f"从 {cam} 移除了 {removed_count} 个重叠区域的检测目标")
    
    return filtered_results

def create_multicam_bev(sample_token, bev_width=40, bev_length=40, resolution=0.04, save_flag=False, fusion_strategy='', show_flag=False, filter_overlap=True, show_filter_result=False):
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
    - fusion_strategy: fusion strategy, optional 'position_based' or 'contour_based'
    - filter_overlap: 是否过滤重叠区域的重复检测
    - show_filter_result: 是否显示过滤结果对比
    
    Returns:
    - stitched_bev: stitched BEV image
    - bev_results: bev results of each camera
    """    
    # 获取原始BEV结果
    original_bev_results = process_all_cameras_parallel(sample_token, bev_width, bev_length, resolution)
    bev_results = {cam: result.copy() for cam, result in original_bev_results.items()}
    
    # 过滤重叠区域的检测结果
    if filter_overlap:
        print("\n====== 过滤重叠区域检测 ======")
        filtered_results = filter_overlapping_detections(bev_results, bev_width, bev_length, resolution)
        
        # 可视化过滤前后的结果
        if show_filter_result:
            visualize_overlap_filtering(original_bev_results, filtered_results, bev_width, bev_length, resolution, save_flag)
        
        # 用过滤后的结果替换原始结果
        bev_results = filtered_results
    
    if fusion_strategy == 'contour_based':
        stitched_bev = contour_based_stitch_bev_images(bev_results)
    elif fusion_strategy == 'position_based':
        stitched_bev = position_based_stitch_bev_images(bev_results)
    else:
        stitched_bev = stitch_bev_images(bev_results)
        
    if show_flag:
        visualize_multicam_bev(sample_token, bev_results, stitched_bev, save_flag)

    if save_flag:
        cv2.imwrite(f'stitched_bev_{fusion_strategy}_{sample_token[:8]}.png', 
                   cv2.cvtColor(stitched_bev, cv2.COLOR_RGBA2BGRA))
        print('Successfully saved stitched BEV image')
    
    return stitched_bev, bev_results

def visualize_overlap_filtering(bev_results, filtered_results, bev_width=40, bev_length=40, resolution=0.04, save_flag=False):
    """
    可视化相机重叠区域和过滤前后的检测结果
    
    参数:
    - bev_results: 过滤前的BEV结果
    - filtered_results: 过滤后的BEV结果
    - bev_width: BEV宽度(米)
    - bev_length: BEV长度(米)
    - resolution: 分辨率(米/像素)
    - save_flag: 是否保存图像
    """
    # 确定BEV尺寸
    bev_height = int(bev_length / resolution)
    bev_width_pixels = int(bev_width / resolution)
    
    # 优先保留的相机
    priority_cameras = ['CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 需要过滤的相机
    filter_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
    
    # 相机覆盖范围掩码
    camera_masks = {}
    
    # 创建各个相机的覆盖范围掩码
    for cam, result in bev_results.items():
        if 'mask' in result:
            camera_masks[cam] = result['mask']
        elif 'bev' in result and result['bev'].shape[2] == 4:
            # 如果没有直接提供mask，则从alpha通道创建
            camera_masks[cam] = (result['bev'][:, :, 3] > 0).astype(np.uint8)
    
    # 优先相机覆盖区域
    priority_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    for cam in priority_cameras:
        if cam in camera_masks:
            priority_mask = np.logical_or(priority_mask, camera_masks[cam])
    
    # 需要过滤的相机覆盖区域
    filter_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    for cam in filter_cameras:
        if cam in camera_masks:
            filter_mask = np.logical_or(filter_mask, camera_masks[cam])
    
    # 重叠区域
    overlap_mask = np.logical_and(priority_mask, filter_mask).astype(np.uint8)
    
    # 创建可视化图像
    plt.figure(figsize=(15, 10))
    
    # 绘制相机覆盖区域
    plt.subplot(2, 2, 1)
    plt.title('相机覆盖区域')
    
    # 创建覆盖范围可视化图像
    coverage_img = np.zeros((bev_height, bev_width_pixels, 3), dtype=np.uint8)
    
    # 设置不同相机的颜色
    camera_colors = {
        'CAM_FRONT': [255, 0, 0],          # 红色
        'CAM_FRONT_LEFT': [255, 165, 0],   # 橙色
        'CAM_FRONT_RIGHT': [255, 255, 0],  # 黄色
        'CAM_BACK': [0, 0, 255],           # 蓝色
        'CAM_BACK_LEFT': [75, 0, 130],     # 靛蓝色
        'CAM_BACK_RIGHT': [238, 130, 238]  # 紫色
    }
    
    # 绘制各相机覆盖区域
    for cam, mask in camera_masks.items():
        if cam in camera_colors:
            for c in range(3):
                coverage_img[:, :, c][mask > 0] = camera_colors[cam][c]
    
    plt.imshow(coverage_img)
    plt.axis('off')
    
    # 绘制重叠区域
    plt.subplot(2, 2, 2)
    plt.title('需要过滤的重叠区域')
    
    overlap_img = np.zeros((bev_height, bev_width_pixels, 3), dtype=np.uint8)
    overlap_img[overlap_mask > 0] = [255, 255, 255]  # 白色表示重叠区域
    
    plt.imshow(overlap_img)
    plt.axis('off')
    
    # 绘制过滤前的检测结果
    plt.subplot(2, 2, 3)
    plt.title('过滤前的检测结果')
    
    # 颜色映射
    color_map = {
        'person': (255, 0, 0),   # 红色
        'car': (0, 255, 0),      # 绿色
        'truck': (0, 0, 255),    # 蓝色
        'bus': (255, 255, 0),    # 黄色
        'motorcycle': (255, 0, 255), # 洋红色
        'bicycle': (0, 255, 255),    # 青色
    }
    default_color = (128, 128, 128)  # 默认灰色
    
    # 创建检测结果图像
    detection_before_img = np.zeros((bev_height, bev_width_pixels, 3), dtype=np.uint8)
    
    # 添加所有检测结果
    for cam, result in bev_results.items():
        if 'detection_results' in result:
            detections = result['detection_results']
            for box in detections:
                x, y = box['bev_x'], box['bev_y']
                size = box['size']
                cls = box['class']
                
                # 确保坐标在图像范围内
                if 0 <= y < bev_height and 0 <= x < bev_width_pixels:
                    # 根据类别选择颜色
                    color = color_map.get(cls, default_color)
                    
                    # 在图像上绘制圆形标记
                    cv2.circle(detection_before_img, (x, y), size, color, -1)
    
    plt.imshow(detection_before_img)
    plt.axis('off')
    
    # 绘制过滤后的检测结果
    plt.subplot(2, 2, 4)
    plt.title('过滤后的检测结果')
    
    # 创建检测结果图像
    detection_after_img = np.zeros((bev_height, bev_width_pixels, 3), dtype=np.uint8)
    
    # 添加所有检测结果
    for cam, result in filtered_results.items():
        if 'detection_results' in result:
            detections = result['detection_results']
            for box in detections:
                x, y = box['bev_x'], box['bev_y']
                size = box['size']
                cls = box['class']
                
                # 确保坐标在图像范围内
                if 0 <= y < bev_height and 0 <= x < bev_width_pixels:
                    # 根据类别选择颜色
                    color = color_map.get(cls, default_color)
                    
                    # 在图像上绘制圆形标记
                    cv2.circle(detection_after_img, (x, y), size, color, -1)
    
    plt.imshow(detection_after_img)
    plt.axis('off')
    
    plt.tight_layout()
    if save_flag:
        plt.savefig('overlap_filtering_visualization.png')
    plt.show()
    
    # 统计检测结果数量变化
    before_count = 0
    after_count = 0
    
    for cam, result in bev_results.items():
        if 'detection_results' in result:
            before_count += len(result['detection_results'])
    
    for cam, result in filtered_results.items():
        if 'detection_results' in result:
            after_count += len(result['detection_results'])
    
    print(f"\n过滤前检测目标总数: {before_count}")
    print(f"过滤后检测目标总数: {after_count}")
    print(f"移除重复检测目标数: {before_count - after_count}")

if __name__ == "__main__":
    print("=" * 40)
    
    my_sample = nusc.sample[70]
    
    stitched_bev, bev_results = create_multicam_bev(
        my_sample['token'], 
        bev_width=40,
        bev_length=40,
        resolution=0.04,
        save_flag=False,
        fusion_strategy='contour_based',
        show_flag=True,
        filter_overlap=True,
        show_filter_result=True
    ) 