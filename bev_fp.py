import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from forward_projection import extract_data, compute_coordinate_transforms, generate_bird_eye_view_torch, post_process_bev_torch, create_bird_eye_view_torch, visualize_bev
import concurrent.futures
import time

nusc = NuScenes(version='v1.0-mini', dataroot='F:/Project/nuscenes-devkit/v1.0-mini', verbose=False)

def get_camera_channels():
    """
    获取nuScenes中的所有相机通道
    
    返回:
    - camera_channels: 相机通道列表
    """
    camera_channels = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]
    return camera_channels

def process_single_camera_bev(sample_token, cam_channel, bev_width=50, bev_length=50, resolution=0.1):
    """
    处理单个相机的BEV投影，使用forward_projection.py中的PyTorch实现
    
    参数:
    - sample_token: 样本token
    - cam_channel: 相机通道
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    
    返回:
    - processed_bev: 处理后的俯视图
    - projection_mask: 投影掩码
    """
    try:
        # 使用forward_projection.py中的实现
        processed_bev, _ = create_bird_eye_view_torch(
            sample_token, cam_channel, 
            bev_width=bev_width, 
            bev_length=bev_length, 
            resolution=resolution
        )
        
        # 提取Alpha通道作为掩码
        if processed_bev.shape[2] == 4:
            projection_mask = (processed_bev[:, :, 3] > 0).astype(np.uint8)
        else:
            # 兼容性处理：如果输出是RGB，使用非白色像素作为掩码
            is_white = np.all(processed_bev == 255, axis=2)
            projection_mask = (~is_white).astype(np.uint8)
        
        # 可选：保存单个相机的BEV图像用于调试
        # if processed_bev.shape[2] == 4:
        #     save_transparent_bev(processed_bev, f'{cam_channel}_bev_{sample_token[:8]}.png')
            
        return processed_bev, projection_mask
        
    except Exception as e:
        print(f"处理相机 {cam_channel} 时出错: {str(e)}")
        # 优雅地处理错误，返回空图像
        bev_height = int(bev_length / resolution)
        bev_width_pixels = int(bev_width / resolution)
        
        # 创建RGBA空图像
        empty_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
        empty_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
        return empty_bev, empty_mask

def save_transparent_bev(bird_eye_view, filename):
    """
    保存透明背景的BEV图像
    
    参数:
    - bird_eye_view: 带Alpha通道的BEV图像
    - filename: 输出文件名
    """
    # 确保输入图像有4个通道（RGBA）
    if bird_eye_view.shape[2] != 4:
        raise ValueError("图像必须有Alpha通道才能保存为透明PNG")
    
    # 保存为PNG（支持透明度）
    cv2.imwrite(filename, cv2.cvtColor(bird_eye_view, cv2.COLOR_RGBA2BGRA))

def process_camera_worker(params):
    """
    线程工作函数：处理单个相机BEV
    
    参数:
    - params: 包含处理所需参数的字典
      - sample_token: 样本token
      - cam_channel: 相机通道
      - bev_width: 俯视图宽度（米）
      - bev_length: 俯视图长度（米）
      - resolution: 分辨率（米/像素）
    
    返回:
    - 相机通道
    - 处理结果 (bev, mask)
    """
    sample_token = params['sample_token']
    cam_channel = params['cam_channel']
    bev_width = params['bev_width']
    bev_length = params['bev_length']
    resolution = params['resolution']
    
    print(f"\n线程开始处理相机: {cam_channel}")
    start_time = time.time()
    
    try:
        bev, mask = process_single_camera_bev(
            sample_token, cam_channel, bev_width, bev_length, resolution
        )
        processing_time = time.time() - start_time
        print(f"相机 {cam_channel} 处理完成，耗时: {processing_time:.2f}秒")
        return cam_channel, (bev, mask)
    except Exception as e:
        print(f"处理相机 {cam_channel} 时出错: {str(e)}")
        # 返回空结果
        bev_height = int(bev_length / resolution)
        bev_width_pixels = int(bev_width / resolution)
        empty_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
        empty_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
        return cam_channel, (empty_bev, empty_mask)

def process_all_cameras_parallel(sample_token, bev_width=50, bev_length=50, resolution=0.1, max_workers=6):
    """
    并行处理所有相机的BEV投影
    
    参数:
    - sample_token: 样本token
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    - max_workers: 最大线程数
    
    返回:
    - bev_results: 包含所有相机BEV结果的字典
    """
    camera_channels = get_camera_channels()
    bev_results = {}
    
    print(f"并行处理样本 {sample_token} 的所有相机")
    start_time = time.time()
    
    # 准备线程任务参数
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
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_camera_worker, params) for params in tasks]
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            try:
                cam_channel, (bev, mask) = future.result()
                bev_results[cam_channel] = {
                    'bev': bev,
                    'mask': mask
                }
            except Exception as e:
                print(f"获取任务结果时出错: {str(e)}")
    
    # 计算总处理时间
    total_time = time.time() - start_time
    print(f"\n所有相机并行处理完成，总耗时: {total_time:.2f}秒")
    print(f"成功处理的相机数: {len(bev_results)}/{len(camera_channels)}")
    
    return bev_results

def process_all_cameras(sample_token, bev_width=50, bev_length=50, resolution=0.1):
    """
    串行处理所有相机的BEV投影
    
    参数:
    - sample_token: 样本token
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    
    返回:
    - bev_results: 包含所有相机BEV结果的字典
    """
    camera_channels = get_camera_channels()
    bev_results = {}
    
    print(f"串行处理样本 {sample_token} 的所有相机")
    start_time = time.time()
    
    for cam_channel in camera_channels:
        print(f"\n处理相机: {cam_channel}")
        cam_start = time.time()
        try:
            bev, mask = process_single_camera_bev(
                sample_token, cam_channel, bev_width, bev_length, resolution
            )
            bev_results[cam_channel] = {
                'bev': bev,
                'mask': mask
            }
            cam_time = time.time() - cam_start
            print(f"成功处理 {cam_channel}，耗时: {cam_time:.2f}秒")
        except Exception as e:
            print(f"处理 {cam_channel} 时出错: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    print(f"\n所有相机串行处理完成，总耗时: {total_time:.2f}秒")
    print(f"成功处理的相机数: {len(bev_results)}/{len(camera_channels)}")
    
    return bev_results

# 添加计算每个相机图像像素深度的函数
def compute_pixel_distances(sample_token, bev_results, bev_width=40, bev_length=40, resolution=0.1):
    """
    计算每个BEV像素到对应相机的距离（作为深度估计）
    
    参数:
    - sample_token: 样本token
    - bev_results: 包含所有相机BEV结果的字典
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    
    返回:
    - distance_maps: 每个相机BEV像素的距离图，格式与bev_results相同
    """
    # 获取第一张图像的尺寸
    first_cam = list(bev_results.keys())[0]
    bev_height, bev_width_pixels = bev_results[first_cam]['bev'].shape[:2]
    
    # 创建字典存储每个相机的距离图
    distance_maps = {}
    
    # 计算每个相机对应的距离图
    for cam_channel in bev_results.keys():
        # 提取相机数据
        _, _, _, _, cs_record, ego_pose, _ = extract_data(sample_token, cam_channel)
        
        # 计算相机在自车坐标系中的位置
        cam_to_ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
        cam_to_ego_translation = np.array(cs_record['translation'])
        
        # 初始化距离图（无限大初始值）
        distance_map = np.full((bev_height, bev_width_pixels), np.inf, dtype=np.float32)
        
        # 获取有效像素掩码
        if bev_results[cam_channel]['bev'].shape[2] == 4:  # RGBA格式
            valid_mask = bev_results[cam_channel]['bev'][:, :, 3] > 0
        else:  # RGB格式
            valid_mask = bev_results[cam_channel]['mask'] > 0
        
        # 生成BEV坐标网格
        y_coords, x_coords = np.meshgrid(
            np.arange(bev_width_pixels), 
            np.arange(bev_height)
        )
        
        # 只处理有效像素
        valid_y = y_coords[valid_mask]
        valid_x = x_coords[valid_mask]
        
        # 将BEV像素坐标转换为自车坐标系下的物理坐标
        ego_x = bev_length/2 - valid_x * resolution  # BEV的x方向对应自车的前方
        ego_y = -bev_width/2 + valid_y * resolution  # BEV的y方向对应自车的左右
        ego_z = np.zeros_like(ego_x)  # 地平面z=0
        
        # 计算每个有效BEV像素到相机的距离（即深度）
        for i in range(len(valid_x)):
            # 计算地面点到相机的欧氏距离
            dx = ego_x[i] - cam_to_ego_translation[0]
            dy = ego_y[i] - cam_to_ego_translation[1]
            dz = ego_z[i] - cam_to_ego_translation[2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # 更新距离图
            distance_map[valid_x[i], valid_y[i]] = distance
        
        # 存储结果
        distance_maps[cam_channel] = distance_map
    
    return distance_maps

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

def fuse_bev_images(bev_results, strategy='max_intensity', sample_token=None, bev_width=40, bev_length=40, resolution=0.1):
    """
    融合多个相机的BEV图像
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - strategy: 融合策略，'max_intensity'、'weighted_average'、'min_depth'或'smooth_blend'
    - sample_token: 样本token (仅当strategy='min_depth'或'smooth_blend'时需要)
    - bev_width: 俯视图宽度（米）(仅当strategy='min_depth'或'smooth_blend'时需要)
    - bev_length: 俯视图长度（米）(仅当strategy='min_depth'或'smooth_blend'时需要)
    - resolution: 分辨率（米/像素）(仅当strategy='min_depth'或'smooth_blend'时需要)
    
    返回:
    - fused_bev: 融合后的BEV图像
    - combined_mask: 组合的掩码
    """
    if not bev_results:
        raise ValueError("没有有效的BEV结果可融合")
    
    # 获取BEV尺寸（假设所有BEV图像尺寸相同）
    first_cam = list(bev_results.keys())[0]
    bev_height, bev_width_pixels = bev_results[first_cam]['bev'].shape[:2]
    
    # 检查是否为RGBA图像
    is_rgba = bev_results[first_cam]['bev'].shape[2] == 4
    
    # 初始化融合结果和组合掩码
    if is_rgba:
        fused_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
    else:
        fused_bev = np.ones((bev_height, bev_width_pixels, 3), dtype=np.uint8) * 255
    
    combined_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    
    # 记录每个像素点的贡献图像数
    contribution_count = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    
    # 首先，组合所有相机掩码，找出所有有效区域
    for cam_channel, result in bev_results.items():
        if is_rgba:
            # 对于RGBA图像，使用Alpha通道作为掩码
            cam_mask = result['bev'][:, :, 3] > 0
        else:
            cam_mask = result['mask'] > 0
        combined_mask = np.logical_or(combined_mask, cam_mask).astype(np.uint8)
        contribution_count[cam_mask] += 1
    
    # 添加新的平滑融合策略
    if strategy == 'smooth_blend':
        # 先平滑每个相机的边缘
        print("检测相机边缘区域...")
        edge_maps, overlap_map = detect_camera_edges(bev_results, dilate_size=5)
        
        print("应用边缘平滑处理...")
        smoothed_bevs = smooth_camera_transitions(bev_results, edge_maps, blur_radius=7)
        
        # 计算过渡权重
        print("创建过渡权重...")
        weight_maps = create_transition_weights(smoothed_bevs, edge_maps, transition_width=15)
        
        # 如果还需要基于深度融合，可以结合min_depth策略
        if sample_token is not None:
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
                
                # 更新最小深度图
                update_pixels = np.logical_and(valid_pixels, distance_map < min_depth_map)
                min_depth_map[update_pixels] = distance_map[update_pixels]
                
                # 存储相机索引
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
                    fused_bev[:, :, c] += (bev[:, :, c] * cam_weight[:, :, 0]).astype(np.uint8)
                
                # 更新Alpha通道（如果有）
                if is_rgba:
                    fused_bev[:, :, 3][valid_weights] = 255
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
                normalized_weight = weight_maps[cam_channel] / weight_sum
                weight_3d = np.expand_dims(normalized_weight, axis=2)
                
                # 对RGB通道应用权重融合
                for c in range(3):
                    fused_bev[:, :, c] += (bev[:, :, c] * weight_3d[:, :, 0]).astype(np.uint8)
                
                # 更新Alpha通道（如果有）
                if is_rgba:
                    alpha_weight = normalized_weight * (bev[:, :, 3] > 0).astype(np.float32)
                    fused_bev[:, :, 3] += (255 * alpha_weight).astype(np.uint8)
            
            # 确保Alpha通道不超过255
            if is_rgba:
                fused_bev[:, :, 3] = np.clip(fused_bev[:, :, 3], 0, 255)
    
    # 使用最小深度策略
    elif strategy == 'min_depth':
        # 检查必要参数
        if sample_token is None:
            raise ValueError("使用min_depth策略时需要提供sample_token参数")
        
        # 先平滑边缘
        print("检测相机边缘并应用平滑...")
        edge_maps, overlap_map = detect_camera_edges(bev_results, dilate_size=5)
        smoothed_bevs = smooth_camera_transitions(bev_results, edge_maps, blur_radius=7)
        
        # 计算每个相机的像素深度
        print("计算像素深度...")
        distance_maps = compute_pixel_distances(sample_token, smoothed_bevs, bev_width, bev_length, resolution)
        
        # 初始化深度图（用于跟踪最小深度）
        min_depth_map = np.full((bev_height, bev_width_pixels), np.inf, dtype=np.float32)
        
        # 基于最小深度进行融合
        print("基于最小深度融合...")
        for cam_channel, result in smoothed_bevs.items():
            bev = result['bev']
            distance_map = distance_maps[cam_channel]
            
            if is_rgba:
                # 对于RGBA图像，只在Alpha > 0的区域融合
                valid_pixels = bev[:, :, 3] > 0
            else:
                valid_pixels = result['mask'] > 0
            
            if np.any(valid_pixels):
                # 找出该相机中深度小于当前最小深度的像素
                update_pixels = np.logical_and(
                    valid_pixels,
                    distance_map < min_depth_map
                )
                
                # 更新这些像素
                if np.any(update_pixels):
                    if is_rgba:
                        # 更新RGBA图像
                        for c in range(4):  # 包括Alpha通道
                            fused_bev[:, :, c][update_pixels] = bev[:, :, c][update_pixels]
                    else:
                        # 更新RGB图像
                        for c in range(3):
                            fused_bev[:, :, c][update_pixels] = bev[:, :, c][update_pixels]
                    
                    # 更新最小深度图
                    min_depth_map[update_pixels] = distance_map[update_pixels]
        
        # 平滑边缘区域中的像素
        print("平滑相机边缘...")
        for cam_channel, edge in edge_maps.items():
            if np.any(edge):
                # 对融合图像中的边缘区域进行高斯模糊
                for c in range(3):  # 只处理RGB通道
                    channel = fused_bev[:, :, c].copy()
                    blurred = cv2.GaussianBlur(channel, (15, 15), 0)
                    channel[edge == 1] = blurred[edge == 1]
                    fused_bev[:, :, c] = channel
        
        print(f"融合完成，有效像素占比: {np.sum(min_depth_map < np.inf) / (bev_height * bev_width_pixels):.2%}")
    
    elif strategy == 'max_intensity':
        # 最大亮度策略 - 保留最强的信号
        for cam_channel, result in bev_results.items():
            bev = result['bev']
            
            if is_rgba:
                # 对于RGBA图像，只在Alpha > 0的区域融合
                valid_pixels = bev[:, :, 3] > 0
                
                # 计算当前像素亮度
                if np.any(valid_pixels):
                    current_intensity = np.sum(fused_bev[:, :, 0:3].astype(np.float32), axis=2)
                    new_intensity = np.sum(bev[:, :, 0:3].astype(np.float32), axis=2)
                    
                    # 找出新图像亮度更高的像素
                    replace_pixels = np.logical_and(valid_pixels, 
                                                   np.logical_or(fused_bev[:, :, 3] == 0,  # 原区域透明
                                                                new_intensity > current_intensity))  # 或新区域更亮
                    
                    # 更新融合图像
                    if np.any(replace_pixels):
                        for c in range(3):  # RGB通道
                            fused_bev[:, :, c][replace_pixels] = bev[:, :, c][replace_pixels]
                        
                        # 同时更新Alpha通道
                        fused_bev[:, :, 3][replace_pixels] = bev[:, :, 3][replace_pixels]
            else:
                # 原始RGB图像处理逻辑
                mask = result['mask']
                valid_pixels = mask > 0
                
                if np.any(valid_pixels):
                    current_intensity = np.sum(fused_bev.astype(np.float32), axis=2)
                    new_intensity = np.sum(bev.astype(np.float32), axis=2)
                    
                    replace_pixels = np.logical_and(valid_pixels, new_intensity > current_intensity)
                    
                    if np.any(replace_pixels):
                        for c in range(3):
                            fused_bev[:, :, c][replace_pixels] = bev[:, :, c][replace_pixels]
        
        # 增强最终图像对比度和亮度
        if is_rgba:
            # 对于RGBA图像，只处理有效区域
            valid_area = fused_bev[:, :, 3] > 0
            if np.any(valid_area):
                # 增强对比度 - 只对RGB通道处理
                for c in range(3):
                    channel = fused_bev[:, :, c].astype(np.float32)
                    if np.any(channel[valid_area] > 0):
                        # 计算有效区域的最小和最大值
                        min_val = np.min(channel[valid_area])
                        max_val = np.max(channel[valid_area])
                        
                        if max_val > min_val:
                            # 线性拉伸到完整范围
                            channel[valid_area] = np.clip(((channel[valid_area] - min_val) / (max_val - min_val) * 255), 0, 255)
                            fused_bev[:, :, c][valid_area] = channel[valid_area].astype(np.uint8)
    
    elif strategy == 'weighted_average':
        # 加权平均策略 - 适用于高密度重叠区域
        # 初始化权重累加器
        weight_sum = np.zeros((bev_height, bev_width_pixels, 1), dtype=np.float32)
        weighted_rgb = np.zeros((bev_height, bev_width_pixels, 3), dtype=np.float32)
        
        # 所有相机的权重总和
        total_cameras = len(bev_results)
        
        for cam_channel, result in bev_results.items():
            bev = result['bev']
            
            if is_rgba:
                # 以Alpha通道作为权重
                alpha = bev[:, :, 3].astype(np.float32) / 255.0
                weight = np.expand_dims(alpha, axis=2)
                
                # 加权累积RGB值
                for c in range(3):
                    weighted_rgb[:, :, c] += bev[:, :, c].astype(np.float32) * weight[:, :, 0]
                
                # 累积权重
                weight_sum += weight
            else:
                # 原始RGB图像处理逻辑
                mask = result['mask'].astype(np.float32)
                weight = np.expand_dims(mask, axis=2) / total_cameras
                
                for c in range(3):
                    weighted_rgb[:, :, c] += bev[:, :, c].astype(np.float32) * weight[:, :, 0]
                
                weight_sum += weight
        
        # 计算最终加权平均值
        valid_pixels = weight_sum[:, :, 0] > 0
        if np.any(valid_pixels):
            for c in range(3):
                weighted_rgb[:, :, c][valid_pixels] /= weight_sum[:, :, 0][valid_pixels]
            
            # 将结果转换回uint8
            fused_bev[:, :, 0:3][valid_pixels] = weighted_rgb[valid_pixels].reshape(-1, 3).astype(np.uint8)
            
            # 设置Alpha通道（如果有）
            if is_rgba:
                # 权重和作为新的Alpha值
                fused_bev[:, :, 3][valid_pixels] = np.clip(weight_sum[:, :, 0][valid_pixels] * 255, 0, 255).astype(np.uint8)
    
    # 确保未贡献区域为完全透明（RGBA）或白色（RGB）
    no_contribution = contribution_count == 0
    if is_rgba:
        fused_bev[:, :, 3][no_contribution] = 0  # 透明
    else:
        fused_bev[no_contribution] = [255, 255, 255]  # 白色
    
    # 对最终结果进行后处理，平滑边缘
    if strategy != 'smooth_blend':  # smooth_blend策略已经包含了平滑
        # 找出所有相机的边缘区域
        edge_pixels = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
        
        for cam_channel, result in bev_results.items():
            if is_rgba:
                mask = result['bev'][:, :, 3] > 0
            else:
                mask = result['mask'] > 0
                
            # 找出边缘
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel)
            eroded = cv2.erode(mask.astype(np.uint8), kernel)
            edge = dilated - eroded
            
            # 累积所有边缘
            edge_pixels = np.logical_or(edge_pixels, edge).astype(np.uint8)
        
        # 对边缘区域应用高斯模糊
        if np.any(edge_pixels):
            for c in range(3):  # 只处理RGB通道
                channel = fused_bev[:, :, c].copy()
                blurred = cv2.GaussianBlur(channel, (7, 7), 0)
                channel[edge_pixels == 1] = blurred[edge_pixels == 1]
                fused_bev[:, :, c] = channel
    
    return fused_bev, combined_mask

def create_colored_bev_visualization(bev_results):
    """
    创建彩色编码的BEV可视化图像，每个相机使用不同颜色
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    
    返回:
    - colored_bev: 彩色编码的BEV图像
    """
    # 为每个相机分配颜色
    camera_colors = {
        'CAM_FRONT': [255, 0, 0],          # 红色
        'CAM_FRONT_LEFT': [255, 165, 0],   # 橙色
        'CAM_FRONT_RIGHT': [255, 255, 0],  # 黄色
        'CAM_BACK': [0, 0, 255],           # 蓝色
        'CAM_BACK_LEFT': [75, 0, 130],     # 靛蓝色
        'CAM_BACK_RIGHT': [238, 130, 238]  # 紫色
    }
    
    # 获取BEV尺寸
    first_cam = list(bev_results.keys())[0]
    bev_height, bev_width = bev_results[first_cam]['bev'].shape[:2]
    
    # 检查是否为RGBA格式
    is_rgba = bev_results[first_cam]['bev'].shape[2] == 4
    
    # 初始化彩色BEV图像
    if is_rgba:
        # 带透明通道
        colored_bev = np.zeros((bev_height, bev_width, 4), dtype=np.uint8)
    else:
        # 白色背景
        colored_bev = np.ones((bev_height, bev_width, 3), dtype=np.uint8) * 255
    
    # 添加半透明的相机覆盖范围
    for cam_channel, result in bev_results.items():
        # 确定掩码
        if is_rgba:
            mask = result['bev'][:, :, 3] > 0
        else:
            mask = result['mask'] > 0
            
        if cam_channel in camera_colors:
            color = np.array(camera_colors[cam_channel], dtype=np.uint8)
            
            # 应用颜色到掩码区域
            if is_rgba:
                # 对于RGBA图像，设置RGB通道并更新Alpha通道
                for c in range(3):
                    colored_pixels = colored_bev[:, :, 3] > 0  # 已有颜色的像素
                    new_pixels = mask & ~colored_pixels  # 新添加的像素
                    overlap_pixels = mask & colored_pixels  # 重叠区域
                    
                    # 新区域直接设置颜色
                    colored_bev[:, :, c][new_pixels] = color[c]
                    
                    # 重叠区域混合颜色 (50-50混合)
                    if np.any(overlap_pixels):
                        colored_bev[:, :, c][overlap_pixels] = (
                            colored_bev[:, :, c][overlap_pixels] * 0.5 + 
                            color[c] * 0.5
                        ).astype(np.uint8)
                
                # 设置Alpha通道为不透明
                colored_bev[:, :, 3][mask] = 255
            else:
                # 对于RGB图像，应用颜色混合
                for c in range(3):
                    colored_bev[:, :, c][mask] = (
                        colored_bev[:, :, c][mask] * 0.2 +  # 保留20%原始颜色
                        color[c] * 0.8  # 添加80%新颜色
                    ).astype(np.uint8)
    
    # 如果是透明图像，保存PNG
    # if is_rgba:
    #     cv2.imwrite('camera_coverage_viz.png', cv2.cvtColor(colored_bev, cv2.COLOR_RGBA2BGRA))
    
    return colored_bev

def visualize_multicam_bev(sample_token, bev_results, fused_bev):
    """
    可视化多相机BEV结果
    
    参数:
    - sample_token: 样本token
    - bev_results: 包含所有相机BEV结果的字典
    - fused_bev: 融合后的BEV图像
    """
    # 计算绘图所需的行数和列数
    num_cameras = len(bev_results)
    n_cols = 3
    n_rows = (num_cameras // n_cols) + 1 + 1  # +1 为融合结果和彩色可视化
    
    plt.figure(figsize=(n_cols * 5, n_rows * 5))
    
    # 检查是否为RGBA格式
    is_rgba = len(fused_bev.shape) == 3 and fused_bev.shape[2] == 4
    
    # 绘制每个相机的BEV
    for i, (cam_channel, result) in enumerate(bev_results.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f'BEV: {cam_channel}')
        
        bev_img = result['bev']
        if is_rgba and bev_img.shape[2] == 4:
            # 为RGBA图像创建白色背景
            white_bg = np.ones((bev_img.shape[0], bev_img.shape[1], 3), dtype=np.uint8) * 255
            alpha = bev_img[:, :, 3].astype(float) / 255
            alpha = np.expand_dims(alpha, axis=2)
            rgb = bev_img[:, :, 0:3]
            composite = rgb * alpha + white_bg * (1 - alpha)
            plt.imshow(composite.astype(np.uint8))
        else:
            plt.imshow(bev_img)
        plt.axis('off')
    
    # 绘制融合BEV
    plt.subplot(n_rows, n_cols, num_cameras + 1)
    plt.title('Fused BEV (All Cameras)')
    
    if is_rgba:
        # 为RGBA图像创建白色背景
        white_bg = np.ones((fused_bev.shape[0], fused_bev.shape[1], 3), dtype=np.uint8) * 255
        alpha = fused_bev[:, :, 3].astype(float) / 255
        alpha = np.expand_dims(alpha, axis=2)
        rgb = fused_bev[:, :, 0:3]
        composite = rgb * alpha + white_bg * (1 - alpha)
        plt.imshow(composite.astype(np.uint8))
    else:
        plt.imshow(fused_bev)
    plt.axis('off')
    
    # 绘制彩色编码的可视化
    colored_viz = create_colored_bev_visualization(bev_results)
    plt.subplot(n_rows, n_cols, num_cameras + 2)
    plt.title('Camera Coverage Visualization')
    plt.imshow(colored_viz)
    plt.axis('off')
    
    plt.tight_layout()
    # plt.savefig(f'multicam_bev_fusion_{sample_token[:8]}.png')
    plt.show()
    
    # 单独保存融合BEV
    plt.figure(figsize=(10, 10))
    plt.title('Fused Bird\'s Eye View (All Cameras)')
    
    if is_rgba:
        # 显示RGB通道
        plt.imshow(fused_bev[:, :, 0:3])
        # 使用Alpha通道作为透明度
        plt.imshow(np.zeros_like(fused_bev[:, :, 0]), alpha=fused_bev[:, :, 3]/255)
    else:
        plt.imshow(fused_bev)
    plt.axis('off')
    # plt.savefig(f'fused_bev_{sample_token[:8]}.png', transparent=is_rgba)

def apply_channel_gain_correction(bev_results, reference_cam='CAM_FRONT'):
    """
    应用通道增益校正，解决不同摄像头曝光差异问题
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - reference_cam: 作为参考的相机通道（其他相机会校正到这个相机的色彩分布）
    
    返回:
    - corrected_results: 校正后的BEV结果
    """
    if reference_cam not in bev_results:
        # 如果参考相机不存在，使用第一个可用相机
        reference_cam = list(bev_results.keys())[0]
        print(f"参考相机 {reference_cam} 不可用，使用 {reference_cam} 作为参考")
    
    # 创建校正后的结果字典
    corrected_results = {}
    
    # 获取参考相机的有效区域
    ref_bev = bev_results[reference_cam]['bev']
    if ref_bev.shape[2] == 4:  # RGBA格式
        ref_mask = ref_bev[:, :, 3] > 0
    else:
        ref_mask = bev_results[reference_cam]['mask'] > 0
    
    # 如果参考相机没有有效区域，返回原始结果
    if not np.any(ref_mask):
        print(f"参考相机 {reference_cam} 没有有效区域，跳过校正")
        return bev_results
    
    # 提取参考相机有效区域的RGB通道平均值
    ref_means = []
    for c in range(3):  # RGB通道
        channel_values = ref_bev[:, :, c][ref_mask]
        if len(channel_values) > 0:
            ref_means.append(np.mean(channel_values))
        else:
            ref_means.append(128)  # 默认中性值
    
    ref_means = np.array(ref_means)
    print(f"参考相机 {reference_cam} RGB均值: {ref_means}")
    
    # 对每个相机进行校正
    for cam_channel, result in bev_results.items():
        bev = result['bev'].copy()
        
        # 跳过参考相机
        if cam_channel == reference_cam:
            corrected_results[cam_channel] = result.copy()
            continue
        
        # 获取当前相机的有效区域
        if bev.shape[2] == 4:  # RGBA格式
            cam_mask = bev[:, :, 3] > 0
        else:
            cam_mask = result['mask'] > 0
        
        # 如果当前相机没有有效区域，直接使用原始结果
        if not np.any(cam_mask):
            corrected_results[cam_channel] = result.copy()
            continue
        
        # 提取当前相机有效区域的RGB通道平均值
        cam_means = []
        for c in range(3):  # RGB通道
            channel_values = bev[:, :, c][cam_mask]
            if len(channel_values) > 0:
                cam_means.append(np.mean(channel_values))
            else:
                cam_means.append(128)  # 默认中性值
        
        cam_means = np.array(cam_means)
        print(f"相机 {cam_channel} RGB均值: {cam_means}")
        
        # 计算增益因子 (参考相机均值 / 当前相机均值)
        # 避免除以零或接近零的值
        gain_factors = np.ones(3)
        for c in range(3):
            if cam_means[c] > 5.0:  # 防止除以太小的值
                gain_factors[c] = ref_means[c] / cam_means[c]
                # 限制增益因子在合理范围内
                gain_factors[c] = np.clip(gain_factors[c], 0.5, 2.0)
        
        print(f"相机 {cam_channel} 增益因子: {gain_factors}")
        
        # 应用增益校正
        corrected_bev = bev.copy()
        for c in range(3):  # RGB通道
            # 仅对有效区域应用增益
            channel = corrected_bev[:, :, c].astype(np.float32)
            channel[cam_mask] = np.clip(channel[cam_mask] * gain_factors[c], 0, 255)
            corrected_bev[:, :, c] = channel.astype(np.uint8)
        
        # 保存校正后的结果
        corrected_result = result.copy()
        corrected_result['bev'] = corrected_bev
        corrected_results[cam_channel] = corrected_result
    
    return corrected_results

def histogram_matching_correction(bev_results, reference_cam='CAM_FRONT'):
    """
    使用直方图匹配进行更精细的颜色校正
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - reference_cam: 作为参考的相机通道
    
    返回:
    - corrected_results: 校正后的BEV结果
    """
    if reference_cam not in bev_results:
        reference_cam = list(bev_results.keys())[0]
        print(f"参考相机 {reference_cam} 不可用，使用 {reference_cam} 作为参考")
    
    # 创建校正后的结果字典
    corrected_results = {}
    
    # 获取参考相机的有效区域和图像
    ref_bev = bev_results[reference_cam]['bev']
    if ref_bev.shape[2] == 4:  # RGBA格式
        ref_mask = ref_bev[:, :, 3] > 0
    else:
        ref_mask = bev_results[reference_cam]['mask'] > 0
    
    # 如果参考相机没有有效区域，返回原始结果
    if not np.any(ref_mask):
        print(f"参考相机 {reference_cam} 没有有效区域，跳过校正")
        return bev_results
    
    # 提取参考相机的有效RGB像素
    ref_rgb = []
    for c in range(3):  # RGB通道
        ref_rgb.append(ref_bev[:, :, c][ref_mask])
    
    # 对每个相机进行校正
    for cam_channel, result in bev_results.items():
        bev = result['bev'].copy()
        
        # 跳过参考相机
        if cam_channel == reference_cam:
            corrected_results[cam_channel] = result.copy()
            continue
        
        # 获取当前相机的有效区域
        if bev.shape[2] == 4:  # RGBA格式
            cam_mask = bev[:, :, 3] > 0
        else:
            cam_mask = result['mask'] > 0
        
        # 如果当前相机没有有效区域，直接使用原始结果
        if not np.any(cam_mask):
            corrected_results[cam_channel] = result.copy()
            continue
        
        # 应用直方图匹配
        corrected_bev = bev.copy()
        for c in range(3):  # RGB通道
            # 提取当前通道的有效像素
            src_values = bev[:, :, c][cam_mask]
            if len(src_values) > 0 and len(ref_rgb[c]) > 0:
                # 计算累积分布函数(CDF)
                src_hist, src_bins = np.histogram(src_values, 256, [0, 256])
                ref_hist, ref_bins = np.histogram(ref_rgb[c], 256, [0, 256])
                
                # 计算累积分布
                src_cdf = src_hist.cumsum() / src_hist.sum()
                ref_cdf = ref_hist.cumsum() / ref_hist.sum()
                
                # 创建查找表
                lookup_table = np.zeros(256, dtype=np.uint8)
                src_idx = 0
                for ref_idx in range(256):
                    while src_idx < 255 and src_cdf[src_idx] < ref_cdf[ref_idx]:
                        src_idx += 1
                    lookup_table[ref_idx] = src_idx
                
                # 应用查找表进行直方图匹配
                channel = corrected_bev[:, :, c].copy()
                corrected_bev[:, :, c][cam_mask] = lookup_table[channel[cam_mask]]
        
        # 保存校正后的结果
        corrected_result = result.copy()
        corrected_result['bev'] = corrected_bev
        corrected_results[cam_channel] = corrected_result
    
    return corrected_results

def balance_camera_exposure(bev_results, method='channel_gain', reference_cam='CAM_FRONT'):
    """
    平衡不同相机之间的曝光差异
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - method: 曝光平衡方法，可选'channel_gain'（通道增益校正）或'histogram_matching'（直方图匹配）
    - reference_cam: 作为参考的相机通道
    
    返回:
    - corrected_results: 曝光平衡后的BEV结果
    """
    print(f"使用 {method} 方法平衡相机曝光...")
    
    # 检查参考相机是否存在
    if reference_cam not in bev_results:
        # 如果参考相机不存在，使用第一个可用相机
        reference_cam = list(bev_results.keys())[0]
        print(f"参考相机 {reference_cam} 不可用，使用 {reference_cam} 作为参考")
    
    # 根据选择的方法应用相应的曝光平衡
    if method == 'histogram_matching':
        print("应用直方图匹配进行曝光平衡...")
        return histogram_matching_correction(bev_results, reference_cam)
    else:  # 默认使用通道增益校正
        print("应用通道增益校正进行曝光平衡...")
        return apply_channel_gain_correction(bev_results, reference_cam)

def create_multicam_bev(sample_token, bev_width=50, bev_length=50, resolution=0.1, fusion_strategy='smooth_blend', use_parallel=True, apply_color_correction=True, color_correction_method='channel_gain'):
    """
    创建多相机融合的BEV图像，支持并行处理
    
    参数:
    - sample_token: 样本token
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    - fusion_strategy: 融合策略，可选'max_intensity', 'weighted_average', 'min_depth', 'smooth_blend'
    - use_parallel: 是否使用并行处理
    - apply_color_correction: 是否应用颜色校正
    - color_correction_method: 颜色校正方法，可选'channel_gain'（通道增益校正）或'histogram_matching'（直方图匹配）
    
    返回:
    - fused_bev: 融合后的BEV图像
    - bev_results: 各个相机的BEV结果
    """
    start_time = time.time()
    
    # 1. 处理所有相机（并行或串行）
    if use_parallel:
        bev_results = process_all_cameras_parallel(sample_token, bev_width, bev_length, resolution)
    else:
        bev_results = process_all_cameras(sample_token, bev_width, bev_length, resolution)
    
    processing_time = time.time() - start_time
    print(f"所有相机处理时间: {processing_time:.2f}秒")
    
    # 2. 应用曝光平衡（可选）
    if apply_color_correction:
        color_correction_start = time.time()
        print(f"应用曝光平衡 (方法: {color_correction_method})...")
        bev_results = balance_camera_exposure(bev_results, method=color_correction_method)
        color_correction_time = time.time() - color_correction_start
        print(f"曝光平衡时间: {color_correction_time:.2f}秒")
    
    # 3. 融合BEV图像
    fusion_start = time.time()
    # 传递额外参数给min_depth和smooth_blend策略
    if fusion_strategy in ['min_depth', 'smooth_blend']:
        fused_bev, combined_mask = fuse_bev_images(
            bev_results, 
            strategy=fusion_strategy, 
            sample_token=sample_token,
            bev_width=bev_width,
            bev_length=bev_length,
            resolution=resolution
        )
    else:
        fused_bev, combined_mask = fuse_bev_images(bev_results, strategy=fusion_strategy)
    
    fusion_time = time.time() - fusion_start
    print(f"Fusion time: {fusion_time:.2f}s")
    
    # Visualize result
    visualize_multicam_bev(sample_token, bev_results, fused_bev)
    
    # Save image
    if len(fused_bev.shape) == 3 and fused_bev.shape[2] == 4:
        print("Save PNG image...")
        correction_suffix = f"_{color_correction_method}" if apply_color_correction else "_no_correction"
        cv2.imwrite(f'fused_bev_{fusion_strategy}{correction_suffix}_{sample_token[:8]}.png', 
                   cv2.cvtColor(fused_bev, cv2.COLOR_RGBA2BGRA))
    
    # Total time
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    
    return fused_bev, bev_results

if __name__ == "__main__":
    my_sample = nusc.sample[120]
    
    print("=" * 60)
    print("Multi-Camera BEV Fusion (Parallel Processing)")
    print("=" * 60)
    
    # Create multicamera BEV
    fused_bev, bev_results = create_multicam_bev(
        my_sample['token'], 
        bev_width=40,
        bev_length=40,
        resolution=0.05,
        fusion_strategy='smooth_blend',
        use_parallel=True,
        apply_color_correction=True,
        color_correction_method='channel_gain'
    )