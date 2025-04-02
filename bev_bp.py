import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
import time
import concurrent.futures
from backward_projection_cpu import create_bev_view, smooth_bev_image
from pyquaternion import Quaternion
from calculate_projection_matrix import calculate_projection_matrix

# 初始化NuScenes实例
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

def get_camera_info(sample_token, cam_channel):
    """
    获取相机在车辆坐标系中的位置和方向信息
    
    参数:
    - sample_token: 样本token
    - cam_channel: 相机通道
    
    返回:
    - camera_info: 包含相机信息的字典
    """
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    # 获取相机内参
    intrinsic = np.array(cs_record['camera_intrinsic'])
    
    # 获取相机外参
    translation = np.array(cs_record['translation'])
    rotation = Quaternion(cs_record['rotation']).rotation_matrix
    
    # 获取自车位姿
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation']).rotation_matrix
    
    # 返回相机信息
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
    处理单个相机的BEV投影
    
    参数:
    - sample_token: 样本token
    - cam_channel: 相机通道
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    
    返回:
    - bird_eye_view: 处理后的俯视图
    - projection_mask: 投影掩码
    - camera_info: 相机信息
    """
    try:
        # 获取相机信息
        camera_info = get_camera_info(sample_token, cam_channel)
        
        # 使用backward_projection_cpu.py中的create_top_view函数
        bird_eye_view, _ = create_bev_view(
            sample_index=1, cam_channel=cam_channel, 
            bev_width=bev_width, 
            bev_length=bev_length, 
            resolution=resolution
        )
        bird_eye_view = smooth_bev_image(bird_eye_view, resolution=0.04)
        
        # 创建投影掩码（Alpha > 0的区域）
        projection_mask = (bird_eye_view[:, :, 3] > 0).astype(np.uint8)
        
        return bird_eye_view, projection_mask, camera_info
        
    except Exception as e:
        print(f"处理相机 {cam_channel} 时出错: {str(e)}")
        # 错误处理，返回空图像
        bev_height = int(bev_length / resolution)
        bev_width_pixels = int(bev_width / resolution)
        empty_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
        empty_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
        
        # 返回空相机信息
        empty_camera_info = {
            'translation': np.zeros(3),
            'rotation': np.eye(3),
            'intrinsic': np.eye(3),
            'ego_translation': np.zeros(3),
            'ego_rotation': np.eye(3)
        }
        
        return empty_bev, empty_mask, empty_camera_info

def process_camera_worker(params):
    """
    线程工作函数：处理单个相机BEV
    
    参数:
    - params: 包含处理所需参数的字典
    
    返回:
    - 相机通道
    - 处理结果 (bev, mask, camera_info)
    """
    sample_token = params['sample_token']
    cam_channel = params['cam_channel']
    bev_width = params['bev_width']
    bev_length = params['bev_length']
    resolution = params['resolution']
    
    print(f"\n线程开始处理相机: {cam_channel}")
    start_time = time.time()
    
    try:
        bev, mask, camera_info = process_single_camera(
            sample_token, cam_channel, bev_width, bev_length, resolution
        )
        processing_time = time.time() - start_time
        print(f"相机 {cam_channel} 处理完成，耗时: {processing_time:.2f}秒")
        return cam_channel, (bev, mask, camera_info)
    except Exception as e:
        print(f"处理相机 {cam_channel} 时出错: {str(e)}")
        # 返回空结果
        bev_height = int(bev_length / resolution)
        bev_width_pixels = int(bev_width / resolution)
        empty_bev = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
        empty_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
        
        # 返回空相机信息
        empty_camera_info = {
            'translation': np.zeros(3),
            'rotation': np.eye(3),
            'intrinsic': np.eye(3),
            'ego_translation': np.zeros(3),
            'ego_rotation': np.eye(3)
        }
        
        return cam_channel, (empty_bev, empty_mask, empty_camera_info)

def process_all_cameras_parallel(sample_token, bev_width=40, bev_length=40, resolution=0.04, max_workers=6):
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
                cam_channel, (bev, mask, camera_info) = future.result()
                bev_results[cam_channel] = {
                    'bev': bev,
                    'mask': mask,
                    'camera_info': camera_info
                }
            except Exception as e:
                print(f"获取任务结果时出错: {str(e)}")
    
    # 计算总处理时间
    total_time = time.time() - start_time
    print(f"\n所有相机并行处理完成，总耗时: {total_time:.2f}秒")
    print(f"成功处理的相机数: {len(bev_results)}/{len(camera_channels)}")
    
    return bev_results

def process_all_cameras(sample_token, bev_width=40, bev_length=40, resolution=0.04):
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
            bev, mask, camera_info = process_single_camera(
                sample_token, cam_channel, bev_width, bev_length, resolution
            )
            bev_results[cam_channel] = {
                'bev': bev,
                'mask': mask,
                'camera_info': camera_info
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

def position_based_stitch_bev_images(bev_results, bev_width=40, bev_length=40, resolution=0.04):
    """
    基于相机位置信息拼接多个相机的BEV图像
    
    参数:
    - bev_results: 包含所有相机BEV结果的字典
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    
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
    
    # 初始化累积权重图，用于平滑混合
    weight_map = np.zeros((bev_height, bev_width_pixels), dtype=np.float32)
    
    # 创建相机贡献图，用于存储每个相机对每个像素的贡献
    if is_rgba:
        camera_contributions = np.zeros((bev_height, bev_width_pixels, len(bev_results), 4), dtype=np.float32)
    else:
        camera_contributions = np.zeros((bev_height, bev_width_pixels, len(bev_results), 3), dtype=np.float32)
    
    # 创建相机权重图
    camera_weights = np.zeros((bev_height, bev_width_pixels, len(bev_results)), dtype=np.float32)
    
    # 相机位置的字典 
    camera_positions = {}
    
    # 优先级顺序：前部相机 > 侧前相机 > 侧后相机 > 后部相机
    # 这是因为前视图通常质量更好，后视图可能有扭曲
    camera_priority = {
        'CAM_FRONT': 1,
        'CAM_FRONT_LEFT': 2,
        'CAM_FRONT_RIGHT': 2,
        'CAM_BACK': 4,
        'CAM_BACK_LEFT': 3,
        'CAM_BACK_RIGHT': 3
    }
    
    # 第一步：收集每个相机的贡献
    for i, (cam_channel, result) in enumerate(bev_results.items()):
        # 提取相机信息和BEV图像
        bev = result['bev'].astype(np.float32)
        camera_info = result['camera_info']
        translation = camera_info['translation']
        
        # 记录相机在车辆坐标系中的位置
        camera_positions[cam_channel] = translation
        
        # 找出有效区域（Alpha > 0的像素）
        if is_rgba:
            valid_pixels = bev[:, :, 3] > 0
        else:
            valid_pixels = result['mask'] > 0
            
        if not np.any(valid_pixels):
            continue
            
        # 计算该相机的基础优先级（数值越小优先级越高）
        base_priority = camera_priority.get(cam_channel, 5)
        
        # 找出所有有效像素的坐标
        y_indices, x_indices = np.where(valid_pixels)
        
        # 计算边缘距离图（用于平滑过渡）
        # 先创建二值掩码
        mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
        mask[y_indices, x_indices] = 1
        
        # 应用距离变换，计算每个像素到边缘的距离
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # 归一化距离变换结果
        max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1.0
        dist_transform = dist_transform / max_dist
        
        # 对每个有效像素，存储相机贡献
        for j in range(len(y_indices)):
            y, x = y_indices[j], x_indices[j]
            
            # 计算基于距离变换的权重（中心像素权重高，边缘像素权重低）
            edge_weight = dist_transform[y, x]
            
            # 最终权重是优先级和边缘距离的组合
            pixel_weight = base_priority * (1.0 - 0.3 * edge_weight)  # 边缘权重影响30%
            
            # 存储像素贡献和权重
            if is_rgba:
                camera_contributions[y, x, i, :] = bev[y, x, :]
            else:
                camera_contributions[y, x, i, :] = bev[y, x, :3]
            
            camera_weights[y, x, i] = 1.0 / pixel_weight  # 权重反比于优先级值
    
    # 第二步：合并所有相机的贡献
    for y in range(bev_height):
        for x in range(bev_width_pixels):
            # 获取所有相机在此像素的权重
            pixel_weights = camera_weights[y, x, :]
            
            # 如果没有任何相机贡献，跳过
            if np.sum(pixel_weights) == 0:
                continue
            
            # 归一化权重
            pixel_weights = pixel_weights / np.sum(pixel_weights)
            
            # 混合所有相机的贡献
            blended_pixel = np.zeros(4 if is_rgba else 3, dtype=np.float32)
            
            for i in range(len(bev_results)):
                if pixel_weights[i] > 0:
                    blended_pixel += pixel_weights[i] * camera_contributions[y, x, i, :]
            
            # 应用混合像素
            if is_rgba:
                # 如果Alpha大于0，则更新像素
                if blended_pixel[3] > 0:
                    stitched_bev[y, x, :] = blended_pixel.astype(np.uint8)
            else:
                stitched_bev[y, x, :] = blended_pixel.astype(np.uint8)
    
    # 第三步：应用后处理以平滑边缘
    if is_rgba:
        # 提取RGB和Alpha通道
        rgb_channels = stitched_bev[:, :, :3]
        alpha_channel = stitched_bev[:, :, 3]
        
        # 创建二值掩码（Alpha > 0的区域）
        mask = (alpha_channel > 0).astype(np.uint8)
        
        # 应用形态学闭操作填充边缘间隙
        kernel_size = max(3, int(5 * resolution / 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 对RGB应用高斯模糊，仅在掩码内
        blur_size = max(3, int(3 * resolution / 0.1))
        if blur_size % 2 == 0:
            blur_size += 1
        
        # 创建边缘掩码（仅包含边缘区域）
        edge_mask = closed_mask - mask
        
        # 对边缘区域应用模糊
        blurred_rgb = cv2.GaussianBlur(rgb_channels, (blur_size, blur_size), 0)
        
        # 在边缘区域使用模糊结果
        edge_indices = np.where(edge_mask > 0)
        rgb_channels[edge_indices] = blurred_rgb[edge_indices]
        
        # 更新Alpha通道
        alpha_channel = cv2.morphologyEx(alpha_channel, cv2.MORPH_CLOSE, kernel)
        alpha_channel = cv2.GaussianBlur(alpha_channel, (blur_size, blur_size), 0)
        
        # 重新组合RGB和Alpha
        stitched_bev = np.dstack((rgb_channels, alpha_channel))
    else:
        # 对RGB图像应用高斯模糊
        kernel_size = max(3, int(5 * resolution / 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        stitched_bev = cv2.morphologyEx(stitched_bev, cv2.MORPH_CLOSE, kernel)
        
        blur_size = max(3, int(3 * resolution / 0.1))
        if blur_size % 2 == 0:
            blur_size += 1
        stitched_bev = cv2.GaussianBlur(stitched_bev, (blur_size, blur_size), 0)
    
    print("相机在车辆坐标系中的位置:")
    for cam, pos in camera_positions.items():
        print(f"{cam}: {pos}")
    
    return stitched_bev

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
                    colored_bev[:, :, c][mask] = color[c]
                
                # 设置Alpha通道为不透明
                colored_bev[:, :, 3][mask] = 255
            else:
                # 对于RGB图像，应用颜色
                for c in range(3):
                    colored_bev[:, :, c][mask] = color[c]
    
    return colored_bev

def camera_pixel_to_bev_pixel(camera_channel, translation, bev_height, bev_width_pixels, resolution=0.04):
    """
    将相机在车辆坐标系中的位置转换为BEV图像中的像素坐标
    
    参数:
    - camera_channel: 相机通道名称
    - translation: 相机在车辆坐标系中的位置
    - bev_height: BEV图像高度（像素）
    - bev_width_pixels: BEV图像宽度（像素）
    - resolution: 分辨率（米/像素）
    
    返回:
    - pixel_x, pixel_y: 相机在BEV图像中的像素坐标
    """
    camera_x = translation[0]
    camera_y = translation[1]
    
    # 对于后部相机，特殊处理坐标映射
    if 'BACK' in camera_channel:
        # 后部相机需要映射到图像的上部区域
        pixel_y = int(bev_height/2 - camera_x / resolution)
    else:
        # 前部和侧部相机使用原有映射
        pixel_y = bev_height - 1 - int(camera_x / resolution)
    
    # 横向坐标映射保持不变
    pixel_x = int(bev_width_pixels / 2 + camera_y / resolution)
    
    return pixel_x, pixel_y

def visualize_multicam_bev(sample_token, bev_results, stitched_bev=None, save_flag=False):
    """
    可视化多相机BEV结果
    
    参数:
    - sample_token: 样本token
    - bev_results: 包含所有相机BEV结果的字典
    - stitched_bev: 拼接后的BEV图像（可选）
    - save_flag: 是否保存图像
    """
    # 计算绘图所需的行数和列数
    num_cameras = len(bev_results)
    n_cols = 3
    n_rows = (num_cameras // n_cols) + 1  # +1 为拼接结果和彩色可视化
    
    plt.figure(figsize=(n_cols * 5, n_rows * 5))
    
    # 检查是否为RGBA格式
    first_cam = list(bev_results.keys())[0]
    is_rgba = bev_results[first_cam]['bev'].shape[2] == 4
    
    # 绘制每个相机的BEV
    for i, (cam_channel, result) in enumerate(bev_results.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f'BEV: {cam_channel}')
        
        bev_img = result['bev']
        if is_rgba:
            # 为RGBA图像创建白色背景
            white_bg = np.ones((bev_img.shape[0], bev_img.shape[1], 3), dtype=np.uint8) * 255
            alpha = bev_img[:, :, 3].astype(float) / 255
            alpha = np.expand_dims(alpha, axis=2)
            rgb = bev_img[:, :, 0:3]
            composite = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            plt.imshow(composite)
        else:
            plt.imshow(bev_img)
        plt.axis('off')
        
        # 获取图像中心坐标（表示车辆位置）
        bev_height, bev_width_pixels = bev_img.shape[:2]
        center_x = bev_width_pixels // 2
        center_y = bev_height // 2
        
        # 在图像中心标记车辆位置
        plt.plot(center_x, center_y, 'o', color='red', markersize=8)
        
        # 标记相机位置
        translation = result['camera_info']['translation']
        cam_x = translation[0]  # 相机X坐标（前/后）
        cam_y = translation[1]  # 相机Y坐标（左/右）
        
        # 计算相机在图像中的位置（相对于中心点）- 适应水平镜像
        cam_pixel_x = center_x - int(cam_y / 0.04)  # Y轴对应横向，镜像后负号变为减号
        cam_pixel_y = center_y - int(cam_x / 0.04)  # X轴对应纵向（正值向上）
        
        # 标记相机位置
        if 0 <= cam_pixel_y < bev_height and 0 <= cam_pixel_x < bev_width_pixels:
            plt.plot(cam_pixel_x, cam_pixel_y, 'o', color='blue', markersize=6)
            plt.text(cam_pixel_x + 3, cam_pixel_y + 3, cam_channel.split('_')[1], 
                     color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    
    # 如果提供了拼接BEV，绘制它
    if stitched_bev is not None:
        plt.subplot(n_rows, n_cols, num_cameras + 1)
        plt.title('Stitched BEV (All Cameras)')
        
        if is_rgba:
            # 为RGBA图像创建白色背景
            white_bg = np.ones((stitched_bev.shape[0], stitched_bev.shape[1], 3), dtype=np.uint8) * 255
            alpha = stitched_bev[:, :, 3].astype(float) / 255
            alpha = np.expand_dims(alpha, axis=2)
            rgb = stitched_bev[:, :, 0:3]
            composite = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            plt.imshow(composite)
        else:
            plt.imshow(stitched_bev)
        plt.axis('off')
        
        # 在拼接图像中，标记车辆中心位置
        bev_height, bev_width_pixels = stitched_bev.shape[:2]
        center_x = bev_width_pixels // 2
        center_y = bev_height // 2
        
        # 绘制车辆中心位置
        plt.plot(center_x, center_y, 'o', color='red', markersize=8)
        plt.text(center_x + 5, center_y + 5, 'EGO', color='white', 
                 bbox=dict(facecolor='black', alpha=0.5))
        
        # 绘制所有相机位置（以不同颜色标记）
        for cam_channel, result in bev_results.items():
            # 获取相机位置
            translation = result['camera_info']['translation']
            cam_x = translation[0]  # 相机X坐标（前/后）
            cam_y = translation[1]  # 相机Y坐标（左/右）
            
            # 计算相机在图像中的位置（相对于中心点）- 适应水平镜像
            cam_pixel_x = center_x - int(cam_y / 0.04)  # Y轴对应横向，镜像后负号变为减号
            cam_pixel_y = center_y - int(cam_x / 0.04)  # X轴对应纵向（正值向上）
            
            # 确保坐标在图像范围内
            if 0 <= cam_pixel_y < bev_height and 0 <= cam_pixel_x < bev_width_pixels:
                # 根据相机类型选择颜色
                if 'FRONT' in cam_channel and 'LEFT' not in cam_channel and 'RIGHT' not in cam_channel:
                    color = 'red'  # 前部相机
                elif 'FRONT_LEFT' in cam_channel:
                    color = 'orange'  # 左前相机
                elif 'FRONT_RIGHT' in cam_channel:
                    color = 'yellow'  # 右前相机
                elif 'BACK' in cam_channel and 'LEFT' not in cam_channel and 'RIGHT' not in cam_channel:
                    color = 'blue'  # 后部相机
                elif 'BACK_LEFT' in cam_channel:
                    color = 'indigo'  # 左后相机
                elif 'BACK_RIGHT' in cam_channel:
                    color = 'purple'  # 右后相机
                else:
                    color = 'white'
                
                plt.plot(cam_pixel_x, cam_pixel_y, 'o', color=color, markersize=6)
                plt.text(cam_pixel_x + 3, cam_pixel_y + 3, cam_channel.split('_')[1], 
                         color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    
    # 绘制彩色编码的可视化
    colored_viz = create_colored_bev_visualization(bev_results)
    plt.subplot(n_rows, n_cols, num_cameras + 2)
    plt.title('Camera Coverage Visualization')
    plt.imshow(colored_viz)
    plt.axis('off')
    
    plt.tight_layout()
    if save_flag:
        plt.savefig(f'multicam_bev_stitched_{sample_token[:8]}.png')
    plt.show()
    
    # 单独显示拼接BEV
    if stitched_bev is not None:
        plt.figure(figsize=(10, 10))
        plt.title('Stitched Bird\'s Eye View (All Cameras)')
        
        if is_rgba:
            # 为RGBA图像创建白色背景
            white_bg = np.ones((stitched_bev.shape[0], stitched_bev.shape[1], 3), dtype=np.uint8) * 255
            alpha = stitched_bev[:, :, 3].astype(float) / 255
            alpha = np.expand_dims(alpha, axis=2)
            rgb = stitched_bev[:, :, 0:3]
            composite = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            plt.imshow(composite)
        else:
            plt.imshow(stitched_bev)
        plt.axis('off')
        
        # 在拼接图像中，标记车辆中心位置
        bev_height, bev_width_pixels = stitched_bev.shape[:2]
        center_x = bev_width_pixels // 2
        center_y = bev_height // 2
        
        # 绘制车辆中心位置
        plt.plot(center_x, center_y, 'o', color='red', markersize=8)
        plt.text(center_x + 5, center_y + 5, 'EGO', color='white', 
                 bbox=dict(facecolor='black', alpha=0.5))
        
        # 绘制所有相机位置（以不同颜色标记）
        for cam_channel, result in bev_results.items():
            # 获取相机位置
            translation = result['camera_info']['translation']
            cam_x = translation[0]  # 相机X坐标（前/后）
            cam_y = translation[1]  # 相机Y坐标（左/右）
            
            # 计算相机在图像中的位置（相对于中心点）- 适应水平镜像
            cam_pixel_x = center_x - int(cam_y / 0.04)  # Y轴对应横向，镜像后负号变为减号
            cam_pixel_y = center_y - int(cam_x / 0.04)  # X轴对应纵向（正值向上）
            
            # 确保坐标在图像范围内
            if 0 <= cam_pixel_y < bev_height and 0 <= cam_pixel_x < bev_width_pixels:
                # 根据相机类型选择颜色
                if 'FRONT' in cam_channel and 'LEFT' not in cam_channel and 'RIGHT' not in cam_channel:
                    color = 'red'  # 前部相机
                elif 'FRONT_LEFT' in cam_channel:
                    color = 'orange'  # 左前相机
                elif 'FRONT_RIGHT' in cam_channel:
                    color = 'yellow'  # 右前相机
                elif 'BACK' in cam_channel and 'LEFT' not in cam_channel and 'RIGHT' not in cam_channel:
                    color = 'blue'  # 后部相机
                elif 'BACK_LEFT' in cam_channel:
                    color = 'indigo'  # 左后相机
                elif 'BACK_RIGHT' in cam_channel:
                    color = 'purple'  # 右后相机
                else:
                    color = 'white'
                
                plt.plot(cam_pixel_x, cam_pixel_y, 'o', color=color, markersize=6)
                plt.text(cam_pixel_x + 3, cam_pixel_y + 3, cam_channel.split('_')[1], 
                         color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
        
        if save_flag:
            plt.savefig(f'stitched_bev_{sample_token[:8]}.png', transparent=is_rgba)
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

def create_multicam_bev(sample_token, bev_width=40, bev_length=40, resolution=0.04, use_parallel=True, save_flag=False, blend_factor=0.3, fusion_strategy='smooth_blend'):
    """
    创建多相机拼接的BEV图像
    
    参数:
    - sample_token: 样本token
    - bev_width: 俯视图宽度（米）
    - bev_length: 俯视图长度（米）
    - resolution: 分辨率（米/像素）
    - use_parallel: 是否使用并行处理
    - save_flag: 是否保存图像
    - blend_factor: 混合因子(0-1)，越大边缘越平滑，默认0.3
    - fusion_strategy: 融合策略，可选'position_based'或'smooth_blend'
    
    返回:
    - stitched_bev: 拼接后的BEV图像
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
    
    # 2. 根据选择的策略拼接BEV图像
    stitching_start = time.time()
    
    if fusion_strategy == 'smooth_blend':
        # 使用平滑混合策略
        transition_width = int(10 * blend_factor)  # 根据blend_factor调整过渡宽度
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
        # 使用基于位置的拼接（默认）
        stitched_bev = position_based_stitch_bev_images(bev_results, bev_width, bev_length, resolution)
    
    stitching_time = time.time() - stitching_start
    print(f"图像拼接时间: {stitching_time:.2f}秒")
    
    # 3. 额外应用边缘平滑处理
    if fusion_strategy != 'smooth_blend' and blend_factor > 0:
        # 如果使用的不是smooth_blend策略，则应用额外的边缘平滑
        print("应用额外的边缘平滑...")
        
        # 检查是否为RGBA图像
        is_rgba = stitched_bev.shape[2] == 4
        
        # 应用额外的边缘平滑
        kernel_size = max(3, int(7 * resolution / 0.1 * blend_factor))
        if kernel_size % 2 == 0:  # 确保kernel_size是奇数
            kernel_size += 1
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if is_rgba:
            # 分离通道
            rgb_channels = stitched_bev[:, :, :3].copy()
            alpha_channel = stitched_bev[:, :, 3].copy()
            
            # 只对有内容的区域应用模糊
            mask = (alpha_channel > 0).astype(np.uint8)
            
            # 应用边缘检测找出拼接边界
            edges = cv2.Canny(mask, 50, 150)
            
            # 扩大边缘区域
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 对边缘区域应用高斯模糊
            blur_size = max(5, int(5 * blend_factor * resolution / 0.04))
            if blur_size % 2 == 0:
                blur_size += 1
                
            blurred_rgb = cv2.GaussianBlur(rgb_channels, (blur_size, blur_size), 0)
            
            # 将模糊结果应用到边缘区域
            edge_indices = np.where(dilated_edges > 0)
            if len(edge_indices[0]) > 0:  # 确保有边缘像素
                rgb_channels[edge_indices] = blurred_rgb[edge_indices]
            
            # 重新合并通道
            stitched_bev = np.dstack((rgb_channels, alpha_channel))
        else:
            # 应用边缘检测找出拼接边界
            gray = cv2.cvtColor(stitched_bev, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 扩大边缘区域
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 对边缘区域应用高斯模糊
            blur_size = max(5, int(5 * blend_factor * resolution / 0.04))
            if blur_size % 2 == 0:
                blur_size += 1
                
            blurred_img = cv2.GaussianBlur(stitched_bev, (blur_size, blur_size), 0)
            
            # 将模糊结果应用到边缘区域
            edge_indices = np.where(dilated_edges > 0)
            if len(edge_indices[0]) > 0:  # 确保有边缘像素
                for c in range(3):  # RGB通道
                    stitched_bev[edge_indices[0], edge_indices[1], c] = blurred_img[edge_indices[0], edge_indices[1], c]
    
    # 4. 可视化结果
    visualize_multicam_bev(sample_token, bev_results, stitched_bev, save_flag)
    
    # 总时间
    total_time = time.time() - start_time
    print(f"总处理时间: {total_time:.2f}秒")
    
    # 5. 保存拼接图像
    if save_flag and stitched_bev.shape[2] == 4:  # RGBA格式
        print("保存PNG格式图像...")
        cv2.imwrite(f'stitched_bev_{fusion_strategy}_{sample_token[:8]}.png', 
                   cv2.cvtColor(stitched_bev, cv2.COLOR_RGBA2BGRA))
    
    return stitched_bev, bev_results

if __name__ == "__main__":
    print("=" * 60)
    print("多相机BEV拼接（基于相机位置信息）")
    print("=" * 60)
    
    # 获取示例数据
    my_sample = nusc.sample[120]
    
    # 创建多相机BEV图像并拼接
    stitched_bev, bev_results = create_multicam_bev(
        my_sample['token'], 
        bev_width=40,
        bev_length=40,
        resolution=0.04,
        use_parallel=True,
        save_flag=True,
        blend_factor=0.6,  # 增加混合因子以加强边缘平滑效果
        fusion_strategy='smooth_blend'  # 使用平滑混合策略
    ) 