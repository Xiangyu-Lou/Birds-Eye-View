import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 初始化nuScenes数据集
nusc = NuScenes(version='v1.0-mini', dataroot='F:/Project/nuscenes-devkit/v1.0-mini', verbose=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def extract_data(sample_token, cam_channel='CAM_FRONT'):
    """
    从nuScenes数据集提取相关数据

    返回:
    - img: 图像 (numpy.ndarray, H×W×3, RGB)
    - img_height, img_width: 图像尺寸
    - K: 相机内参 (3×3, numpy)
    - cs_record, ego_pose, cam_data: 传感器标定/自车姿态/相机数据
    """
    sample = nusc.get('sample', sample_token)

    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)

    # 获取相机图片路径并读取图像
    img_path = nusc.get_sample_data_path(cam_token)
    print(f"Read Image: {img_path}\n")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_height, img_width = img.shape[:2]

    # 获取标定数据
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    # 自车姿态
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    # 相机内参
    K = np.array(cs_record['camera_intrinsic'], dtype=np.float32)

    return img, img_height, img_width, K, cs_record, ego_pose, cam_data


def compute_coordinate_transforms(cs_record, ego_pose):
    """
    计算坐标系转换矩阵
    返回 (4×4) 的 cam_to_ego、ego_to_world、cam_to_world、world_to_cam
    """
    cam_to_ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    cam_to_ego_translation = np.array(cs_record['translation'])

    cam_to_ego = np.eye(4, dtype=np.float32)
    cam_to_ego[:3, :3] = cam_to_ego_rotation
    cam_to_ego[:3, 3] = cam_to_ego_translation

    ego_to_world_rotation = Quaternion(ego_pose['rotation']).rotation_matrix
    ego_to_world_translation = np.array(ego_pose['translation'])

    ego_to_world = np.eye(4, dtype=np.float32)
    ego_to_world[:3, :3] = ego_to_world_rotation
    ego_to_world[:3, 3] = ego_to_world_translation

    cam_to_world = ego_to_world @ cam_to_ego
    world_to_cam = np.linalg.inv(cam_to_world)

    return cam_to_ego, ego_to_world, cam_to_world, world_to_cam


def generate_bird_eye_view_torch(img, img_width, img_height, K,
                                 ego_to_world, world_to_cam,
                                 bev_width=40, bev_length=40,
                                 resolution=0.1):
    """
    使用PyTorch在GPU上生成BEV:
      1) 地面网格生成+变换
      2) 投影+筛选
      3) 深度冲突用CPU循环做最小深度过滤

    返回: 
    - bird_eye_view (numpy.ndarray, [H, W, 4], RGBA)
    - projection_mask (numpy.ndarray, [H, W], 0/1)
    - mapped_pixels (int)
    """

    # 1) 数据 -> Torch GPU
    torch_img = torch.from_numpy(img).float().to(device)       # [H, W, 3], float32
    K_torch = torch.from_numpy(K).float().to(device)           # [3, 3]
    ego_to_world_torch = torch.from_numpy(ego_to_world).float().to(device)  # [4, 4]
    world_to_cam_torch = torch.from_numpy(world_to_cam).float().to(device)  # [4, 4]

    bev_h = int(bev_length / resolution)
    bev_w = int(bev_width / resolution)

    # 2) 生成更密集的地面网格
    ground_resolution = resolution * 0.5
    x_range = torch.arange(-bev_length/2, bev_length/2, ground_resolution, device=device)
    y_range = torch.arange(-bev_width/2, bev_width/2, ground_resolution, device=device)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')   # [Nx, Ny]
    Nx, Ny = xx.shape

    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)

    # 齐次坐标 (x, y, 0, 1)
    ones = torch.ones_like(xx_flat, device=device)
    ground_points_ego = torch.stack([xx_flat, yy_flat, torch.zeros_like(xx_flat), ones], dim=1)  # [N,4]

    # ego -> world
    ground_points_world = (ego_to_world_torch @ ground_points_ego.t()).t()  # [N,4]
    # world -> cam
    ground_points_cam = (world_to_cam_torch @ ground_points_world.t()).t()  # [N,4]

    # 3) 筛选相机前方
    z_cam = ground_points_cam[:, 2]
    valid_mask = z_cam > 0
    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        # 全部在相机后方
        empty_bev = np.zeros((bev_h, bev_w, 4), dtype=np.uint8)
        return empty_bev, np.zeros((bev_h, bev_w), dtype=np.uint8), 0

    ground_points_cam = ground_points_cam[valid_idx]
    xx_ego_valid = xx_flat[valid_idx]
    yy_ego_valid = yy_flat[valid_idx]

    # 4) 相机内参投影
    x_cam = ground_points_cam[:, 0]
    y_cam = ground_points_cam[:, 1]
    z_cam = ground_points_cam[:, 2]

    u = K_torch[0, 0] * (x_cam / z_cam) + K_torch[0, 2]
    v = K_torch[1, 1] * (y_cam / z_cam) + K_torch[1, 2]

    within_width = (u >= 0) & (u < img_width)
    within_height = (v >= 0) & (v < img_height)
    valid_proj = within_width & within_height
    proj_idx = valid_proj.nonzero(as_tuple=True)[0]
    if proj_idx.numel() == 0:
        # 没有落在图像有效区的点
        empty_bev = np.zeros((bev_h, bev_w, 4), dtype=np.uint8)
        return empty_bev, np.zeros((bev_h, bev_w), dtype=np.uint8), 0

    # 最终
    u_final = u[proj_idx].long()
    v_final = v[proj_idx].long()
    z_final = z_cam[proj_idx]
    x_ego_final = xx_ego_valid[proj_idx]
    y_ego_final = yy_ego_valid[proj_idx]

    # 5) BEV 平面坐标计算
    i_bev = (bev_length/2 - x_ego_final) / resolution
    j_bev = (bev_width/2 - y_ego_final) / resolution
    i_bev = i_bev.long()
    j_bev = j_bev.long()

    in_bev_h = (i_bev >= 0) & (i_bev < bev_h)
    in_bev_w = (j_bev >= 0) & (j_bev < bev_w)
    in_bev = in_bev_h & in_bev_w
    valid_bev_idx = in_bev.nonzero(as_tuple=True)[0]
    if valid_bev_idx.numel() == 0:
        empty_bev = np.zeros((bev_h, bev_w, 4), dtype=np.uint8)
        return empty_bev, np.zeros((bev_h, bev_w), dtype=np.uint8), 0

    i_bev = i_bev[valid_bev_idx]
    j_bev = j_bev[valid_bev_idx]
    z_final = z_final[valid_bev_idx]
    u_final = u_final[valid_bev_idx]
    v_final = v_final[valid_bev_idx]

    # 从原图采样颜色
    color_array = torch_img[v_final, u_final, :]  # [M, 3], float

    # 6) CPU 上做“最小深度”筛选
    #    i_bev、j_bev 组合成 idx_1d = i*bev_w + j
    idx_1d = i_bev * bev_w + j_bev

    # 将上述索引与 z_final, color_array 都搬回CPU，做循环
    idx_1d_cpu = idx_1d.cpu().numpy()
    z_cpu = z_final.cpu().numpy()
    color_cpu = color_array.cpu().numpy()  # shape [M, 3]

    M = idx_1d_cpu.shape[0]
    # 初始化深度图与颜色映射
    depth_map = np.full((bev_h * bev_w), np.inf, dtype=np.float32)
    color_map = np.zeros((bev_h * bev_w, 3), dtype=np.float32)

    for i in range(M):
        pix_id = idx_1d_cpu[i]
        z_val = z_cpu[i]
        if z_val < depth_map[pix_id]:
            depth_map[pix_id] = z_val
            color_map[pix_id] = color_cpu[i]

    # 有效投影像素位置: depth_map < inf
    valid_pixel_mask = depth_map < np.inf
    mapped_pixels = np.count_nonzero(valid_pixel_mask)

    # 7) 构造输出 RGBA
    # color_map形状 (bev_h*bev_w, 3)
    color_map_2d = color_map.reshape(bev_h, bev_w, 3).astype(np.uint8)
    alpha_2d = np.where(valid_pixel_mask.reshape(bev_h, bev_w), 255, 0).astype(np.uint8)
    bird_eye_view = np.concatenate([color_map_2d, alpha_2d[..., None]], axis=-1)  # [bev_h, bev_w, 4]

    # projection_mask
    projection_mask = (alpha_2d > 0).astype(np.uint8)

    return bird_eye_view, projection_mask, mapped_pixels


def morphological_dilation_torch(mask, kernel_size=3):
    """
    使用 max_pool2d 实现二值膨胀 (PyTorch)
    mask: [H, W] (float)
    """
    mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    pad = kernel_size // 2
    dilated_4d = F.max_pool2d(mask_4d, kernel_size, stride=1, padding=pad)
    dilated = dilated_4d.squeeze(0).squeeze(0)
    return dilated


# def gaussian_blur_torch(img, kernel_size=3, sigma=1.0):

def gaussian_blur_torch(img, kernel_size=3, sigma=1.0):
    """
    img 可以是 [H, W], [H, W, 1], [H, W, 3], [H, W, 4] 或已经是 [C, H, W]
    """
    need_transpose = False
    # 如果是三维 [H, W, C]，并且 C in {1,3,4}，则做 permute
    if img.dim() == 3 and img.shape[-1] in (1,3,4):
        img = img.permute(2, 0, 1).contiguous()  # 变为 [C, H, W]
        need_transpose = True

    # 此时若是二维 [H, W] 也行——可以视为单通道 [1,H,W]，不过要再做一次unsqueeze(0)或改逻辑
    if img.dim() == 2:
        # 说明是 [H, W], 视作单通道 => [1, H, W]
        img = img.unsqueeze(0)
        need_transpose = False  # 因为原本就不是 [H,W,C]

    c, h, w = img.shape

    # 构造高斯核
    coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device)
    coords -= (kernel_size - 1) / 2.
    g = torch.exp(-(coords**2)/(2*sigma*sigma))
    g /= g.sum()

    gauss_2d = g[:, None] * g[None, :]  # [k, k]
    gauss_2d = gauss_2d.view(1,1,kernel_size,kernel_size).repeat(c,1,1,1)

    pad = kernel_size // 2
    # [N=1, C, H, W]  => pad => conv2d
    img_4d = img.unsqueeze(0)  # => [1, c, h, w]

    # 如果 (h, w) 太小，也无法做 reflect padding. 
    # 一般要求 h,w >= kernel_size. 
    # 可在此做检查，或者改成 'replicate'/'constant'
    x_pad = F.pad(img_4d, (pad,pad,pad,pad), mode='reflect')

    blurred = F.conv2d(x_pad, gauss_2d, groups=c)
    blurred = blurred.squeeze(0)

    if need_transpose:
        blurred = blurred.permute(1, 2, 0).contiguous()

    return blurred

    """
    使用 PyTorch 卷积实现简单高斯模糊 (通道独立)
    img: [C, H, W] or [H, W, C]
    """
    need_transpose = False
    if img.dim() == 3 and img.shape[-1] in (3,4):
        # [H, W, C] => [C, H, W]
        img = img.permute(2, 0, 1).contiguous()
        need_transpose = True

    c, h, w = img.shape
    coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device)
    coords -= (kernel_size - 1) / 2.
    g = torch.exp(-(coords**2)/(2*sigma*sigma))
    g /= g.sum()
    gauss_2d = g[:,None] * g[None,:]   # [k, k]

    # 卷积核形状: [C, 1, k, k] (Depthwise)
    gauss_2d = gauss_2d.view(1,1,kernel_size,kernel_size)
    gauss_2d = gauss_2d.repeat(c,1,1,1)

    pad = kernel_size // 2
    x_pad = F.pad(img.unsqueeze(0), (pad,pad,pad,pad), mode='reflect')
    blurred = F.conv2d(x_pad, gauss_2d, groups=c)
    blurred = blurred.squeeze(0)

    if need_transpose:
        blurred = blurred.permute(1,2,0).contiguous()

    return blurred


def post_process_bev_torch(bird_eye_view, projection_mask, mapped_pixels, resolution=0.1):
    """
    可选后处理: 形态学膨胀 + 高斯模糊 (PyTorch)
    bird_eye_view: [H, W, 4] (uint8)
    projection_mask: [H, W] (uint8)
    """
    if mapped_pixels == 0:
        return bird_eye_view

    bev_torch = torch.from_numpy(bird_eye_view).float().to(device)     # [H, W, 4]
    mask_torch = torch.from_numpy(projection_mask).float().to(device)  # [H, W]

    kernel_size = max(3, int(5 * resolution / 0.1))
    dilated_mask = morphological_dilation_torch(mask_torch, kernel_size=kernel_size)

    blur_size = max(3, int(3 * resolution / 0.1))
    if blur_size % 2 == 0:
        blur_size += 1

    rgb = bev_torch[:, :, :3]
    alpha = bev_torch[:, :, 3:4]

    # 分别模糊
    rgb_blurred = gaussian_blur_torch(rgb, kernel_size=blur_size, sigma=blur_size/6.0)
    alpha_blurred = gaussian_blur_torch(alpha, kernel_size=blur_size, sigma=blur_size/6.0)

    dilated_mask_3 = dilated_mask.unsqueeze(-1)
    processed_rgb = rgb * (1 - dilated_mask_3) + rgb_blurred * dilated_mask_3
    processed_alpha = alpha * (1 - dilated_mask_3) + alpha_blurred * dilated_mask_3

    processed_bev_torch = torch.cat([processed_rgb, processed_alpha], dim=-1)
    processed_bev = processed_bev_torch.clamp(0, 255).byte().cpu().numpy()

    return processed_bev


def visualize_bev(bird_eye_view, original_img, title="Modular BEV Projection"):
    """
    可视化俯视图结果
    """
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title)
    if bird_eye_view.shape[2] == 4:
        white_bg = np.ones((bird_eye_view.shape[0], bird_eye_view.shape[1], 3), dtype=np.uint8) * 255
        alpha = bird_eye_view[:, :, 3].astype(float) / 255
        alpha = np.expand_dims(alpha, axis=2)
        rgb = bird_eye_view[:, :, :3]
        composite = rgb * alpha + white_bg * (1 - alpha)
        plt.imshow(composite.astype(np.uint8))
    else:
        plt.imshow(bird_eye_view)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_bird_eye_view_torch(sample_token, cam_channel='CAM_FRONT',
                               bev_width=30, bev_length=30, resolution=0.05):
    """
    生成BEV (在PyTorch 1.13环境下, 深度冲突处理使用CPU循环)
    """
    # 1. 提取数据
    img, img_h, img_w, K, cs_record, ego_pose, _ = extract_data(sample_token, cam_channel)

    # 2. 坐标系变换矩阵
    cam_to_ego, ego_to_world, cam_to_world, world_to_cam = compute_coordinate_transforms(cs_record, ego_pose)

    # 3. 生成俯视图 (Torch)
    bird_eye_view, projection_mask, mapped_pixels = generate_bird_eye_view_torch(
        img, img_w, img_h, K, ego_to_world, world_to_cam,
        bev_width=bev_width, bev_length=bev_length, resolution=resolution
    )

    # 4. 后处理 (可自行关闭)
    processed_bev = post_process_bev_torch(bird_eye_view, projection_mask, mapped_pixels, resolution)

    return processed_bev, img


if __name__ == "__main__":
    my_sample = nusc.sample[0]

    print("=" * 40)
    print("PyTorch 1.13 + CUDA BEV Projection")
    print("=" * 40)

    bev_img, orig_img = create_bird_eye_view_torch(my_sample['token'], 'CAM_FRONT',
                                                   bev_width=30, bev_length=30,
                                                   resolution=0.05)
    visualize_bev(bev_img, orig_img, "PyTorch 1.13 BEV Projection")
