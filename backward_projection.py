import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

nusc = NuScenes(version='v1.0-mini', dataroot='v1.0-mini', verbose=True)
print(f'use GPU: {torch.cuda.is_available()}')

# This function is not use anymore
def create_bev_view_cpu_oldversion(sample_token='6402fd1ffaf041d0b9162bd92a7ba0a2', cam_channel='CAM_FRONT', bev_width=40, bev_length=40, resolution=0.04):
    """
    Create bev view
    
    Args:
    - sample_token: sample token
    - cam_channel: camera channel
    - bev_width: bev view width (meters)
    - bev_length: bev view length (meters)
    - resolution: resolution, the actual distance per pixel (meters/pixel)
    
    Returns:
    - bird_eye_view: bev view image
    - img: original image
    """
    # Read sample data
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    # Camera intrinsic matrix
    K = np.array(cs_record['camera_intrinsic'])
    img_path = nusc.get_sample_data_path(cam_token)
    print("="*40)
    
    # Read image
    print(f"Read image: {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img_height, img_width = img.shape[:2]
    
    # Camera
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_to_ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    cam_to_ego_translation = np.array(cs_record['translation'])
    
    # Camera coordinate to ego coordinate
    cam_to_ego = np.eye(4)
    cam_to_ego[:3, :3] = cam_to_ego_rotation
    cam_to_ego[:3, 3] = cam_to_ego_translation
    
    # Camera position in ego
    cam_pos_ego = cam_to_ego_translation
    
    # Size
    bev_length_pixels = int(bev_length / resolution)
    bev_width_pixels = int(bev_width / resolution)
    
    # Create transparent bev view and weight map
    bird_eye_view = np.zeros((bev_length_pixels, bev_width_pixels, 4), dtype=np.uint8)
    weight_map = np.zeros((bev_length_pixels, bev_width_pixels), dtype=np.float32)
    print(f"Image size: {img_width} x {img_height}")
    print(f"Top view size: {bev_width_pixels} x {bev_length_pixels}")
    
    # Vehicle center position
    ego_center_x = bev_width_pixels // 2
    ego_center_y = bev_length_pixels // 2
    
    base_step = 1
    mapped_pixels = 0

    # Iterate over all pixels
    for v in range(0, img_height, base_step):
        for u in range(0, img_width, base_step):
            # Normalization
            x_norm = (u - K[0, 2]) / K[0, 0]
            y_norm = (v - K[1, 2]) / K[1, 1]
            
            # Ray direction in camera coordinate system
            ray_camera = np.array([x_norm, y_norm, 1.0])
            
            # Convert ray direction to vehicle coordinate system
            ray_ego = cam_to_ego[:3, :3] @ ray_camera
            ray_ego = ray_ego / np.linalg.norm(ray_ego)
            
            # If ray is upward or almost parallel to ground, it will not intersect with ground
            if ray_ego[2] >= -0.1:
                continue
            
            # Calculate intersection of ray with ground (Z=0 plane) in vehicle coordinate system
            t = -cam_pos_ego[2] / ray_ego[2]
            ground_point_ego = cam_pos_ego + t * ray_ego
            
            # Extract x,y coordinates in vehicle coordinate system
            x_ego, y_ego = ground_point_ego[0], ground_point_ego[1]
            
            # Filter condition in vehicle coordinate system
            half_length = bev_length / 2
            half_width = bev_width / 2
            
            # Filter using vehicle-centered coordinates: ensure point is within BEV range
            if x_ego < -half_length or x_ego > half_length:
                continue
                
            if y_ego < -half_width or y_ego > half_width:
                continue
            
            # Calculate distance to vehicle center, for weight calculation
            distance = np.sqrt(x_ego**2 + y_ego**2)
            weight = np.exp(-distance / 20.0)
            
            # Map vehicle coordinate system point to BEV image coordinates (horizontal mirror)
            i = ego_center_y - int(x_ego / resolution)    # x_ego positive (forward) corresponds to upper part of image
            j = ego_center_x - int(y_ego / resolution)    # y_ego positive (left) corresponds to left half of image
            
            # Ensure coordinates are within image range
            if 0 <= i < bev_length_pixels and 0 <= j < bev_width_pixels:
                # Select pixel based on weight
                if weight > weight_map[i, j]:
                    bird_eye_view[i, j] = img[v, u]
                    bird_eye_view[i, j, 3] = 255
                    weight_map[i, j] = weight
                    mapped_pixels += 1
    
    print(f"\nSuccessfully mapped pixels: {mapped_pixels}")
    
    return bird_eye_view, img

def smooth_bev_image(image, cam_channel='CAM_FRONT', flag_blur=False):
    """
    Apply smoothing operations to the BEV image

    Args:
    - image: BEV image to smooth
    - resolution: resolution in meters/pixel

    Returns:
    - smoothed_image: processed image
    """
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological closing operation
    if cam_channel == 'CAM_BACK':
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Apply Gaussian blur
    if flag_blur:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return image

def visualize_bev_view(bev, original_img, flag_save=False, save_path='images/bev_only.png'):
    """
    Visualize the bev view transformation result
    
    Args:
    - sample_token: sample token
    - cam_channel: camera channel
    """
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Projection Image')
    plt.imshow(bev)
    plt.axis('off')
    
    plt.tight_layout()
    
    if flag_save:
        plt.savefig(save_path, transparent=False)
        
    plt.show()

def create_bev_view(sample_token='6402fd1ffaf041d0b9162bd92a7ba0a2', cam_channel='CAM_FRONT', bev_width=40, bev_length=40, resolution=0.04):
    """
    Use Torch acceleration to create a bird's-eye view
    
    Args:
    - sample_token: sample token
    - cam_channel: camera channel
    - bev_width: bev view width (meters)
    - bev_length: bev view length (meters)
    - resolution: resolution, the actual distance per pixel (meters/pixel)
    
    Returns:
    - bird_eye_view: bev view image
    - img: original image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initial variables
    # BEV image size
    bev_length_pixels = int(bev_length / resolution)
    bev_width_pixels = int(bev_width / resolution)
    # Bev image and weight map
    bird_eye_view = torch.zeros((bev_length_pixels, bev_width_pixels, 4), dtype=torch.uint8, device=device)
    weight_map = torch.zeros((bev_length_pixels, bev_width_pixels), dtype=torch.float32, device=device)
    # Get sample
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    # Camera intrinsic matrix
    K = np.array(cs_record['camera_intrinsic'])
    K_tensor = torch.from_numpy(K).float().to(device)
    img_path = nusc.get_sample_data_path(cam_token)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img_tensor = torch.from_numpy(img).to(device)
    img_height, img_width = img_tensor.shape[:2]
    # Camera extrinsic 
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_to_ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    cam_to_ego_translation = np.array(cs_record['translation'])
    
    # Camera coordinate to vehical coordinate
    cam_to_ego = np.eye(4)
    cam_to_ego[:3, :3] = cam_to_ego_rotation
    cam_to_ego[:3, 3] = cam_to_ego_translation
    cam_to_ego_tensor = torch.from_numpy(cam_to_ego).float().to(device)
    cam_pos_ego = cam_to_ego_translation
    cam_pos_ego_tensor = torch.from_numpy(cam_pos_ego).float().to(device)
    
    # Place the origin ego system at the center of the image
    ego_center_x = bev_width_pixels // 2
    ego_center_y = bev_length_pixels // 2
    
    # Create grid coordinates
    v_indices, u_indices = torch.meshgrid(
        torch.arange(0, img_height, device=device),
        torch.arange(0, img_width, device=device),
        indexing='ij'
    )
    v_indices = v_indices.flatten()
    u_indices = u_indices.flatten()
    
    # Normalization
    x_norm = (u_indices - K_tensor[0, 2]) / K_tensor[0, 0]
    y_norm = (v_indices - K_tensor[1, 2]) / K_tensor[1, 1]
    
    # Ray direction in camera coordinate system
    ray_camera = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=1)
    
    # Convert to ray direction in ego coordinate system
    ray_ego = torch.matmul(cam_to_ego_tensor[:3, :3], ray_camera.unsqueeze(-1)).squeeze(-1)
    ray_norm = torch.norm(ray_ego, dim=1, keepdim=True)
    ray_ego = ray_ego / ray_norm
    
    # Filter out rays pointing upward or parallel to the ground
    valid_rays = ray_ego[:, 2] < -0.1
    

    # Keep valid rays(pointing to the ground)
    valid_u = u_indices[valid_rays]
    valid_v = v_indices[valid_rays]
    valid_ray_ego = ray_ego[valid_rays]
    
    # Calculate the intersection of the ray with the ground(Z=0 plane)
    t = -cam_pos_ego_tensor[2] / valid_ray_ego[:, 2]
    ground_point_ego = cam_pos_ego_tensor.unsqueeze(0) + t.unsqueeze(1) * valid_ray_ego
    
    # Extract x,y coordinates in ego coordinate system
    x_ego, y_ego = ground_point_ego[:, 0], ground_point_ego[:, 1]
    
    # Filter condition in ego coordinate system
    half_length = bev_length / 2
    half_width = bev_width / 2
    
    # Ensure the point is within the BEV range
    in_range = (x_ego >= -half_length) & (x_ego <= half_length) & (y_ego >= -half_width) & (y_ego <= half_width)
    
    # Keep the points within the range
    x_ego = x_ego[in_range]
    y_ego = y_ego[in_range]
    valid_u = valid_u[in_range]
    valid_v = valid_v[in_range]
    
    # Calculate the distance to the vehicle center, for weight calculation
    distance = torch.sqrt(x_ego**2 + y_ego**2)
    weight = torch.exp(-distance / 20.0)
    
    # Map the points in ego coordinate system to the BEV image coordinates
    i = ego_center_y - (x_ego / resolution).long()  # x_ego positive (forward) corresponds to the upper part of the image
    j = ego_center_x - (y_ego / resolution).long()  # y_ego positive (left) corresponds to the left half of the image
    
    # Ensure the coordinates are within the image range
    valid_coords = (i >= 0) & (i < bev_length_pixels) & (j >= 0) & (j < bev_width_pixels)

    # Keep the valid coordinates
    i = i[valid_coords]
    j = j[valid_coords]
    weight = weight[valid_coords]
    valid_u = valid_u[valid_coords]
    valid_v = valid_v[valid_coords]
    
    # Create indices for atomic operations
    indices = i * bev_width_pixels + j
    
    # Get the current weight and image
    current_weights = weight_map.view(-1)[indices]
    
    # Find the pixels with larger weights
    better_pixels = weight > current_weights

    # Filter out the pixels with larger weights
    filtered_weight = weight[better_pixels]
    filtered_u = valid_u[better_pixels]
    filtered_v = valid_v[better_pixels]
    filtered_indices = indices[better_pixels]
    
    # Update the weights
    weight_map.view(-1)[filtered_indices] = filtered_weight
    
    # Get the pixel colors
    pixel_colors = img_tensor[filtered_v, filtered_u]
    
    # Update the BEV image
    bird_eye_view.view(-1, 4)[filtered_indices, :3] = pixel_colors[:, :3]
    bird_eye_view.view(-1, 4)[filtered_indices, 3] = 255

    bird_eye_view_np = bird_eye_view.cpu().numpy()
    
    return bird_eye_view_np, img

if __name__ == "__main__":
    my_sample = nusc.sample[65]
    bev, img = create_bev_view(sample_token=my_sample['token'])
    bev = smooth_bev_image(bev, cam_channel='CAM_FRONT')
    visualize_bev_view(bev, img)