import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# Initialize nuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot='F:/Project/nuscenes-devkit/v1.0-mini', verbose=False)

def extract_data(sample_token, cam_channel='CAM_FRONT'):
    """
    Extract data from nuScenes dataset
    
    Args:
        sample_token: Sample token
        cam_channel: Camera channel
    
    Returns:
        Image data, dimensions, camera matrix, calibration record and ego pose
    """
    # Get sample data
    sample = nusc.get('sample', sample_token)
    
    # Get camera data
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    
    # Load image
    img_path = nusc.get_sample_data_path(cam_token)
    print(f"Read Image: {img_path}\n")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Image dimensions
    img_height, img_width = img.shape[:2]
    
    # Get calibration data (camera to ego transform)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    # Get ego pose data (ego to world transform)
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    
    # Camera intrinsic matrix
    K = np.array(cs_record['camera_intrinsic'])
    
    return img, img_height, img_width, K, cs_record, ego_pose, cam_data

def compute_coordinate_transforms(cs_record, ego_pose):
    """
    Compute coordinate transformation matrices
    
    Args:
        cs_record: Camera calibration record
        ego_pose: Ego vehicle pose record
    
    Returns:
        Camera-to-ego, ego-to-world, camera-to-world, and world-to-camera matrices
    """
    # Camera to ego transform
    cam_to_ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    cam_to_ego_translation = np.array(cs_record['translation'])
    
    # Build camera to ego transform matrix
    cam_to_ego = np.eye(4)
    cam_to_ego[:3, :3] = cam_to_ego_rotation
    cam_to_ego[:3, 3] = cam_to_ego_translation
    
    # Ego to world transform
    ego_to_world_rotation = Quaternion(ego_pose['rotation']).rotation_matrix
    ego_to_world_translation = np.array(ego_pose['translation'])
    
    # Build ego to world transform matrix
    ego_to_world = np.eye(4)
    ego_to_world[:3, :3] = ego_to_world_rotation
    ego_to_world[:3, 3] = ego_to_world_translation
    
    # Camera to world transform (cam -> ego -> world)
    cam_to_world = ego_to_world @ cam_to_ego
    
    # World to camera transform (inverse of camera to world)
    world_to_cam = np.linalg.inv(cam_to_world)
    
    return cam_to_ego, ego_to_world, cam_to_world, world_to_cam

def generate_bird_eye_view(img, img_width, img_height, K, ego_to_world, world_to_cam,
                           bev_width=40, bev_length=40, resolution=0.1):
    """
    Generate bird's eye view projection
    
    Args:
        img: Original image
        img_width, img_height: Image dimensions
        K: Camera intrinsic matrix
        ego_to_world: Ego to world transform matrix
        world_to_cam: World to camera transform matrix
        bev_width: BEV width in meters
        bev_length: BEV length in meters
        resolution: Resolution in meters/pixel
    
    Returns:
        BEV image, projection mask, and pixel count
    """
    # Set BEV parameters
    bev_height = int(bev_length / resolution)
    bev_width_pixels = int(bev_width / resolution)
    
    # Create empty image with alpha channel
    bird_eye_view = np.zeros((bev_height, bev_width_pixels, 4), dtype=np.uint8)
    
    # Mask to track projected pixels
    projection_mask = np.zeros((bev_height, bev_width_pixels), dtype=np.uint8)
    
    # Depth map for occlusion handling
    depth_map = np.full((bev_height, bev_width_pixels), np.inf)
    
    print(f"Image Size: {img_width} x {img_height}")
    print(f"Projection Size: {bev_width_pixels} x {bev_height}")
    
    # Forward projection method
    mapped_pixels = 0
    
    # Generate denser ground grid based on resolution
    ground_resolution = resolution * 0.5
    
    # Generate ground grid points in ego-centered coordinates
    for x_ego in np.arange(-bev_length/2, bev_length/2, ground_resolution):
        for y_ego in np.arange(-bev_width/2, bev_width/2, ground_resolution):
            # Ground point height set to 0 (flat ground)
            ground_point_ego = np.array([x_ego, y_ego, 0, 1])
            
            # Ego to world coordinates
            ground_point_world = ego_to_world @ ground_point_ego
            
            # World to camera coordinates
            ground_point_cam = world_to_cam @ ground_point_world
            
            # Skip if point is behind camera
            if ground_point_cam[2] <= 0:
                continue
            
            # Project to image plane
            x_cam = ground_point_cam[0]
            y_cam = ground_point_cam[1]
            z_cam = ground_point_cam[2]
            
            u = K[0, 0] * (x_cam / z_cam) + K[0, 2]
            v = K[1, 1] * (y_cam / z_cam) + K[1, 2]
            
            # Check if projection is within image bounds
            if 0 <= u < img_width and 0 <= v < img_height:
                # Calculate BEV coordinates
                i = int((bev_length/2 - x_ego) / resolution)  # Fix front-back flip
                j = int((bev_width/2 - y_ego) / resolution)   # Fix mirroring: flip y-axis mapping
                
                # Ensure coordinates are within BEV image bounds
                if 0 <= i < bev_height and 0 <= j < bev_width_pixels:
                    # Handle occlusion with depth
                    if z_cam < depth_map[i, j]:
                        # Sample from original image and map to BEV
                        pixel_color = img[int(v), int(u)]
                        bird_eye_view[i, j, 0:3] = pixel_color
                        bird_eye_view[i, j, 3] = 255  # Set alpha to opaque
                        depth_map[i, j] = z_cam
                        projection_mask[i, j] = 1
                        mapped_pixels += 1
    
    print(f"\nSuccessfully mapped pixels: {mapped_pixels}")
    
    return bird_eye_view, projection_mask, mapped_pixels

def post_process_bev(bird_eye_view, projection_mask, mapped_pixels, resolution=0.1):
    """
    Post-process BEV image to improve visual quality
    
    Args:
        bird_eye_view: BEV image
        projection_mask: Projection mask
        mapped_pixels: Number of mapped pixels
        resolution: Resolution in meters/pixel
    
    Returns:
        Processed BEV image
    """
    if mapped_pixels == 0:
        return bird_eye_view
    
    # Create adaptive kernel size
    kernel_size = max(3, int(5 * resolution / 0.1))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological operations to projected area
    processed_mask = cv2.dilate(projection_mask, kernel, iterations=1)
    
    # Apply Gaussian blur to smooth image
    blur_size = max(3, int(3 * resolution / 0.1))
    if blur_size % 2 == 0:
        blur_size += 1
    
    # Separate RGB and alpha channels
    rgb = bird_eye_view[:, :, 0:3]
    alpha = bird_eye_view[:, :, 3]
    
    # Apply blur only to projected area
    mask_3ch = np.stack([processed_mask, processed_mask, processed_mask], axis=2)
    blurred_rgb = cv2.GaussianBlur(rgb, (blur_size, blur_size), 0)
    
    # Apply blurred image only to projected area
    processed_rgb = np.where(mask_3ch > 0, blurred_rgb, rgb)
    
    # Update alpha channel - extend alpha values at edges
    blurred_alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
    processed_alpha = np.where(processed_mask > 0, blurred_alpha, alpha)
    
    # Recombine RGB and alpha channels
    processed_bev = np.zeros_like(bird_eye_view)
    processed_bev[:, :, 0:3] = processed_rgb
    processed_bev[:, :, 3] = processed_alpha
    
    return processed_bev

def visualize_bev(bird_eye_view, original_img, title="Modular BEV Projection"):
    """
    Visualize BEV projection results
    
    Args:
        bird_eye_view: BEV image
        original_img: Original image
        title: Image title
    """
    # Visualization
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(title)
    # Display with transparency
    if bird_eye_view.shape[2] == 4:  # If has alpha channel
        # Create white background
        white_bg = np.ones((bird_eye_view.shape[0], bird_eye_view.shape[1], 3), dtype=np.uint8) * 255
        # Normalize alpha channel
        alpha = bird_eye_view[:, :, 3].astype(float) / 255
        alpha = np.expand_dims(alpha, axis=2)
        # Blend RGB and background
        rgb = bird_eye_view[:, :, 0:3]
        composite = rgb * alpha + white_bg * (1 - alpha)
        plt.imshow(composite.astype(np.uint8))
    else:
        plt.imshow(bird_eye_view)
    plt.axis('off')
    
    plt.tight_layout()
    # plt.savefig(f'{title.lower().replace(" ", "_")}_result.png')
    plt.show()
    
    # Save separate BEV for easy viewing
    plt.figure(figsize=(10, 10))
    plt.title(title)
    if bird_eye_view.shape[2] == 4:  # If has alpha channel
        # Display with transparent background
        plt.imshow(bird_eye_view[:, :, 0:3])
        plt.imshow(bird_eye_view[:, :, 3], alpha=bird_eye_view[:, :, 3]/255)
    else:
        plt.imshow(bird_eye_view)
    plt.axis('off')
    # plt.savefig(f'{title.lower().replace(" ", "_")}_only.png', transparent=True)

def save_transparent_bev(bird_eye_view, filename):
    """
    Save BEV image with transparent background
    
    Args:
        bird_eye_view: BEV image with alpha channel
        filename: Output filename
    """
    # Ensure image has 4 channels (RGBA)
    if bird_eye_view.shape[2] != 4:
        raise ValueError("Image must have alpha channel to save as transparent PNG")
    
    # Save as PNG (supports transparency)
    cv2.imwrite(filename, cv2.cvtColor(bird_eye_view, cv2.COLOR_RGBA2BGRA))

def create_bird_eye_view(sample_token, cam_channel='CAM_FRONT', 
                        bev_width=30, bev_length=30, resolution=0.05):
    """
    Create bird's eye view using modular approach
    
    Args:
        sample_token: Sample token
        cam_channel: Camera channel
        bev_width: BEV width in meters
        bev_length: BEV length in meters
        resolution: Resolution in meters/pixel
    
    Returns:
        Processed BEV image and original image
    """
    # 1. Extract data
    img, img_height, img_width, K, cs_record, ego_pose, _ = extract_data(sample_token, cam_channel)
    
    # 2. Compute transformation matrices
    cam_to_ego, ego_to_world, cam_to_world, world_to_cam = compute_coordinate_transforms(cs_record, ego_pose)
    
    # 3. Generate BEV
    bird_eye_view, projection_mask, mapped_pixels = generate_bird_eye_view(
        img, img_width, img_height, K, ego_to_world, world_to_cam,
        bev_width, bev_length, resolution
    )
    
    # 4. Post-process
    processed_bev = post_process_bev(bird_eye_view, projection_mask, mapped_pixels, resolution)
    
    return processed_bev, img

if __name__ == "__main__":
    # Use sample #10
    my_sample = nusc.sample[10]
    
    print("=" * 40)
    print("Modular BEV Projection")
    print("=" * 40)
    
    # Create and visualize BEV
    bev_img, orig_img = create_bird_eye_view(my_sample['token'], 'CAM_FRONT')
    visualize_bev(bev_img, orig_img, "Modular BEV Projection") 