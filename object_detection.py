import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
import backward_projection as bp
from pyquaternion import Quaternion
import os

nusc = NuScenes(version='v1.0-mini', dataroot='v1.0-mini', verbose=False)

def project_detections_to_bev(results, K, cam_to_ego, bev_width_pixels=1000, bev_length_pixels=1000, resolution=0.04):
    """
    Project detections from image coordinates to BEV coordinates
    
    Args:
    - results: YOLOv8 detection results
    - K: Camera intrinsic matrix
    - cam_to_ego: Camera to ego coordinate transformation matrix
    - img_dims: Image dimensions (width, height)
    - bev_width_pixels: BEV width in pixels
    - bev_length_pixels: BEV length in pixels
    - resolution: Resolution (meters/pixel)
    
    Returns:
    - detection_points: Detection points in BEV
    """   
    # BEV center point
    ego_center_x = bev_width_pixels // 2
    ego_center_y = bev_length_pixels // 2
    # Store projected detection results
    detection_points = []
    
    boxes = results[0].boxes
    for _, box in enumerate(boxes):
        # Get bottom center point of bounding box
        x1, _, x2, y2 = box.xyxy[0].cpu().numpy()
        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2
        
        # Normalize
        x_norm = (bottom_center_x - K[0, 2]) / K[0, 0]
        y_norm = (bottom_center_y - K[1, 2]) / K[1, 1]
        
        # Ray direction in camera coordinate system
        ray_camera = np.array([x_norm, y_norm, 1.0])
        
        # Convert to ego coordinate system
        ray_ego = cam_to_ego[:3, :3] @ ray_camera
        ray_ego = ray_ego / np.linalg.norm(ray_ego)
        
        # Filter out rays pointing upward or parallel to the ground
        if ray_ego[2] >= -0.1:
            continue
        
        # Calculate intersection with ground (Z=0 plane)
        cam_pos_ego = cam_to_ego[:3, 3]
        t = -cam_pos_ego[2] / ray_ego[2]
        ground_point_ego = cam_pos_ego + t * ray_ego

        # Extract x,y coordinates in ego coordinate system
        x_ego, y_ego = ground_point_ego[0], ground_point_ego[1]
        
        # Convert to BEV coordinates
        bev_x = ego_center_x - int(y_ego / resolution)
        bev_y = ego_center_y - int(x_ego / resolution)
        
        # Make sure the projected point is within the BEV range
        if 0 <= bev_x < bev_width_pixels and 0 <= bev_y < bev_length_pixels:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = results[0].names[cls_id]
            
            detection_points.append({
                'bev_x': bev_x,
                'bev_y': bev_y,
                'class': cls_name,
                'confidence': conf
            })
    
    return detection_points

def visualize_bev_with_detections(bev_img, detection_img, projected_boxes, flag_save=False, save_path='images/bev_with_detections.png'):
    """
    Visualize the BEV image with detection results
    
    Args:
    - bev_img: BEV image
    - original_img: Original image
    - detection_img: Image with detection boxes
    - projected_boxes: Projected detection boxes to BEV
    """
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('Detection result')
    plt.imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('BEV with Detection result')
    bev_with_detections = bev_img.copy()

    color_map = {
        'person': (255, 0, 0, 255),   # Red
        'car': (0, 255, 0, 255),      # Green
        'truck': (0, 0, 255, 255),    # Blue
        'bus': (255, 255, 0, 255),    # Yellow
        'motorcycle': (255, 0, 255, 255), # Magenta
        'bicycle': (0, 255, 255, 255),    # Cyan
    }
    
    # Detection boxes
    for box in projected_boxes:
        x, y = box['bev_x'], box['bev_y']
        cls = box['class']
        
        # Select color based on class
        color = color_map.get(cls)
        
        # Draw circle marker on BEV image
        cv2.circle(bev_with_detections, (x, y), 7, color, -1)
        
        # Add class label
        cv2.putText(bev_with_detections, cls, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1, cv2.LINE_AA)
    
    plt.imshow(bev_with_detections)
    plt.axis('off')
    if flag_save:
        plt.savefig(save_path, transparent=False)
    plt.show()

def run_detection_on_sample(sample_token, cam_channel='CAM_FRONT', bev_width=40, bev_length=40, resolution=0.04, conf_thresh=0.3):
    """
    Run detection on a specified sample and generate BEV view
    
    Args:
    - sample_token: Sample token
    - cam_channel: Camera channel
    - bev_width: BEV view width (meters)
    - bev_length: BEV view length (meters)
    - resolution: Resolution (meters/pixel)
    - conf_thresh: Detection confidence threshold
    
    Returns:
    - bev_img: BEV image
    - original_img: Original image
    - detection_img: Image with detection boxes
    - detection_points: Detection points in BEV
    """
    
    # Get sample
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    
    # Get camera parameters
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    K = np.array(cs_record['camera_intrinsic'])
    cam_to_ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    cam_to_ego_translation = np.array(cs_record['translation'])
    
    # Camera to ego coordinate transformation
    cam_to_ego = np.eye(4)
    cam_to_ego[:3, :3] = cam_to_ego_rotation
    cam_to_ego[:3, 3] = cam_to_ego_translation
    
    # Read image
    img_path = nusc.get_sample_data_path(cam_token)
    original_img = cv2.imread(img_path)
    
    # Get image size
    bev_length_pixels = int(bev_length / resolution)
    bev_width_pixels = int(bev_width / resolution)
    
    # Get BEV image
    bev_img, _ = bp.create_bev_view(sample_token, cam_channel, bev_width, bev_length, resolution)
    # Smooth BEV image
    bev_img = bp.smooth_bev_image(bev_img, cam_channel)
    
    # Detection
    model = YOLO('models/yolov8n.pt')
    results = model(original_img, conf=conf_thresh, classes=[0, 2, 3, 5, 7]) # person, car, motorcycle, bus, truck
    detection_img = results[0].plot() 
    
    # Project detection results to BEV view
    detection_points = project_detections_to_bev(results, K, cam_to_ego, bev_width_pixels, bev_length_pixels, resolution)
    
    return bev_img, original_img, detection_img, detection_points

def save_detection_results(bev_img, detection_img, detection_points, output_dir="output"):
    """
    Save detection results  
    
    Args:
    - bev_img: BEV image
    - detection_img: Image with detection boxes
    - detection_points: Detection points in BEV
    - output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detection image
    cv2.imwrite(os.path.join(output_dir, "detection.png"), detection_img)
    
    # Create BEV image with detection boxes
    bev_with_detections = bev_img.copy()
    
    # Color mapping
    color_map = {
        'person': (255, 0, 0, 255),
        'car': (0, 255, 0, 255),
        'truck': (0, 0, 255, 255),
        'bus': (255, 255, 0, 255),
        'motorcycle': (255, 0, 255, 255),
        'bicycle': (0, 255, 255, 255),
    }
    
    default_color = (128, 128, 128, 255)
    
    # Draw detection boxes
    for box in detection_points:
        x, y = box['bev_x'], box['bev_y']
        size = box['size']
        cls = box['class']
        
        color = color_map.get(cls, default_color)
        cv2.circle(bev_with_detections, (x, y), size, color, -1)
        cv2.putText(bev_with_detections, cls, (x, y - size - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1, cv2.LINE_AA)
    
    # Save BEV image with detection boxes
    cv2.imwrite(os.path.join(output_dir, "bev_with_detections.png"), cv2.cvtColor(bev_with_detections, cv2.COLOR_RGBA2BGRA))
    
    # Save original BEV image
    cv2.imwrite(os.path.join(output_dir, "bev.png"), cv2.cvtColor(bev_img, cv2.COLOR_RGBA2BGRA))
    
    print(f"Detection results saved to {output_dir} directory")

if __name__ == "__main__":
    sample_token = nusc.sample[100]['token']
    
    bev_img, original_img, detection_img, detection_points = run_detection_on_sample(
        sample_token=sample_token,
        cam_channel='CAM_FRONT',
        conf_thresh=0.3
    )

    visualize_bev_with_detections(bev_img, detection_img, detection_points)
