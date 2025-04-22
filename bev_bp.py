import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
import concurrent.futures
from pyquaternion import Quaternion
from object_detection import run_detection_on_sample

nusc = NuScenes(version='v1.0-mini', dataroot='v1.0-mini', verbose=False)
camera_channels = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
        ]

def process_camera_worker(params):
    """
    Worker function to process a single camera's BEV projection and gather its info.
    Combines logic previously in get_camera_info and process_single_camera.
    
    Args:
    - params: dictionary containing processing parameters:
        - sample_token: sample token
        - cam_channel: camera channel
        - bev_width: BEV width (meters)
        - bev_length: BEV length (meters)
        - resolution: resolution (meters/pixel)
    
    Returns:
    - camera_data: dictionary containing camera information and processing results
    """
    sample_token = params['sample_token']
    cam_channel = params['cam_channel']
    bev_width = params['bev_width']
    bev_length = params['bev_length']
    resolution = params['resolution']

    # Initial sample data
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    intrinsic = np.array(cs_record['camera_intrinsic'])
    translation = np.array(cs_record['translation'])
    rotation_quat = Quaternion(cs_record['rotation'])
    rotation_matrix = rotation_quat.rotation_matrix
    camera_info = {
        'translation': translation,
        'rotation': rotation_matrix,
        'intrinsic': intrinsic
    }

    # Run detection and BEV projection
    bev, _, _, detection_points = run_detection_on_sample(
        sample_token=sample_token, 
        cam_channel=cam_channel, 
        bev_width=bev_width, 
        bev_length=bev_length, 
        resolution=resolution
    )

    # Prepare the result dictionary
    camera_data = {
        'cam_channel': cam_channel,
        'bev': bev,
        'camera_info': camera_info,
        'detection_points': detection_points
    }
    return camera_data

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
                'camera_info': result['camera_info'],
                'detection_points': result['detection_points']
            }
    
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
    
    return stitched_bev

def contour_based_stitch_bev_images(bev_results):
    """
    Stitch multiple camera BEV (bird‑eye‑view) images into a single RGBA frame.
    
    Args:
    - bev_results: dictionary containing all camera BEV results
    
    Returns:
    - stitched_bev: stitched BEV image
    """
    # Initialize variables
    first_cam = next(iter(bev_results))
    h, w = bev_results[first_cam]['bev'].shape[:2]
    stitched = np.zeros((h, w, 4), np.uint8) # final output
    weight_maps = {k: np.ones((h, w), np.float32) for k in bev_results} # per‑cam weights

    # Visibility mask per camera
    cam_masks = {}
    for cam, res in bev_results.items():
        bev = res['bev']
        if bev.shape[2] == 4: # RGBA supplied → use alpha
            cam_masks[cam] = (bev[:, :, 3] > 0).astype(np.uint8)
        else: # RGB supplied → fall back to provided mask or full‑frame
            cam_masks[cam] = res.get('mask', np.ones((h, w), bool)).astype(np.uint8)

    # Compute pair‑wise contour distance weights
    kernel = np.ones((5, 5), np.uint8) # small structuring element for closing
    eps = 1e-6 # avoids divide‑by‑zero at the contour
    cams = list(bev_results)

    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a, b = cams[i], cams[j]
            overlap = cam_masks[a] & cam_masks[b] # shared pixels
            if not np.any(overlap):
                continue # no overlap → skip

            # Extract pixels *unique* to each camera (needed to find contours)
            uniq_a = cv2.morphologyEx(cam_masks[a] - overlap, cv2.MORPH_CLOSE, kernel)
            uniq_b = cv2.morphologyEx(cam_masks[b] - overlap, cv2.MORPH_CLOSE, kernel)

            # Largest contour = boundary around exclusive region
            ca, _ = cv2.findContours(uniq_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cb, _ = cv2.findContours(uniq_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not ca or not cb:
                continue # one camera lacks exclusive area

            pa = max(ca, key=cv2.contourArea)
            pb = max(cb, key=cv2.contourArea)
            # Slight simplification for speed
            pa = cv2.approxPolyDP(pa, 0.01 * cv2.arcLength(pa, True), True)
            pb = cv2.approxPolyDP(pb, 0.01 * cv2.arcLength(pb, True), True)

            # Compute a weight for every overlapping pixel (could vectorise, kept simple for clarity)
            ys, xs = np.where(overlap)
            for y, x in zip(ys, xs):
                p = (float(x), float(y))
                d_a = max(abs(cv2.pointPolygonTest(pa, p, True)), eps)
                d_b = max(abs(cv2.pointPolygonTest(pb, p, True)), eps)

                w_a = d_b**2 / (d_a**2 + d_b**2) # camera a weight
                w_b = 1.0 - w_a # camera b weight

                # Keep the *smallest* weight encountered (pixel might be in more than one pair)
                weight_maps[a][y, x] = min(weight_maps[a][y, x], w_a)
                weight_maps[b][y, x] = min(weight_maps[b][y, x], w_b)

    # Accumulate weighted RGB & combined alpha
    rgb_acc = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), bool)

    for cam, res in bev_results.items():
        bev = res['bev']
        if bev.shape[2] == 3: # add full‑opacity alpha if absent
            bev = np.dstack([bev, np.full((h, w), 255, bev.dtype)])
        rgb_acc += bev[:, :, :3].astype(np.float32) * weight_maps[cam][:, :, None]
        alpha |= bev[:, :, 3] > 0

    # Compose final RGBA frame
    stitched[:, :, :3] = np.clip(rgb_acc, 0, 255).astype(np.uint8)
    stitched[:, :, 3] = alpha.astype(np.uint8) * 255
    return stitched

def brightness_balance(stitched_bev_rgba):
    """
    Soft brightness/contrast adjustment using CLAHE with mild parameters.

    Args:
        stitched_bev_rgba (np.ndarray): Input RGBA image

    Returns:
        np.ndarray: Brightness-balanced RGBA image
    """
    if stitched_bev_rgba is None or stitched_bev_rgba.shape[2] != 4:
        return stitched_bev_rgba

    # Separate RGB and Alpha
    rgb = stitched_bev_rgba[:, :, :3]
    alpha = stitched_bev_rgba[:, :, 3]

    # Convert RGB to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge and convert back to RGB
    limg = cv2.merge((cl, a, b))
    adjusted_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Combine with alpha
    adjusted_bev_rgba = np.dstack((adjusted_rgb, alpha))

    return adjusted_bev_rgba

def visualize_multicam_bev(bev_results, stitched_bev, flag_save=False, image_path='images/BEV.png'):
    """
    Visualize ONLY the final stitched multicamera BEV result.
    Includes detection results if available.
    
    Args:
    - bev_results: dictionary containing all camera BEV results (used for detections)
    - stitched_bev: stitched BEV image (required)
    - save_flag: whether to save image
    """
    # Set color map for detection results
    color_map = {
        'person': (255, 0, 0, 255),   # Red
        'car': (0, 255, 0, 255),      # Green
        'truck': (0, 0, 255, 255),    # Blue
        'bus': (255, 255, 0, 255),    # Yellow
        'motorcycle': (255, 0, 255, 255), # Magenta
        'bicycle': (0, 255, 255, 255),    # Cyan
    }

    # Prepare image to draw on (start with stitched)
    stitched_display = stitched_bev.copy()

    # Collect all detection results
    all_detections = []
    for _, result in bev_results.items():
        if 'detection_points' in result and result['detection_points']:
            all_detections.extend(result['detection_points'])

    # If detections exist, draw them and update title/filename
    show_legend = False
    if all_detections:
        title = 'Stitched BEV with Object Detection'
        show_legend = True
        for box in all_detections:
            x, y = box['bev_x'], box['bev_y']
            cls = box['class']
            color = color_map.get(cls)
            cv2.circle(stitched_display, (x, y), 5, color, -1)
            # Add class label
            label_text = f"{cls}"
            cv2.putText(stitched_display, label_text, (x + 7, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1, cv2.LINE_AA)
            # Add distance if available
            if 'distance' in box:
                distance = box['distance']
                dist_text = f"{distance:.1f}m"
                cv2.putText(stitched_display, dist_text, (x + 7, y + 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[:3], 1, cv2.LINE_AA)

    # Create the figure and plot the final image
    # plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(stitched_display)
    plt.axis('off')
    plt.axis('off')

    # Add legend if detections were shown
    if show_legend:
        legend_elements = []
        plotted_classes = set()
        # Create legend handles dynamically based on actual detections
        for box in all_detections:
             cls = box['class']
             if cls not in plotted_classes:
                 color = color_map.get(cls)
                 # Use plot for legend handle creation
                 handle, = plt.plot([], [], 'o', color=[c/255 for c in color[:3]], label=cls, markersize=5)
                 legend_elements.append(handle)
                 plotted_classes.add(cls)
        if legend_elements: # Only show legend if there are elements
             plt.legend(handles=legend_elements, loc='upper right')
             
    if flag_save:
        plt.savefig(image_path, transparent=False)
    plt.show()

def create_multicam_bev(sample_token, bev_width=40, bev_length=40, resolution=0.04, save_flag=False, fusion_strategy='contour_based', balance_brightness_flag=False):
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
    - balance_brightness_flag: Apply post-stitching brightness adjustment (default: False)
    
    Returns:
    - stitched_bev: stitched BEV image
    - bev_results: bev results of each camera
    """    
    bev_results = process_all_cameras_parallel(sample_token, bev_width, bev_length, resolution)
    
    if fusion_strategy == 'contour_based':
        stitched_bev = contour_based_stitch_bev_images(bev_results)
    elif fusion_strategy == 'position_based':
        stitched_bev = position_based_stitch_bev_images(bev_results)
        
    # Apply brightness adjustment AFTER stitching, if requested
    if balance_brightness_flag:
        print("Applying post-stitching brightness adjustment...")
        stitched_bev = brightness_balance(stitched_bev)

    # Corrected call: removed sample_token as it's not a parameter of visualize_multicam_bev
    visualize_multicam_bev(bev_results, stitched_bev, flag_save=save_flag)

    if save_flag:
        cv2.imwrite(f'stitched_bev_{fusion_strategy}_{sample_token[:8]}.png', 
                   cv2.cvtColor(stitched_bev, cv2.COLOR_RGBA2BGRA))
        print('Successfully saved stitched BEV image')
    
    return stitched_bev, bev_results

if __name__ == "__main__":
    print("=" * 40)
    my_sample = nusc.sample[159]
    
    stitched_bev, bev_results = create_multicam_bev(
        my_sample['token'], 
        bev_width=40,
        bev_length=40,
        resolution=0.04,
        save_flag=False,
        fusion_strategy='contour_based',
        balance_brightness_flag=True
    ) 