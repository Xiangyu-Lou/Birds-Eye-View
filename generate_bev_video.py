import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from bev_bp import create_multicam_bev
import matplotlib
# Set Matplotlib to use non-interactive backend
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt

def generate_bev_video(nusc, output_path, num_frames=6, start_frame=0, fps=2, fusion_strategy='contour_based', include_detections=False):
    """
    Generate BEV video from nuScenes samples with object detection
    
    Args:
    - nusc: NuScenes instance
    - output_path: output video path
    - num_frames: number of frames to process (None = all)
    - start_frame: index of the first frame to process
    - fps: frames per second in output video
    - fusion_strategy: fusion strategy ('contour_based' or 'position_based')
    - include_detections: whether to include object detection markers (default: False)
    """
    # Create temp directory for frames
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get samples
    samples = nusc.sample
    
    # Apply start_frame index
    if start_frame > 0:
        if start_frame >= len(samples):
            print(f"Warning: start_frame ({start_frame}) exceeds number of available samples ({len(samples)})")
            start_frame = 0
        samples = samples[start_frame:]
        print(f"Starting from frame index {start_frame}")
    
    # If num_frames is specified, limit the number of samples
    if num_frames is not None:
        samples = samples[:min(num_frames, len(samples))]
    
    print(f"Processing {len(samples)} samples (from index {start_frame} to {start_frame + len(samples) - 1})...")
    
    # Color map for detection results
    color_map = {
        'person': (255, 0, 0, 255),   # Red
        'car': (0, 255, 0, 255),      # Green
        'truck': (0, 0, 255, 255),    # Blue
        'bus': (255, 255, 0, 255),    # Yellow
        'motorcycle': (255, 0, 255, 255), # Magenta
        'bicycle': (0, 255, 255, 255),    # Cyan
    }
    
    # Create legend image before processing to avoid threading issues
    legend_frame_path = os.path.join(temp_dir, "legend.png")
    create_legend_image(legend_frame_path, color_map)
    
    # Process each sample
    for i, sample in enumerate(tqdm(samples)):
        # Get sample token
        sample_token = sample['token']
        
        try:
            # Generate BEV (detection results are always computed by create_multicam_bev)
            stitched_bev, bev_results = create_multicam_bev(
                sample_token=sample_token,
                fusion_strategy=fusion_strategy
            )
            
            # Decide which frame to save based on the include_detections flag
            if include_detections:
                # Create a copy of the stitched BEV for adding detection markers
                stitched_with_detections = stitched_bev.copy()
                all_detections = []
                
                # Collect all detection results
                for cam_channel, result in bev_results.items():
                    if 'detection_points' in result and result['detection_points']:
                        all_detections.extend(result['detection_points'])
                
                # Draw detection markers on the BEV image if detections exist
                if all_detections:
                    for box in all_detections:
                        x, y = box['bev_x'], box['bev_y']
                        cls = box['class']
                        
                        # Choose color based on class
                        color = color_map.get(cls)
                        
                        # Draw circular marker with a fixed radius
                        marker_radius = 5
                        cv2.circle(stitched_with_detections, (x, y), marker_radius, color, -1)
                        
                        # Add class label and distance
                        label_text = f"{cls}"
                        cv2.putText(stitched_with_detections, label_text, (x + marker_radius + 2, y - marker_radius), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1, cv2.LINE_AA)
                        if 'distance' in box:
                            distance = box['distance']
                            dist_text = f"{distance:.1f}m"
                            # Position distance text below the class label
                            cv2.putText(stitched_with_detections, dist_text, (x + marker_radius + 2, y + 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[:3], 1, cv2.LINE_AA)
                
                    # Use the BEV with detections for the frame
                    frame_to_save = stitched_with_detections
                else:
                    # If include_detections is True but no detections found, use original BEV
                    frame_to_save = stitched_bev
            else:
                # If include_detections is False, always use the original BEV
                frame_to_save = stitched_bev
                
            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            
            # Check if image is RGBA and convert if needed
            if frame_to_save.shape[2] == 4:
                # For video, convert RGBA to RGB
                # First create a white background
                background = np.ones((frame_to_save.shape[0], frame_to_save.shape[1], 3), dtype=np.uint8) * 255
                
                # Extract RGB channels from the BEV
                rgb = frame_to_save[:, :, :3]
                
                # Extract alpha channel
                alpha = frame_to_save[:, :, 3:4] / 255.0
                
                # Blend the BEV RGB with white background using alpha
                blended = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
                
                cv2.imwrite(frame_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(frame_path, cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR))
                
            print(f"Saved frame {i+1}/{len(samples)} (original index: {start_frame + i})")
                
        except Exception as e:
            print(f"Error processing sample {i} (original index: {start_frame + i}): {e}")
    
    # Generate video from frames
    print("Generating video...")
    frame_path = os.path.join(temp_dir, "frame_%04d.png")
    
    # Get the first frame to determine video size
    first_frame = cv2.imread(os.path.join(temp_dir, "frame_0000.png"))
    if first_frame is None:
        print("No frames were generated successfully. Cannot create video.")
        return
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    # Use 'mp4v' for H.264 encoding in an MP4 container
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for i in range(len(samples)):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to {output_path}")

def create_legend_image(output_path, color_map):
    """
    Create legend image
    
    Args:
    - output_path: output path
    - color_map: color map
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.title("BEV with Object Detection - Legend")
        # Add legend items
        for cls, color in color_map.items():
            plt.plot([], [], 'o', color=[c/255 for c in color[:3]], markersize=15, label=cls)
        plt.legend(loc='center', fontsize=12)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        print("Saved legend frame")
    except Exception as e:
        print(f"Error creating legend image: {e}")

def main():
    """
    Main function to parse arguments and generate BEV video
    """
    parser = argparse.ArgumentParser(description='Generate BEV video from nuScenes samples with object detection')
    parser.add_argument('--output', type=str, default='bev_video.mp4', help='Output video path (MP4 format)')
    parser.add_argument('--frames', type=int, default=120, help='Number of frames to process')
    parser.add_argument('--start_frame', type=int, default=40, help='Index of the first frame to process')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second in output video')
    parser.add_argument('--dataroot', type=str, default='F:/Project/Birds-Eye-View/v1.0-mini', help='NuScenes data root')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='NuScenes version')
    parser.add_argument('--fusion', type=str, default='contour_based', choices=['contour_based', 'position_based'], 
                        help='Fusion strategy for BEV generation')
    parser.add_argument('--wd', '--with_detection', action='store_true', 
                        help='Include object detection markers in the video')
    
    args = parser.parse_args()
    
    # Initialize nuScenes
    print(f"Loading NuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    # Generate BEV video
    generate_bev_video(
        nusc=nusc,
        output_path=args.output,
        num_frames=args.frames,
        start_frame=args.start_frame,
        fps=args.fps,
        fusion_strategy=args.fusion,
        include_detections=args.wd
    )

if __name__ == "__main__":
    main() 