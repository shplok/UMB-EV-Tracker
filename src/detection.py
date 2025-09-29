"""
Particle Detection Module for EV Detection Pipeline

This module detects extracellular vesicles (EVs) in enhanced microscopy frames
using template matching with specialized filters. It identifies particle
locations and scores their detection confidence.

Functions:
- detect_particles_in_all_frames(): Main detection function
- visualize_detection_results(): Create detection visualizations
- analyze_detection_quality(): Basic detection assessment
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Any


def detect_particles_in_all_frames(enhanced_frames: np.ndarray, 
                                 ev_filter: np.ndarray,
                                 threshold: float = 0.35, # confidence threshold for detection (tweakable)
                                 min_distance: int = 30) -> Dict[int, Dict[str, List]]:
    """
    Detect particles in all enhanced frames using the specialized EV filter
    
    Args:
        enhanced_frames (np.ndarray): Enhanced image frames
        ev_filter (np.ndarray): EV detection filter
        threshold (float): Minimum correlation score for detection
        min_distance (int): Minimum distance between detections (non-max suppression)
        
    Returns:
        Dict[int, Dict[str, List]]: Dictionary with frame indices as keys and 
                                   lists of particle positions and scores as values
    """
    print("Detecting particles in all frames...")
    num_frames = len(enhanced_frames)
    all_particles = {}
    
    total_detections = 0
    
    for i in tqdm(range(num_frames), desc="Detecting particles"):
        # Prepare frame for template matching
        frame_norm = cv2.normalize(enhanced_frames[i], None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        # Apply filter to detect particles using normalized cross-correlation
        correlation = cv2.matchTemplate(frame_norm, ev_filter, cv2.TM_CCORR_NORMED)
        
        # Find local maxima (non-maximum suppression)
        data_max = ndimage.maximum_filter(correlation, size=min_distance)
        maxima = (correlation == data_max) & (correlation > threshold)
        
        # Get particle coordinates
        y_peaks, x_peaks = np.where(maxima)
        filter_size = ev_filter.shape[0]
        half_size = filter_size // 2
        
        # Adjust coordinates to account for filter size
        centers = [(x + half_size, y + half_size) for x, y in zip(x_peaks, y_peaks)]
        
        # Get correlation scores
        scores = [correlation[y, x] for y, x in zip(y_peaks, x_peaks)]
        
        # Store particles for this frame
        all_particles[i] = {
            'positions': centers,
            'scores': scores
        }
        
        total_detections += len(centers)
    
    print(f"Detection complete: {total_detections} total detections across {num_frames} frames")
    print(f"Average detections per frame: {total_detections/num_frames:.1f}")
    
    return all_particles


def visualize_detection_results(enhanced_frames: np.ndarray,
                              all_particles: Dict[int, Dict[str, List]],
                              output_dir: str,
                              num_samples: int = 6) -> List[str]:
    """
    Create visualizations showing particle detection results
    
    Args:
        enhanced_frames (np.ndarray): Enhanced frames
        all_particles (Dict): Detection results
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of sample frames to visualize
        
    Returns:
        List[str]: Paths to created visualization files
    """
    print("Creating detection result visualizations...")
    
    num_frames = len(enhanced_frames)
    sample_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    
    # Create detection overlay figure
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, frame_idx in enumerate(sample_indices):
        # Original enhanced frame
        axes[0, i].imshow(enhanced_frames[frame_idx], cmap='gray')
        axes[0, i].set_title(f'Enhanced Frame {frame_idx}')
        axes[0, i].axis('off')
        
        # Frame with detection overlays
        frame_with_detections = cv2.cvtColor(enhanced_frames[frame_idx], cv2.COLOR_GRAY2RGB)
        
        if frame_idx in all_particles:
            positions = all_particles[frame_idx]['positions']
            scores = all_particles[frame_idx]['scores']
            
            for pos, score in zip(positions, scores):
                x, y = int(pos[0]), int(pos[1])
                
                # Color code by detection confidence
                if score > 0.6:
                    color = (255, 0, 0)  # Red for high confidence
                elif score > 0.4:
                    color = (255, 255, 0)  # Yellow for medium confidence
                else:
                    color = (0, 255, 0)  # Green for low confidence
                
                # Draw circle and score
                cv2.circle(frame_with_detections, (x, y), 12, color, 2)
                cv2.putText(frame_with_detections, f'{score:.2f}', (x+15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        axes[1, i].imshow(frame_with_detections)
        axes[1, i].set_title(f'Detections Frame {frame_idx}\n({len(all_particles.get(frame_idx, {}).get("positions", []))} particles)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, "detection_overlays.png")
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detection video
    video_path = create_detection_video(enhanced_frames, all_particles, output_dir)
    
    print(f"Detection visualizations saved:")
    print(f"  - Overlays: {overlay_path}")
    print(f"  - Video: {video_path}")
    
    return [overlay_path, video_path]


def create_detection_video(enhanced_frames: np.ndarray,
                         all_particles: Dict[int, Dict[str, List]],
                         output_dir: str,
                         fps: int = 5,
                         show_scores: bool = True) -> str:
    """
    Create a video showing particle detections across all frames
    
    Args:
        enhanced_frames (np.ndarray): Enhanced frames
        all_particles (Dict): Detection results
        output_dir (str): Output directory
        fps (int): Frames per second for output video
        show_scores (bool): Whether to display detection scores
        
    Returns:
        str: Path to the created video file
    """
    print("Creating detection video...")
    
    # Get dimensions
    height, width = enhanced_frames[0].shape
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, "detection_video.mp4")
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx in tqdm(range(len(enhanced_frames)), desc="Creating video"):
        # Get current frame and convert to RGB
        frame = enhanced_frames[frame_idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Add detections if they exist for this frame
        if frame_idx in all_particles:
            positions = all_particles[frame_idx]['positions']
            scores = all_particles[frame_idx]['scores']
            
            for pos, score in zip(positions, scores):
                x, y = int(pos[0]), int(pos[1])
                
                # Color code by detection confidence
                if score > 0.6:
                    color = (255, 0, 0)  # Red for high confidence
                elif score > 0.4:
                    color = (255, 255, 0)  # Yellow for medium confidence
                else:
                    color = (0, 255, 0)  # Green for low confidence
                
                # Draw circle around detection
                cv2.circle(frame_rgb, (x, y), 12, color, 2)
                
                # Optionally show scores
                if show_scores:
                    cv2.putText(frame_rgb, f'{score:.2f}', (x+15, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add frame number and detection count
        detection_count = len(all_particles.get(frame_idx, {}).get('positions', []))
        cv2.putText(frame_rgb, f"Frame: {frame_idx}, Detections: {detection_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write frame to video
        video.write(frame_rgb)
    
    # Release the video
    video.release()
    print(f"Detection video saved to: {video_path}")
    
    return video_path


def analyze_detection_quality(all_particles: Dict[int, Dict[str, List]],
                            enhanced_frames: np.ndarray,
                            detection_params: Dict[str, Any],
                            output_dir: str) -> str:
    """
    Basic analysis of detection quality and characteristics
    
    Args:
        all_particles (Dict): Detection results
        enhanced_frames (np.ndarray): Enhanced frames
        detection_params (Dict): Detection parameters used
        output_dir (str): Output directory
        
    Returns:
        str: Path to the quality analysis file
    """
    # Collect basic detection statistics
    all_scores = []
    detections_per_frame = []
    
    for frame_idx, frame_data in all_particles.items():
        scores = frame_data['scores']
        positions = frame_data['positions']
        
        all_scores.extend(scores)
        detections_per_frame.append(len(positions))
    
    total_detections = len(all_scores)
    total_frames = len(enhanced_frames)
    
    if total_detections == 0:
        print("Warning: No particles detected. Consider lowering threshold or adjusting filter parameters.")
        return ""
    
    avg_detections_per_frame = total_detections / total_frames
    avg_score = np.mean(all_scores)
    high_confidence_detections = sum(1 for s in all_scores if s > 0.6)
    high_confidence_ratio = high_confidence_detections / total_detections
    
    print(f"Detection Quality Summary:")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Average detections per frame: {avg_detections_per_frame:.2f}")
    print(f"  - Average detection score: {avg_score:.3f}")
    print(f"  - High confidence ratio: {high_confidence_ratio:.2f}")
    
    return output_dir
