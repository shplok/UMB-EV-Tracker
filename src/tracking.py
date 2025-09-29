"""
Particle Tracking Module for EV Detection Pipeline

This module links particle detections across frames to create trajectories,
calculates motion properties, and provides comprehensive tracking analysis.

Functions:
- track_particles_across_frames(): Link detections into tracks
- calculate_track_properties(): Compute motion and intensity properties
- visualize_tracking_results(): Create track visualizations
- analyze_tracking_quality(): Basic tracking assessment
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Any


def track_particles_across_frames(all_particles: Dict[int, Dict[str, List]],
                                max_distance: int = 25,
                                min_track_length: int = 5,
                                max_frame_gap: int = 3) -> Dict[int, Dict[str, Any]]:
    """
    Track particles across frames by connecting detections
    
    Args:
        all_particles (Dict): Particle detections from detect_particles_in_all_frames
        max_distance (int): Maximum distance a particle can move between frames
        min_track_length (int): Minimum number of frames for a valid track
        max_frame_gap (int): Maximum frames a track can be missing before termination
        
    Returns:
        Dict[int, Dict[str, Any]]: Dictionary of particle tracks with properties
    """
    print("Tracking particles across frames...")
    num_frames = max(all_particles.keys()) + 1 if all_particles else 0
    
    # Initialize tracks
    tracks = {}
    next_track_id = 1
    active_tracks = {}  # Currently active tracks
    
    # Process frames in order
    for frame_idx in tqdm(range(num_frames), desc="Tracking particles"):
        # Get particles detected in current frame
        if frame_idx not in all_particles or not all_particles[frame_idx]['positions']:
            # No detections in this frame, check for track terminations
            tracks_to_remove = []
            for track_id in active_tracks:
                track = tracks[track_id]
                last_frame = track['frames'][-1]
                if frame_idx - last_frame > max_frame_gap:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                active_tracks.pop(track_id)
            continue
            
        current_particles = all_particles[frame_idx]['positions']
        current_scores = all_particles[frame_idx]['scores']
        
        # If we have active tracks, try to extend them
        if active_tracks:
            assigned_detections = set()
            tracks_to_remove = []
            
            # For each active track, find the closest detection
            for track_id in list(active_tracks.keys()):
                track = tracks[track_id]
                last_pos = track['positions'][-1]
                last_frame = track['frames'][-1]
                
                # Skip if track already updated in this frame
                if last_frame == frame_idx:
                    continue
                    
                # Skip if track has been inactive for too long
                if frame_idx - last_frame > max_frame_gap:
                    tracks_to_remove.append(track_id)
                    continue
                
                # Find closest detection
                best_match = None
                best_distance = float('inf')
                
                for i, pos in enumerate(current_particles):
                    if i in assigned_detections:
                        continue
                        
                    # Calculate distance
                    distance = np.sqrt((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)
                    
                    if distance < max_distance and distance < best_distance:
                        best_match = i
                        best_distance = distance
                
                # If we found a match, extend the track
                if best_match is not None:
                    pos = current_particles[best_match]
                    score = current_scores[best_match]
                    
                    track['positions'].append(pos)
                    track['frames'].append(frame_idx)
                    track['scores'].append(score)
                    track['distances'].append(best_distance)
                    assigned_detections.add(best_match)
            
            # Remove inactive tracks
            for track_id in tracks_to_remove:
                active_tracks.pop(track_id)
            
            # Create new tracks for unassigned detections
            for i, pos in enumerate(current_particles):
                if i not in assigned_detections:
                    track_id = next_track_id
                    next_track_id += 1
                    
                    tracks[track_id] = {
                        'positions': [pos],
                        'frames': [frame_idx],
                        'scores': [current_scores[i]],
                        'distances': [0]  # First detection has no movement
                    }
                    
                    active_tracks[track_id] = True
        else:
            # No active tracks, create new tracks for all detections
            for i, pos in enumerate(current_particles):
                track_id = next_track_id
                next_track_id += 1
                
                tracks[track_id] = {
                    'positions': [pos],
                    'frames': [frame_idx],
                    'scores': [current_scores[i]],
                    'distances': [0]
                }
                
                active_tracks[track_id] = True
    
    # Filter tracks by minimum length
    filtered_tracks = {}
    for track_id, track in tracks.items():
        if len(track['frames']) >= min_track_length:
            filtered_tracks[track_id] = track
    
    print(f"Tracking complete: {len(filtered_tracks)} tracks with ≥{min_track_length} frames")
    print(f"Total track segments created: {len(tracks)}")
    print(f"Tracks meeting length criteria: {len(filtered_tracks)}")
    
    return filtered_tracks


def calculate_track_properties(tracks: Dict[int, Dict[str, Any]], 
                             image_stack: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    Calculate comprehensive properties for each track
    
    Args:
        tracks (Dict): Track data from track_particles_across_frames
        image_stack (np.ndarray): Original image stack for intensity measurements
        
    Returns:
        Dict[int, Dict[str, Any]]: Tracks with calculated properties
    """
    print("Calculating track properties...")
    
    for track_id, track in tqdm(tracks.items(), desc="Analyzing tracks"):
        # Calculate instantaneous velocities
        velocities = []
        for i in range(1, len(track['frames'])):
            pos1 = track['positions'][i-1]
            pos2 = track['positions'][i]
            frame1 = track['frames'][i-1]
            frame2 = track['frames'][i]
            
            # Distance in pixels
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            # Time in frames
            time = frame2 - frame1
            
            # Velocity in pixels per frame
            velocity = distance / time if time > 0 else 0
            velocities.append(velocity)
        
        track['velocities'] = velocities
        track['avg_velocity'] = np.mean(velocities) if velocities else 0
        track['max_velocity'] = max(velocities) if velocities else 0
        track['velocity_std'] = np.std(velocities) if velocities else 0
        
        # Calculate accelerations (change in velocity)
        accelerations = []
        for i in range(1, len(velocities)):
            accel = velocities[i] - velocities[i-1]
            accelerations.append(accel)
        
        track['accelerations'] = accelerations
        track['avg_acceleration'] = np.mean(accelerations) if accelerations else 0
        
        # Calculate intensities at each position
        intensities = []
        for frame_idx, pos in zip(track['frames'], track['positions']):
            if frame_idx >= len(image_stack):
                continue
                
            x, y = int(pos[0]), int(pos[1])
            
            # Create circular mask for intensity measurement
            size = 10  # Radius around particle
            y_min = max(0, y - size)
            y_max = min(image_stack.shape[1], y + size)
            x_min = max(0, x - size)
            x_max = min(image_stack.shape[2], x + size)
            
            # Extract region and calculate mean intensity within circular area
            region = image_stack[frame_idx, y_min:y_max, x_min:x_max]
            local_y, local_x = np.ogrid[0:y_max-y_min, 0:x_max-x_min]
            local_center_y = y - y_min
            local_center_x = x - x_min
            mask = ((local_x - local_center_x)**2 + (local_y - local_center_y)**2) <= size**2
            
            if np.any(mask):
                intensity = np.mean(region[mask])
                intensities.append(intensity)
            else:
                intensities.append(0)
        
        track['intensities'] = intensities
        track['avg_intensity'] = np.mean(intensities) if intensities else 0
        track['intensity_std'] = np.std(intensities) if intensities else 0
        
        # Calculate displacement and path metrics
        if len(track['positions']) >= 2:
            first_pos = track['positions'][0]
            last_pos = track['positions'][-1]
            displacement = np.sqrt((last_pos[0] - first_pos[0])**2 + 
                                 (last_pos[1] - first_pos[1])**2)
            
            # Total path length
            path_length = sum(np.sqrt((track['positions'][i][0] - track['positions'][i-1][0])**2 + 
                                     (track['positions'][i][1] - track['positions'][i-1][1])**2)
                             for i in range(1, len(track['positions'])))
            
            # Directness ratio (displacement / path length)
            directness = displacement / path_length if path_length > 0 else 0
            
            # Track duration
            duration = track['frames'][-1] - track['frames'][0] + 1
            
            track['displacement'] = displacement
            track['path_length'] = path_length
            track['directness'] = directness
            track['duration'] = duration
            track['avg_speed'] = displacement / duration if duration > 0 else 0
        else:
            track['displacement'] = 0
            track['path_length'] = 0
            track['directness'] = 0
            track['duration'] = 1
            track['avg_speed'] = 0
        
        # Calculate track quality metrics
        track['avg_detection_score'] = np.mean(track['scores'])
        track['min_detection_score'] = min(track['scores'])
        track['score_consistency'] = 1 - (np.std(track['scores']) / (np.mean(track['scores']) + 1e-6))
        
        # Calculate movement patterns
        track['straightness'] = track['directness']
        
        # Calculate turning angles
        turning_angles = []
        if len(track['positions']) >= 3:
            for i in range(1, len(track['positions']) - 1):
                p1 = np.array(track['positions'][i-1])
                p2 = np.array(track['positions'][i])
                p3 = np.array(track['positions'][i+1])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate angle between vectors
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    turning_angles.append(angle)
        
        track['turning_angles'] = turning_angles
        track['avg_turning_angle'] = np.mean(turning_angles) if turning_angles else 0
        track['movement_variability'] = np.std(turning_angles) if turning_angles else 0
    
    return tracks


def visualize_tracking_results(image_stack: np.ndarray, 
                             tracks: Dict[int, Dict[str, Any]],
                             output_dir: str) -> List[str]:
    """
    Create basic visualizations of tracking results
    
    Args:
        image_stack (np.ndarray): Original image stack
        tracks (Dict): Track data with calculated properties
        output_dir (str): Directory to save visualizations
        
    Returns:
        List[str]: Paths to created visualization files
    """
    print("Creating tracking visualizations...")
    
    # Create track overview
    overview_path = create_track_overview(image_stack, tracks, output_dir)
    
    # Create detailed track visualizations for top tracks
    details_path = create_detailed_track_visualizations(image_stack, tracks, output_dir)
    
    print(f"Tracking visualizations saved:")
    print(f"  - Overview: {overview_path}")
    print(f"  - Detailed Tracks: {details_path}")
    
    return [overview_path, details_path]


def create_track_overview(image_stack: np.ndarray,
                         tracks: Dict[int, Dict[str, Any]],
                         output_dir: str) -> str:
    """Create an overview visualization showing all tracks"""
    
    # Use middle frame as background
    mid_frame = len(image_stack) // 2
    base_frame = image_stack[mid_frame]
    frame_norm = cv2.normalize(base_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
    
    # Define colors for different track qualities
    colors = [
        (255, 0, 0),   # Red - high quality
        (0, 255, 0),   # Green - medium quality  
        (0, 0, 255),   # Blue - lower quality
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255)  # Yellow
    ]
    
    # Sort tracks by quality (average detection score)
    sorted_tracks = sorted(tracks.items(), 
                          key=lambda x: x[1]['avg_detection_score'], 
                          reverse=True)
    
    # Draw tracks
    tracks_drawn = 0
    for track_id, track in sorted_tracks:
        if tracks_drawn >= 20:  # Limit to top 20 tracks for clarity
            break
            
        avg_score = track['avg_detection_score']
        if avg_score < 0.3:  # Skip low-quality tracks
            continue
            
        color = colors[tracks_drawn % len(colors)]
        positions = track['positions']
        
        # Draw track path
        for i in range(1, len(positions)):
            pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            cv2.line(frame_rgb, pt1, pt2, color, 2)
        
        # Draw start and end markers
        if positions:
            start_pt = (int(positions[0][0]), int(positions[0][1]))
            end_pt = (int(positions[-1][0]), int(positions[-1][1]))
            
            # Start: filled circle
            cv2.circle(frame_rgb, start_pt, 8, color, -1)
            # End: circle with X
            cv2.circle(frame_rgb, end_pt, 8, color, 2)
            cv2.line(frame_rgb, (end_pt[0]-5, end_pt[1]-5), (end_pt[0]+5, end_pt[1]+5), color, 2)
            cv2.line(frame_rgb, (end_pt[0]-5, end_pt[1]+5), (end_pt[0]+5, end_pt[1]-5), color, 2)
            
            # Add track ID and quality score
            cv2.putText(frame_rgb, f"#{track_id} ({avg_score:.2f})", 
                       (start_pt[0] + 10, start_pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        tracks_drawn += 1
    
    # Add legend
    legend_y = 30
    cv2.putText(frame_rgb, f"Showing top {tracks_drawn} tracks (score > 0.3)", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_rgb, "● = start, ⊗ = end", 
               (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    overview_path = os.path.join(output_dir, "tracking_overview.png")
    cv2.imwrite(overview_path, frame_rgb)
    
    return overview_path


def create_detailed_track_visualizations(image_stack: np.ndarray,
                                       tracks: Dict[int, Dict[str, Any]],
                                       output_dir: str,
                                       num_tracks: int = 5) -> str:
    """Create detailed visualizations for the best tracks"""
    
    # Sort tracks by quality and select top ones
    sorted_tracks = sorted(tracks.items(),
                          key=lambda x: (x[1]['avg_detection_score'], len(x[1]['frames'])),
                          reverse=True)
    
    top_tracks = sorted_tracks[:num_tracks]
    
    # Create summary figure showing all top tracks
    fig, axes = plt.subplots(num_tracks, 3, figsize=(15, 3*num_tracks))
    if num_tracks == 1:
        axes = axes.reshape(1, -1)
    
    for i, (track_id, track) in enumerate(top_tracks):
        # Track path plot
        positions = track['positions']
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        axes[i, 0].plot(x_coords, y_coords, 'b-o', markersize=3)
        axes[i, 0].plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
        axes[i, 0].plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
        axes[i, 0].set_title(f'Track #{track_id} Path')
        axes[i, 0].set_xlabel('X Position (pixels)')
        axes[i, 0].set_ylabel('Y Position (pixels)')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axis('equal')
        
        # Velocity profile
        frames = track['frames'][1:]  # Skip first frame (no velocity)
        velocities = track['velocities']
        
        axes[i, 1].plot(frames, velocities, 'r-o', markersize=3)
        axes[i, 1].axhline(track['avg_velocity'], color='k', linestyle='--',
                          label=f'Avg: {track["avg_velocity"]:.2f}')
        axes[i, 1].set_title(f'Track #{track_id} Velocity')
        axes[i, 1].set_xlabel('Frame Number')
        axes[i, 1].set_ylabel('Velocity (pixels/frame)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Detection scores
        axes[i, 2].plot(track['frames'], track['scores'], 'g-o', markersize=3)
        axes[i, 2].axhline(track['avg_detection_score'], color='k', linestyle='--',
                          label=f'Avg: {track["avg_detection_score"]:.3f}')
        axes[i, 2].set_title(f'Track #{track_id} Detection Scores')
        axes[i, 2].set_xlabel('Frame Number')
        axes[i, 2].set_ylabel('Detection Score')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    details_path = os.path.join(output_dir, "detailed_track_analysis.png")
    plt.savefig(details_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return details_path


def analyze_tracking_quality(tracks: Dict[int, Dict[str, Any]],
                           all_particles: Dict[int, Dict[str, List]],
                           tracking_params: Dict[str, Any],
                           output_dir: str) -> str:
    """Basic tracking quality analysis"""
    
    # Calculate linking efficiency
    total_detections = sum(len(frame_data['positions']) for frame_data in all_particles.values())
    linked_detections = sum(len(track['positions']) for track in tracks.values())
    linking_efficiency = linked_detections / total_detections if total_detections > 0 else 0
    
    # Basic track statistics
    if tracks:
        avg_track_length = np.mean([len(track['frames']) for track in tracks.values()])
        avg_score = np.mean([track['avg_detection_score'] for track in tracks.values()])
        avg_directness = np.mean([track['directness'] for track in tracks.values()])
    else:
        avg_track_length = 0
        avg_score = 0
        avg_directness = 0
    
    print(f"Tracking Quality Analysis:")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Successfully linked: {linked_detections}")
    print(f"  - Linking efficiency: {linking_efficiency*100:.1f}%")
    print(f"  - Number of tracks: {len(tracks)}")
    print(f"  - Average track length: {avg_track_length:.1f}")
    print(f"  - Average detection score: {avg_score:.3f}")
    print(f"  - Average directness: {avg_directness:.3f}")
    
    return output_dir