import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional


def export_detections_to_csv(
    all_particles: Dict[int, Dict[str, List]],
    tracks: Dict[int, Dict[str, Any]],
    tiff_filename: str,
    output_path: str,
    include_untracked: bool = True
) -> str:
    """
    Export all detections to CSV with particle tracking information.
    
    Args:
        all_particles: Detection results from all frames
        tracks: Track assignments and properties
        tiff_filename: Name of the input TIFF file
        output_path: Path where CSV will be saved
        include_untracked: Whether to include detections not assigned to any track
        
    Returns:
        str: Path to the created CSV file
    """
    print("Exporting detection results to CSV...")
    
    # Create reverse mapping: frame -> position -> track_id
    position_to_track = {}
    for track_id, track in tracks.items():
        for frame_idx, pos in zip(track['frames'], track['positions']):
            if frame_idx not in position_to_track:
                position_to_track[frame_idx] = {}
            # Use position tuple as key (rounded to avoid floating point issues)
            pos_key = (round(pos[0], 2), round(pos[1], 2))
            position_to_track[frame_idx][pos_key] = track_id
    
    # Collect all detection data
    detection_records = []
    
    for frame_idx in sorted(all_particles.keys()):
        frame_data = all_particles[frame_idx]
        
        for pos, score in zip(frame_data['positions'], frame_data['scores']):
            # Find matching track ID
            pos_key = (round(pos[0], 2), round(pos[1], 2))
            track_id = None
            
            if frame_idx in position_to_track:
                # Check for exact match first
                if pos_key in position_to_track[frame_idx]:
                    track_id = position_to_track[frame_idx][pos_key]
                else:
                    # Check for nearby matches (within 1 pixel)
                    for tracked_pos, tid in position_to_track[frame_idx].items():
                        dist = np.sqrt((tracked_pos[0] - pos[0])**2 + (tracked_pos[1] - pos[1])**2)
                        if dist < 1.0:
                            track_id = tid
                            break
            
            # Only include if tracked or if include_untracked is True
            if track_id is not None or include_untracked:
                record = {
                    'tiff_filename': os.path.basename(tiff_filename),
                    'particle_id': track_id if track_id is not None else -1,  # -1 for untracked
                    'frame_number': frame_idx,
                    'x_center_of_mass': round(pos[0], 3),
                    'y_center_of_mass': round(pos[1], 3),
                    'detection_confidence': round(score, 4),
                    'tracked': 'Yes' if track_id is not None else 'No'
                }
                detection_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(detection_records)
    
    # Sort by particle_id and frame_number
    df = df.sort_values(['particle_id', 'frame_number'], ascending=[True, True])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Print summary
    total_detections = len(df)
    tracked_detections = len(df[df['tracked'] == 'Yes'])
    num_particles = len(df[df['particle_id'] != -1]['particle_id'].unique())
    
    print(f"\nExport Summary:")
    print(f"  Total detections exported: {total_detections}")
    print(f"  Tracked detections: {tracked_detections}")
    print(f"  Untracked detections: {total_detections - tracked_detections}")
    print(f"  Unique particle tracks: {num_particles}")
    print(f"  CSV saved to: {output_path}")
    
    return output_path


def export_tracks_to_csv(
    tracks: Dict[int, Dict[str, Any]],
    tiff_filename: str,
    output_path: str
) -> str:
    """
    Export track-level summary statistics to CSV.
    
    Args:
        tracks: Track data with calculated properties
        tiff_filename: Name of the input TIFF file
        output_path: Path where CSV will be saved
        
    Returns:
        str: Path to the created CSV file
    """
    print("Exporting track summary to CSV...")
    
    track_records = []
    
    for track_id, track in tracks.items():
        track_length = len(track['frames'])
        record = {
            'tiff_filename': os.path.basename(tiff_filename),
            'particle_id': track_id,
            'track_length': track_length,
            'start_frame': min(track['frames']),
            'end_frame': max(track['frames']),
            'duration': track['duration'],
            'num_gaps': track.get('num_gaps', 0),
            'avg_detection_confidence': round(track['avg_detection_score'], 4),
            'min_detection_confidence': round(track['min_detection_score'], 4),
            'avg_velocity_px_per_frame': round(track['avg_velocity'], 3),
            'max_velocity_px_per_frame': round(track['max_velocity'], 3),
            'velocity_std': round(track['velocity_std'], 3),
            'displacement_px': round(track['displacement'], 3),
            'path_length_px': round(track['path_length'], 3),
            'directness_ratio': round(track['directness'], 4),
            'avg_intensity': round(track['avg_intensity'], 2),
            'intensity_std': round(track['intensity_std'], 2),
            'avg_turning_angle_rad': round(track['avg_turning_angle'], 4),
            'movement_variability': round(track['movement_variability'], 4),
            'start_x': round(track['positions'][0][0], 3),
            'start_y': round(track['positions'][0][1], 3),
            'end_x': round(track['positions'][-1][0], 3),
            'end_y': round(track['positions'][-1][1], 3),
        }
        track_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(track_records)
    
    # Sort by track length (longest first) and particle ID
    df = df.sort_values(['track_length', 'particle_id'], ascending=[False, True])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"  Tracks exported: {len(df)}")
    print(f"  CSV saved to: {output_path}")
    
    return output_path


def export_all_results(
    all_particles: Dict[int, Dict[str, List]],
    tracks: Dict[int, Dict[str, Any]],
    tiff_filename: str,
    output_dir: str,
    include_untracked: bool = True
) -> Dict[str, str]:
    """
    Export all results to CSV files.
    
    Args:
        all_particles: Detection results from all frames
        tracks: Track assignments and properties
        tiff_filename: Name of the input TIFF file
        output_dir: Directory where CSVs will be saved
        include_untracked: Whether to include untracked detections
        
    Returns:
        Dict[str, str]: Paths to created CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export detections
    detections_path = os.path.join(output_dir, "all_detections.csv")
    export_detections_to_csv(
        all_particles, 
        tracks, 
        tiff_filename, 
        detections_path,
        include_untracked
    )
    
    # Export track summaries
    tracks_path = os.path.join(output_dir, f"{os.path.basename(tiff_filename)}_summaries.csv")
    export_tracks_to_csv(tracks, tiff_filename, tracks_path)
    
    print("\n" + "="*60)
    print("ALL RESULTS EXPORTED SUCCESSFULLY")
    print("="*60)
    
    return {
        'detections': detections_path,
        'tracks': tracks_path
    }