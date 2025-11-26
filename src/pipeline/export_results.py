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
    
    # Get just the filename, not the full path
    base_filename = os.path.basename(tiff_filename)
    
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
                    'tiff_filename': base_filename,
                    'EV_ID': track_id if track_id is not None else -1,  # -1 for untracked
                    'frame_number': frame_idx + 1,  # Add 1 to make it 1-indexed
                    'X_COM': round(pos[0], 0),      # Rounded to integer per your example
                    'Y_COM': round(pos[1], 0),      # Rounded to integer per your example
                    'detection_confidence': round(score, 4),
                    'tracked': 'Yes' if track_id is not None else 'No'
                }
                detection_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(detection_records)
    
    # Define column order explicitly to match your request
    columns = ['tiff_filename', 'EV_ID', 'frame_number', 'X_COM', 'Y_COM', 'detection_confidence', 'tracked']
    
    # If DataFrame is empty (no detections), create it with these columns
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        df = df[columns]
    
    # Sort by EV_ID and frame_number
    # Put tracked items first (positive IDs), then untracked (-1)
    if not df.empty:
        df = df.sort_values(['EV_ID', 'frame_number'], ascending=[True, True])
        
        # Optional: Cast X_COM and Y_COM to int if you want strictly integer look in CSV
        df['X_COM'] = df['X_COM'].astype(int)
        df['Y_COM'] = df['Y_COM'].astype(int)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Print summary
    total_detections = len(df)
    tracked_detections = len(df[df['tracked'] == 'Yes'])
    num_particles = len(df[df['EV_ID'] != -1]['EV_ID'].unique()) if not df.empty else 0
    
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
    """Export track summaries (one row per track)"""

    print("Exporting track summary to CSV...")
    
    track_records = []
    base_filename = os.path.basename(tiff_filename)
    
    for track_id, track in tracks.items():
        record = {
            'tiff_filename': base_filename,
            'EV_ID': track_id,
            'track_length': len(track['frames']),
            'start_frame': min(track['frames']) + 1,  # 1-indexed
            'end_frame': max(track['frames']) + 1,    # 1-indexed
            'duration': track.get('duration', 0),
            'avg_detection_confidence': round(track.get('avg_detection_score', 0), 4),
            'avg_velocity_px_per_frame': round(track.get('avg_velocity', 0), 3),
            'displacement_px': round(track.get('displacement', 0), 3),
            'path_length_px': round(track.get('path_length', 0), 3),
            'directness_ratio': round(track.get('directness', 0), 4),
            'start_x': round(track['positions'][0][0], 1),
            'start_y': round(track['positions'][0][1], 1),
            'end_x': round(track['positions'][-1][0], 1),
            'end_y': round(track['positions'][-1][1], 1),
        }
        track_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(track_records)
    
    if not df.empty:
        # Sort by track length (longest first)
        df = df.sort_values(['track_length', 'EV_ID'], ascending=[False, True])
    
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

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export detections (This is the one you asked for)
    detections_path = os.path.join(output_dir, "all_detections.csv")
    export_detections_to_csv(
        all_particles, 
        tracks, 
        tiff_filename, 
        detections_path,
        include_untracked
    )
    
    # Export track summaries
    tracks_path = os.path.join(output_dir, f"track_summaries.csv")
    export_tracks_to_csv(tracks, tiff_filename, tracks_path)
    
    print("\n" + "="*60)
    print("ALL RESULTS EXPORTED SUCCESSFULLY")
    print("="*60)
    
    return {
        'detections': detections_path,
        'tracks': tracks_path
    }