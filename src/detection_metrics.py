"""
Tracking Metrics Module for EV Detection Pipeline

Evaluates tracking performance when ground truth represents a single particle
tracked across multiple frames (as opposed to multiple separate particles).

Functions:
- evaluate_tracking_performance(): Main evaluation for tracked particles
- calculate_frame_detection_rate(): Per-frame detection success
- calculate_position_accuracy(): Spatial accuracy of detections
- match_track_to_ground_truth(): Find which track corresponds to GT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.distance import cdist
import os


def load_ground_truth_track(csv_path: str) -> Dict[str, Any]:
    """
    Load ground truth for a tracked particle
    
    Returns:
        Dictionary with 'frames', 'positions', and metadata
    """
    print(f"Loading ground truth track from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for EV_ID to determine if multiple particles
    if 'EV_ID' in df.columns:
        unique_ids = df['EV_ID'].nunique()
        print(f"Found {unique_ids} unique particle ID(s) in ground truth")
        
        if unique_ids > 1:
            print(f"Warning: Multiple particles detected. Using first particle (ID={df['EV_ID'].iloc[0]})")
            df = df[df['EV_ID'] == df['EV_ID'].iloc[0]]
    
    # Extract frame numbers and positions - KEEP ORIGINAL SLICE NUMBERS
    frames = df['Slice'].values
    positions = list(zip(df['X_COM'].values, df['Y_COM'].values))
    
    gt_track = {
        'frames': frames,
        'positions': positions,
        'start_frame': int(frames.min()),
        'end_frame': int(frames.max()),
        'num_frames': len(frames),
        'particle_id': df['EV_ID'].iloc[0] if 'EV_ID' in df.columns else 1
    }
    
    print(f"Ground truth track: {gt_track['num_frames']} frames "
          f"({gt_track['start_frame']} to {gt_track['end_frame']})")
    
    return gt_track


def match_track_to_ground_truth(tracks: Dict[int, Dict[str, Any]],
                                gt_track: Dict[str, Any],
                                distance_threshold: float = 50.0) -> Tuple[Optional[int], float]:
    """
    Find which detected track best matches the ground truth track
    
    Args:
        tracks: All detected tracks
        gt_track: Ground truth track data
        distance_threshold: Max average distance to consider a match
    
    Returns:
        (best_track_id, avg_distance) or (None, inf) if no match
    """
    if not tracks:
        return None, float('inf')
    
    gt_positions = np.array(gt_track['positions'])
    gt_frames = set(gt_track['frames'])
    
    best_track_id = None
    best_avg_distance = float('inf')
    
    for track_id, track in tracks.items():
        # Find overlapping frames
        track_frames = set(track['frames'])
        overlap_frames = gt_frames.intersection(track_frames)
        
        if len(overlap_frames) < 3:  # Need at least 3 overlapping frames
            continue
        
        # Calculate distances for overlapping frames
        distances = []
        for frame in overlap_frames:
            gt_idx = np.where(gt_track['frames'] == frame)[0][0]
            track_idx = track['frames'].index(frame)
            
            gt_pos = gt_positions[gt_idx]
            track_pos = track['positions'][track_idx]
            
            dist = np.sqrt((gt_pos[0] - track_pos[0])**2 + (gt_pos[1] - track_pos[1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        if avg_distance < best_avg_distance and avg_distance < distance_threshold:
            best_avg_distance = avg_distance
            best_track_id = track_id
    
    return best_track_id, best_avg_distance


def calculate_frame_detection_rate(all_particles: Dict[int, Dict[str, List]],
                                   gt_track: Dict[str, Any],
                                   distance_threshold: float = 20.0) -> Dict[str, Any]:
    """
    Calculate per-frame detection success rate
    Only evaluates frames where GT exists AND detections were performed
    """
    print(f"Calculating frame detection rate (threshold={distance_threshold}px)...")
    
    gt_frames = gt_track['frames']  # These are actual Slice numbers (239-314)
    gt_positions = np.array(gt_track['positions'])
    
    detected_frames = []
    missed_frames = []
    detection_distances = []
    detection_scores = []
    
    for i, frame in enumerate(gt_frames):
        gt_pos = gt_positions[i]
        
        # Check if this GT frame exists in our detections
        # GT frame 239 should match detection frame 239
        if frame not in all_particles:
            missed_frames.append(frame)
            continue
        
        # Check if any detection is close to GT position
        frame_detections = all_particles[frame]['positions']
        frame_scores = all_particles[frame]['scores']
        
        if not frame_detections:
            missed_frames.append(frame)
            continue
        
        # Find closest detection
        det_array = np.array(frame_detections)
        distances = np.sqrt(np.sum((det_array - gt_pos)**2, axis=1))
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        if min_dist <= distance_threshold:
            detected_frames.append(frame)
            detection_distances.append(min_dist)
            detection_scores.append(frame_scores[min_dist_idx])
        else:
            missed_frames.append(frame)
    
    # Calculate summary statistics AFTER the loop
    num_detected = len(detected_frames)
    num_missed = len(missed_frames)
    total_frames = len(gt_frames)
    
    detection_rate = num_detected / total_frames if total_frames > 0 else 0
    
    results = {
        'detection_rate': detection_rate,
        'frames_detected': num_detected,
        'frames_missed': num_missed,
        'total_frames': total_frames,
        'detected_frame_list': detected_frames,
        'missed_frame_list': missed_frames,
        'avg_detection_distance': np.mean(detection_distances) if detection_distances else 0,
        'std_detection_distance': np.std(detection_distances) if detection_distances else 0,
        'avg_detection_score': np.mean(detection_scores) if detection_scores else 0,
        'detection_distances': detection_distances,
        'detection_scores': detection_scores
    }
    
    print(f"  Detected in {num_detected}/{total_frames} frames ({detection_rate*100:.1f}%)")
    print(f"  Avg position error: {results['avg_detection_distance']:.2f}px")
    print(f"  Avg detection score: {results['avg_detection_score']:.3f}")
    
    return results


def calculate_track_metrics(matched_track: Dict[str, Any],
                            gt_track: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate metrics for the matched track
    
    Args:
        matched_track: The detected track that matches GT
        gt_track: Ground truth track
    
    Returns:
        Dictionary with track quality metrics
    """
    print("Calculating track quality metrics...")
    
    # Frame overlap
    track_frames = set(matched_track['frames'])
    gt_frames = set(gt_track['frames'])
    overlap_frames = track_frames.intersection(gt_frames)
    
    track_recall = len(overlap_frames) / len(gt_frames) if gt_frames else 0
    track_precision = len(overlap_frames) / len(track_frames) if track_frames else 0
    
    # Position accuracy for overlapping frames
    position_errors = []
    gt_positions = np.array(gt_track['positions'])
    
    for frame in overlap_frames:
        gt_idx = np.where(gt_track['frames'] == frame)[0][0]
        track_idx = matched_track['frames'].index(frame)
        
        gt_pos = gt_positions[gt_idx]
        track_pos = matched_track['positions'][track_idx]
        
        error = np.sqrt((gt_pos[0] - track_pos[0])**2 + (gt_pos[1] - track_pos[1])**2)
        position_errors.append(error)
    
    # Track continuity (gaps)
    track_frame_nums = sorted(matched_track['frames'])
    gaps = []
    for i in range(len(track_frame_nums) - 1):
        gap = track_frame_nums[i+1] - track_frame_nums[i] - 1
        if gap > 0:
            gaps.append(gap)
    
    results = {
        'track_recall': track_recall,
        'track_precision': track_precision,
        'track_f1': 2 * (track_precision * track_recall) / (track_precision + track_recall + 1e-10),
        'frames_in_track': len(track_frames),
        'frames_in_gt': len(gt_frames),
        'overlapping_frames': len(overlap_frames),
        'avg_position_error': np.mean(position_errors) if position_errors else 0,
        'max_position_error': np.max(position_errors) if position_errors else 0,
        'std_position_error': np.std(position_errors) if position_errors else 0,
        'num_gaps': len(gaps),
        'avg_gap_size': np.mean(gaps) if gaps else 0,
        'max_gap_size': max(gaps) if gaps else 0,
        'track_length': len(matched_track['frames']),
        'avg_velocity': matched_track.get('avg_velocity', 0),
        'avg_score': matched_track.get('avg_detection_score', 0)
    }
    
    print(f"  Track recall: {track_recall*100:.1f}%")
    print(f"  Track precision: {track_precision*100:.1f}%")
    print(f"  Avg position error: {results['avg_position_error']:.2f}px")
    print(f"  Track gaps: {results['num_gaps']}")
    
    return results


def evaluate_tracking_performance(all_particles: Dict[int, Dict[str, List]],
                                  tracks: Dict[int, Dict[str, Any]],
                                  ground_truth_csv: str,
                                  output_dir: str,
                                  distance_threshold: float = 10.0,
                                  visualize: bool = True) -> Dict[str, Any]:
    """
    Main evaluation function for tracking a single particle
    
    Args:
        all_particles: Detection results
        tracks: Tracking results
        ground_truth_csv: Path to ground truth CSV
        output_dir: Directory to save results
        distance_threshold: Distance threshold for matching (pixels)
        visualize: Whether to create visualizations
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print("\n" + "="*80)
    print("TRACKING EVALUATION - Single Particle Tracking")
    print("="*80 + "\n")
    
    # Load ground truth
    gt_track = load_ground_truth_track(ground_truth_csv)
    
    # Calculate frame-by-frame detection rate
    frame_metrics = calculate_frame_detection_rate(
        all_particles, gt_track, distance_threshold
    )

    print("\nDEBUG - Checking GT frame range:")
    for frame in range(239, 245):  # Check first 6 GT frames
        if frame in all_particles:
            print(f"  Frame {frame}: {len(all_particles[frame]['positions'])} detections")
        else:
            print(f"  Frame {frame}: NOT IN all_particles dict")
    
    # Match tracks to ground truth
    matched_track_id, match_distance = match_track_to_ground_truth(
        tracks, gt_track, distance_threshold=50.0
    )
    
    if matched_track_id is not None:
        print(f"\nMatched track ID: {matched_track_id} (avg distance: {match_distance:.2f}px)")
        matched_track = tracks[matched_track_id]
        
        track_metrics = calculate_track_metrics(matched_track, gt_track)
    else:
        print("\nWarning: No track matched ground truth!")
        track_metrics = {
            'track_recall': 0,
            'track_precision': 0,
            'track_f1': 0,
            'avg_position_error': float('inf')
        }
    
    # Compile results
    results = {
        'frame_detection_rate': frame_metrics['detection_rate'],
        'frames_detected': frame_metrics['frames_detected'],
        'frames_missed': frame_metrics['frames_missed'],
        'total_gt_frames': frame_metrics['total_frames'],
        'avg_position_error': frame_metrics['avg_detection_distance'],
        'std_position_error': frame_metrics['std_detection_distance'],
        'avg_detection_score': frame_metrics['avg_detection_score'],
        'matched_track_id': matched_track_id,
        'track_match_distance': match_distance,
        'distance_threshold': distance_threshold,
        'frame_metrics': frame_metrics,
        'track_metrics': track_metrics if matched_track_id else None,
        'ground_truth': gt_track
    }
    
    # Print summary
    print("\n" + "="*80)
    print("TRACKING PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Frame Detection Rate:  {frame_metrics['detection_rate']*100:.1f}% "
          f"({frame_metrics['frames_detected']}/{frame_metrics['total_frames']} frames)")
    print(f"Avg Position Error:    {frame_metrics['avg_detection_distance']:.2f} ± "
          f"{frame_metrics['std_detection_distance']:.2f} px")
    print(f"Avg Detection Score:   {frame_metrics['avg_detection_score']:.3f}")
    
    if matched_track_id:
        print(f"\nMatched Track #{matched_track_id}:")
        print(f"  Track Recall:        {track_metrics['track_recall']*100:.1f}%")
        print(f"  Track Precision:     {track_metrics['track_precision']*100:.1f}%")
        print(f"  Track F1 Score:      {track_metrics['track_f1']:.3f}")
        print(f"  Track Length:        {track_metrics['track_length']} frames")
        print(f"  Track Gaps:          {track_metrics['num_gaps']}")
    
    # Create visualizations
    if visualize:
        viz_paths = visualize_tracking_performance(
            results, all_particles, tracks, output_dir
        )
        results['visualization_paths'] = viz_paths
        
        # Save report
        report_path = save_tracking_report(results, output_dir)
        results['report_path'] = report_path
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")
    
    return results


def visualize_tracking_performance(results: Dict[str, Any],
                                   all_particles: Dict[int, Dict[str, List]],
                                   tracks: Dict[int, Dict[str, Any]],
                                   output_dir: str) -> List[str]:
    """
    Create visualizations for tracking performance
    """
    viz_paths = []
    
    # Figure 1: Detection rate and position error over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    gt_track = results['ground_truth']
    frame_metrics = results['frame_metrics']
    
    # Plot 1: Frame detection status
    ax1 = axes[0, 0]
    all_frames = gt_track['frames']
    detected = [1 if f in frame_metrics['detected_frame_list'] else 0 for f in all_frames]
    
    colors = ['green' if d else 'red' for d in detected]
    ax1.scatter(all_frames, detected, c=colors, s=50, alpha=0.7)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Detected (1) / Missed (0)')
    ax1.set_title(f'Frame-by-Frame Detection Status\nDetection Rate: {results["frame_detection_rate"]*100:.1f}%')
    ax1.set_ylim([-0.2, 1.2])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position errors over time
    ax2 = axes[0, 1]
    if frame_metrics['detection_distances']:
        detected_frames = frame_metrics['detected_frame_list']
        ax2.plot(detected_frames, frame_metrics['detection_distances'], 'b-o', markersize=4)
        ax2.axhline(results['avg_position_error'], color='red', linestyle='--',
                   label=f'Avg: {results["avg_position_error"]:.2f}px')
        ax2.axhline(results['distance_threshold'], color='orange', linestyle=':',
                   label=f'Threshold: {results["distance_threshold"]}px')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Position Error (pixels)')
        ax2.set_title('Detection Position Accuracy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detection scores over time
    ax3 = axes[1, 0]
    if frame_metrics['detection_scores']:
        detected_frames = frame_metrics['detected_frame_list']
        ax3.plot(detected_frames, frame_metrics['detection_scores'], 'g-o', markersize=4)
        ax3.axhline(results['avg_detection_score'], color='red', linestyle='--',
                   label=f'Avg: {results["avg_detection_score"]:.3f}')
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Detection Score')
        ax3.set_title('Detection Confidence Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance summary bars
    ax4 = axes[1, 1]
    if results['matched_track_id']:
        tm = results['track_metrics']
        metrics = ['Detection\nRate', 'Track\nRecall', 'Track\nPrecision', 'Track\nF1']
        values = [results['frame_detection_rate'], tm['track_recall'], 
                 tm['track_precision'], tm['track_f1']]
        
        bars = ax4.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylim([0, 1.1])
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Summary')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path1 = os.path.join(output_dir, 'tracking_performance.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(path1)
    
    print(f"Visualization saved: {path1}")
    
    return viz_paths


def save_tracking_report(results: Dict[str, Any], output_dir: str) -> str:
    """
    Save detailed tracking evaluation report
    """
    report_path = os.path.join(output_dir, 'tracking_evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRACKING EVALUATION REPORT\n")
        f.write("Single Particle Tracking Performance\n")
        f.write("="*80 + "\n\n")
        
        f.write("FRAME-BY-FRAME DETECTION\n")
        f.write("-"*80 + "\n")
        f.write(f"Detection Rate:        {results['frame_detection_rate']*100:.1f}%\n")
        f.write(f"Frames Detected:       {results['frames_detected']}/{results['total_gt_frames']}\n")
        f.write(f"Frames Missed:         {results['frames_missed']}\n")
        f.write(f"Avg Position Error:    {results['avg_position_error']:.2f} px\n")
        f.write(f"Std Position Error:    {results['std_position_error']:.2f} px\n")
        f.write(f"Avg Detection Score:   {results['avg_detection_score']:.3f}\n")
        f.write(f"Distance Threshold:    {results['distance_threshold']:.1f} px\n\n")
        
        if results['matched_track_id']:
            tm = results['track_metrics']
            f.write("TRACK QUALITY\n")
            f.write("-"*80 + "\n")
            f.write(f"Matched Track ID:      #{results['matched_track_id']}\n")
            f.write(f"Track Recall:          {tm['track_recall']*100:.1f}%\n")
            f.write(f"Track Precision:       {tm['track_precision']*100:.1f}%\n")
            f.write(f"Track F1 Score:        {tm['track_f1']:.3f}\n")
            f.write(f"Track Length:          {tm['track_length']} frames\n")
            f.write(f"Overlapping Frames:    {tm['overlapping_frames']}\n")
            f.write(f"Track Gaps:            {tm['num_gaps']}\n")
            if tm['num_gaps'] > 0:
                f.write(f"Avg Gap Size:          {tm['avg_gap_size']:.1f} frames\n")
                f.write(f"Max Gap Size:          {tm['max_gap_size']} frames\n")
            f.write(f"Avg Position Error:    {tm['avg_position_error']:.2f} px\n")
            f.write(f"Max Position Error:    {tm['max_position_error']:.2f} px\n")
            f.write(f"Avg Velocity:          {tm['avg_velocity']:.2f} px/frame\n")
            f.write(f"Avg Detection Score:   {tm['avg_score']:.3f}\n\n")
        else:
            f.write("TRACK QUALITY\n")
            f.write("-"*80 + "\n")
            f.write("No track matched ground truth\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*80 + "\n")
        dr = results['frame_detection_rate']
        if dr > 0.9:
            f.write("Excellent tracking performance (>90% frames detected)\n")
        elif dr > 0.75:
            f.write("Good tracking performance (>75% frames detected)\n")
        elif dr > 0.5:
            f.write("Moderate tracking performance (>50% frames detected)\n")
        else:
            f.write("Poor tracking performance (<50% frames detected)\n")
        
        if results['avg_position_error'] < 5:
            f.write("Excellent position accuracy (<5px error)\n")
        elif results['avg_position_error'] < 10:
            f.write("Good position accuracy (<10px error)\n")
        else:
            f.write("Moderate position accuracy (>10px error)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Tracking report saved: {report_path}")
    return report_path