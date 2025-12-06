import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import os


def load_ground_truth_track(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    
    if 'EV_ID' in df.columns and df['EV_ID'].nunique() > 1:
        df = df[df['EV_ID'] == df['EV_ID'].iloc[0]]

    # CSV 'Slice' is 1-indexed (e.g. 1..N). Convert to 0-indexed for internal use.
    frames = np.array(df['Slice'].astype(int)) - 1
    # Ensure no negative indices if CSV had unexpected zeros
    frames = frames.clip(min=0)
    
    positions = list(zip(df['X_COM'].values, df['Y_COM'].values))
    
    return {
        'frames': frames,
        'positions': positions,
        'start_frame': int(frames.min()),
        'end_frame': int(frames.max()),
        'num_frames': len(frames)
    }


def match_track_to_ground_truth(tracks: Dict[int, Dict[str, Any]],
                                gt_track: Dict[str, Any],
                                distance_threshold: float = 50.0) -> Tuple[Optional[int], float]:
    """Find which detected track best matches ground truth"""
    if not tracks:
        return None, float('inf')
    
    gt_positions = np.array(gt_track['positions'])
    gt_frames = set(gt_track['frames'])
    
    best_track_id = None
    best_avg_distance = float('inf')
    
    for track_id, track in tracks.items():
        track_frames = set(track['frames'])
        overlap_frames = gt_frames.intersection(track_frames)
        
        if len(overlap_frames) < 3:
            continue
        
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
    """Calculate per-frame detection success rate"""
    gt_frames = gt_track['frames']
    gt_positions = np.array(gt_track['positions'])
    
    detected_frames = []
    missed_frames = []
    false_positive_frames = []
    detection_distances = []
    detection_scores = []
    
    # First handle all detection frames
    for frame, particles in all_particles.items():
        if not particles['positions']:
            continue
            
        if frame in gt_frames:
            # Frame has ground truth - check if detection matches
            gt_idx = np.where(gt_track['frames'] == frame)[0][0]
            gt_pos = gt_positions[gt_idx]
            
            # Find closest detection
            det_array = np.array(particles['positions'])
            distances = np.sqrt(np.sum((det_array - gt_pos)**2, axis=1))
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            if min_dist <= distance_threshold:
                detected_frames.append(frame)
                detection_distances.append(min_dist)
                detection_scores.append(particles['scores'][min_dist_idx])
            else:
                missed_frames.append(frame)
        else:
            # No ground truth for this frame - false positive
            false_positive_frames.append(frame)
    
    # Add remaining missed frames
    for frame in gt_frames:
        if frame not in detected_frames and frame not in missed_frames:
            missed_frames.append(frame)
    
    total_frames = len(all_particles)
    gt_frame_set = set(gt_frames)
    all_frame_set = set(all_particles.keys())

    # Correct detections (true positives)
    TP = len(detected_frames)

    # Missed detections (false negatives)
    FN = len(missed_frames)

    # False positives
    FP = len(false_positive_frames)

    # True negatives: frames with no GT and no detections
    TN = len([
        f for f in all_frame_set
        if f not in gt_frame_set and not all_particles[f]['positions']
    ])

    accuracy = (TP + TN) / total_frames if total_frames > 0 else 0

    
    return {
        'overall_frame_accuracy': accuracy,
        'true_positives': TP,
        'false_negatives': FN,
        'true_negatives': TN,
        'false_positives': FP,
        'frames_detected': TP,  # ADD THIS LINE
        'frames_missed': len(missed_frames),
        'total_frames': total_frames,
        'detected_frame_list': detected_frames,
        'missed_frame_list': missed_frames,
        'false_positive_frame_list': false_positive_frames,
        'avg_detection_distance': np.mean(detection_distances) if detection_distances else 0,
        'std_detection_distance': np.std(detection_distances) if detection_distances else 0,
        'avg_detection_score': np.mean(detection_scores) if detection_scores else 0,
        'detection_distances': detection_distances,
        'detection_scores': detection_scores,
        'detection_rate': len(detected_frames) / len(gt_frames) if len(gt_frames) > 0 else 0,
        'false_positive_rate': len(false_positive_frames) / total_frames if total_frames > 0 else 0,
    }

def calculate_track_metrics(matched_track: Dict[str, Any],
                            gt_track: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate metrics for the matched track"""
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
    
    # Track continuity
    track_frame_nums = sorted(matched_track['frames'])
    gaps = [track_frame_nums[i+1] - track_frame_nums[i] - 1 
            for i in range(len(track_frame_nums) - 1) if track_frame_nums[i+1] - track_frame_nums[i] > 1]
    
    return {
        'track_recall': track_recall,
        'track_precision': track_precision,
        'track_f1': 2 * (track_precision * track_recall) / (track_precision + track_recall + 1e-10),
        'overlapping_frames': len(overlap_frames),
        'avg_position_error': np.mean(position_errors) if position_errors else 0,
        'max_position_error': np.max(position_errors) if position_errors else 0,
        'num_gaps': len(gaps),
        'track_length': len(matched_track['frames']),
        'avg_velocity': matched_track.get('avg_velocity', 0)
    }

def evaluate_tracking_performance(all_particles: Dict[int, Dict[str, List]],
                                  tracks: Dict[int, Dict[str, Any]],
                                  ground_truth_csv: str,
                                  output_dir: str,
                                  distance_threshold: float = 20.0,
                                  visualize: bool = True,
                                  image_stack: np.ndarray = None) -> Dict[str, Any]:
    """Main evaluation function for tracking a single particle"""
    
    # Load ground truth
    gt_track = load_ground_truth_track(ground_truth_csv)
    
    # Calculate frame-by-frame detection rate
    frame_metrics = calculate_frame_detection_rate(all_particles, gt_track, distance_threshold)
    
    # Match tracks to ground truth
    matched_track_id, match_distance = match_track_to_ground_truth(tracks, gt_track, distance_threshold=50.0)
    
    if matched_track_id is not None:
        matched_track = tracks[matched_track_id]
        track_metrics = calculate_track_metrics(matched_track, gt_track)
    else:
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
        'matched_track_id': matched_track_id,
        'distance_threshold': distance_threshold,
        'frame_metrics': frame_metrics,
        'track_metrics': track_metrics if matched_track_id else None,
        'ground_truth': gt_track
    }
    
    # Print summary
    print("\n" + "="*80)
    print("TRACKING PERFORMANCE METRICS")
    print("="*80)
    print(f"\nFrame Detection Rate:  {frame_metrics['detection_rate']*100:.1f}% "
          f"({frame_metrics['frames_detected']}/{frame_metrics['total_frames']} frames)")
    print(f"False Positive Rate:   {frame_metrics['false_positive_rate']*100:.1f}% "
          f"({frame_metrics['false_positives']} false detections)")
    print(f"Position Accuracy:     {frame_metrics['avg_detection_distance']:.2f} ± "
          f"{frame_metrics['std_detection_distance']:.2f} px")
    print(f"Overall Frame Accuracy: {frame_metrics['overall_frame_accuracy']*100:.1f}% "
      f"({frame_metrics['true_positives']} TP, {frame_metrics['true_negatives']} TN)")

    
    if matched_track_id:
        print(f"\nTrack Quality (ID #{matched_track_id}):")
        print(f"  Recall:              {track_metrics['track_recall']*100:.1f}%")
        print(f"  Precision:           {track_metrics['track_precision']*100:.1f}%")
        print(f"  F1 Score:            {track_metrics['track_f1']:.3f}")
        print(f"  Track Length:        {track_metrics['track_length']} frames")
        print(f"  Gaps:                {track_metrics['num_gaps']}")
    
    print("="*80 + "\n")
    
    # Create visualizations and report
    if visualize:
        # Original visualization
        viz_paths = visualize_tracking_performance(results, all_particles, tracks, output_dir)
        results['visualization_paths'] = viz_paths
        
        # NEW: Create comprehensive track report if image_stack is provided
        if image_stack is not None:
            try:
                from src.metrics.improved_visualizations import create_comprehensive_track_report
                
                comprehensive_path = create_comprehensive_track_report(
                    image_stack=image_stack,
                    tracks=tracks,
                    all_particles=all_particles,
                    gt_track=gt_track,
                    frame_metrics=frame_metrics,
                    output_dir=output_dir,
                    top_n_tracks=5
                )
                
                results['comprehensive_report_path'] = comprehensive_path
                print(f"\n✓ Comprehensive track report created: {comprehensive_path}")
                
            except Exception as e:
                print(f"\n⚠ Warning: Could not create comprehensive report: {str(e)}")
        
        report_path = save_tracking_report(results, output_dir)
        results['report_path'] = report_path
    
    return results


def visualize_tracking_performance(results: Dict[str, Any],
                                   all_particles: Dict[int, Dict[str, List]],
                                   tracks: Dict[int, Dict[str, Any]],
                                   output_dir: str) -> List[str]:
    """Create performance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    gt_track = results['ground_truth']
    frame_metrics = results['frame_metrics']
    
    # Plot 1: Frame detection status with false positives
    ax1 = axes[0, 0]
    
    # Plot ground truth frames
    all_frames = sorted(list(set(gt_track['frames']).union(frame_metrics['false_positive_frame_list'])))
    detection_status = []
    colors = []
    
    for frame in all_frames:
        if frame in frame_metrics['detected_frame_list']:
            detection_status.append(1)
            colors.append('green')  # True positive
        elif frame in frame_metrics['false_positive_frame_list']:
            detection_status.append(0.5)  # Place false positives at y=0.5
            colors.append('yellow')  # False positive
        else:
            detection_status.append(0)
            colors.append('red')  # Missed detection
    
    ax1.scatter(all_frames, detection_status, c=colors, s=50, alpha=0.7)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Detection Status')
    ax1.set_title(f'Detection Status: {results["frame_detection_rate"]*100:.1f}% detected\n'
                  f'False Positive Rate: {frame_metrics["false_positive_rate"]*100:.1f}%')
    ax1.set_ylim([-0.2, 1.2])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels(['Missed', 'False\nPositive', 'Detected'])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position errors
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
        ax2.set_title('Position Accuracy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detection scores
    ax3 = axes[1, 0]
    if frame_metrics['detection_scores']:
        detected_frames = frame_metrics['detected_frame_list']
        ax3.plot(detected_frames, frame_metrics['detection_scores'], 'g-o', markersize=4)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Detection Score')
        ax3.set_title('Detection Confidence Over Time')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance summary
    ax4 = axes[1, 1]
    if results['matched_track_id']:
        tm = results['track_metrics']
        metrics = ['Detection\nRate', 'Track\nRecall', 'Track\nPrecision', 'Track\nF1']
        values = [results['frame_detection_rate'], tm['track_recall'], 
                 tm['track_precision'], tm['track_f1']]
        
        bars = ax4.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylim([0, 1.1])
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Summary')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'tracking_performance.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [path]


def save_tracking_report(results: Dict[str, Any], output_dir: str) -> str:
    """Save tracking evaluation report"""
    report_path = os.path.join(output_dir, 'tracking_evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRACKING EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DETECTION PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Detection Rate:        {results['frame_detection_rate']*100:.1f}%\n")
        f.write(f"Frames Detected:       {results['frames_detected']}/{results['total_gt_frames']}\n")
        f.write(f"False Positive Rate:   {results['frame_metrics']['false_positive_rate']*100:.1f}%\n")
        f.write(f"False Positives:       {results['frame_metrics']['false_positives']} frames\n")
        f.write(f"Avg Position Error:    {results['avg_position_error']:.2f} ± "
                f"{results['std_position_error']:.2f} px\n")
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
            f.write(f"Avg Position Error:    {tm['avg_position_error']:.2f} px\n")
            f.write(f"Max Position Error:    {tm['max_position_error']:.2f} px\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*80 + "\n")
        dr = results['frame_detection_rate']
        if dr > 0.75:
            f.write("Excellent/Good tracking performance (>75% frames detected)\n")
        elif dr > 0.5:
            f.write("Moderate tracking performance (>50% frames detected)\n")
        else:
            f.write("Poor tracking performance (<50% frames detected)\n")
        
        if results['avg_position_error'] < 15:
            f.write("Good position accuracy (<15px error)\n")
        else:
            f.write("Moderate position accuracy (>15px error)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    return report_path