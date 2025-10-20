"""
PR/ROC Curve Analysis and Multi-Track Evaluation for EV Detection Pipeline

This module provides comprehensive evaluation tools including:
- PR and ROC curve generation
- Per-track performance visualization
- Multi-file batch evaluation with aggregate metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import os
from pathlib import Path


def calculate_detection_labels_and_scores(all_particles: Dict[int, Dict[str, List]],
                                          gt_track: Dict[str, Any],
                                          distance_threshold: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create binary labels and scores for all detections across all frames."""
    labels = []
    scores = []
    
    gt_frames = set(gt_track['frames'])
    gt_positions = np.array(gt_track['positions'])
    gt_frame_to_idx = {frame: idx for idx, frame in enumerate(gt_track['frames'])}
    
    for frame, particles in all_particles.items():
        if not particles['positions']:
            continue
        
        det_positions = np.array(particles['positions'])
        det_scores = np.array(particles['scores'])
        
        if frame in gt_frames:
            gt_pos = gt_positions[gt_frame_to_idx[frame]]
            distances = np.sqrt(np.sum((det_positions - gt_pos)**2, axis=1))
            
            for dist, score in zip(distances, det_scores):
                labels.append(1 if dist <= distance_threshold else 0)
                scores.append(score)
        else:
            for score in det_scores:
                labels.append(0)
                scores.append(score)
    
    return np.array(labels), np.array(scores)


def calculate_per_frame_metrics(all_particles: Dict[int, Dict[str, List]],
                                gt_track: Dict[str, Any],
                                distance_threshold: float = 20.0) -> Dict[str, Any]:
    """Calculate frame-by-frame detection metrics including false positives."""
    gt_frames = gt_track['frames']
    gt_positions = np.array(gt_track['positions'])
    gt_frame_to_idx = {frame: idx for idx, frame in enumerate(gt_track['frames'])}
    
    frame_data = {
        'frames': [],
        'detection_scores': [],
        'position_errors': [],
        'detected': [],
        'false_positives': [],
        'num_detections': []
    }
    
    # Process each ground truth frame
    for frame in gt_frames:
        gt_pos = gt_positions[gt_frame_to_idx[frame]]
        
        if frame in all_particles and all_particles[frame]['positions']:
            det_positions = np.array(all_particles[frame]['positions'])
            det_scores = np.array(all_particles[frame]['scores'])
            
            # Find closest detection
            distances = np.sqrt(np.sum((det_positions - gt_pos)**2, axis=1))
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            frame_data['frames'].append(frame)
            frame_data['num_detections'].append(len(det_positions))
            
            if min_dist <= distance_threshold:
                # True positive
                frame_data['detected'].append(True)
                frame_data['detection_scores'].append(det_scores[min_dist_idx])
                frame_data['position_errors'].append(min_dist)
                # False positives = all other detections in this frame
                frame_data['false_positives'].append(len(det_positions) - 1)
            else:
                # False negative (missed detection)
                frame_data['detected'].append(False)
                frame_data['detection_scores'].append(0.0)
                frame_data['position_errors'].append(min_dist)
                # All detections are false positives
                frame_data['false_positives'].append(len(det_positions))
        else:
            # No detections in frame (missed)
            frame_data['frames'].append(frame)
            frame_data['detected'].append(False)
            frame_data['detection_scores'].append(0.0)
            frame_data['position_errors'].append(np.inf)
            frame_data['false_positives'].append(0)
            frame_data['num_detections'].append(0)
    
    return frame_data


def plot_single_track_performance(all_particles: Dict[int, Dict[str, List]],
                                  tracks: Dict[int, Dict[str, Any]],
                                  gt_track: Dict[str, Any],
                                  matched_track_id: Optional[int],
                                  output_path: str,
                                  filename: str,
                                  distance_threshold: float = 20.0) -> Dict[str, float]:
    """Create comprehensive visualization for a single track."""
    
    # Calculate frame-by-frame metrics
    frame_metrics = calculate_per_frame_metrics(all_particles, gt_track, distance_threshold)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # --- Plot 1: Track visualization (spans 2 rows) ---
    ax1 = fig.add_subplot(gs[0:2, 0])
    
    # Plot ground truth
    gt_positions = np.array(gt_track['positions'])
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], 'g-', linewidth=2, 
             label='Ground Truth', alpha=0.7)
    ax1.scatter(gt_positions[:, 0], gt_positions[:, 1], c='green', s=50, 
                marker='o', alpha=0.7, zorder=5)
    
    # Plot matched track if exists
    if matched_track_id is not None and matched_track_id in tracks:
        track = tracks[matched_track_id]
        track_positions = np.array(track['positions'])
        ax1.plot(track_positions[:, 0], track_positions[:, 1], 'b--', 
                linewidth=2, label=f'Detected Track #{matched_track_id}', alpha=0.7)
        ax1.scatter(track_positions[:, 0], track_positions[:, 1], c='blue', 
                   s=30, marker='x', alpha=0.7, zorder=5)
    
    # Mark start and end
    ax1.scatter(gt_positions[0, 0], gt_positions[0, 1], c='lime', s=200, 
               marker='*', edgecolors='black', linewidths=2, label='Start', zorder=10)
    ax1.scatter(gt_positions[-1, 0], gt_positions[-1, 1], c='red', s=200, 
               marker='*', edgecolors='black', linewidths=2, label='End', zorder=10)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=11)
    ax1.set_ylabel('Y Position (pixels)', fontsize=11)
    ax1.set_title(f'Track Visualization\n{filename}', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # --- Plot 2: Detection Score Over Time ---
    ax2 = fig.add_subplot(gs[0, 1])
    frames = frame_metrics['frames']
    scores = frame_metrics['detection_scores']
    detected = frame_metrics['detected']
    
    # Color by detection status
    colors = ['green' if d else 'red' for d in detected]
    ax2.scatter(frames, scores, c=colors, s=40, alpha=0.7)
    ax2.plot(frames, scores, 'k-', alpha=0.3, linewidth=1)
    
    # Add mean line
    valid_scores = [s for s, d in zip(scores, detected) if d]
    if valid_scores:
        ax2.axhline(np.mean(valid_scores), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(valid_scores):.3f}')
    
    ax2.set_xlabel('Frame Number', fontsize=10)
    ax2.set_ylabel('Detection Score', fontsize=10)
    ax2.set_title('Detection Confidence Over Time', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # --- Plot 3: Position Error Over Time ---
    ax3 = fig.add_subplot(gs[1, 1])
    errors = [e for e, d in zip(frame_metrics['position_errors'], detected) if d and e != np.inf]
    error_frames = [f for f, d, e in zip(frames, detected, frame_metrics['position_errors']) 
                    if d and e != np.inf]
    
    if errors:
        ax3.plot(error_frames, errors, 'b-o', markersize=4, linewidth=1.5)
        ax3.axhline(np.mean(errors), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(errors):.2f}px')
        ax3.axhline(distance_threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Threshold: {distance_threshold}px')
    
    ax3.set_xlabel('Frame Number', fontsize=10)
    ax3.set_ylabel('Position Error (pixels)', fontsize=10)
    ax3.set_title('Position Accuracy Over Time', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: False Positives Over Time ---
    ax4 = fig.add_subplot(gs[2, 0])
    fps = frame_metrics['false_positives']
    ax4.bar(frames, fps, color='orange', alpha=0.7, width=1.0)
    ax4.axhline(np.mean(fps), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(fps):.2f}')
    
    ax4.set_xlabel('Frame Number', fontsize=10)
    ax4.set_ylabel('False Positives per Frame', fontsize=10)
    ax4.set_title('False Positive Rate Over Time', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 5: Performance Summary ---
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Calculate summary metrics
    detection_rate = sum(detected) / len(detected) if detected else 0
    valid_scores = [s for s, d in zip(scores, detected) if d]
    avg_score = np.mean(valid_scores) if valid_scores else 0
    avg_error = np.mean(errors) if errors else np.inf
    avg_fps = np.mean(fps)
    
    metrics = ['Detection\nRate', 'Avg\nScore', 'Avg Error\n(norm)', 'Avg FPs\nper Frame']
    # Normalize error to 0-1 scale for visualization
    norm_error = 1 - min(avg_error / (distance_threshold * 2), 1.0) if avg_error != np.inf else 0
    values = [detection_rate, avg_score, norm_error, 1 - min(avg_fps / 5, 1.0)]  # Normalize FPs too
    
    bars = ax5.bar(metrics, values, color=['blue', 'green', 'purple', 'orange'], alpha=0.7)
    
    # Add value labels
    for bar, val, orig_val in zip(bars, values, [detection_rate, avg_score, avg_error, avg_fps]):
        height = bar.get_height()
        if orig_val == avg_error:
            label_text = f'{orig_val:.1f}px'
        elif orig_val == avg_fps:
            label_text = f'{orig_val:.2f}'
        else:
            label_text = f'{orig_val:.3f}'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                label_text, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax5.set_ylim([0, 1.1])
    ax5.set_ylabel('Score (normalized)', fontsize=10)
    ax5.set_title('Performance Summary', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return summary metrics for aggregation
    return {
        'detection_rate': detection_rate,
        'avg_score': avg_score,
        'avg_error': avg_error,
        'avg_fps': avg_fps,
        'num_frames': len(detected)
    }


def plot_aggregate_summary(all_track_metrics: List[Dict[str, Any]],
                          all_labels: List[np.ndarray],
                          all_scores: List[np.ndarray],
                          output_path: str) -> None:
    """Create summary plots across all tracks."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Plot 1: Mean Performance Metrics ---
    ax1 = axes[0, 0]
    
    # Aggregate metrics
    detection_rates = [m['detection_rate'] for m in all_track_metrics]
    avg_scores = [m['avg_score'] for m in all_track_metrics]
    avg_errors = [m['avg_error'] for m in all_track_metrics if m['avg_error'] != np.inf]
    avg_fps = [m['avg_fps'] for m in all_track_metrics]
    
    metrics_names = ['Detection\nRate', 'Avg\nScore', 'Avg Error\n(px)', 'Avg FPs\nper Frame']
    means = [np.mean(detection_rates), np.mean(avg_scores), 
             np.mean(avg_errors) if avg_errors else 0, np.mean(avg_fps)]
    stds = [np.std(detection_rates), np.std(avg_scores),
            np.std(avg_errors) if avg_errors else 0, np.std(avg_fps)]
    
    bars = ax1.bar(metrics_names, means, yerr=stds, capsize=5,
                   color=['blue', 'green', 'purple', 'orange'], alpha=0.7)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Mean Performance Across All Tracks', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 2: PR Curve ---
    ax2 = axes[0, 1]
    
    # Combine all labels and scores
    combined_labels = np.concatenate(all_labels)
    combined_scores = np.concatenate(all_scores)
    
    if len(combined_labels) > 0 and np.sum(combined_labels) > 0:
        precision, recall, _ = precision_recall_curve(combined_labels, combined_scores)
        avg_precision = average_precision_score(combined_labels, combined_scores)
        
        ax2.plot(recall, precision, 'b-', linewidth=2.5, label=f'AP = {avg_precision:.3f}')
        ax2.fill_between(recall, precision, alpha=0.2)
        
        ax2.set_xlabel('Recall', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title('Precision-Recall Curve (All Tracks)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1.05])
        ax2.set_ylim([0, 1.05])
    
    # --- Plot 3: Metric Distributions ---
    ax3 = axes[1, 0]
    
    box_data = [detection_rates, avg_scores, 
                [e / 20 for e in avg_errors] if avg_errors else [0],  # Normalize to 0-1
                [min(fp / 5, 1) for fp in avg_fps]]  # Normalize to 0-1
    box_labels = ['Detection\nRate', 'Avg\nScore', 'Avg Error\n(norm)', 'Avg FPs\n(norm)']
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    colors = ['blue', 'green', 'purple', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Metric Distributions Across Tracks', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 4: Summary Statistics ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}
    
    Number of Tracks: {len(all_track_metrics)}
    Total Frames: {sum(m['num_frames'] for m in all_track_metrics)}
    
    Detection Rate: {np.mean(detection_rates):.3f} ± {np.std(detection_rates):.3f}
    Range: [{min(detection_rates):.3f}, {max(detection_rates):.3f}]
    
    Avg Detection Score: {np.mean(avg_scores):.3f} ± {np.std(avg_scores):.3f}
    Range: [{min(avg_scores):.3f}, {max(avg_scores):.3f}]
    
    Avg Position Error: {np.mean(avg_errors) if avg_errors else 0:.2f} ± {np.std(avg_errors) if avg_errors else 0:.2f} px
    Range: [{min(avg_errors) if avg_errors else 0:.2f}, {max(avg_errors) if avg_errors else 0:.2f}] px
    
    Avg False Positives: {np.mean(avg_fps):.2f} ± {np.std(avg_fps):.2f} per frame
    Range: [{min(avg_fps):.2f}, {max(avg_fps):.2f}]
    
    Average Precision (AP): {avg_precision:.3f}
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_multiple_tracks(tiff_dir: str, csv_dir: str, output_dir: str,
                            pipeline_function,
                            distance_threshold: float = 30.0,
                            parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Evaluate tracking performance across multiple TIFF/CSV pairs.
    
    Args:
        tiff_dir: Directory containing TIFF files
        csv_dir: Directory containing ground truth CSV files
        output_dir: Directory to save results
        pipeline_function: Function to run detection/tracking pipeline
        distance_threshold: Distance threshold for matching (pixels)
        parameters: Pipeline parameters
    """
    from detection_metrics import load_ground_truth_track, match_track_to_ground_truth
    
    tiff_dir = Path(tiff_dir)
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIFF files
    tiff_files = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    
    print(f"\nFound {len(tiff_files)} TIFF files")
    
    all_track_metrics = []
    all_labels = []
    all_scores = []
    
    for tiff_file in tiff_files:
        # Find corresponding CSV (match first 20 characters)
        tiff_prefix = tiff_file.stem[:20]
        csv_candidates = [f for f in csv_dir.glob("*.csv") if f.stem[:20] == tiff_prefix]
        
        if not csv_candidates:
            print(f"Warning: No CSV found matching first 20 chars of {tiff_file.name}, skipping...")
            print(f"  Looking for CSV starting with: '{tiff_prefix}'")
            continue
        
        csv_file = csv_candidates[0]
        print(f"\n{'='*80}")
        print(f"Processing: {tiff_file.name}")
        print(f"Ground Truth: {csv_file.name}")
        print(f"{'='*80}")
        
        try:
            # Run pipeline
            results = pipeline_function(
                tiff_file=str(tiff_file),
                output_dir=None,
                parameters=parameters,
                ground_truth_csv=str(csv_file)
            )
            
            if not results['success']:
                print(f"Pipeline failed for {tiff_file.name}")
                continue
            
            # Extract results
            all_particles = results['stage_results']['detection']['all_particles']
            tracks = results['stage_results']['tracking']
            
            # Load ground truth
            gt_track = load_ground_truth_track(str(csv_file))
            
            # Get track dict
            if isinstance(tracks, dict) and 'tracks' not in tracks:
                # tracks is already the track dict from tracking module
                track_dict = tracks
            elif isinstance(tracks, dict) and 'tracks' in tracks:
                # tracks is results dict containing track dict
                track_dict = tracks['tracks']
            else:
                print(f"Warning: Unexpected tracks format for {tiff_file.name}")
                track_dict = {}
                
            # Match track
            matched_track_id, _ = match_track_to_ground_truth(
                track_dict, gt_track, distance_threshold=distance_threshold
            )
            
            # Generate per-track plot
            plot_path = output_dir / f"{tiff_file.stem}_track_performance.png"
            track_metrics = plot_single_track_performance(
                all_particles=all_particles,
                tracks=track_dict,
                gt_track=gt_track,
                matched_track_id=matched_track_id,
                output_path=str(plot_path),
                filename=tiff_file.name,
                distance_threshold=distance_threshold
            )
            
            all_track_metrics.append(track_metrics)
            
            # Calculate labels and scores for PR curve
            labels, scores = calculate_detection_labels_and_scores(
                all_particles, gt_track, distance_threshold
            )
            all_labels.append(labels)
            all_scores.append(scores)
            
            print(f"✓ Completed: Detection Rate = {track_metrics['detection_rate']:.3f}")
            
        except Exception as e:
            print(f"Error processing {tiff_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate aggregate summary
    if all_track_metrics:
        summary_path = output_dir / "aggregate_summary.png"
        plot_aggregate_summary(all_track_metrics, all_labels, all_scores, str(summary_path))
        
        # Save summary CSV
        summary_df = pd.DataFrame(all_track_metrics)
        summary_df['filename'] = [f.stem for f in tiff_files[:len(all_track_metrics)]]
        summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Tracks evaluated: {len(all_track_metrics)}")
        print(f"Results saved to: {output_dir}")
        print(f"Mean detection rate: {np.mean([m['detection_rate'] for m in all_track_metrics]):.3f}")
    
    return {
        'track_metrics': all_track_metrics,
        'output_dir': str(output_dir)
    }