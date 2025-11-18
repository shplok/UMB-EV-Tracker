"""
Visual Overlay Tool: Ground Truth COM vs Detected Positions

This module creates side-by-side visualizations showing:
- Ground truth particle centers (from CSV)
- Detected particle positions (from detection algorithm)
- Match quality assessment

Author: Created for EV tracking validation
"""

import numpy as np
import cv2
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def load_ground_truth(csv_path: str, frame_column: str = "Slice") -> Dict[int, List[Tuple[float, float]]]:
    """
    Load ground truth center-of-mass positions from CSV.
    
    Args:
        csv_path: Path to ground truth CSV file
        frame_column: Column name containing frame numbers (default: "Slice")
    
    Returns:
        Dictionary mapping frame_idx -> list of (x, y) positions
    """
    df = pd.read_csv(csv_path)
    
    print(f"Loading ground truth from: {csv_path}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Total rows: {len(df)}")
    
    if frame_column not in df.columns:
        raise ValueError(f"Frame column '{frame_column}' not found in CSV. Available: {df.columns.tolist()}")
    
    # Group by frame and extract positions
    gt_positions = {}
    for frame_idx, group in df.groupby(frame_column):
        positions = list(zip(group['X_COM'].values, group['Y_COM'].values))
        gt_positions[int(frame_idx)] = positions
    
    print(f"  Loaded GT for {len(gt_positions)} frames")
    print(f"  Frame range: {min(gt_positions.keys())} to {max(gt_positions.keys())}")
    
    return gt_positions


def overlay_com_vs_detections(
    enhanced_frames: np.ndarray,
    all_particles: Dict[int, Dict[str, List]],
    ground_truth_csv: str,
    output_dir: str,
    num_examples: int = 5,
    distance_threshold: float = 20.0,
    use_matplotlib: bool = True
):
    """
    Create visual overlays of ground truth vs detected positions.
    
    Args:
        enhanced_frames: Array of enhanced frames [N, H, W]
        all_particles: Detection results from detect_particles_in_all_frames
        ground_truth_csv: Path to ground truth CSV
        output_dir: Directory to save overlay images
        num_examples: Number of example frames to visualize
        distance_threshold: Max distance to consider detection as matching GT
        use_matplotlib: If True, use matplotlib for better visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating COM overlay visualizations...")
    print(f"  Enhanced frames shape: {enhanced_frames.shape}")
    print(f"  Total detection frames: {len(all_particles)}")
    
    # Load ground truth
    gt_positions = load_ground_truth(ground_truth_csv)
    
    # Find frames with both detections and ground truth
    candidate_frames = []
    for idx in range(len(enhanced_frames)):
        has_gt = idx in gt_positions and len(gt_positions[idx]) > 0
        has_det = idx in all_particles and len(all_particles[idx]['positions']) > 0
        if has_gt and has_det:
            candidate_frames.append(idx)
    
    print(f"  Candidate frames (with both GT and detections): {len(candidate_frames)}")
    
    if len(candidate_frames) == 0:
        print("  WARNING: No frames with both GT and detections found!")
        print(f"  GT frames: {sorted(gt_positions.keys())[:10]}...")
        print(f"  Detection frames: {sorted(all_particles.keys())[:10]}...")
        return
    
    # Select example frames (evenly spaced)
    if len(candidate_frames) <= num_examples:
        selected_frames = candidate_frames
    else:
        indices = np.linspace(0, len(candidate_frames)-1, num_examples, dtype=int)
        selected_frames = [candidate_frames[i] for i in indices]
    
    print(f"  Selected frames for visualization: {selected_frames}")
    
    # Create overlays
    stats = {
        'total_gt': 0,
        'total_detections': 0,
        'matched': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    for frame_idx in selected_frames:
        if use_matplotlib:
            create_matplotlib_overlay(
                frame_idx, enhanced_frames[frame_idx],
                gt_positions[frame_idx],
                all_particles[frame_idx],
                output_dir, distance_threshold, stats
            )
        else:
            create_opencv_overlay(
                frame_idx, enhanced_frames[frame_idx],
                gt_positions[frame_idx],
                all_particles[frame_idx],
                output_dir, distance_threshold, stats
            )
    
    # Print summary statistics
    print(f"\nOverlay Statistics (across {len(selected_frames)} frames):")
    print(f"  Total GT positions: {stats['total_gt']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Matched (TP): {stats['matched']}")
    print(f"  False Positives: {stats['false_positives']}")
    print(f"  False Negatives: {stats['false_negatives']}")
    if stats['total_gt'] > 0:
        print(f"  Detection Rate: {stats['matched']/stats['total_gt']*100:.1f}%")
    
    # Create summary figure
    create_summary_figure(selected_frames, enhanced_frames, gt_positions, 
                         all_particles, output_dir, distance_threshold)
    
    print(f"\nOverlays saved to: {output_dir}")


def create_matplotlib_overlay(
    frame_idx: int,
    frame: np.ndarray,
    gt_positions: List[Tuple[float, float]],
    detections: Dict[str, List],
    output_dir: str,
    distance_threshold: float,
    stats: Dict
):
    """Create overlay using matplotlib for better quality."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Ground Truth only
    ax1 = axes[0]
    ax1.imshow(frame, cmap='gray')
    for x_gt, y_gt in gt_positions:
        circle = Circle((x_gt, y_gt), 10, color='lime', fill=False, linewidth=2)
        ax1.add_patch(circle)
        ax1.plot(x_gt, y_gt, 'g+', markersize=10, markeredgewidth=2)
    ax1.set_title(f'Frame {frame_idx}: Ground Truth ({len(gt_positions)} particles)', fontsize=12)
    ax1.axis('off')
    
    # Panel 2: Detections only
    ax2 = axes[1]
    ax2.imshow(frame, cmap='gray')
    det_positions = detections['positions']
    det_scores = detections['scores']
    for (x_det, y_det), score in zip(det_positions, det_scores):
        color = 'red' if score > 0.6 else 'yellow'
        circle = Circle((x_det, y_det), 10, color=color, fill=False, linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x_det+12, y_det-12, f'{score:.2f}', color=color, fontsize=8, 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax2.set_title(f'Detections ({len(det_positions)} particles)', fontsize=12)
    ax2.axis('off')
    
    # Panel 3: Overlay with matching
    ax3 = axes[2]
    ax3.imshow(frame, cmap='gray')
    
    # Match detections to GT
    matched_gt = set()
    matched_det = set()
    
    for i, (x_gt, y_gt) in enumerate(gt_positions):
        stats['total_gt'] += 1
        
        # Find closest detection
        min_dist = float('inf')
        best_match = None
        for j, (x_det, y_det) in enumerate(det_positions):
            dist = np.sqrt((x_gt - x_det)**2 + (y_gt - y_det)**2)
            if dist < min_dist:
                min_dist = dist
                best_match = j
        
        # Draw GT
        if best_match is not None and min_dist <= distance_threshold:
            # Matched - green
            circle = Circle((x_gt, y_gt), 10, color='lime', fill=False, linewidth=2)
            matched_gt.add(i)
            matched_det.add(best_match)
            stats['matched'] += 1
            
            # Draw line to detection
            x_det, y_det = det_positions[best_match]
            ax3.plot([x_gt, x_det], [y_gt, y_det], 'g-', linewidth=1, alpha=0.5)
            ax3.text((x_gt+x_det)/2, (y_gt+y_det)/2, f'{min_dist:.1f}px', 
                    color='lime', fontsize=7, fontweight='bold')
        else:
            # Unmatched GT - magenta (False Negative)
            circle = Circle((x_gt, y_gt), 10, color='magenta', fill=False, linewidth=2)
            stats['false_negatives'] += 1
        
        ax3.add_patch(circle)
        ax3.plot(x_gt, y_gt, '+', color='white', markersize=8, markeredgewidth=2)
    
    # Draw unmatched detections (False Positives)
    for j, ((x_det, y_det), score) in enumerate(zip(det_positions, det_scores)):
        stats['total_detections'] += 1
        if j not in matched_det:
            stats['false_positives'] += 1
            circle = Circle((x_det, y_det), 10, color='cyan', fill=False, linewidth=2, linestyle='--')
            ax3.add_patch(circle)
            ax3.text(x_det+12, y_det+12, 'FP', color='cyan', fontsize=8, fontweight='bold')
    
    ax3.set_title(f'Overlay (Green=TP, Magenta=FN, Cyan=FP)', fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_overlay.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_opencv_overlay(
    frame_idx: int,
    frame: np.ndarray,
    gt_positions: List[Tuple[float, float]],
    detections: Dict[str, List],
    output_dir: str,
    distance_threshold: float,
    stats: Dict
):
    """Create overlay using OpenCV (faster but lower quality)."""
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    det_positions = detections['positions']
    det_scores = detections['scores']
    
    # Match detections to GT
    matched_det = set()
    
    for x_gt, y_gt in gt_positions:
        stats['total_gt'] += 1
        
        # Find closest detection
        min_dist = float('inf')
        best_match = None
        for j, (x_det, y_det) in enumerate(det_positions):
            dist = np.sqrt((x_gt - x_det)**2 + (y_gt - y_det)**2)
            if dist < min_dist:
                min_dist = dist
                best_match = j
        
        # Draw GT position
        if best_match is not None and min_dist <= distance_threshold:
            # Matched - green
            cv2.circle(frame_rgb, (int(x_gt), int(y_gt)), 12, (0, 255, 0), 2)
            matched_det.add(best_match)
            stats['matched'] += 1
            
            # Draw line to detection
            x_det, y_det = det_positions[best_match]
            cv2.line(frame_rgb, (int(x_gt), int(y_gt)), (int(x_det), int(y_det)), (0, 255, 0), 1)
        else:
            # Unmatched - magenta
            cv2.circle(frame_rgb, (int(x_gt), int(y_gt)), 12, (255, 0, 255), 2)
            stats['false_negatives'] += 1
        
        cv2.putText(frame_rgb, "GT", (int(x_gt)+5, int(y_gt)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw unmatched detections
    for j, ((x_det, y_det), score) in enumerate(zip(det_positions, det_scores)):
        stats['total_detections'] += 1
        if j not in matched_det:
            stats['false_positives'] += 1
            color = (0, 255, 255)  # Cyan for FP
            cv2.circle(frame_rgb, (int(x_det), int(y_det)), 12, color, 2)
            cv2.putText(frame_rgb, f"FP {score:.2f}", (int(x_det)+5, int(y_det)+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_overlay.png")
    cv2.imwrite(output_path, frame_rgb)
    print(f"  Saved: {output_path}")


def create_summary_figure(
    selected_frames: List[int],
    enhanced_frames: np.ndarray,
    gt_positions: Dict,
    all_particles: Dict,
    output_dir: str,
    distance_threshold: float
):
    """Create a summary figure with all selected frames."""
    
    n_frames = len(selected_frames)
    fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))
    
    if n_frames == 1:
        axes = axes.reshape(2, 1)
    
    for i, frame_idx in enumerate(selected_frames):
        frame = enhanced_frames[frame_idx]
        gt_pos = gt_positions[frame_idx]
        det_pos = all_particles[frame_idx]['positions']
        det_scores = all_particles[frame_idx]['scores']
        
        # Top row: Ground truth
        axes[0, i].imshow(frame, cmap='gray')
        for x, y in gt_pos:
            axes[0, i].plot(x, y, 'g+', markersize=10, markeredgewidth=2)
            circle = Circle((x, y), 10, color='lime', fill=False, linewidth=1.5)
            axes[0, i].add_patch(circle)
        axes[0, i].set_title(f'Frame {frame_idx}\nGT: {len(gt_pos)}', fontsize=10)
        axes[0, i].axis('off')
        
        # Bottom row: Detections
        axes[1, i].imshow(frame, cmap='gray')
        for (x, y), score in zip(det_pos, det_scores):
            color = 'red' if score > 0.6 else 'yellow'
            axes[1, i].plot(x, y, 'x', color=color, markersize=8, markeredgewidth=2)
            circle = Circle((x, y), 10, color=color, fill=False, linewidth=1.5)
            axes[1, i].add_patch(circle)
        axes[1, i].set_title(f'Detections: {len(det_pos)}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "summary_overlay.png")
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary figure saved: {summary_path}")


# Standalone execution example
if __name__ == "__main__":
    print("This module provides COM overlay visualization tools.")
    print("Import and use overlay_com_vs_detections() in your pipeline.")