"""
Simple Detection Metrics Module for EV Detection Pipeline

Calculates mAP (mean Average Precision) and AUC (Area Under Curve) for detection performance
using ground truth data from CSV files.

Functions:
- load_ground_truth_csv(): Load ground truth from CSV
- calculate_map(): Calculate mean Average Precision
- calculate_precision_recall_curve(): Generate precision-recall data
- evaluate_detections(): Main evaluation function
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import cdist
import os


def load_ground_truth_csv(csv_path: str) -> Dict[int, List[Tuple[float, float]]]:
    """
    Load ground truth from CSV file
    
    Supports multiple CSV formats:
    - Format 1: Slice, X_COM, Y_COM columns
    - Format 2: Slice, X_COM, Y_COM with other columns (Area, Mean, etc.)
    
    Args:
        csv_path: Path to ground truth CSV file
    
    Returns:
        Dictionary mapping frame indices to list of (x, y) positions
    """
    print(f"Loading ground truth from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check which columns are available
    required_cols = ['Slice', 'X_COM', 'Y_COM']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}. "
                        f"Available columns: {list(df.columns)}")
    
    print(f"CSV format detected with columns: {list(df.columns)}")
    
    # Group by Slice (frame number) and extract positions
    ground_truth = {}
    
    for slice_num in df['Slice'].unique():
        frame_particles = df[df['Slice'] == slice_num]
        positions = list(zip(frame_particles['X_COM'], frame_particles['Y_COM']))
        ground_truth[slice_num] = positions
    
    total_particles = len(df)
    num_frames = len(ground_truth)
    
    print(f"Loaded {total_particles} ground truth particles across {num_frames} frames")
    print(f"Average particles per frame: {total_particles/num_frames:.1f}")
    
    # Print frame range for verification
    frame_range = f"{min(ground_truth.keys())} to {max(ground_truth.keys())}"
    print(f"Frame range: {frame_range}")
    
    return ground_truth


def match_detections_to_ground_truth(detections: List[Tuple[float, float]],
                                     detection_scores: List[float],
                                     ground_truth: List[Tuple[float, float]],
                                     distance_threshold: float = 10.0) -> Tuple[List[bool], List[float]]:
    """
    Match detections to ground truth using distance threshold
    
    Args:
        detections: List of (x, y) detection positions
        detection_scores: Confidence scores for each detection
        ground_truth: List of (x, y) ground truth positions
        distance_threshold: Maximum distance for a match (pixels)
    
    Returns:
        (match_flags, scores) where match_flags[i] is True if detection i matched GT
    """
    if len(detections) == 0:
        return [], []
    
    if len(ground_truth) == 0:
        # No ground truth, all detections are false positives
        return [False] * len(detections), detection_scores
    
    # Calculate distance matrix
    det_array = np.array(detections)
    gt_array = np.array(ground_truth)
    distances = cdist(det_array, gt_array, metric='euclidean')
    
    # Match each detection to nearest GT if within threshold
    matched_gt = set()
    match_flags = []
    
    for i in range(len(detections)):
        min_dist_idx = np.argmin(distances[i])
        min_dist = distances[i, min_dist_idx]
        
        if min_dist <= distance_threshold and min_dist_idx not in matched_gt:
            match_flags.append(True)
            matched_gt.add(min_dist_idx)
        else:
            match_flags.append(False)
    
    return match_flags, detection_scores


def calculate_precision_recall_curve(all_particles: Dict[int, Dict[str, List]],
                                     ground_truth: Dict[int, List[Tuple[float, float]]],
                                     distance_threshold: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve by varying score threshold
    
    Returns:
        (precisions, recalls, thresholds)
    """
    print("Calculating precision-recall curve...")
    
    # Collect all detections across all frames with their scores
    all_detections = []
    
    for frame_idx, frame_data in all_particles.items():
        positions = frame_data['positions']
        scores = frame_data['scores']
        gt = ground_truth.get(frame_idx, [])
        
        # Match detections to ground truth
        match_flags, det_scores = match_detections_to_ground_truth(
            positions, scores, gt, distance_threshold
        )
        
        for pos, score, is_match in zip(positions, det_scores, match_flags):
            all_detections.append({
                'score': score,
                'is_true_positive': is_match,
                'frame': frame_idx
            })
    
    # Sort detections by score (descending)
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate total number of ground truth objects
    total_gt = sum(len(gt_list) for gt_list in ground_truth.values())
    
    # Calculate precision and recall at each threshold
    precisions = []
    recalls = []
    thresholds = []
    
    tp_cumsum = 0
    fp_cumsum = 0
    
    for i, detection in enumerate(all_detections):
        if detection['is_true_positive']:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt if total_gt > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(detection['score'])
    
    return np.array(precisions), np.array(recalls), np.array(thresholds)


def calculate_average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using the 11-point interpolation method
    """
    # Use 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def calculate_auc(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Area Under Curve using trapezoidal rule
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls_sorted = recalls[sorted_indices]
    precisions_sorted = precisions[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(precisions_sorted, recalls_sorted)
    
    return auc


def calculate_map(all_particles: Dict[int, Dict[str, List]],
                 ground_truth: Dict[int, List[Tuple[float, float]]],
                 distance_threshold: float = 10.0,
                 iou_thresholds: List[float] = None) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP)
    
    For detection tasks, we use distance threshold instead of IoU.
    mAP is calculated as AP averaged over multiple thresholds.
    """
    print(f"Calculating mAP with distance threshold: {distance_threshold}px")
    
    if iou_thresholds is None:
        # Use single threshold for simplicity
        iou_thresholds = [distance_threshold]
    
    aps = []
    
    for threshold in iou_thresholds:
        precisions, recalls, _ = calculate_precision_recall_curve(
            all_particles, ground_truth, threshold
        )
        
        if len(precisions) > 0:
            ap = calculate_average_precision(precisions, recalls)
            aps.append(ap)
            print(f"  AP @ {threshold}px: {ap:.4f}")
    
    map_score = np.mean(aps) if aps else 0.0
    
    return {
        'mAP': map_score,
        'AP_per_threshold': dict(zip(iou_thresholds, aps))
    }


def evaluate_detections(all_particles: Dict[int, Dict[str, List]],
                       ground_truth_csv: str,
                       output_dir: str,
                       distance_threshold: float = 10.0,
                       visualize: bool = True) -> Dict[str, Any]:
    """
    Main evaluation function - calculates mAP and AUC
    
    Args:
        all_particles: Detection results from pipeline
        ground_truth_csv: Path to ground truth CSV file
        output_dir: Directory to save results
        distance_threshold: Maximum distance for matching (pixels)
        visualize: Whether to create visualizations
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*80)
    print("DETECTION EVALUATION - mAP and AUC")
    print("="*80 + "\n")
    
    # Load ground truth
    ground_truth = load_ground_truth_csv(ground_truth_csv)
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = calculate_precision_recall_curve(
        all_particles, ground_truth, distance_threshold
    )
    
    # Calculate metrics
    map_results = calculate_map(all_particles, ground_truth, distance_threshold)
    auc = calculate_auc(recalls, precisions)
    
    # Calculate additional metrics at best threshold
    if len(thresholds) > 0:
        # Find threshold with best F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
        best_f1 = f1_scores[best_idx]
    else:
        best_threshold = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
    
    results = {
        'mAP': map_results['mAP'],
        'AUC': auc,
        'best_threshold': best_threshold,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_f1': best_f1,
        'precisions': precisions,
        'recalls': recalls,
        'thresholds': thresholds,
        'distance_threshold': distance_threshold
    }
    
    # Print results
    print("\nRESULTS:")
    print("-"*80)
    print(f"mAP (mean Average Precision): {map_results['mAP']:.4f}")
    print(f"AUC (Area Under PR Curve):    {auc:.4f}")
    print(f"\nBest Operating Point (max F1):")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  Precision: {best_precision:.3f}")
    print(f"  Recall:    {best_recall:.3f}")
    print(f"  F1 Score:  {best_f1:.3f}")
    
    # Create visualizations
    if visualize:
        viz_path = visualize_pr_curve(results, output_dir)
        results['visualization_path'] = viz_path
        
        # Save detailed report
        report_path = save_evaluation_report(results, ground_truth, all_particles, output_dir)
        results['report_path'] = report_path
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")
    
    return results


def visualize_pr_curve(results: Dict[str, Any], output_dir: str) -> str:
    """
    Create precision-recall curve visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Precision-Recall Curve
    ax1 = axes[0]
    ax1.plot(results['recalls'], results['precisions'], 'b-', linewidth=2, label='PR Curve')
    ax1.scatter([results['best_recall']], [results['best_precision']], 
               c='red', s=100, zorder=5, label=f'Best F1 (threshold={results["best_threshold"]:.3f})')
    ax1.fill_between(results['recalls'], results['precisions'], alpha=0.3)
    
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Precision-Recall Curve\nAUC = {results["AUC"]:.4f}, mAP = {results["mAP"]:.4f}', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # F1 Score vs Threshold
    ax2 = axes[1]
    if len(results['thresholds']) > 0:
        f1_scores = 2 * (results['precisions'] * results['recalls']) / \
                   (results['precisions'] + results['recalls'] + 1e-10)
        
        ax2.plot(results['thresholds'], f1_scores, 'g-', linewidth=2, label='F1 Score')
        ax2.plot(results['thresholds'], results['precisions'], 'b--', alpha=0.7, label='Precision')
        ax2.plot(results['thresholds'], results['recalls'], 'r--', alpha=0.7, label='Recall')
        ax2.axvline(results['best_threshold'], color='black', linestyle=':', 
                   linewidth=2, label=f'Best threshold: {results["best_threshold"]:.3f}')
        
        ax2.set_xlabel('Detection Threshold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title(f'Metrics vs Threshold\nBest F1 = {results["best_f1"]:.3f}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
    
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'detection_evaluation.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {viz_path}")
    return viz_path


def save_evaluation_report(results: Dict[str, Any],
                          ground_truth: Dict[int, List[Tuple[float, float]]],
                          all_particles: Dict[int, Dict[str, List]],
                          output_dir: str) -> str:
    """
    Save detailed evaluation report to text file
    """
    report_path = os.path.join(output_dir, 'detection_evaluation_report.txt')
    
    # Calculate frame-by-frame statistics
    total_gt = sum(len(gt_list) for gt_list in ground_truth.values())
    total_detections = sum(len(frame_data['positions']) for frame_data in all_particles.values())
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETECTION EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"mAP (mean Average Precision): {results['mAP']:.4f}\n")
        f.write(f"AUC (Area Under PR Curve):    {results['AUC']:.4f}\n")
        f.write(f"Distance Threshold:           {results['distance_threshold']:.1f} pixels\n\n")
        
        f.write("BEST OPERATING POINT (Maximum F1 Score)\n")
        f.write("-"*80 + "\n")
        f.write(f"Detection Threshold: {results['best_threshold']:.3f}\n")
        f.write(f"Precision:           {results['best_precision']:.3f}\n")
        f.write(f"Recall:              {results['best_recall']:.3f}\n")
        f.write(f"F1 Score:            {results['best_f1']:.3f}\n\n")
        
        f.write("DETECTION STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Ground Truth Particles: {total_gt}\n")
        f.write(f"Total Detections:             {total_detections}\n")
        f.write(f"Detection/GT Ratio:           {total_detections/total_gt:.2f}\n")
        f.write(f"Frames Evaluated:             {len(ground_truth)}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*80 + "\n")
        if results['mAP'] > 0.8:
            f.write("Excellent detection performance (mAP > 0.8)\n")
        elif results['mAP'] > 0.6:
            f.write("Good detection performance (mAP > 0.6)\n")
        elif results['mAP'] > 0.4:
            f.write("Moderate detection performance (mAP > 0.4)\n")
        else:
            f.write("Poor detection performance (mAP < 0.4)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Evaluation report saved: {report_path}")
    return report_path