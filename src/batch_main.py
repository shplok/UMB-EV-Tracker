import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

# Import pipeline components
from main import run_ev_detection_pipeline
from metrics.detection_metrics import load_ground_truth_track

def calculate_detection_labels_for_file(all_particles, gt_track, distance_threshold=20.0):
    """
    Helper to calculate labels/scores for a SINGLE file.
    Exact copy of logic from compute_pr_roc.py to ensure consistency.
    """
    labels = []
    scores = []
    
    gt_frames = set(gt_track['frames'])
    gt_positions = np.array(gt_track['positions'])
    gt_frame_to_idx = {frame: idx for idx, frame in enumerate(gt_track['frames'])}
    
    # Track which GT frames were successfully detected
    detected_gt_frames = set()
    
    # Process all detections
    for frame, particles in all_particles.items():
        if not particles['positions']:
            continue
        
        det_positions = np.array(particles['positions'])
        det_scores = np.array(particles['scores'])
        
        if frame in gt_frames:
            # Frame has ground truth - find best match
            gt_pos = gt_positions[gt_frame_to_idx[frame]]
            distances = np.sqrt(np.sum((det_positions - gt_pos)**2, axis=1))
            
            # Find the BEST (closest) detection
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            
            # Label the best match
            if best_dist <= distance_threshold:
                labels.append(1)  # True Positive
                scores.append(det_scores[best_idx])
                detected_gt_frames.add(frame)
            else:
                labels.append(0)  # False Positive (detected but too far)
                scores.append(det_scores[best_idx])
            
            # All OTHER detections in this frame are False Positives
            for i in range(len(det_positions)):
                if i != best_idx:
                    labels.append(0)
                    scores.append(det_scores[i])
        else:
            # No ground truth in this frame - all detections are False Positives
            for score in det_scores:
                labels.append(0)
                scores.append(score)
    
    # CRITICAL: Add False Negatives as lowest-confidence "detections"
    for frame in gt_frames:
        if frame not in detected_gt_frames:
            labels.append(1)  # This is a positive sample we failed to detect
            scores.append(0.0)  # Assign lowest possible score
            
    return labels, scores

def run_global_batch_analysis(dataset_list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_output_dir = os.path.join("UMB-EV-Tracker", "out", "global_metrics", f"run_{timestamp}")
    os.makedirs(global_output_dir, exist_ok=True)

    print(f"Starting Global Analysis on {len(dataset_list)} files...")
    print(f"Output Directory: {global_output_dir}\n")

    # 1. Containers for Aggregation
    global_labels = []
    global_scores = []
    file_summaries = []

    # 2. Fixed Parameters (Must use very low threshold to capture curve!)
    BATCH_PARAMS = {
        'filter_radius': 10,
        'filter_size': 41,
        'filter_sigma': 2.0,
        'bg_window_size': 15,
        'blur_kernel_size': 5,
        'clahe_clip_limit': 2.0,
        'clahe_grid_size': (8, 8),
        # CRITICAL: Use low threshold (0.1) so we can see the full curve behavior
        # The PR curve calculation will filter this later.
        'detection_threshold': 0.1, 
        'min_distance': 30,
        'max_distance': 25,
        'min_track_length': 5,
        'max_frame_gap': 3,
        'num_sample_frames': 6,
        'num_top_tracks': 5
    }

    # 3. Process Files
    for tiff_file, csv_file in dataset_list:
        filename = os.path.basename(tiff_file)
        print(f"Processing: {filename}...")
        
        if not os.path.exists(tiff_file) or not os.path.exists(csv_file):
            print("  Skipping (File not found)")
            continue

        # Run pipeline
        results = run_ev_detection_pipeline(
            tiff_file=tiff_file,
            output_dir=None, # Auto-generate
            parameters=BATCH_PARAMS,
            ground_truth_csv=csv_file
        )

        if results['success']:
            # Extract raw data for global metrics
            all_particles = results['stage_results']['detection']['all_particles']
            gt_track = load_ground_truth_track(csv_file)
            
            # Calculate labels for THIS file
            labels, scores = calculate_detection_labels_for_file(
                all_particles, gt_track, distance_threshold=30.0
            )
            
            # Add to global lists
            global_labels.extend(labels)
            global_scores.extend(scores)
            
            # Store summary
            metrics = results['stage_results'].get('pr_roc', {}).get('pr_roc_data', {})
            file_summaries.append({
                'filename': filename,
                'file_ap': metrics.get('avg_precision', 0),
                'total_detections': len(scores)
            })
            print(f"  > Added {len(scores)} data points to global pool.")

    # 4. Compute Global Metrics
    print("\n" + "="*60)
    print("COMPUTING GLOBAL STATISTICS")
    print("="*60)

    if not global_labels:
        print("No data collected.")
        return

    g_labels = np.array(global_labels)
    g_scores = np.array(global_scores)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(g_labels, g_scores)
    global_ap = average_precision_score(g_labels, g_scores)

    # ROC Curve
    fpr, tpr, _ = roc_curve(g_labels, g_scores)
    global_auc = auc(fpr, tpr)

    print(f"Global Micro-Average AP:  {global_ap:.4f}")
    print(f"Global Micro-Average AUC: {global_auc:.4f}")
    print(f"Total Data Points:        {len(g_labels)}")

    # 5. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PR Curve
    axes[0].plot(recall, precision, 'b-', linewidth=3, label=f'Global AP={global_ap:.3f}')
    axes[0].set_title(f'Global Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].legend(loc='lower left')
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    axes[1].plot(fpr, tpr, 'r-', linewidth=3, label=f'Global AUC={global_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_title(f'Global ROC Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(global_output_dir, "global_performance_curves.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plots to: {plot_path}")

    # 6. Save Data
    # Save the raw curve data for custom plotting later
    curve_data = pd.DataFrame({'Recall': recall, 'Precision': precision})
    curve_data.to_csv(os.path.join(global_output_dir, "global_pr_curve_data.csv"), index=False)
    
    # Save file summaries
    pd.DataFrame(file_summaries).to_csv(os.path.join(global_output_dir, "file_summaries.csv"), index=False)


if __name__ == "__main__":
    # Define your dataset here
    DATASET = [
        (
            r"UMB-EV-Tracker/data/tiff/new/xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2_MMStack_Pos0.ome.tif",
            r"UMB-EV-Tracker/data/csv/new/xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2.csv"
        ),
        (
            r"UMB-EV-Tracker\data\tiff\new\xslot_BT747_PT00_xp1_1500uLhr_z40um_mov_6_flush_adj_MMStack_Pos0.ome.tif",
            r"UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_BT747_PT00_xp1_1500uLhr_z40um_mov_6_flush_adj_MMStack_Pos0.ome.csv"
        ),
        (
            r"UMB-EV-Tracker\data\tiff\new\xslot_HCC1954_PT03_xp4_1250uLhr_z40um_mov_1_MMStack_Pos0.ome.tif",
            r"UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_HCC1954_PT03_xp4_1250uLhr_z40um_mov_1_MMStack_Pos0.ome.csv"
        ),
        (
            r"UMB-EV-Tracker\data\tiff\new\xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1_MMStack_Pos0.ome.tif",
            r"UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1.csv"
        ),
        (
            r"UMB-EV-Tracker\data\tiff\xslot_BT747_03_1000uLhr_z35um_adjSP_mov_2_MMStack_Pos0.ome.tif",
            r"UMB-EV-Tracker\data\csv\xslot_BT747_03_1000uLhr_z35um_adjSP_mov_2.csv"
        ),
        (
            r"UMB-EV-Tracker\data\tiff\xslot_HCC1954_01_500uLhr_z35um_mov_1_MMStack_Pos0.ome.tif",
            r"UMB-EV-Tracker\data\csv\xslot_HCC1954_01_500uLhr_z35um_mov_1.csv"
        ),
        (
            r"UMB-EV-Tracker\data\tiff\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif",
            r"UMB-EV-Tracker\data\csv\xslot_HCC1954_01_500uLhr_z40um_mov_1.csv"
        ),
    ]
    
    run_global_batch_analysis(DATASET)