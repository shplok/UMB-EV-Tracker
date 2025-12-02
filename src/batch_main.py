import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

# Import your existing pipeline
from main import run_ev_detection_pipeline
from metrics.detection_metrics import load_ground_truth_track
from metrics.compute_pr_roc import calculate_detection_labels_and_scores

def run_batch_processing(dataset_list, base_output_dir="out/global_results"):
    """
    Runs the pipeline on multiple files and computes global metrics.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_output_dir = os.path.join("UMB-EV-Tracker", base_output_dir, f"batch_run_{timestamp}")
    os.makedirs(global_output_dir, exist_ok=True)

    # 1. Containers for Global Aggregation
    # -----------------------------------
    # To compute a "Global" curve, we need every single detection decision 
    # from every file in one big array.
    global_labels = [] # 1 for TP, 0 for FP
    global_scores = [] # Confidence scores
    
    # To compute mAP (Macro-average), we track individual APs
    file_metrics = []

    print(f"Starting Batch Processing on {len(dataset_list)} files...")
    print(f"Global Output: {global_output_dir}\n")

    # 2. Parameters (Global settings for this batch)
    # -----------------------------------
    # You can tweak these once here for the whole batch
    BATCH_PARAMETERS = {
        'filter_radius': 10,
        'filter_size': 41,
        'filter_sigma': 2.0,
        'bg_window_size': 15,
        'blur_kernel_size': 5,
        'clahe_clip_limit': 2.0,
        'clahe_grid_size': (8, 8),
        'detection_threshold': 0.55,
        'min_distance': 30,
        'max_distance': 25,
        'min_track_length': 5,
        'max_frame_gap': 3,
        'num_sample_frames': 6,
        'num_top_tracks': 5
    }

    # 3. Loop through files
    # -----------------------------------
    for i, (tiff_file, csv_file) in enumerate(dataset_list):
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE {i+1}/{len(dataset_list)}: {os.path.basename(tiff_file)}")
        print(f"{'='*60}")

        if not os.path.exists(tiff_file) or not os.path.exists(csv_file):
            print(f"Skipping {tiff_file} (File not found)")
            continue

        # Run the standard pipeline for this file
        # We assume specific output dirs aren't needed per file, 
        # but we let the pipeline auto-generate them inside 'out/'
        file_results = run_ev_detection_pipeline(
            tiff_file=tiff_file,
            output_dir=None, # Let it auto-generate specific timestamped folder
            parameters=BATCH_PARAMETERS,
            ground_truth_csv=csv_file
        )

        if not file_results['success']:
            print(f"Failed to process {tiff_file}")
            continue

        # 4. Extract Data for Global Metrics
        # -----------------------------------
        # We need to dig into the results to get the raw data for aggregation
        try:
            # Load GT helper
            gt_track = load_ground_truth_track(csv_file)
            all_particles = file_results['stage_results']['detection']['all_particles']
            
            # Re-calculate labels/scores for this specific file using your helper
            # (We do this to ensure we get the raw lists for concatenation)
            labels, scores = calculate_detection_labels_and_scores(
                all_particles, 
                gt_track, 
                distance_threshold=30.0 # Match metrics threshold
            )
            
            # Aggregate for Micro-Average (Global Curve)
            global_labels.extend(labels)
            global_scores.extend(scores)

            # Store for Macro-Average (mAP)
            # We look at the 'metrics' stage result from the pipeline
            pr_roc_data = file_results['stage_results'].get('pr_roc', {}).get('pr_roc_data', {})
            
            file_metrics.append({
                'filename': os.path.basename(tiff_file),
                'ap': pr_roc_data.get('avg_precision', 0),
                'auc': pr_roc_data.get('roc_auc', 0),
                'f1': pr_roc_data.get('optimal_f1', 0),
                'detections': len(scores),
                'output_dir': file_results['output_dir']
            })

        except Exception as e:
            print(f"Error extracting metrics for global batch: {e}")

    # 5. Compute Global Statistics
    # -----------------------------------
    print("\n" + "="*60)
    print("COMPUTING GLOBAL METRICS")
    print("="*60)

    if not global_labels:
        print("No valid metrics collected.")
        return

    global_labels_arr = np.array(global_labels)
    global_scores_arr = np.array(global_scores)

    # A. Global Micro-Averaged Curves (Treating all videos as one giant video)
    gl_precision, gl_recall, _ = precision_recall_curve(global_labels_arr, global_scores_arr)
    gl_ap = average_precision_score(global_labels_arr, global_scores_arr)
    
    gl_fpr, gl_tpr, _ = roc_curve(global_labels_arr, global_scores_arr)
    gl_auc = auc(gl_fpr, gl_tpr)

    # B. Mean Average Precision (mAP) - Macro Average
    # Average of the individual AP scores
    aps = [m['ap'] for m in file_metrics]
    mAP = np.mean(aps) if aps else 0

    print(f"Global Micro-Average AP: {gl_ap:.4f}")
    print(f"Global Micro-Average AUC: {gl_auc:.4f}")
    print(f"Mean Average Precision (mAP): {mAP:.4f}")

    # 6. Visualization
    # -----------------------------------
    plot_global_performance(
        global_output_dir, 
        gl_recall, gl_precision, gl_ap,
        gl_fpr, gl_tpr, gl_auc,
        file_metrics,
        mAP
    )

    # 7. Save Summary CSV
    # -----------------------------------
    df = pd.DataFrame(file_metrics)
    df.loc['Mean'] = df.mean(numeric_only=True)
    df.at['Mean', 'filename'] = 'GLOBAL MEAN'
    csv_path = os.path.join(global_output_dir, 'batch_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved batch summary to: {csv_path}")


def plot_global_performance(output_dir, recall, precision, gl_ap, fpr, tpr, gl_auc, file_metrics, map_score):
    """
    Plots the Global curve overlaid with faint individual curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: PR Curve ---
    ax1 = axes[0]
    
    # Plot individual curves (faintly) - we don't have the raw arrays for individual files 
    # stored in 'file_metrics' to save memory, but we can plot the points as scatter or just skip.
    # Instead, let's plot the Global Curve Boldly.
    
    ax1.plot(recall, precision, 'k-', linewidth=3, label=f'Global Micro-Avg (AP={gl_ap:.3f})')
    ax1.set_title(f'Global Precision-Recall\n(mAP = {map_score:.3f})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])

    # --- Plot 2: Per-File Performance Bar Chart ---
    ax2 = axes[1]
    filenames = [m['filename'] for m in file_metrics]
    # Shorten filenames for display
    filenames = [f[:15]+"..." if len(f)>15 else f for f in filenames]
    
    aps = [m['ap'] for m in file_metrics]
    
    bars = ax2.bar(filenames, aps, color='steelblue', alpha=0.8)
    ax2.axhline(map_score, color='red', linestyle='--', label=f'Mean AP: {map_score:.3f}')
    
    ax2.set_title('AP Score per Dataset', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Precision (AP)')
    ax2.set_xticklabels(filenames, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0, 1.1])
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_performance.png"), dpi=300)
    plt.close()


if __name__ == "__main__":

    DATASET = [
        # Newer Files
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
    
    run_batch_processing(DATASET)