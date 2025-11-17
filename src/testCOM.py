import numpy as np
import cv2
import pandas as pd
import os
from tifffile import imread
from detection import detect_particles_in_all_frames  # import your detection function

def load_ground_truth(csv_path: str, frame_column: str = "Slice"):
    df = pd.read_csv(csv_path)
    if frame_column in df.columns:
        gt_positions = {int(f_idx): list(zip(group['X_COM'], group['Y_COM']))
                        for f_idx, group in df.groupby(frame_column)}
    else:
        gt_positions = {0: list(zip(df['X_COM'], df['Y_COM']))}
    return gt_positions

def overlay_com_vs_detections(
    enhanced_frames: np.ndarray,
    ev_filter: np.ndarray,
    ground_truth_csv: str,
    output_dir: str,
    num_examples: int = 2,
    detection_threshold: float = 0.58,
    min_distance: int = 30
):
    os.makedirs(output_dir, exist_ok=True)

    # Run detection
    all_particles = detect_particles_in_all_frames(
        enhanced_frames,
        ev_filter,
        threshold=detection_threshold,
        min_distance=min_distance
    )

    # Load CSV COMs
    gt_positions = load_ground_truth(ground_truth_csv)

    # Pick frames with both detections and GT
    candidate_frames = [idx for idx in range(len(enhanced_frames))
                        if idx in all_particles and idx in gt_positions]
    selected_frames = candidate_frames[:num_examples]

    for frame_idx in selected_frames:
        frame = cv2.cvtColor(enhanced_frames[frame_idx], cv2.COLOR_GRAY2RGB)

        # Ground truth COM (green)
        for x_gt, y_gt in gt_positions[frame_idx]:
            cv2.circle(frame, (int(x_gt), int(y_gt)), 10, (0, 255, 0), 2)
            cv2.putText(frame, "GT", (int(x_gt)+5, int(y_gt)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Detected particles (red/yellow)
        for (x_det, y_det), score in zip(all_particles[frame_idx]['positions'], all_particles[frame_idx]['scores']):
            color = (255, 0, 0) if score > 0.6 else (255, 255, 0)
            cv2.circle(frame, (int(x_det), int(y_det)), 10, color, 2)
            cv2.putText(frame, f"{score:.2f}", (int(x_det)+5, int(y_det)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        output_path = os.path.join(output_dir, f"frame_{frame_idx}_overlay.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved overlay for frame {frame_idx} -> {output_path}")

    return all_particles
