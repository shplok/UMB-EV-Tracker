import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # Add this import

def compute_ap_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Sort by recall (ascending)
    df = df.sort_values(by="recall")

    # Update to use trapezoid instead of deprecated trapz
    ap = np.trapezoid(df["precision"], df["recall"])
    return ap

csv_files = glob.glob(r"UMB_EV_Tracker\out\*\07_metrics\threshold_analysis.csv")
aps = [compute_ap_from_csv(f) for f in csv_files]
map_score = np.mean(aps)
output_dir = r"UMB_EV_Tracker\out\mAP_results"
os.makedirs(output_dir, exist_ok=True)

# Create the plot
plt.figure(figsize=(8, 6))

for f in csv_files:
    df = pd.read_csv(f)
    df = df.sort_values(by="recall")
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
    plt.plot(df["recall"], df["precision"], label=dataset_name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curves (mAP = {map_score:.3f})")
plt.legend(title="Dataset", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_path = os.path.join(output_dir, "precision_recall_curves.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to: {output_path}")