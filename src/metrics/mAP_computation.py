import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_ap_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Sort by recall (ascending)
    df = df.sort_values(by="recall")

    # Compute area under Precision–Recall curve
    ap = np.trapz(df["precision"], df["recall"])
    return ap

csv_files = glob.glob(r"UMB-EV-Tracker\out\*\07_metrics\threshold_analysis.csv")
aps = [compute_ap_from_csv(f) for f in csv_files]
map_score = np.mean(aps)

print(f"mAP across {len(aps)} datasets: {map_score:.4f}")

plt.figure(figsize=(8, 6))

for f in csv_files:
    df = pd.read_csv(f)
    df = df.sort_values(by="recall")
    plt.plot(df["recall"], df["precision"], label=f"{f.split('/')[-2]}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curves (mAP = {map_score:.3f})")
plt.legend(title="Dataset", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
