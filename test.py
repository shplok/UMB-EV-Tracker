from src.ev_tracker import quick_analyze
from src.ev_tracker import EVTracker

tracker = EVTracker()
results = tracker.run_batch(
    tiff_file=r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif",
    ground_truth_csv=r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1.csv"
)

if results['success']:
    print(f"Average Precision: {results['global_ap']:.3f}")
    print(f"ROC AUC: {results['global_auc']:.3f}")
    print(f"Output saved to: {results['output_dir']}")

# results = quick_analyze(r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif", 
#                         r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1.csv", 
#                         threshold=0.6)

# thresholds = [0.4, 0.5, 0.6, 0.7]

# best_ap = 0
# best_thresh = None

# for thresh in thresholds:
#     tracker.set_params(threshold=thresh)
#     results = tracker.run(tiff_file=r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif",
#                         ground_truth_csv=r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1.csv")
#     ap = results['global_ap']
    
#     print(f"Threshold {thresh}: AP = {ap:.3f}")
    
#     if ap > best_ap:
#         best_ap = ap
#         best_thresh = thresh

# print(f"\nBest threshold: {best_thresh} (AP = {best_ap:.3f})")