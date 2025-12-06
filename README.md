# EV Tracker - Extracellular Vesicle Detection and Tracking

A simplified, object-oriented interface for analyzing extracellular vesicle (EV) movements in microscopy image sequences.

## Overview

EV Tracker provides automated detection, tracking, and analysis of extracellular vesicles in TIFF image stacks. The pipeline includes:

- **Background subtraction** - Temporal median filtering
- **Enhancement** - CLAHE contrast enhancement and noise reduction
- **Detection** - Template matching with correlation-based particle detection
- **Tracking** - Frame-to-frame particle linking with gap handling
- **Metrics** - Precision-recall curves, ROC analysis, and performance metrics

## Quick Start

```python
from ev_tracker import EVTracker

# Create tracker
tracker = EVTracker()

# Set parameters
tracker.set_params(threshold=0.55, min_distance=30)

# Single file analysis
results = tracker.run("movie.tif", "ground_truth.csv")
print(f"Average Precision: {results['global_ap']:.3f}")

# Batch analysis
datasets = [("movie1.tif", "gt1.csv"), ("movie2.tif", "gt2.csv")]
results = tracker.run_batch(datasets)
print(f"Global AP: {results['global_ap']:.3f}")
```

## Installation

### Requirements

- Python 3.7+
- NumPy
- OpenCV
- Matplotlib
- Pandas
- SciPy
- scikit-learn
- tifffile
- tqdm

### Setup

1. Clone or download this repository

2. Install dependencies:
   ```bash
   cd UMB-EV-Tracker
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "from src.ev_tracker import EVTracker; print('✓ Ready!')"
   ```

## Important: Running Location

**All Python scripts must be run from the project root directory (`UMB-EV-Tracker/`), NOT from the `src/` directory.**

### Correct:
```bash
cd UMB-EV-Tracker/
python src/test_all_features.py
python -c "from src.ev_tracker import EVTracker; tracker = EVTracker()"
```

### Incorrect:
```bash
cd UMB-EV-Tracker/src/
python test_all_features.py  # This will fail with import errors!
```

This is because the code uses imports like `from helpers.batch_main import ...` which expect `src/` to be in the Python path, which only works when running from the parent directory.

## Usage

### Basic Single File Analysis

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()
tracker.set_params(threshold=0.55, min_distance=30)
results = tracker.run("path/to/movie.tif", "path/to/ground_truth.csv")
```

### Batch Analysis

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()
datasets = [
    ("movie1.tif", "ground_truth1.csv"),
    ("movie2.tif", "ground_truth2.csv"),
    ("movie3.tif", "ground_truth3.csv")
]

results = tracker.run_batch(datasets)
print(f"Global Average Precision: {results['global_ap']:.3f}")
```

### Quick Analysis (One-liner)

```python
from src.ev_tracker import quick_analyze

results = quick_analyze("movie.tif", "ground_truth.csv", threshold=0.6)
```

### Method Chaining

```python
from src.ev_tracker import EVTracker

results = (EVTracker()
          .set_params(threshold=0.55, min_distance=30, max_distance=25)
          .run("movie.tif", "ground_truth.csv"))
```

## Understanding `run()` vs `run_batch()`

### CRITICAL DIFFERENCE

**`run()`** and **`run_batch()`** behave differently with the threshold parameter:

| Method | Threshold Behavior | Use When |
|--------|-------------------|----------|
| **`run()`** | Uses the specified threshold | Single file, want specific threshold results |
| **`run_batch()`** | Overrides threshold to 0.1 | Multiple files, want comprehensive PR curves |

### `run()` - Uses Your Threshold

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()
tracker.set_params(threshold=0.6)  # This WILL be used

results = tracker.run("movie.tif", "ground_truth.csv")
# Detections made at threshold=0.6
# Results reflect performance at YOUR chosen threshold
```

**Use `run()` when:**
- Testing a single file
- You want results at a specific threshold
- Doing parameter tuning
- You care about the exact threshold value

### `run_batch()` - Overrides to 0.1 for PR Curves

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()
tracker.set_params(threshold=0.6)  # This will be IGNORED

datasets = [
    ("movie1.tif", "gt1.csv"),
    ("movie2.tif", "gt2.csv")
]

results = tracker.run_batch(datasets)
# !!! Threshold is overridden to 0.1 for comprehensive PR curve
# Captures ALL possible detections (even weak ones)
# Computes PR/ROC curves across the full threshold range
```

**Use `run_batch()` when:**
- Analyzing multiple files
- You want comprehensive PR/ROC curves showing performance at ALL thresholds
- Publication-quality metrics
- You need global performance metrics across a dataset

### Why the Difference?

To compute an accurate **Precision-Recall curve**, you need:
1. ALL possible detections (even weak ones with low confidence)
2. Their confidence scores
3. Then vary the threshold POST-HOC to plot precision vs recall

If you only detect at threshold=0.6, you miss all detections scored 0.1-0.59, and your PR curve is incomplete.

**Solution:**
- `run()` - Normal detection at YOUR threshold
- `run_batch()` - Get ALL detections (threshold=0.1) to compute full PR curve

### Quick Reference

```python
# Use YOUR threshold (0.6)
tracker.set_params(threshold=0.6)
tracker.run("movie.tif", "gt.csv")  # Uses 0.6

# Override to 0.1 for comprehensive PR curves
tracker.set_params(threshold=0.6)
tracker.run_batch([("movie.tif", "gt.csv")])  # Uses 0.1
```

**Note:** All other parameters (min_distance, max_distance, filter_radius, etc.) are respected by BOTH methods.

## Parameters

### All Available Parameters

EVTracker exposes **13 configurable parameters** organized into 4 categories:

#### Detection Parameters

| Parameter | Type | Default | Description | Recommended Range |
|-----------|------|---------|-------------|-------------------|
| `threshold` | float | 0.55 | Detection confidence (0-1). Higher = fewer detections | 0.4-0.7 |
| `min_distance` | int | 30 | Minimum separation between particles (pixels) | 20-40 |
| `filter_radius` | int | 10 | Expected particle radius (pixels) | 5-15 |
| `filter_size` | int | 41 | Size of detection filter matrix (pixels) | 21-61 (odd) |
| `filter_sigma` | float | 2.0 | Gaussian smoothing for filter | 1.0-4.0 |

#### Background & Enhancement Parameters

| Parameter | Type | Default | Description | Recommended Range |
|-----------|------|---------|-------------|-------------------|
| `bg_window_size` | int | 15 | Temporal window for background subtraction (frames) | 5-30 |
| `blur_kernel_size` | int | 7 | Noise reduction kernel size (pixels) | 3-15 (odd) |
| `clahe_clip_limit` | float | 2.0 | Contrast enhancement limit | 1.0-4.0 |
| `clahe_grid_size` | tuple | (8, 8) | Contrast enhancement tile size (width, height) | (4,4)-(16,16) |

#### Tracking Parameters

| Parameter | Type | Default | Description | Recommended Range |
|-----------|------|---------|-------------|-------------------|
| `max_distance` | int | 25 | Maximum particle movement per frame (pixels) | 15-40 |
| `min_track_length` | int | 5 | Minimum frames to keep a track | 3-15 |
| `max_frame_gap` | int | 3 | Maximum gap in frames for a track | 1-10 |

#### Metrics Parameters

| Parameter | Type | Default | Description | Recommended Range |
|-----------|------|---------|-------------|-------------------|
| `distance_threshold` | float | 30.0 | Distance threshold for metrics evaluation (pixels) | 15-50 |

### Setting Parameters

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()

# Set basic parameters (most common)
tracker.set_params(
    threshold=0.55,
    min_distance=30,
    max_distance=25
)

# Set advanced parameters
tracker.set_params(
    filter_size=41,
    filter_sigma=2.0,
    blur_kernel_size=7,
    clahe_clip_limit=2.0,
    clahe_grid_size=(8, 8)
)

# Set tracking parameters
tracker.set_params(
    min_track_length=5,
    max_frame_gap=3,
    distance_threshold=30.0
)

# Or set everything at once
tracker.set_params(
    # Detection
    threshold=0.55,
    min_distance=30,
    filter_radius=10,
    filter_size=41,
    filter_sigma=2.0,
    # Background & Enhancement
    bg_window_size=15,
    blur_kernel_size=7,
    clahe_clip_limit=2.0,
    clahe_grid_size=(8, 8),
    # Tracking
    max_distance=25,
    min_track_length=5,
    max_frame_gap=3,
    # Metrics
    distance_threshold=30.0
)

# View current parameters
tracker.print_params()
```

## Output

### Results Dictionary

```python
results = {
    'success': True,                 # Whether analysis completed
    'global_ap': 0.82,              # Average Precision (PR curve)
    'global_auc': 0.88,             # ROC AUC
    'total_points': 5432,           # Total detections analyzed
    'output_dir': 'path/to/results', # Output directory
    'file_summaries': [...]         # Per-file metrics (list of dicts)
}
```

### Output Files

Each analysis creates timestamped directories with:

**Visualizations:**
- Detection overlays with particle positions
- Track paths showing particle movement
- Performance plots (PR curves, ROC curves)
- Enhanced frame comparisons

**Data Exports:**
- `*_all_detections.csv` - All detections with coordinates, confidence, and track IDs
- `track_summaries.csv` - Per-track statistics (length, velocity, etc.)
- `threshold_analysis.csv` - Performance at different thresholds

**Metrics:**
- Precision-recall curves
- ROC curves
- Per-file performance summaries
- Global metrics across all files (for batch analysis)

### Output Directory Structure

```
UMB-EV-Tracker/
├── out/
│   ├── global_metrics/
│   │   └── run_20241205_143022/
│   │       ├── global_performance_curves.png
│   │       ├── global_pr_curve_data.csv
│   │       └── file_summaries.csv
│   └── ev_detection_results_20241205_143022/
│       ├── 02_filter_creation/
│       ├── 03_background_subtraction/
│       ├── 04_enhancement/
│       ├── 05_detection/
│       ├── 06_tracking/
│       └── 07_metrics/
```

## Parameter Tuning

### Detection Sensitivity

```python
# More sensitive (finds dim particles, more false positives)
tracker.set_params(threshold=0.45)

# Less sensitive (fewer false positives, may miss dim particles)
tracker.set_params(threshold=0.65)
```

### Particle Size

```python
# For larger particles (~25px diameter)
tracker.set_params(filter_radius=12, min_distance=40)

# For smaller particles (~10px diameter)
tracker.set_params(filter_radius=5, min_distance=20)
```

### Tracking Behavior

```python
# Fast-moving particles
tracker.set_params(max_distance=35, max_frame_gap=5)

# Slow-moving particles
tracker.set_params(max_distance=15, max_frame_gap=2)
```

## Performance Metrics

### Available Metrics

- **Detection Rate**: Percentage of ground truth frames with successful detection
- **Position Error**: Average distance between detections and ground truth (pixels)
- **Average Precision (AP)**: Area under precision-recall curve (0-1)
- **ROC AUC**: Area under receiver operating characteristic curve (0-1)
- **Track F1 Score**: Harmonic mean of track precision and recall

### Interpreting Results

**Average Precision (AP):**
- **AP > 0.8**: Excellent detection performance
- **AP 0.6-0.8**: Good detection performance
- **AP < 0.6**: Consider parameter tuning

**Position Error:**
- **< 10px**: High accuracy
- **10-20px**: Moderate accuracy
- **> 20px**: Review detection threshold or verify frame alignment with ground truth

**Detection Rate:**
- **> 75%**: Good tracking
- **50-75%**: Moderate tracking
- **< 50%**: Poor tracking, adjust parameters

## Examples

### Example 1: Basic Analysis

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()
results = tracker.run(
    tiff_file="data/tiff/movie.tif",
    ground_truth_csv="data/csv/ground_truth.csv"
)

if results['success']:
    print(f"Average Precision: {results['global_ap']:.3f}")
    print(f"ROC AUC: {results['global_auc']:.3f}")
    print(f"Output saved to: {results['output_dir']}")
```

### Example 2: Parameter Sweep

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()
thresholds = [0.4, 0.5, 0.6, 0.7]

best_ap = 0
best_thresh = None

for thresh in thresholds:
    tracker.set_params(threshold=thresh)
    results = tracker.run("movie.tif", "ground_truth.csv")
    ap = results['global_ap']
    
    print(f"Threshold {thresh}: AP = {ap:.3f}")
    
    if ap > best_ap:
        best_ap = ap
        best_thresh = thresh

print(f"\nBest threshold: {best_thresh} (AP = {best_ap:.3f})")
```

### Example 3: Batch Processing

```python
from src.ev_tracker import EVTracker
import glob

# Find all TIFF files
tiff_files = sorted(glob.glob("data/tiff/*.tif"))
csv_files = sorted(glob.glob("data/csv/*.csv"))

# Create dataset list (ensure matching pairs)
datasets = list(zip(tiff_files, csv_files))

# Run batch analysis
tracker = EVTracker()
tracker.set_params(threshold=0.55, min_distance=30)
results = tracker.run_batch(datasets)

print(f"Global AP: {results['global_ap']:.3f}")
print(f"Global AUC: {results['global_auc']:.3f}")
print(f"Processed {len(datasets)} files")
print(f"Total data points: {results['total_points']}")
```

### Example 4: Custom Output Directory

```python
from src.ev_tracker import EVTracker

tracker = EVTracker(output_dir="my_results/experiment_1")
tracker.set_params(threshold=0.55, min_distance=30)
results = tracker.run("movie.tif", "ground_truth.csv")
```

## Testing

Run the comprehensive test suite:

```bash
# From project root (UMB-EV-Tracker/)
python src/test_all_features.py
```

The test suite includes:
- Import and initialization tests
- Parameter setting and validation
- Single file analysis
- Batch analysis
- Method chaining
- Parameter sweeps

**Note:** Update the file paths in `test_all_features.py` (`TEST_TIFF`, `TEST_CSV`, `BATCH_DATASET`) to point to your actual data files before running tests.

## Troubleshooting

### Common Issues

**1. Import Error**
```
ImportError: No module named 'src'
```
**Solution:** Make sure you're running from the project root directory:
```bash
cd UMB-EV-Tracker/
python src/test_all_features.py
```

**2. Too Many False Positives**
```python
# Increase threshold to be more selective
tracker.set_params(threshold=0.65)
```

**3. Missing Particles**
```python
# Decrease threshold to detect dimmer particles
tracker.set_params(threshold=0.45)
```

**4. Fragmented Tracks**
```python
# Allow larger movements and more frame gaps
tracker.set_params(max_distance=35, max_frame_gap=5)
```

**5. File Not Found Errors**
```python
# Use absolute paths or verify relative paths
import os
tiff_path = os.path.abspath("data/tiff/movie.tif")
csv_path = os.path.abspath("data/csv/ground_truth.csv")
```

**6. Memory Issues with Large TIFF Stacks**
- Process files individually instead of batch
- Reduce `bg_window_size` parameter
- Close other applications to free memory

## API Reference

### EVTracker Class

```python
class EVTracker(output_dir="../out")
```

Initialize the tracker with optional output directory.

**Parameters:**
- `output_dir` (str): Directory for output files. Default: `"../out"` (relative to `src/`)

#### Methods

**`set_params(**kwargs) -> EVTracker`**

Set pipeline parameters. Returns `self` for method chaining.

**All Available Parameters (13 total):**

*Detection:*
- `threshold` (float): Detection confidence threshold (0.0-1.0)
- `min_distance` (int): Minimum separation between particles (pixels)
- `filter_radius` (int): Expected particle radius (pixels)
- `filter_size` (int): Size of detection filter matrix (pixels, must be odd)
- `filter_sigma` (float): Gaussian smoothing for filter

*Background & Enhancement:*
- `bg_window_size` (int): Temporal window for background subtraction (frames)
- `blur_kernel_size` (int): Noise reduction kernel size (pixels, must be odd)
- `clahe_clip_limit` (float): Contrast enhancement limit
- `clahe_grid_size` (tuple): Contrast enhancement tile size (width, height)

*Tracking:*
- `max_distance` (int): Maximum particle movement per frame (pixels)
- `min_track_length` (int): Minimum frames for valid track
- `max_frame_gap` (int): Maximum gap in frames for a track

*Metrics:*
- `distance_threshold` (float): Distance threshold for metrics evaluation (pixels)

**Example:**
```python
tracker.set_params(
    threshold=0.55, 
    min_distance=30,
    filter_size=41,
    clahe_clip_limit=2.0
)
```

**`run(tiff_file: str, ground_truth_csv: str = None) -> Dict[str, Any]`**

Run analysis on a single TIFF file **using your specified threshold**.

**Parameters:**
- `tiff_file` (str): Path to TIFF image stack
- `ground_truth_csv` (str, optional): Path to ground truth CSV file

**Threshold Behavior:**
- **Uses YOUR configured threshold** (e.g., 0.55, 0.6, etc.)
- All other parameters are also respected

**Returns:** Dictionary with keys:
- `success` (bool): Whether analysis completed successfully
- `global_ap` (float): Average Precision at your threshold
- `global_auc` (float): ROC AUC based on detection scores
- `total_points` (int): Total detections
- `output_dir` (str): Output directory path
- `file_summaries` (list): Per-file metrics

**Example:**
```python
tracker = EVTracker()
tracker.set_params(threshold=0.6)  # This threshold WILL be used

results = tracker.run("movie.tif", "ground_truth.csv")
print(f"AP at threshold 0.6: {results['global_ap']:.3f}")
```

**`run_batch(dataset_list: List[Tuple[str, str]]) -> Dict[str, Any]`**

Run batch analysis with global metrics across multiple files.

** IMPORTANT:** This method **overrides your threshold to 0.1** to capture all possible detections for computing comprehensive PR/ROC curves across the full threshold range.

**Parameters:**
- `dataset_list` (list): List of (tiff_file, csv_file) tuples

**Threshold Behavior:**
- **Threshold is OVERRIDDEN to 0.1** (ignores your set_params threshold)
- All other parameters (min_distance, max_distance, etc.) ARE respected
- This is intentional to generate comprehensive PR curves

**Returns:** Same as `run()` but with global metrics aggregated across all files

**Example:**
```python
tracker = EVTracker()
tracker.set_params(
    threshold=0.6,      # ⚠️ This will be IGNORED (overridden to 0.1)
    min_distance=30,    # ✅ This WILL be used
    max_distance=25     # ✅ This WILL be used
)

datasets = [
    ("movie1.tif", "gt1.csv"),
    ("movie2.tif", "gt2.csv")
]

results = tracker.run_batch(datasets)
print(f"Global AP: {results['global_ap']:.3f}")  # Based on threshold=0.1
```

**Why the override?**  
To compute accurate Precision-Recall curves, we need ALL possible detections (even weak ones) with their confidence scores. Using threshold=0.1 ensures we capture the full range of detections, then PR/ROC curves show performance at all thresholds.

**`print_params()`**

Display current parameter settings in a formatted table.

### Functions

**`quick_analyze(tiff_file: str, ground_truth_csv: str = None, threshold: float = 0.55) -> Dict[str, Any]`**

Quick one-liner analysis with minimal setup.

**Parameters:**
- `tiff_file` (str): Path to TIFF file
- `ground_truth_csv` (str, optional): Path to ground truth
- `threshold` (float): Detection threshold

**Returns:** Results dictionary (same as `run()`)

**Example:**
```python
from src.ev_tracker import quick_analyze
results = quick_analyze("movie.tif", "ground_truth.csv", threshold=0.6)
```

## Ground Truth Format

Ground truth CSV files should have the following columns:

- `Slice`: Frame number (1-indexed)
- `X_COM`: X coordinate (pixels)
- `Y_COM`: Y coordinate (pixels)
- `EV_ID` (optional): Particle ID for tracking multiple particles

**Example CSV:**
```csv
Slice,X_COM,Y_COM,EV_ID
1,123.4,456.7,1
2,125.1,458.3,1
3,126.8,460.2,1
```

## Project Structure

```
UMB-EV-Tracker/
├── src/
│   ├── ev_tracker.py          # Main API interface
│   ├── test_all_features.py   # Test suite
│   ├── requirements.txt       # Dependencies
│   ├── helpers/
│   │   ├── main.py           # Core pipeline
│   │   └── batch_main.py     # Batch processing
│   ├── metrics/
│   │   ├── detection_metrics.py
│   │   ├── compute_pr_roc.py
│   │   └── improved_visualizations.py
│   └── pipeline/
│       ├── image_loader.py
│       ├── filter_creation.py
│       ├── background_subtraction.py
│       ├── enhancement.py
│       ├── detection.py
│       ├── tracking.py
│       └── export_results.py
├── data/
│   ├── tiff/                 # TIFF image stacks
│   └── csv/                  # Ground truth files
├── out/                      # Output directory
└── README.md
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Citation

If you use this software in your research, please cite:

```
EV Tracker - Extracellular Vesicle Detection and Tracking
University of Massachusetts Boston
2024-2025
```

## License

[Add your license here]

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: s.bowerman.cs@gmail.com

## Acknowledgments

Developed for extracellular vesicle tracking research at the University of Massachusetts Boston.

Special thanks to the UMass Boston Computer Science and Biology departments for their support.

---

**Last Updated:** December 2024  
**Version:** 1.0  
**Python:** 3.7+