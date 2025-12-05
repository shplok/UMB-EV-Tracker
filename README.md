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

**⚠️ All Python scripts must be run from the project root directory (`UMB-EV-Tracker/`), NOT from the `src/` directory.**

### ✅ Correct:
```bash
cd UMB-EV-Tracker/
python src/test_all_features.py
python -c "from src.ev_tracker import EVTracker; tracker = EVTracker()"
```

### ❌ Incorrect:
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

## Parameters

### Detection Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `threshold` | Detection confidence (0-1) | 0.55 | 0.4-0.7 |
| `min_distance` | Minimum particle separation (pixels) | 30 | 20-40 |
| `filter_radius` | Expected particle radius (pixels) | 10 | 5-15 |
| `bg_window_size` | Background subtraction window (frames) | 15 | 5-30 |

### Tracking Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `max_distance` | Maximum movement per frame (pixels) | 25 | 15-40 |
| `min_track_length` | Minimum frames for valid track | 5 | 3-15 |
| `max_frame_gap` | Maximum missing frames in track | 3 | 1-10 |

### Setting Parameters

```python
from src.ev_tracker import EVTracker

tracker = EVTracker()

# Set individual parameters
tracker.set_params(threshold=0.6)
tracker.set_params(min_distance=35)

# Set multiple parameters at once
tracker.set_params(
    threshold=0.55,
    min_distance=30,
    max_distance=25,
    min_track_length=5
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

**Available Parameters:**
- `threshold` (float): Detection confidence threshold (0.0-1.0)
- `min_distance` (int): Minimum separation between particles (pixels)
- `max_distance` (int): Maximum particle movement per frame (pixels)
- `min_track_length` (int): Minimum frames for valid track
- `filter_radius` (int): Expected particle radius (pixels)
- `bg_window_size` (int): Background subtraction temporal window (frames)

**Example:**
```python
tracker.set_params(threshold=0.55, min_distance=30)
```

**`run(tiff_file: str, ground_truth_csv: str = None) -> Dict[str, Any]`**

Run analysis on a single TIFF file.

**Parameters:**
- `tiff_file` (str): Path to TIFF image stack
- `ground_truth_csv` (str, optional): Path to ground truth CSV file

**Returns:** Dictionary with keys:
- `success` (bool): Whether analysis completed successfully
- `global_ap` (float): Average Precision
- `global_auc` (float): ROC AUC
- `total_points` (int): Total detections
- `output_dir` (str): Output directory path
- `file_summaries` (list): Per-file metrics

**`run_batch(dataset_list: List[Tuple[str, str]]) -> Dict[str, Any]`**

Run batch analysis with global metrics across multiple files.

**Parameters:**
- `dataset_list` (list): List of (tiff_file, csv_file) tuples

**Returns:** Same as `run()` but with global metrics across all files

**Example:**
```python
datasets = [
    ("movie1.tif", "gt1.csv"),
    ("movie2.tif", "gt2.csv")
]
results = tracker.run_batch(datasets)
```

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
