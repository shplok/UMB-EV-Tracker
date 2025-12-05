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
- NumPy >= 1.19.0
- OpenCV >= 4.5.0
- Matplotlib >= 3.3.0
- Pandas >= 1.1.0
- SciPy >= 1.5.0
- scikit-learn >= 0.24.0
- tifffile >= 2020.9.3
- tqdm >= 4.50.0 (optional, for progress bars)

### Setup

1. Clone or download this repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy opencv-python matplotlib pandas scipy scikit-learn tifffile tqdm
   ```

3. Verify installation:
   ```bash
   python -c "from ev_tracker import EVTracker; print('✓ Ready!')"
   ```

## Usage

### Basic Single File Analysis

```python
from ev_tracker import EVTracker

tracker = EVTracker()
tracker.set_params(threshold=0.55, min_distance=30)
results = tracker.run("path/to/movie.tif", "path/to/ground_truth.csv")
```

### Batch Analysis

```python
from ev_tracker import EVTracker

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
from ev_tracker import quick_analyze

results = quick_analyze("movie.tif", "ground_truth.csv", threshold=0.6)
```

### Method Chaining

```python
results = (EVTracker()
          .set_params(threshold=0.55, min_distance=30, max_distance=25)
          .run("movie.tif", "ground_truth.csv"))
```

## Parameters

### Detection Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `threshold` | Detection confidence (0-1) | 0.55 | 0.0-1.0 |
| `min_distance` | Minimum particle separation (pixels) | 30 | 20-40 |
| `filter_radius` | Expected particle radius (pixels) | 10 | 5-15 |
| `bg_window_size` | Background subtraction window (frames) | 15 | 5-30 |

### Tracking Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_distance` | Maximum movement per frame (pixels) | 25 | 15-40 |
| `min_track_length` | Minimum frames for valid track | 5 | 3-15 |
| `max_frame_gap` | Maximum missing frames in track | 3 | 1-10 |

### Setting Parameters

```python
tracker = EVTracker()

# Set individual parameters
tracker.set_params(threshold=0.6)
tracker.set_params(min_distance=35)

# Set multiple parameters
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
    'file_summaries': [...]         # Per-file metrics
}
```

### Output Files

Each analysis creates:

- **Visualizations**: Detection overlays, tracking paths, performance plots
- **CSV Exports**: All detections with coordinates and confidence scores
- **Metrics**: Precision-recall curves, ROC curves, performance summaries
- **Videos**: Detection visualization videos (optional)

Files are organized in timestamped directories:
```
UMB-EV-Tracker/out/
├── global_metrics/
│   └── run_20241205_143022/
│       ├── global_performance_curves.png
│       ├── global_pr_curve_data.csv
│       └── file_summaries.csv
└── ev_detection_results_20241205_143022/
    ├── 02_filter_creation/
    ├── 03_background_subtraction/
    ├── 04_enhancement/
    ├── 05_detection/
    ├── 06_tracking/
    └── 07_metrics/
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
# For larger particles (~25px)
tracker.set_params(filter_radius=12, min_distance=40)

# For smaller particles (~10px)
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

- **Detection Rate**: Percentage of ground truth frames detected
- **Position Error**: Average distance between detections and ground truth (pixels)
- **Average Precision (AP)**: Area under precision-recall curve
- **ROC AUC**: Area under receiver operating characteristic curve
- **Track F1 Score**: Harmonic mean of track precision and recall

### Interpreting Results

- **AP > 0.8**: Excellent detection performance
- **AP 0.6-0.8**: Good detection performance
- **AP < 0.6**: Consider parameter tuning

- **Position Error < 10px**: High accuracy
- **Position Error 10-20px**: Moderate accuracy
- **Position Error > 20px**: Review detection threshold or double check that frames are lining up.

## Examples

### Example 1: Basic Analysis

```python
from ev_tracker import EVTracker

tracker = EVTracker()
results = tracker.run(
    tiff_file="data/movie.tif",
    ground_truth_csv="data/ground_truth.csv"
)

if results['success']:
    print(f"Average Precision: {results['global_ap']:.3f}")
    print(f"Output saved to: {results['output_dir']}")
```

### Example 2: Parameter Sweep

```python
from ev_tracker import EVTracker

tracker = EVTracker()
thresholds = [0.4, 0.5, 0.6, 0.7]

for thresh in thresholds:
    tracker.set_params(threshold=thresh)
    results = tracker.run("movie.tif", "ground_truth.csv")
    print(f"Threshold {thresh}: AP = {results['global_ap']:.3f}")
```

### Example 3: Batch Processing

```python
from ev_tracker import EVTracker
import glob

# Find all TIFF files
tiff_files = glob.glob("data/tiff/*.tif")
csv_files = glob.glob("data/csv/*.csv")

# Create dataset list
datasets = list(zip(tiff_files, csv_files))

# Run batch analysis
tracker = EVTracker()
tracker.set_params(threshold=0.55, min_distance=30)
results = tracker.run_batch(datasets)

print(f"Global AP: {results['global_ap']:.3f}")
print(f"Processed {len(datasets)} files")
```

The test suite includes:
- Import and initialization tests
- Parameter setting and validation
- Single file analysis
- Batch analysis
- Method chaining
- Parameter sweeps

## Troubleshooting

### Common Issues

**Import Error**
```python
ImportError: cannot import name 'EVTracker'
```
 Make sure you're in the project directory containing `ev_tracker.py`

**Too Many False Positives**
```python
tracker.set_params(threshold=0.65)  # Increase threshold
```

**Missing Particles**
```python
tracker.set_params(threshold=0.45)  # Decrease threshold
```

**Fragmented Tracks**
```python
tracker.set_params(max_distance=35, max_frame_gap=5)
```

## API Reference

### EVTracker Class

```python
class EVTracker(output_dir="UMB-EV-Tracker/out")
```

Initialize the tracker with optional output directory.

#### Methods

**`set_params(**kwargs)`**
- Set pipeline parameters
- Returns: `self` (for method chaining)
- Parameters: `threshold`, `min_distance`, `max_distance`, `min_track_length`, `filter_radius`, `bg_window_size`

**`run(tiff_file, ground_truth_csv=None)`**
- Run analysis on a single file
- Returns: Results dictionary with metrics and output paths

**`run_batch(dataset_list)`**
- Run batch analysis with global metrics
- Parameters: List of (tiff_file, csv_file) tuples
- Returns: Global results dictionary

**`print_params()`**
- Display current parameter settings

### Functions

**`quick_analyze(tiff_file, ground_truth_csv=None, threshold=0.55)`**
- Quick one-liner analysis
- Returns: Results dictionary


## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: s.bowerman.cs@gmail.com

## Acknowledgments

Developed for extracellular vesicle tracking research at University of Massachusetts - Boston.

---
