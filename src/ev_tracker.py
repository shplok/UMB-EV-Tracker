import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class EVTracker:

    def __init__(self, output_dir: str = "../out"):
        # Detection parameters
        self.threshold = 0.55
        self.min_distance = 30
        self.filter_radius = 10
        self.filter_size = 41
        self.filter_sigma = 2.0
        
        # Background & Enhancement
        self.bg_window_size = 15
        self.blur_kernel_size = 7
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)
        
        # Tracking parameters
        self.max_distance = 25
        self.min_track_length = 5
        self.max_frame_gap = 3
        
        # Distance threshold for metrics
        self.distance_threshold = 30.0
        
        self.output_base_dir = output_dir
        
        print("EV Tracker initialized")
    
    def set_params(self, 
                   threshold: Optional[float] = None,
                   min_distance: Optional[int] = None,
                   max_distance: Optional[int] = None,
                   min_track_length: Optional[int] = None,
                   filter_radius: Optional[int] = None,
                   bg_window_size: Optional[int] = None,
                   filter_size: Optional[int] = None,
                   filter_sigma: Optional[float] = None,
                   blur_kernel_size: Optional[int] = None,
                   clahe_clip_limit: Optional[float] = None,
                   clahe_grid_size: Optional[Tuple[int, int]] = None,
                   max_frame_gap: Optional[int] = None,
                   distance_threshold: Optional[float] = None) -> 'EVTracker':

        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            self.threshold = threshold
            
        if min_distance is not None:
            self.min_distance = min_distance
            
        if max_distance is not None:
            self.max_distance = max_distance
            
        if min_track_length is not None:
            self.min_track_length = min_track_length
            
        if filter_radius is not None:
            self.filter_radius = filter_radius
            
        if bg_window_size is not None:
            self.bg_window_size = bg_window_size
            
        if filter_size is not None:
            self.filter_size = filter_size
            
        if filter_sigma is not None:
            self.filter_sigma = filter_sigma
            
        if blur_kernel_size is not None:
            self.blur_kernel_size = blur_kernel_size
            
        if clahe_clip_limit is not None:
            self.clahe_clip_limit = clahe_clip_limit
            
        if clahe_grid_size is not None:
            self.clahe_grid_size = clahe_grid_size
            
        if max_frame_gap is not None:
            self.max_frame_gap = max_frame_gap
            
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold
        
        print("Parameters updated")
        return self
    
    def run(self, 
            tiff_file: str, 
            ground_truth_csv: Optional[str] = None) -> Dict[str, Any]:

        # Single file = batch of one
        dataset = [(tiff_file, ground_truth_csv)] if ground_truth_csv else [(tiff_file, None)]
        
        print(f"\n{'='*60}")
        print(f"Running: {os.path.basename(tiff_file)}")
        print(f"{'='*60}")
        
        # Use batch pipeline
        from src.helpers.batch_main import run_global_batch_analysis
        
        results = run_global_batch_analysis(
            dataset_list=dataset,
            batch_params=self._get_params_dict(),
            distance_threshold=self.distance_threshold,
            override_threshold_for_pr_curve=False  # Use specified threshold
        )
        
        return results
    
    def run_batch(self, 
                  dataset_list: List[Tuple[str, str]]) -> Dict[str, Any]:

        print(f"\n{'='*60}")
        print(f"Batch Analysis: {len(dataset_list)} files")
        print(f"{'='*60}")
        
        from src.helpers.batch_main import run_global_batch_analysis
        
        results = run_global_batch_analysis(
            dataset_list=dataset_list,
            batch_params=self._get_params_dict(),
            distance_threshold=self.distance_threshold
        )
        
        return results
    
    def print_params(self):
        """Print all current parameter settings in organized groups."""
        print(f"\n{'='*70}")
        print("CURRENT PARAMETERS")
        print(f"{'='*70}")
        
        print(f"\nDetection:")
        print(f"  threshold:        {self.threshold}")
        print(f"  min_distance:     {self.min_distance}px")
        print(f"  filter_radius:    {self.filter_radius}px")
        print(f"  filter_size:      {self.filter_size}px")
        print(f"  filter_sigma:     {self.filter_sigma}")
        
        print(f"\nBackground & Enhancement:")
        print(f"  bg_window_size:   {self.bg_window_size} frames")
        print(f"  blur_kernel_size: {self.blur_kernel_size}px")
        print(f"  clahe_clip_limit: {self.clahe_clip_limit}")
        print(f"  clahe_grid_size:  {self.clahe_grid_size}")
        
        print(f"\nTracking:")
        print(f"  max_distance:     {self.max_distance}px")
        print(f"  min_track_length: {self.min_track_length} frames")
        print(f"  max_frame_gap:    {self.max_frame_gap} frames")
        
        print(f"\nMetrics:")
        print(f"  distance_threshold: {self.distance_threshold}px")
        
        print(f"{'='*70}\n")
    
    def _get_params_dict(self) -> Dict[str, Any]:
        return {
            'filter_radius': self.filter_radius,
            'filter_size': self.filter_size,
            'filter_sigma': self.filter_sigma,
            'bg_window_size': self.bg_window_size,
            'blur_kernel_size': self.blur_kernel_size,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_grid_size': self.clahe_grid_size,
            'detection_threshold': self.threshold,
            'min_distance': self.min_distance,
            'max_distance': self.max_distance,
            'min_track_length': self.min_track_length,
            'max_frame_gap': self.max_frame_gap,
            'num_sample_frames': 6,
            'num_top_tracks': 5
        }


# Convenience function
def quick_analyze(tiff_file: str, 
                 ground_truth_csv: Optional[str] = None,
                 threshold: float = 0.55) -> Dict[str, Any]:
    return EVTracker().set_params(threshold=threshold).run(tiff_file, ground_truth_csv)