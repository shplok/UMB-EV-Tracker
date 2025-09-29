"""
EV Detection Pipeline - Visualization Only
Simplified pipeline that focuses on creating visual outputs at each processing stage
without metrics evaluation or ground truth comparison.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import pipeline modules
from image_loader import (
    load_tiff_stack, 
    validate_image_stack, 
    get_stack_info
)
from filter_creation import (
    create_large_ev_filter,
    visualize_filter,
    save_filter_data
)
from background_subtraction import (
    create_temporal_background,
    subtract_background_from_stack,
    visualize_background_subtraction
)
from detection import (
    detect_particles_in_all_frames,
    visualize_detection_results
)
from tracking import (
    track_particles_across_frames,
    calculate_track_properties,
    visualize_tracking_results
)


def create_output_directory(base_dir: str = "ev_visualization") -> str:
    """Create a timestamped output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = "out"
    os.makedirs(root, exist_ok=True)
    output_dir = os.path.join(root, f"{base_dir}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    subdirs = [
        "01_raw_frames",
        "02_filter",
        "03_background_subtraction",
        "04_enhancement_comparison",
        "05_detection",
        "06_tracking"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir


def visualize_raw_frames(image_stack: np.ndarray, output_dir: str, num_samples: int = 6):
    """Visualize sample raw frames from the image stack"""
    print("Creating raw frame visualizations...")
    
    num_frames = len(image_stack)
    sample_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(sample_indices):
        axes[i].imshow(image_stack[frame_idx], cmap='gray')
        axes[i].set_title(f'Raw Frame {frame_idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_raw_frames", "raw_frames_sample.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Raw frames saved: {save_path}")
    return save_path


def create_enhancement_comparison(subtracted_frames: np.ndarray, 
                                 output_dir: str,
                                 sample_frame_idx: int = None,
                                 blur_kernel: int = 7,
                                 clahe_clip: float = 2.0):
    """Create side-by-side comparison of all enhancement methods"""
    print("Creating enhancement method comparison...")
    
    if sample_frame_idx is None:
        sample_frame_idx = len(subtracted_frames) // 2
    
    frame = subtracted_frames[sample_frame_idx]
    
    # Create all enhancement variations
    # 1. Raw normalized
    raw = cv2.normalize(np.abs(frame), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. CLAHE only
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    clahe_only = clahe.apply(raw)
    
    # 3. Blur only
    blur_only = cv2.GaussianBlur(raw, (blur_kernel, blur_kernel), 0)
    
    # 4. CLAHE + Blur
    clahe_blur = cv2.GaussianBlur(clahe_only, (blur_kernel, blur_kernel), 0)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(raw, cmap='gray')
    axes[0, 0].set_title(f'Raw Normalized\nFrame {sample_frame_idx}', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(clahe_only, cmap='gray')
    axes[0, 1].set_title(f'CLAHE Only\nFrame {sample_frame_idx}', fontsize=12, weight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(blur_only, cmap='gray')
    axes[1, 0].set_title(f'Gaussian Blur Only\nFrame {sample_frame_idx}', fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(clahe_blur, cmap='gray')
    axes[1, 1].set_title(f'CLAHE + Blur\nFrame {sample_frame_idx}', fontsize=12, weight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('Enhancement Method Comparison', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "04_enhancement_comparison", 
                            f"enhancement_comparison_frame_{sample_frame_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhancement comparison saved: {save_path}")
    return save_path


def create_multi_frame_enhancement_comparison(subtracted_frames: np.ndarray,
                                             output_dir: str,
                                             num_samples: int = 4):
    """Create enhancement comparison across multiple frames"""
    print("Creating multi-frame enhancement comparison...")
    
    num_frames = len(subtracted_frames)
    sample_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    enhancement_methods = [
        ('Raw', lambda x: cv2.normalize(np.abs(x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)),
        ('CLAHE', lambda x: clahe.apply(cv2.normalize(np.abs(x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))),
        ('Blur', lambda x: cv2.GaussianBlur(cv2.normalize(np.abs(x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (7, 7), 0)),
        ('CLAHE+Blur', lambda x: cv2.GaussianBlur(clahe.apply(cv2.normalize(np.abs(x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)), (7, 7), 0))
    ]
    
    for i, frame_idx in enumerate(sample_indices):
        frame = subtracted_frames[frame_idx]
        
        for j, (method_name, method_func) in enumerate(enhancement_methods):
            processed = method_func(frame)
            axes[i, j].imshow(processed, cmap='gray')
            
            if i == 0:
                axes[i, j].set_title(method_name, fontsize=12, weight='bold')
            
            axes[i, j].set_ylabel(f'Frame {frame_idx}', fontsize=10)
            axes[i, j].axis('off')
    
    plt.suptitle('Enhancement Methods Across Multiple Frames', fontsize=16, weight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "04_enhancement_comparison", 
                            "multi_frame_enhancement_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-frame enhancement comparison saved: {save_path}")
    return save_path


def apply_enhancement(subtracted_frames: np.ndarray,
                     use_clahe: bool = True,
                     use_blur: bool = True) -> np.ndarray:
    """Apply enhancement selectively"""
    num_frames = len(subtracted_frames)
    enhanced_frames = np.zeros_like(subtracted_frames, dtype=np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
    
    for i in range(num_frames):
        diff_abs = np.abs(subtracted_frames[i])
        diff_norm = cv2.normalize(diff_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if use_clahe and clahe is not None:
            enhanced = clahe.apply(diff_norm)
        else:
            enhanced = diff_norm
        
        if use_blur:
            enhanced = cv2.GaussianBlur(enhanced, (7, 7), 0)
        
        enhanced_frames[i] = enhanced
    
    return enhanced_frames


def run_visualization_pipeline(tiff_file: str,
                               output_dir: str = None,
                               parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run visualization-focused pipeline without metrics evaluation
    """
    
    start_time = time.time()
    
    if parameters is None:
        parameters = {
            'filter_radius': 10,
            'filter_size': 41,
            'filter_sigma': 2.0,
            'bg_window_size': 15,
            'detection_threshold': 0.7,
            'min_distance': 30,
            'max_distance': 25,
            'min_track_length': 5,
            'max_frame_gap': 3,
            'num_sample_frames': 6
        }
    
    if output_dir is None:
        output_dir = create_output_directory()
    
    print(f"\nInput file: {tiff_file}")
    print(f"Output directory: {output_dir}\n")
    
    results = {
        'input_file': tiff_file,
        'output_dir': output_dir,
        'parameters': parameters,
        'visualizations': []
    }
    
    try:
        # Stage 1: Load Images
        print("="*60)
        print("STAGE 1: LOADING IMAGES")
        print("="*60)
        
        image_stack = load_tiff_stack(tiff_file)
        if image_stack is None or not validate_image_stack(image_stack):
            raise ValueError("Failed to load or validate image stack")
        
        stack_info = get_stack_info(image_stack)
        print(f"Loaded {stack_info['num_frames']} frames of size {stack_info['height']}x{stack_info['width']}")
        
        # Visualize raw frames
        raw_vis = visualize_raw_frames(image_stack, output_dir, parameters['num_sample_frames'])
        results['visualizations'].append(raw_vis)
        
        # Stage 2: Filter Creation
        print("\n" + "="*60)
        print("STAGE 2: FILTER CREATION")
        print("="*60)
        
        ev_filter = create_large_ev_filter(
            radius=parameters['filter_radius'],
            size=parameters['filter_size'],
            sigma=parameters['filter_sigma']
        )
        
        filter_dir = os.path.join(output_dir, "02_filter")
        filter_params = {
            'radius': parameters['filter_radius'],
            'size': parameters['filter_size'],
            'sigma': parameters['filter_sigma']
        }
        filter_vis = visualize_filter(ev_filter, filter_dir, filter_params)
        save_filter_data(ev_filter, filter_params, filter_dir)
        results['visualizations'].append(filter_vis)
        
        # Stage 3: Background Subtraction
        print("\n" + "="*60)
        print("STAGE 3: BACKGROUND SUBTRACTION")
        print("="*60)
        
        background_models = create_temporal_background(
            image_stack, 
            window_size=parameters['bg_window_size']
        )
        subtracted_frames = subtract_background_from_stack(image_stack, background_models)
        
        bg_dir = os.path.join(output_dir, "03_background_subtraction")
        bg_vis = visualize_background_subtraction(
            image_stack, background_models, subtracted_frames, bg_dir,
            num_samples=parameters['num_sample_frames']
        )
        results['visualizations'].extend(bg_vis)
        
        # Stage 4: Enhancement Comparison
        print("\n" + "="*60)
        print("STAGE 4: ENHANCEMENT COMPARISON")
        print("="*60)
        
        # Single frame comparison
        enh_comp = create_enhancement_comparison(subtracted_frames, output_dir)
        results['visualizations'].append(enh_comp)
        
        # Multi-frame comparison
        multi_enh_comp = create_multi_frame_enhancement_comparison(
            subtracted_frames, output_dir, num_samples=4
        )
        results['visualizations'].append(multi_enh_comp)
        
        # Apply enhancement for detection (using CLAHE + Blur)
        enhanced_frames = apply_enhancement(subtracted_frames, use_clahe=True, use_blur=True)
        
        # Stage 5: Detection
        print("\n" + "="*60)
        print("STAGE 5: PARTICLE DETECTION")
        print("="*60)
        
        all_particles = detect_particles_in_all_frames(
            enhanced_frames, ev_filter,
            threshold=parameters['detection_threshold'],
            min_distance=parameters['min_distance']
        )
        
        total_detections = sum(len(frame_data['positions']) for frame_data in all_particles.values())
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/len(image_stack):.2f}")
        
        det_dir = os.path.join(output_dir, "05_detection")
        det_vis = visualize_detection_results(
            enhanced_frames, all_particles, det_dir,
            num_samples=parameters['num_sample_frames']
        )
        results['visualizations'].extend(det_vis)
        results['total_detections'] = total_detections
        
        # Stage 6: Tracking
        print("\n" + "="*60)
        print("STAGE 6: PARTICLE TRACKING")
        print("="*60)
        
        tracks = track_particles_across_frames(
            all_particles,
            max_distance=parameters['max_distance'],
            min_track_length=parameters['min_track_length'],
            max_frame_gap=parameters['max_frame_gap']
        )
        
        tracks = calculate_track_properties(tracks, image_stack)
        
        print(f"Total tracks: {len(tracks)}")
        if tracks:
            avg_track_length = np.mean([len(t['frames']) for t in tracks.values()])
            print(f"Average track length: {avg_track_length:.1f} frames")
        
        track_dir = os.path.join(output_dir, "06_tracking")
        track_vis = visualize_tracking_results(image_stack, tracks, track_dir)
        results['visualizations'].extend(track_vis)
        results['total_tracks'] = len(tracks)
        
        # Summary
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['success'] = True
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total runtime: {total_time:.1f} seconds")
        print(f"Total detections: {total_detections}")
        print(f"Total tracks: {len(tracks)}")
        print(f"\nAll visualizations saved to: {output_dir}")
        
        # Create summary document
        create_summary_document(results, output_dir)
        
        return results
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        results['success'] = False
        results['error'] = str(e)
        return results


def create_summary_document(results: Dict[str, Any], output_dir: str):
    """Create a summary text file"""
    summary_path = os.path.join(output_dir, "pipeline_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("EV DETECTION PIPELINE - VISUALIZATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {results['input_file']}\n")
        f.write(f"Output Directory: {results['output_dir']}\n")
        f.write(f"Total Runtime: {results.get('total_time', 0):.1f} seconds\n\n")
        
        f.write("DETECTION RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Detections: {results.get('total_detections', 0)}\n")
        f.write(f"Total Tracks: {results.get('total_tracks', 0)}\n\n")
        
        f.write("PARAMETERS USED\n")
        f.write("-" * 30 + "\n")
        for key, value in results['parameters'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nVISUALIZATIONS CREATED\n")
        f.write("-" * 30 + "\n")
        f.write("1. Raw frame samples\n")
        f.write("2. Filter visualization (2D, 3D, cross-sections)\n")
        f.write("3. Background subtraction comparison\n")
        f.write("4. Enhancement method comparison (single frame)\n")
        f.write("5. Enhancement method comparison (multiple frames)\n")
        f.write("6. Detection overlays and video\n")
        f.write("7. Particle tracking overview and detailed tracks\n")
    
    print(f"Summary document saved: {summary_path}")


if __name__ == "__main__":
    # Configuration
    TIFF_FILE = "data\\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif"
    
    PARAMETERS = {
        'filter_radius': 10,
        'filter_size': 41,
        'filter_sigma': 2.0,
        'bg_window_size': 15,
        'detection_threshold': 0.7,
        'min_distance': 30,
        'max_distance': 25,
        'min_track_length': 5,
        'max_frame_gap': 3,
        'num_sample_frames': 6
    }
    
    print("Starting EV Detection Pipeline - Visualization Mode")
    print("This pipeline focuses on creating visual outputs at each stage")
    print("=" * 60 + "\n")
    
    if not os.path.exists(TIFF_FILE):
        print(f"Error: Input file not found: {TIFF_FILE}")
        sys.exit(1)
    
    results = run_visualization_pipeline(
        tiff_file=TIFF_FILE,
        parameters=PARAMETERS
    )
    
    if results['success']:
        print(f"\nSuccess! Check the output directory for all visualizations:")
        print(f"{results['output_dir']}")
    else:
        print(f"\nPipeline failed: {results.get('error', 'Unknown error')}")