import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

# Import all pipeline modules
from src.pipeline.image_loader import (
    load_tiff_stack, 
    validate_image_stack, 
    get_stack_info
)
from src.pipeline.filter_creation import (
    create_large_ev_filter,
    visualize_filter,
    save_filter_data
)
from src.pipeline.background_subtraction import (
    create_temporal_background,
    subtract_background_from_stack,
    visualize_background_subtraction
)
from src.pipeline.enhancement import (
    enhance_movement_frames,
    visualize_enhancement
)
from src.pipeline.detection import (
    detect_particles_in_all_frames,
    visualize_detection_results,
    analyze_detection_quality
)
from src.pipeline.tracking import (
    track_particles_across_frames,
    calculate_track_properties,
    visualize_tracking_results,
    analyze_tracking_quality
)
from src.metrics.detection_metrics import evaluate_tracking_performance
from src.metrics.compute_pr_roc import evaluate_with_pr_roc
from src.metrics.testCOM import overlay_com_vs_detections
from src.pipeline.export_results import export_all_results

def create_output_directory(base_dir: str = "ev_detection_results") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Place all outputs under an `out/` root directory for easier discovery
    root = "out"
    output_dir = os.path.join("UMB-EV-Tracker", root, f"{base_dir}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for organized output
    subdirs = [
        "02_filter_creation", 
        "03_background_subtraction",
        "04_enhancement",
        "05_detection",
        "06_tracking",
        "07_metrics"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir


def run_ev_detection_pipeline(tiff_file: str, 
                            output_dir: str = None,
                            parameters: Dict[str, Any] = None,
                            ground_truth_csv: str = None) -> Dict[str, Any]:
    
    start_time = time.time()
    
    # Set default parameters
    if parameters is None:
        parameters = {
            'input_file': tiff_file,
            'filter_radius': 10,
            'filter_size': 41,
            'filter_sigma': 2.0,
            'bg_window_size': 15,
            'blur_kernel_size': 7,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8),
            'detection_threshold': 0.58,
            'min_distance': 30,
            'max_distance': 25,
            'min_track_length': 5,
            'max_frame_gap': 3,
            'num_sample_frames': 6,
            'num_top_tracks': 5
        }
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_output_directory()
    
    print(f"Input file: {tiff_file}")
    print(f"Output directory: {output_dir}")
    
    results = {
        'input_file': tiff_file,
        'output_dir': output_dir,
        'parameters': parameters,
        'start_time': start_time,
        'stage_times': {},
        'stage_results': {}
    }
    
    try:
        # Stage 1: Image Loading
        print("STAGE 1: IMAGE LOADING")
        print("-" * 30)
        stage_start = time.time()
        
        image_stack = load_tiff_stack(tiff_file)
        if image_stack is None:
            raise ValueError(f"Failed to load TIFF file: {tiff_file}")
        
        if not validate_image_stack(image_stack):
            raise ValueError("Image stack validation failed")
        
        stack_info = get_stack_info(image_stack)
        
        results['stage_times']['image_loading'] = time.time() - stage_start
        results['stage_results']['image_loading'] = {
            'stack_shape': image_stack.shape,
            'stack_info': stack_info
        }
        
        # Stage 2: Filter Creation
        print("STAGE 2: FILTER CREATION")
        print("-" * 30)
        stage_start = time.time()
        
        filter_params = {
            'radius': parameters['filter_radius'],
            'size': parameters['filter_size'],
            'sigma': parameters['filter_sigma']
        }
        
        ev_filter = create_large_ev_filter(**filter_params)
        filter_dir = os.path.join(output_dir, "02_filter_creation")
        
        visualize_filter(ev_filter, filter_dir, filter_params)
        save_filter_data(ev_filter, filter_params, filter_dir)
        
        results['stage_times']['filter_creation'] = time.time() - stage_start
        results['stage_results']['filter_creation'] = {
            'filter_shape': ev_filter.shape,
            'filter_params': filter_params
        }
        
        # Stage 3: Background Subtraction
        print("STAGE 3: BACKGROUND SUBTRACTION")
        print("-" * 30)
        stage_start = time.time()
        
        background_models = create_temporal_background(
            image_stack, 
            window_size=parameters['bg_window_size']
        )
        subtracted_frames = subtract_background_from_stack(image_stack, background_models)
        
        bg_dir = os.path.join(output_dir, "03_background_subtraction")
        visualize_background_subtraction(
            image_stack, background_models, subtracted_frames, bg_dir,
            num_samples=parameters['num_sample_frames']
        )
        
        results['stage_times']['background_subtraction'] = time.time() - stage_start
        results['stage_results']['background_subtraction'] = {
            'window_size': parameters['bg_window_size'],
            'frames_processed': len(subtracted_frames)
        }
        
        # Stage 4: Enhancement
        print("STAGE 4: ENHANCEMENT")
        print("-" * 30)
        stage_start = time.time()
        
        enhancement_params = {
            'blur_kernel_size': parameters['blur_kernel_size'],
            'clahe_clip_limit': parameters['clahe_clip_limit'],
            'clahe_grid_size': parameters['clahe_grid_size']
        }
        
        enhanced_frames = enhance_movement_frames(subtracted_frames, **enhancement_params)
        
        enh_dir = os.path.join(output_dir, "04_enhancement")
        visualize_enhancement(
            subtracted_frames, enhanced_frames, enh_dir,
            num_samples=parameters['num_sample_frames']
        )
        
        results['stage_times']['enhancement'] = time.time() - stage_start
        results['stage_results']['enhancement'] = {
            'enhancement_params': enhancement_params,
            'frames_processed': len(enhanced_frames)
        }
        
        # Stage 5: Detection
        print("STAGE 5: PARTICLE DETECTION")
        print("-" * 30)
        stage_start = time.time()
        
        detection_params = {
            'threshold': parameters['detection_threshold'],
            'min_distance': parameters['min_distance'],
            'filter_size': parameters['filter_size']
        }
        
        all_particles = detect_particles_in_all_frames(
            enhanced_frames, ev_filter,
            threshold=parameters['detection_threshold'],
            min_distance=parameters['min_distance']
        )

        # --------------------
        if ground_truth_csv:
            from src.metrics.testCOM import overlay_com_vs_detections
            overlay_dir = os.path.join(output_dir, "05_detection", "COM_overlays")
            overlay_com_vs_detections(
                enhanced_frames=enhanced_frames,
                ev_filter=ev_filter,
                ground_truth_csv=ground_truth_csv,
                output_dir=overlay_dir,
                num_examples=2,
                detection_threshold=parameters['detection_threshold'],
                min_distance=parameters['min_distance']
            )
        # ----------------------

        
        det_dir = os.path.join(output_dir, "05_detection")
        visualize_detection_results(
            enhanced_frames, all_particles, det_dir,
            num_samples=parameters['num_sample_frames']
        )
        analyze_detection_quality(all_particles, enhanced_frames, detection_params, det_dir)
        

        if ground_truth_csv is not None:
            print("\n  Creating COM overlay visualizations...")
            overlay_dir = os.path.join(det_dir, "COM_overlays")
            overlay_com_vs_detections(
                enhanced_frames=enhanced_frames,
                all_particles=all_particles,
                ground_truth_csv=ground_truth_csv,
                output_dir=overlay_dir,
                num_examples=5,  # Number of example frames
                distance_threshold=30.0,  # Same as your metrics
                use_matplotlib=True  # Better quality plots
            )

        total_detections = sum(len(frame_data['positions']) for frame_data in all_particles.values())
        
        results['stage_times']['detection'] = time.time() - stage_start
        results['stage_results']['detection'] = {
            'total_detections': total_detections,
            'frames_with_detections': len([f for f in all_particles.values() if f['positions']]),
            'detection_params': detection_params,
            'all_particles': all_particles  # Store for metrics evaluation
        }
        print(f"  Total detections: {total_detections}")
        
        # Stage 6: Tracking
        print("STAGE 6: PARTICLE TRACKING")
        print("-" * 30)

        stage_start = time.time()
        
        tracking_params = {
            'max_distance': parameters['max_distance'],
            'min_track_length': parameters['min_track_length'],
            'max_frame_gap': parameters['max_frame_gap']
        }
        
        tracks = track_particles_across_frames(all_particles, **tracking_params)
        tracks = calculate_track_properties(tracks, image_stack)
        
        track_dir = os.path.join(output_dir, "06_tracking")
        visualize_tracking_results(image_stack, tracks, track_dir)
        analyze_tracking_quality(tracks, all_particles, tracking_params, track_dir)
        
        results['stage_times']['tracking'] = time.time() - stage_start
        results['stage_results']['tracking'] = {
            'total_tracks': len(tracks),
            'avg_track_length': np.mean([len(t['frames']) for t in tracks.values()]) if tracks else 0,
            'tracking_params': tracking_params
        }
        
        # Stage 7: Metrics Evaluation (if ground truth provided)
        if ground_truth_csv is not None:
            print("STAGE 7: METRICS EVALUATION")
            print("-" * 30)
            stage_start = time.time()
            
            metrics_dir = os.path.join(output_dir, "07_metrics")
            
            try:

                print("\nDEBUG - Checking frame numbering:")
                print("Detection frames (first 20):", sorted(all_particles.keys())[:20])
                print("Detection frames (last 20):", sorted(all_particles.keys())[-20:])
                print("Total detection frames:", len(all_particles))

                metrics_results = evaluate_tracking_performance(
                    all_particles=all_particles,
                    tracks=tracks,
                    ground_truth_csv=ground_truth_csv,
                    output_dir=metrics_dir,
                    distance_threshold=30.0,
                    visualize=True,
                    image_stack=image_stack
                )
                pr_roc_results = evaluate_with_pr_roc(
                    all_particles=all_particles,
                    ground_truth_csv=ground_truth_csv,
                    output_dir=metrics_dir,
                    distance_threshold=30.0
                )
                
                results['stage_times']['metrics'] = time.time() - stage_start
                results['stage_results']['metrics'] = metrics_results
                results['stage_results']['pr_roc'] = pr_roc_results
                
                print(f"\n  Frame Detection Rate: {metrics_results['frame_detection_rate']*100:.1f}%")
                print(f"  Avg Position Error: {metrics_results['avg_position_error']:.2f}px")
                if metrics_results['matched_track_id']:
                    print(f"  Track F1 Score: {metrics_results['track_metrics']['track_f1']:.3f}")

                if pr_roc_results:
                    print(f"\n  Average Precision (AP): {pr_roc_results['pr_roc_data']['avg_precision']:.3f}")
                    print(f"  ROC AUC: {pr_roc_results['pr_roc_data']['roc_auc']:.3f}")
                
            except Exception as e:
                print(f"  Warning: Metrics evaluation failed: {str(e)}")
                results['stage_results']['metrics'] = {'error': str(e)}


        print("STAGE 7.5?: CSV EXPORT")
        print("-" * 30)
                
        export_paths = export_all_results(
            all_particles=all_particles,
            tracks=tracks,
            tiff_filename=tiff_file,
            output_dir=output_dir,
            include_untracked=True  # This ensures -1 IDs are included
        )
        results['export_paths'] = export_paths

        # Stage 8: Final Documentation
        print("\nSTAGE 8: FINAL DOCUMENTATION")
        print("-" * 30)
        stage_start = time.time()
        
        # Save the parameters
        with open(os.path.join(output_dir, "pipeline_parameters.txt"), 'w') as f:
            f.write(f"Pipeline Parameters - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input: {tiff_file}\n")
            if ground_truth_csv:
                f.write(f"Ground Truth: {ground_truth_csv}\n")
            f.write("\n")
            for key, value in parameters.items():
                f.write(f"{key}: {value}\n")
        
        results['stage_times']['documentation'] = time.time() - stage_start
        
        # Pipeline completion
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['success'] = True
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total runtime: {total_time:.1f} seconds")
        print(f"Results saved to: {output_dir}")
        
        # Print summary statistics
        if tracks:
            avg_velocity = np.mean([t['avg_velocity'] for t in tracks.values()])
            print(f"\nSummary:")
            print(f"  Total detections: {total_detections}")
            print(f"  Total tracks: {len(tracks)}")
            print(f"  Avg velocity: {avg_velocity:.2f} px/frame")
        
        if 'metrics' in results['stage_results'] and 'mAP' in results['stage_results']['metrics']:
            metrics = results['stage_results']['metrics']
            print(f"\nDetection Performance:")
            print(f"  mAP: {metrics['mAP']:.4f}")
            print(f"  AUC: {metrics['AUC']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        print(f"Error occurred after {time.time() - start_time:.1f}s")
        
        results['success'] = False
        results['error'] = str(e)
        results['total_time'] = time.time() - start_time
        
        return results


if __name__ == "__main__":
    
    # TIFF LIST
    # UMB-EV-Tracker\data\tiff\xslot_BT747_03_1000uLhr_z35um_adjSP_mov_2_MMStack_Pos0.ome.tif -- Done
    # UMB-EV-Tracker\data\tiff\xslot_HCC1954_01_500uLhr_z35um_mov_1_MMStack_Pos0.ome.tif -- Done
    # UMB-EV-Tracker\data\tiff\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif -- Done

    # CSV LIST
    # UMB-EV-Tracker\data\csv\xslot_BT747_03_1000uLhr_z35um_adjSP_mov_2.csv -- Done
    # UMB-EV-Tracker\data\csv\xslot_HCC1954_01_500uLhr_z35um_mov_1.csv -- Done
    # UMB-EV-Tracker\data\csv\xslot_HCC1954_01_500uLhr_z40um_mov_1.csv -- Done


    #----------------NEWER FILES----------------#

    # TIFF LIST
    # UMB-EV-Tracker/data/tiff/new/xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2_MMStack_Pos0.ome.tif
    # UMB-EV-Tracker\data\tiff\new\xslot_BT747_PT00_xp1_1500uLhr_z40um_mov_6_flush_adj_MMStack_Pos0.ome.tif
    # UMB-EV-Tracker\data\tiff\new\xslot_HCC1954_PT03_xp4_1250uLhr_z40um_mov_1_MMStack_Pos0.ome.tif
    # UMB-EV-Tracker\data\tiff\new\xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1_MMStack_Pos0.ome.tif

    # CSV LIST
    # UMB-EV-Tracker/data/csv/new/xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2.csv
    # UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_BT747_PT00_xp1_1500uLhr_z40um_mov_6_flush_adj_MMStack_Pos0.ome.csv
    # UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_HCC1954_PT03_xp4_1250uLhr_z40um_mov_1_MMStack_Pos0.ome.csv
    # UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1.csv

    TIFF_FILE = r"UMB-EV-Tracker\data\tiff\new\xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1_MMStack_Pos0.ome.tif"
    
    # Ground truth CSV (optional - set to None if not available)
    GROUND_TRUTH_CSV = r"UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1.csv"
    
    # Output directory - will be auto-generated with timestamp if None
    OUTPUT_DIR = None
    
    # Pipeline parameters - adjust based on data characteristics
    PARAMETERS = {
        # Filter parameters (for ~20px EVs)
        'filter_radius': 10,          # Bright center radius
        'filter_size': 41,            # Filter matrix size
        'filter_sigma': 2.0,          # Gaussian smoothing
        
        # Background subtraction
        'bg_window_size': 15,         # Temporal median window
        
        # Enhancement  
        'blur_kernel_size': 7,        # Noise reduction kernel
        'clahe_clip_limit': 2.0,      # Contrast enhancement
        'clahe_grid_size': (8, 8),    # CLAHE tile size
        
        # Detection
        'detection_threshold': 0.55,  # Correlation threshold
        'min_distance': 30,           # Min separation between detections
        
        # Tracking
        'max_distance': 25,           # Max movement between frames
        'min_track_length': 5,        # Min detections per track
        'max_frame_gap': 3,           # Max missing frames in track
        
        # Visualization
        'num_sample_frames': 6,       # Sample frames for plots
        'num_top_tracks': 5           # Detailed tracks to analyze
    }
    
    # Run the pipeline
    # ================
    
    print("Starting EV Detection Pipeline...")
    print(f"Input file: {TIFF_FILE}")
    if GROUND_TRUTH_CSV:
        print(f"Ground truth: {GROUND_TRUTH_CSV}")
    
    # Check if input file exists
    if not os.path.exists(TIFF_FILE):
        print(f"Error: Input file not found: {TIFF_FILE}")
        sys.exit(1)
    
    # Check if ground truth exists (if specified)
    if GROUND_TRUTH_CSV and not os.path.exists(GROUND_TRUTH_CSV):
        print(f"Warning: Ground truth file not found: {GROUND_TRUTH_CSV}")
        print("Continuing without metrics evaluation...")
        GROUND_TRUTH_CSV = None
    
    # Execute pipeline
    results = run_ev_detection_pipeline(
        tiff_file=TIFF_FILE,
        output_dir=OUTPUT_DIR,
        parameters=PARAMETERS,
        ground_truth_csv=GROUND_TRUTH_CSV
    )
    
    # Final status
    if results['success']:
        print(f"\n Pipeline completed successfully")
        print(f"  Results directory: {results['output_dir']}")
    else:
        print(f"\n Pipeline failed: {results.get('error', 'Unknown error')}")
        print(f"  Partial results in: {results['output_dir']}")