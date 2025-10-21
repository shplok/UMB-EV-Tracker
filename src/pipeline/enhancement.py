"""
Image Enhancement Module for EV Detection Pipeline

This module enhances background-subtracted frames to improve particle visibility.
It applies contrast enhancement, noise reduction, and normalization techniques
to prepare images for particle detection.

Functions:
- enhance_movement_frames(): Main enhancement pipeline
- visualize_enhancement(): Create before/after comparisons
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import List, Tuple, Dict, Any


def enhance_movement_frames(subtracted_frames: np.ndarray, 
                          blur_kernel_size: int = 7,
                          clahe_clip_limit: float = 2.0,
                          clahe_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:

    print("Enhancing movement frames...")
    num_frames = len(subtracted_frames)
    enhanced_frames = np.zeros_like(subtracted_frames, dtype=np.uint8)
    
    # Create CLAHE object for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    
    for i in tqdm(range(num_frames), desc="Enhancing frames"):
        # Take absolute difference to capture all changes
        diff_abs = np.abs(subtracted_frames[i])
        
        # Normalize to 0-255 range
        diff_norm = cv2.normalize(diff_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE for better contrast
        enhanced = clahe.apply(diff_norm)
        
        # # Additional noise reduction - use larger kernel for 20px EVs
        enhanced = cv2.GaussianBlur(enhanced, (blur_kernel_size, blur_kernel_size), 0)
        
        # Store enhanced frame
        enhanced_frames[i] = enhanced
    
    print(f"Enhancement complete for {num_frames} frames")
    return enhanced_frames


def visualize_enhancement(subtracted_frames: np.ndarray,
                         enhanced_frames: np.ndarray,
                         output_dir: str,
                         num_samples: int = 6) -> List[str]:
    """
    Create visualizations showing the enhancement process
    
    Args:
        subtracted_frames (np.ndarray): Input frames before enhancement
        enhanced_frames (np.ndarray): Output frames after enhancement
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of sample frames to show
        
    Returns:
        List[str]: Paths to created visualization files
    """
    print("Creating enhancement visualizations...")
    
    num_frames = len(subtracted_frames)
    sample_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    
    # Create before/after comparison
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, frame_idx in enumerate(sample_indices):
        # Before enhancement (absolute values of subtracted frame)
        before = np.abs(subtracted_frames[frame_idx])
        before_norm = cv2.normalize(before, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        axes[0, i].imshow(before_norm, cmap='gray')
        axes[0, i].set_title(f'Before Enhancement\nFrame {frame_idx}')
        axes[0, i].axis('off')
        
        # After enhancement
        axes[1, i].imshow(enhanced_frames[frame_idx], cmap='gray')
        axes[1, i].set_title(f'After Enhancement\nFrame {frame_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "enhancement_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhancement visualization saved: {comparison_path}")
    
    return [comparison_path]