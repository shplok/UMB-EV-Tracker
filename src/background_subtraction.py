"""
Background Subtraction Module for EV Detection Pipeline

This module handles the removal of static background elements from image sequences
to enhance the visibility of moving particles (EVs). It uses temporal median filtering
to create adaptive background models.

Functions:
- create_temporal_background(): Creates background models for all frames
- subtract_background_from_stack(): Applies background subtraction
- visualize_background_subtraction(): Creates before/after comparisons
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Tuple, List


def create_temporal_background(image_stack: np.ndarray, window_size: int = 15) -> np.ndarray:

    print(f"Creating temporal background models for all frames (window_size={window_size})...")
    num_frames = len(image_stack)
    background_models = np.zeros_like(image_stack, dtype=np.float32)
    
    for i in tqdm(range(num_frames), desc="Creating backgrounds"):
        # Define window around current frame
        start_idx = max(0, i - window_size)
        end_idx = min(num_frames, i + window_size + 1)
        
        # Get frames in window (excluding current frame)
        window_frames = []
        for j in range(start_idx, end_idx):
            if j != i:  # Exclude the current frame
                window_frames.append(image_stack[j])
        
        # Compute median of surrounding frames as background
        if window_frames:
            window_stack = np.array(window_frames, dtype=np.float32)
            background_models[i] = np.median(window_stack, axis=0)
        else:
            background_models[i] = image_stack[i]  # Fallback if no other frames available
    
    print(f"Background models created for {num_frames} frames")
    return background_models


def subtract_background_from_stack(image_stack: np.ndarray, 
                                 background_models: np.ndarray) -> np.ndarray:
    print("Subtracting background from all frames...")
    num_frames = len(image_stack)
    subtracted_frames = np.zeros_like(image_stack, dtype=np.float32)
    
    for i in tqdm(range(num_frames), desc="Background subtraction"):
        # Subtract background
        subtracted_frames[i] = image_stack[i].astype(np.float32) - background_models[i]
    
    print("Background subtraction complete")
    return subtracted_frames


def visualize_background_subtraction(image_stack: np.ndarray, 
                                   background_models: np.ndarray,
                                   subtracted_frames: np.ndarray,
                                   output_dir: str,
                                   num_samples: int = 6) -> List[str]:

    print("Creating background subtraction visualizations...")
    
    num_frames = len(image_stack)
    sample_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    
    # Create comparison figure
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, frame_idx in enumerate(sample_indices):
        # Original frame
        axes[0, i].imshow(image_stack[frame_idx], cmap='gray')
        axes[0, i].set_title(f'Original Frame {frame_idx}')
        axes[0, i].axis('off')
        
        # Background model
        axes[1, i].imshow(background_models[frame_idx], cmap='gray')
        axes[1, i].set_title(f'Background Model {frame_idx}')
        axes[1, i].axis('off')
        
        # Subtracted frame (show both positive and negative changes)
        subtracted = subtracted_frames[frame_idx]
        # Use symmetric colormap to show both positive and negative changes
        vmax = np.max(np.abs(subtracted))
        im = axes[2, i].imshow(subtracted, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2, i].set_title(f'After Subtraction {frame_idx}')
        axes[2, i].axis('off')
        
        # Add colorbar to the last subtracted frame
        if i == num_samples - 1:
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "background_subtraction_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Background subtraction visualization saved: {comparison_path}")
    
    return [comparison_path]