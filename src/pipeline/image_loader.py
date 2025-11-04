import numpy as np
import tifffile as tiff
import os
from typing import Optional, Tuple, Dict, Any


def load_tiff_stack(tiff_file: str) -> Optional[np.ndarray]:
    print(f"Loading TIFF file: {tiff_file}")
    
    if not os.path.exists(tiff_file):
        print(f"Error: File {tiff_file} does not exist")
        return None
        
    try:
        image_stack = tiff.imread(tiff_file)
        print(f"Successfully loaded {len(image_stack)} frames, shape: {image_stack[0].shape}")
        return image_stack
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None


def validate_image_stack(image_stack: np.ndarray) -> bool:
    if image_stack is None:
        print("Error: Image stack is None")
        return False
    if len(image_stack.shape) != 3:
        print(f"Error: Expected 3D array, got {len(image_stack.shape)}D")
        return False

    print(f"Image stack validation passed: {image_stack.shape}")
    return True


def get_stack_info(image_stack: np.ndarray) -> Dict[str, Any]:
    if image_stack is None:
        return {}
        
    info = {
        'num_frames': image_stack.shape[0],        # Number of frames/slices in the TIFF stack (timepoints or z-slices)
        'height': image_stack.shape[1],            # Pixel height of each frame (number of rows)
        'width': image_stack.shape[2],             # Pixel width of each frame (number of columns)
        'dtype': str(image_stack.dtype),           # NumPy dtype of image pixels (e.g. 'uint16', 'float32')
        'min_value': np.min(image_stack),          # Minimum pixel intensity across the whole stack
        'max_value': np.max(image_stack),          # Maximum pixel intensity across the whole stack
        'mean_value': np.mean(image_stack),        # Mean pixel intensity across the whole stack
        'std_value': np.std(image_stack),          # Standard deviation of pixel intensities (contrast/noise measure)
        'total_size_mb': image_stack.nbytes / (1024 * 1024)  # Total memory size of the stack in megabytes (MB)
    }
    
    return info