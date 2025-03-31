"""
Utility functions for Interior Style Transfer Demo Notebook.

This module provides helper functions for loading sample images, 
displaying results, and ensuring proper use of environment variables.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_sample_images(sample_dir: str = "notebooks/sample_images") -> Dict[str, np.ndarray]:
    """
    Load sample interior images from a directory.
    
    Args:
        sample_dir: Directory containing sample images
        
    Returns:
        Dictionary mapping image names to numpy arrays
    """
    images = {}
    sample_files = glob.glob(os.path.join(sample_dir, "*.jpg")) + \
                  glob.glob(os.path.join(sample_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(sample_dir, "*.png"))
    
    if not sample_files:
        logger.warning(f"No sample images found in {sample_dir}")
        return images
    
    for file_path in sample_files:
        try:
            # Get the filename without extension as the key
            name = os.path.splitext(os.path.basename(file_path))[0]
            # Load the image
            img = Image.open(file_path)
            # Convert to numpy array
            img_array = np.array(img)
            # Add to dictionary
            images[name] = img_array
            logger.info(f"Loaded {name} from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            
    return images


def display_images(images: List[np.ndarray], titles: Optional[List[str]] = None, 
                 figsize: Tuple[int, int] = (15, 10), rows: int = 1) -> None:
    """
    Display multiple images in a row.
    
    Args:
        images: List of images as numpy arrays
        titles: Optional list of titles for each image
        figsize: Figure size (width, height)
        rows: Number of rows for the subplot grid
    """
    if not images:
        logger.warning("No images to display")
        return
        
    cols = len(images) // rows
    if len(images) % rows != 0:
        cols += 1
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    for i, (img, ax) in enumerate(zip(images, axes.flatten())):
        if i < len(images):
            if img.max() <= 1.0 and img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def check_environment_variables() -> bool:
    """
    Check if required environment variables are set.
    
    Returns:
        True if all required variables are set, False otherwise
    """
    required_vars = ['FAL_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Make sure to set these in your .env file or environment")
        return False
    
    logger.info("All required environment variables are set")
    return True


def save_image_result(image: np.ndarray, filename: str, 
                     output_dir: str = "notebooks/outputs") -> str:
    """
    Save an image result to disk.
    
    Args:
        image: Image as numpy array
        filename: Output filename
        output_dir: Output directory
        
    Returns:
        Path to the saved image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to PIL Image
    if image.max() <= 1.0 and image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    img = Image.fromarray(image.astype(np.uint8))
    
    # Ensure filename has extension
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        filename += '.jpg'
        
    output_path = os.path.join(output_dir, filename)
    img.save(output_path)
    logger.info(f"Saved result to {output_path}")
    
    return output_path
