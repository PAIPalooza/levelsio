"""
Test data helper module for Interior Style Transfer POC.

Provides utilities for generating test images for development and testing
following Semantic Seed Coding Standards without requiring large binary files.
"""

import numpy as np
import os
from PIL import Image


def create_test_interior_image(
    output_path: str,
    width: int = 100,
    height: int = 100
) -> str:
    """
    Create a simple synthetic interior image for testing.
    
    Args:
        output_path: Path to save the generated image
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Path to the created image
    """
    # Create a simple interior-like image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Light beige walls
    img[:, :] = [245, 245, 220]
    
    # Brown floor at the bottom third
    img[height*2//3:, :] = [150, 75, 0]
    
    # Ceiling in light gray at the top
    img[:height//6, :] = [230, 230, 230]
    
    # Add a "window" on the right wall
    window_x = width - width//5
    window_y = height//3
    window_w = width//6
    window_h = height//3
    img[window_y:window_y+window_h, window_x:window_x+window_w] = [135, 206, 235]  # Sky blue
    
    # Add a simple "furniture" item
    furniture_x = width//4
    furniture_y = height*2//3 - height//6
    furniture_w = width//3
    furniture_h = height//6
    img[furniture_y:furniture_y+furniture_h, furniture_x:furniture_x+furniture_w] = [160, 82, 45]  # Sienna brown
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    Image.fromarray(img).save(output_path)
    
    return output_path


def get_test_image_path() -> str:
    """
    Get path to a test image, creating one if it doesn't exist.
    
    Returns:
        Path to a test image
    """
    # Define path relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img_dir = os.path.join(base_dir, 'assets', 'test_images')
    test_img_path = os.path.join(test_img_dir, 'test_interior.jpg')
    
    # Create the directory and test image if they don't exist
    if not os.path.exists(test_img_path):
        create_test_interior_image(test_img_path)
    
    return test_img_path
