"""
Mock objects and functions for testing.

This module provides mocks for external services used by the API,
following the Semantic Seed Coding Standards (SSCS) for testability.
"""

import base64
import json
import io
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, List

# Sample base64 encoded small image (1x1 red pixel)
SAMPLE_B64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def create_mock_image(color: str = "red", size: tuple = (100, 100)) -> bytes:
    """Create a mock image with the specified color and size."""
    img = Image.new('RGB', size, color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


def create_mock_b64_image(color: str = "red", size: tuple = (100, 100)) -> str:
    """Create a mock base64 encoded image."""
    return base64.b64encode(create_mock_image(color, size)).decode('utf-8')


class MockFalAPI:
    """Mock for the fal.ai API used in style transfer and segmentation."""
    
    @staticmethod
    def mock_style_transfer_response(style: str) -> Dict[str, Any]:
        """Generate a mock style transfer response."""
        return {
            "styled_image": create_mock_b64_image("blue"),
            "style": style,
            "prompt": f"A {style} style interior with clean lines",
            "processing_time": 2.5
        }
    
    @staticmethod
    def mock_segmentation_response() -> Dict[str, Any]:
        """Generate a mock segmentation response."""
        # Create a simple 100x100 segmentation mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1  # Add a square in the middle
        
        # Convert to base64
        img = Image.fromarray(mask * 255)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "regions": [
                {
                    "label": "wall",
                    "mask": mask_b64,
                    "confidence": 0.95
                },
                {
                    "label": "floor",
                    "mask": mask_b64,
                    "confidence": 0.92
                }
            ],
            "detail_level": "medium",
            "image_width": 100,
            "image_height": 100,
            "processing_time": 1.8
        }
    
    @staticmethod
    def mock_evaluation_response() -> Dict[str, Any]:
        """Generate a mock evaluation response."""
        return {
            "ssim": 0.85,
            "mse": 0.15,
            "psnr": 25.5,
            "style_score": 0.78,
            "processing_time": 0.5
        }


# Patch functions to be used as decorators for tests

def mock_fal_style_transfer():
    """Decorator to mock the style transfer service."""
    def decorator(func):
        @patch('src.api.routes.style_transfer.call_fal_api')
        def wrapper(mock_call_api, *args, **kwargs):
            mock_call_api.return_value = MockFalAPI.mock_style_transfer_response("minimalist")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def mock_fal_segmentation():
    """Decorator to mock the segmentation service."""
    def decorator(func):
        @patch('src.api.routes.segmentation.call_fal_segmentation_api')
        def wrapper(mock_call_api, *args, **kwargs):
            mock_call_api.return_value = MockFalAPI.mock_segmentation_response()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def mock_fal_evaluation():
    """Decorator to mock the evaluation service."""
    def decorator(func):
        @patch('src.api.routes.evaluation.calculate_metrics')
        def wrapper(mock_call_api, *args, **kwargs):
            mock_call_api.return_value = MockFalAPI.mock_evaluation_response()
            return func(*args, **kwargs)
        return wrapper
    return decorator
