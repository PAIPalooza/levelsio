"""
Test module for segmentation functionality.

BDD-style tests for the Segment Anything Model (SAM) integration
following Semantic Seed Coding Standards.
"""

import os
import pytest
import numpy as np
from PIL import Image

# Path manipulations for testing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import SegmentationHandler
from tests.test_data import get_test_image_path


class TestSegmentationHandler:
    """
    Tests for SAM integration and mask generation.
    
    Follows BDD-style testing to verify segmentation functionality.
    """
    
    @pytest.fixture
    def sample_image_path(self):
        """Fixture that provides path to test image."""
        # Use the test_data module to get or create a test image
        return get_test_image_path()
    
    @pytest.fixture
    def segmentation_handler(self):
        """Fixture that provides a segmentation handler instance."""
        # Use vit_b model which is smaller (375MB vs 2.5GB for vit_h)
        return SegmentationHandler(model_type="vit_b")
    
    def test_segmentation_handler_init(self, segmentation_handler):
        """
        GIVEN a segmentation handler
        WHEN it is initialized
        THEN it should have the correct default properties
        """
        assert segmentation_handler.model_type == "vit_b"
        assert segmentation_handler.device in ["cuda", "cpu"]
        
    def test_load_model(self, segmentation_handler):
        """
        GIVEN a segmentation handler
        WHEN the load_model method is called
        THEN it should return True indicating successful loading
        """
        result = segmentation_handler.load_model()
        assert result is True
        
    def test_generate_masks(self, segmentation_handler, sample_image_path):
        """
        GIVEN a segmentation handler and a sample image
        WHEN generate_masks is called on the image
        THEN it should return a dictionary of masks with the expected structure
        """
        # Load the test image
        img = np.array(Image.open(sample_image_path))
        
        # Generate masks
        masks = segmentation_handler.generate_masks(img)
        
        # Verify the output structure
        assert isinstance(masks, dict)
        expected_keys = ["walls", "floor", "ceiling", "windows", "structure"]
        for key in expected_keys:
            assert key in masks
            assert isinstance(masks[key], np.ndarray)
            assert masks[key].shape[:2] == img.shape[:2]  # Same dimensions as input
            
    def test_apply_mask(self, segmentation_handler, sample_image_path):
        """
        GIVEN a segmentation handler, sample image, and mask
        WHEN apply_mask is called
        THEN it should return the masked image
        """
        # Load the test image
        img = np.array(Image.open(sample_image_path))
        h, w = img.shape[:2]
        
        # Create a test mask (top half is 1, bottom half is 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:h//2, :] = 1
        
        # Apply mask
        masked_img = segmentation_handler.apply_mask(img, mask)
        
        # Verify the output
        assert isinstance(masked_img, np.ndarray)
        assert masked_img.shape == img.shape
        
        # Check that bottom half is zeroed out (masked)
        assert np.all(masked_img[h//2:, :, :] == 0)
        # Check that top half is unchanged
        assert np.all(masked_img[:h//2, :, :] == img[:h//2, :, :])
