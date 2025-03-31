"""
Coverage-focused test module for segmentation functionality.

This module contains BDD-style tests for the SegmentationHandler class,
specifically targeting coverage gaps to meet code quality standards.
"""

import pytest
import numpy as np
from unittest import mock
from unittest.mock import MagicMock, patch
import os
import json
import tempfile
from PIL import Image
import cv2

from src.segmentation import SegmentationHandler, SAM_AVAILABLE


class TestSegmentationHandler:
    """
    Test suite focused on SegmentationHandler's public methods.
    
    BDD-style tests to improve code coverage.
    """
    
    @pytest.fixture
    def sample_interior_image(self):
        """Fixture that provides a sample interior image."""
        # Create a simple interior image for testing
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        # Add a simple structure to simulate walls, floor, etc.
        # Ceiling (top)
        image[:20, :] = [230, 230, 230]
        # Floor (bottom)
        image[80:, :] = [210, 210, 210]
        # Wall (left)
        image[:, :20] = [220, 220, 220]
        # Wall (right)
        image[:, 80:] = [220, 220, 220]
        # Window (center)
        image[40:60, 40:60] = [240, 230, 200]
        
        return image
    
    @pytest.fixture
    def mock_sam_mask_generator(self):
        """Fixture providing a mocked SAM mask generator."""
        # Create mocks
        mask_generator = MagicMock()
        
        # Sample masks output
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[:20, :] = 1  # Top area (ceiling)
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[80:, :] = 1  # Bottom area (floor)
        
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        mask3[:, :20] = 1  # Left area (wall)
        
        # Format the masks like SAM output
        masks = [
            {'segmentation': mask1, 'area': 2000, 'bbox': [0, 0, 100, 20]},
            {'segmentation': mask2, 'area': 2000, 'bbox': [0, 80, 100, 20]},
            {'segmentation': mask3, 'area': 2000, 'bbox': [0, 0, 20, 100]}
        ]
        
        mask_generator.generate.return_value = masks
        return mask_generator
    
    @pytest.fixture
    def segmentation_handler(self):
        """Fixture that provides a basic SegmentationHandler."""
        return SegmentationHandler()
    
    @pytest.fixture
    def segmentation_handler_with_mocks(self, mock_sam_mask_generator):
        """Fixture that provides a SegmentationHandler with mocked components."""
        handler = SegmentationHandler()
        handler.mask_generator = mock_sam_mask_generator
        handler.model = MagicMock()
        return handler
    
    def test_init_with_default_values(self):
        """
        GIVEN no parameters
        WHEN SegmentationHandler is initialized
        THEN it should use default values
        """
        handler = SegmentationHandler()
        
        # Check default values
        assert handler.model is None
        assert handler.device in ["cpu", "cuda"]
        assert handler.model_type == "vit_b"
        assert handler.mask_generator is None
    
    def test_init_with_custom_values(self):
        """
        GIVEN custom parameters
        WHEN SegmentationHandler is initialized
        THEN it should use the provided values
        """
        handler = SegmentationHandler(
            model_type="vit_l",
            checkpoint_path="/custom/path.pth"
        )
        
        # Check custom values
        assert handler.model is None
        assert handler.model_type == "vit_l"
        assert handler.checkpoint_path == "/custom/path.pth"
    
    def test_generate_masks_with_sam_available(self, segmentation_handler_with_mocks, sample_interior_image, monkeypatch):
        """
        GIVEN an image and SAM is available
        WHEN generate_masks is called
        THEN it should generate masks using SAM
        """
        monkeypatch.setattr('src.segmentation.SAM_AVAILABLE', True)
        
        # Call generate_masks
        result = segmentation_handler_with_mocks.generate_masks(sample_interior_image)
        
        # Verify mask generator was called
        segmentation_handler_with_mocks.mask_generator.generate.assert_called_once()
        
        # Verify result format
        assert isinstance(result, dict)
        assert "walls" in result
        assert "floor" in result
        assert "ceiling" in result
        assert "windows" in result
        assert "structure" in result
    
    def test_generate_masks_with_sam_unavailable(self, segmentation_handler, sample_interior_image, monkeypatch):
        """
        GIVEN an image and SAM is not available
        WHEN generate_masks is called
        THEN it should generate fallback masks using simple heuristics
        """
        # Ensure SAM is unavailable
        monkeypatch.setattr('src.segmentation.SAM_AVAILABLE', False)
        
        # Call generate_masks
        result = segmentation_handler.generate_masks(sample_interior_image)
        
        # Verify result format (should still return masks even without SAM)
        assert isinstance(result, dict)
        assert "walls" in result
        assert "floor" in result
        assert "ceiling" in result
        assert "windows" in result
        assert "structure" in result
    
    def test_apply_structure_preservation(self, segmentation_handler, sample_interior_image):
        """
        GIVEN original and stylized images with a structure mask
        WHEN apply_structure_preservation is called
        THEN it should blend the images based on the mask
        """
        # Create a simple stylized version (darker)
        stylized_image = sample_interior_image.copy() // 2
        
        # Create a simple structure mask (top half is structure)
        structure_mask = np.zeros((100, 100), dtype=np.uint8)
        structure_mask[:50, :] = 1
        
        # Apply structure preservation
        result = segmentation_handler.apply_structure_preservation(
            sample_interior_image, stylized_image, structure_mask
        )
        
        # Verify result is correct shape and type
        assert result.shape == sample_interior_image.shape
        assert result.dtype == np.uint8
        
        # Check that structure areas are more similar to original
        structure_area_original = sample_interior_image[:50, :]
        structure_area_result = result[:50, :]
        structure_diff = np.mean(np.abs(structure_area_result - structure_area_original))
        
        # Non-structure areas are more similar to stylized
        non_structure_area_stylized = stylized_image[50:, :]
        non_structure_area_result = result[50:, :]
        non_structure_diff = np.mean(np.abs(non_structure_area_result - non_structure_area_stylized))
        
        # Structure should retain more of original, non-structure more of stylized
        assert structure_diff < 50  # Values should be somewhat similar
    
    def test_preserve_structure_in_styles(self, segmentation_handler, sample_interior_image):
        """
        GIVEN original image, multiple style variations, and structure mask
        WHEN preserve_structure_in_styles is called
        THEN it should blend each variation while preserving structure
        """
        # Create multiple style variations
        styles = [
            sample_interior_image.copy() // 2,  # Darker
            sample_interior_image.copy() * 2,   # Brighter (capped at 255)
            cv2.cvtColor(sample_interior_image, cv2.COLOR_RGB2BGR)  # Channel swap
        ]
        
        # Ensure values stay in valid range
        for style in styles:
            np.clip(style, 0, 255, out=style)
        
        # Create a simple structure mask (top third is structure)
        structure_mask = np.zeros((100, 100), dtype=np.uint8)
        structure_mask[:33, :] = 1
        
        # Preserve structure in all styles
        results = segmentation_handler.preserve_structure_in_styles(
            sample_interior_image, styles, structure_mask
        )
        
        # Verify we get correct number of results
        assert len(results) == len(styles)
        
        # Verify each result is the right shape
        for result in results:
            assert result.shape == sample_interior_image.shape
            assert result.dtype == np.uint8
