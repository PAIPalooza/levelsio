"""
Extended test coverage module for segmentation functionality.

This module contains targeted BDD-style tests for the SegmentationHandler class
specifically focusing on previously uncovered lines in segmentation.py to exceed
industry-standard coverage requirements.
"""

import pytest
import numpy as np
from unittest import mock
from unittest.mock import MagicMock, patch
import os
import tempfile
from PIL import Image
import torch
import cv2
import urllib

from src.segmentation import SegmentationHandler, SAM_AVAILABLE


class TestSegmentationHandlerExtended:
    """
    Extended test suite for SegmentationHandler.
    
    These tests specifically target lines 82-109, 149-214, and other uncovered
    sections to improve code coverage to industry-leading standards.
    """
    
    @pytest.fixture
    def sample_interior_image(self):
        """Fixture providing a sample interior image."""
        # Create a complex interior image with multiple features
        height, width = 100, 100
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # More detailed image structure for better testing
        # Ceiling (gradual color change at top)
        for i in range(20):
            image[i, :] = [230 - i, 230 - i, 230 - i]
        
        # Floor (textured bottom)
        for i in range(20):
            image[80 + i, :] = [150 + i, 140 + i, 130 + i]
            # Add floor pattern
            image[80 + i, ::4] = [120 + i, 110 + i, 100 + i]
        
        # Left wall with window
        image[:80, :20] = [210, 210, 210]
        image[30:50, 5:15] = [240, 230, 200]  # Window
        
        # Right wall with door
        image[:80, 80:] = [200, 200, 200]
        image[30:70, 85:95] = [180, 160, 140]  # Door
        
        # Central furniture (sofa)
        image[60:75, 30:70] = [100, 120, 140]
        
        return image
    
    @pytest.fixture
    def segmentation_handler(self):
        """Fixture providing a basic SegmentationHandler."""
        return SegmentationHandler()
    
    @pytest.fixture
    def mocked_sam_model(self):
        """Fixture providing a mocked SAM model."""
        mock_model = MagicMock()
        return mock_model
    
    def test_is_cuda_available(self, segmentation_handler, monkeypatch):
        """
        GIVEN a SegmentationHandler
        WHEN _is_cuda_available is called in different environments
        THEN it should correctly detect CUDA availability
        """
        # Test when CUDA is available
        with patch('torch.cuda.is_available', return_value=True):
            assert segmentation_handler._is_cuda_available() is True
        
        # Test when CUDA is not available
        with patch('torch.cuda.is_available', return_value=False):
            assert segmentation_handler._is_cuda_available() is False
        
        # Test when torch.cuda raises ImportError
        def raise_import_error():
            raise ImportError("CUDA not available")
        
        monkeypatch.setattr('torch.cuda.is_available', raise_import_error)
        assert segmentation_handler._is_cuda_available() is False
    
    def test_compute_md5(self, segmentation_handler, tmp_path):
        """
        GIVEN a file path
        WHEN _compute_md5 is called
        THEN it should return the correct MD5 hash
        """
        # Create a test file with known content
        test_file = os.path.join(tmp_path, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Calculate MD5 hash
        md5_hash = segmentation_handler._compute_md5(test_file)
        
        # Known MD5 hash for "test content"
        expected_hash = "9473fdd0d880a43c21b7778d34872157"
        
        # Check if calculated hash matches expected
        assert md5_hash == expected_hash
    
    def test_verify_checkpoint(self, segmentation_handler, tmp_path, monkeypatch):
        """
        GIVEN a checkpoint file path and expected hash
        WHEN _verify_checkpoint is called
        THEN it should verify the checkpoint's integrity
        """
        # Create a test checkpoint file
        test_checkpoint = os.path.join(tmp_path, "test_checkpoint.pth")
        with open(test_checkpoint, "w") as f:
            f.write("mock checkpoint data")
        
        # Mock _compute_md5
        def mock_compute_md5(file_path):
            return "mock_md5_hash"
        
        monkeypatch.setattr(segmentation_handler, '_compute_md5', mock_compute_md5)
        
        # Verify with matching hash
        assert segmentation_handler._verify_checkpoint(test_checkpoint, "mock_md5_hash") is True
        
        # Verify with non-matching hash
        assert segmentation_handler._verify_checkpoint(test_checkpoint, "different_hash") is False
        
        # Verify non-existent file
        assert segmentation_handler._verify_checkpoint(os.path.join(tmp_path, "nonexistent.pth"), "any_hash") is False
    
    def test_download_checkpoint_patched(self, segmentation_handler, tmp_path, monkeypatch):
        """
        GIVEN a segmentation handler and mocked download functions
        WHEN download_checkpoint is called
        THEN it should handle different scenarios correctly
        """
        # Prepare a mock checkpoint path
        test_checkpoint_path = os.path.join(tmp_path, "sam_vit_b_01ec64.pth")
        segmentation_handler.checkpoint_path = test_checkpoint_path
        
        # Case 1: File exists and hash matches
        with patch('os.path.exists', return_value=True):
            with patch.object(segmentation_handler, '_verify_checkpoint', return_value=True):
                assert segmentation_handler.download_checkpoint() is True
        
        # Case 2: File exists but hash doesn't match (redownload)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"mock_data"]
        
        with patch('os.path.exists', return_value=True):
            with patch.object(segmentation_handler, '_verify_checkpoint', return_value=False):
                with patch('urllib.request.urlopen', return_value=mock_response):
                    with patch('builtins.open', mock.mock_open()) as mock_file:
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(test_checkpoint_path), exist_ok=True)
                        assert segmentation_handler.download_checkpoint() is True
                        
                        # Check that file was opened for writing
                        mock_file.assert_called_with(test_checkpoint_path, 'wb')
    
    def test_download_checkpoint_new_file(self, segmentation_handler, tmp_path, monkeypatch):
        """
        GIVEN a segmentation handler with no existing checkpoint
        WHEN download_checkpoint is called
        THEN it should download and verify the new checkpoint
        """
        # Set up a temporary checkpoint path
        test_checkpoint_dir = os.path.join(tmp_path, "models")
        os.makedirs(test_checkpoint_dir, exist_ok=True)
        test_checkpoint_path = os.path.join(test_checkpoint_dir, "sam_vit_b_01ec64.pth")
        
        # Mock file operations
        def mock_urlretrieve(url, path, reporthook=None):
            with open(path, 'wb') as f:
                f.write(b"mock checkpoint data")
            return path, None
        
        monkeypatch.setattr('urllib.request.urlretrieve', mock_urlretrieve)
        monkeypatch.setattr(segmentation_handler, '_verify_checkpoint', lambda path, hash: True)
        monkeypatch.setattr('os.path.exists', lambda path: False)
        
        # Temporarily set checkpoint_path
        original_path = segmentation_handler.checkpoint_path
        segmentation_handler.checkpoint_path = test_checkpoint_path
        
        try:
            # Test download
            result = segmentation_handler.download_checkpoint()
            assert result is True
        finally:
            # Restore original path
            segmentation_handler.checkpoint_path = original_path
    
    def test_heuristic_segmentation(self, segmentation_handler, sample_interior_image):
        """
        GIVEN an interior image
        WHEN _heuristic_segmentation is called
        THEN it should return reasonable masks based on color/edge heuristics
        """
        # Call the heuristic segmentation method
        masks = segmentation_handler._heuristic_segmentation(sample_interior_image)
        
        # Verify the result structure
        assert isinstance(masks, dict)
        assert "walls" in masks
        assert "floor" in masks
        assert "ceiling" in masks
        assert "windows" in masks
        assert "structure" in masks
        
        # Each mask should be a binary image of the same size as input
        for mask_name, mask in masks.items():
            assert mask.shape == sample_interior_image.shape[:2]
            assert mask.dtype == np.uint8
            assert set(np.unique(mask)).issubset({0, 1})  # Binary mask
        
        # Structure mask should be the union of other masks
        structure_pixels = np.sum(masks["structure"])
        other_masks_union = np.logical_or.reduce([
            masks["walls"] > 0,
            masks["floor"] > 0,
            masks["ceiling"] > 0,
            masks["windows"] > 0
        ]).astype(np.uint8)
        
        other_pixels = np.sum(other_masks_union)
        
        # Structure should have at least as many pixels as the union of other masks
        assert structure_pixels >= other_pixels
    
    def test_simple_edge_detection(self, segmentation_handler, sample_interior_image):
        """
        GIVEN an interior image
        WHEN _simple_edge_detection is called
        THEN it should detect edges in the image
        """
        # Call the edge detection method
        edges = segmentation_handler._simple_edge_detection(sample_interior_image)
        
        # Verify the result is an edge map
        assert edges.shape == sample_interior_image.shape[:2]
        assert edges.dtype == np.uint8
        
        # An edge map should have some non-zero pixels (edges)
        assert np.sum(edges) > 0
    
    def test_initialize_model(self, segmentation_handler, monkeypatch):
        """
        GIVEN a segmentation handler
        WHEN initialize_model is called with mocked SAM components
        THEN it should initialize the model and components correctly
        """
        # Mock the SAM components
        mock_sam = MagicMock()
        mock_predictor = MagicMock()
        mock_mask_generator = MagicMock()
        
        # Set up patch for SAM registry
        monkeypatch.setattr('src.segmentation.sam_model_registry', mock_sam)
        monkeypatch.setattr('src.segmentation.SamPredictor', lambda model: mock_predictor)
        monkeypatch.setattr('src.segmentation.SamAutomaticMaskGenerator', lambda model, points_per_side: mock_mask_generator)
        
        # Patch the download_checkpoint method
        monkeypatch.setattr(segmentation_handler, 'download_checkpoint', lambda: True)
        
        # Set SAM_AVAILABLE to True
        monkeypatch.setattr('src.segmentation.SAM_AVAILABLE', True)
        
        # Call the initialize_model method
        result = segmentation_handler.initialize_model()
        
        # Verify model was initialized
        assert result is True
        assert segmentation_handler.model is not None
        assert segmentation_handler.predictor is not None
        assert segmentation_handler.mask_generator is not None
        
        # Verify the correct model type and device were used
        mock_sam.get.assert_called_once()
        assert mock_sam.get.call_args[0][0] == segmentation_handler.model_type
    
    def test_initialize_model_sam_unavailable(self, segmentation_handler, monkeypatch):
        """
        GIVEN a segmentation handler
        WHEN initialize_model is called but SAM is unavailable
        THEN it should return False and log a warning
        """
        # Set SAM_AVAILABLE to False
        monkeypatch.setattr('src.segmentation.SAM_AVAILABLE', False)
        
        # Mock logging to check for warning
        mock_logger = MagicMock()
        monkeypatch.setattr('src.segmentation.logger', mock_logger)
        
        # Call the initialize_model method
        result = segmentation_handler.initialize_model()
        
        # Verify it returns False when SAM is unavailable
        assert result is False
        
        # Verify it logged a warning
        mock_logger.warning.assert_called_once()
    
    def test_process_sam_masks(self, segmentation_handler):
        """
        GIVEN SAM mask generation results
        WHEN _process_sam_masks is called
        THEN it should correctly classify and combine masks
        """
        # Create mock SAM masks with different positions and sizes
        h, w = 100, 100
        
        # Create segmentation masks positioned in different areas
        ceiling_mask = np.zeros((h, w), dtype=np.uint8)
        ceiling_mask[:20, :] = 1  # Top 20% - likely ceiling
        
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_mask[80:, :] = 1  # Bottom 20% - likely floor
        
        left_wall = np.zeros((h, w), dtype=np.uint8)
        left_wall[:, :20] = 1  # Left 20% - likely wall
        
        right_wall = np.zeros((h, w), dtype=np.uint8)
        right_wall[:, 80:] = 1  # Right 20% - likely wall
        
        window_mask = np.zeros((h, w), dtype=np.uint8)
        window_mask[30:50, 40:60] = 1  # Center area - likely window
        
        # Format masks like SAM output (area and bbox are used for classification)
        masks = [
            {"segmentation": ceiling_mask, "area": 2000, "bbox": [0, 0, 100, 20]},
            {"segmentation": floor_mask, "area": 2000, "bbox": [0, 80, 100, 20]},
            {"segmentation": left_wall, "area": 2000, "bbox": [0, 0, 20, 100]},
            {"segmentation": right_wall, "area": 2000, "bbox": [80, 0, 20, 100]},
            {"segmentation": window_mask, "area": 400, "bbox": [40, 30, 20, 20]}
        ]
        
        # Process the masks
        result = segmentation_handler._process_sam_masks(masks, (h, w))
        
        # Verify the structure of the result
        assert "walls" in result
        assert "floor" in result
        assert "ceiling" in result
        assert "windows" in result
        assert "structure" in result
        
        # Verify each mask has the right shape
        for mask_name, mask in result.items():
            assert mask.shape == (h, w)
            assert mask.dtype == np.uint8
        
        # Verify the combined structure mask has pixels in all the right places
        structure_mask = result["structure"]
        assert np.any(structure_mask[:20, :])  # Ceiling
        assert np.any(structure_mask[80:, :])  # Floor
        assert np.any(structure_mask[:, :20])  # Left wall
        assert np.any(structure_mask[:, 80:])  # Right wall
    
    def test_blend_with_mask(self, segmentation_handler):
        """
        GIVEN two images and a mask
        WHEN _blend_with_mask is called with different alpha values
        THEN it should blend the images according to the mask and alpha
        """
        # Create sample images and mask
        img1 = np.ones((50, 50, 3), dtype=np.uint8) * 100  # Dark gray
        img2 = np.ones((50, 50, 3), dtype=np.uint8) * 200  # Light gray
        
        # Create a mask (1 in top half, 0 in bottom half)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[:25, :] = 1
        
        # Test blending with different alpha values
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = segmentation_handler._blend_with_mask(img1, img2, mask, alpha)
            
            # Check output shape and type
            assert result.shape == img1.shape
            assert result.dtype == np.uint8
            
            # Top half (mask=1) should be closer to img1 as alpha increases
            top_half_avg = np.mean(result[:25, :])
            bottom_half_avg = np.mean(result[25:, :])
            
            # For alpha=0.0, the result should be entirely img2
            if alpha == 0.0:
                assert np.allclose(result, img2)
            
            # For alpha=1.0, masked region should be entirely img1
            elif alpha == 1.0:
                assert np.allclose(result[:25, :], img1[:25, :])
                assert np.allclose(result[25:, :], img2[25:, :])
            
            # For intermediate alphas, the result should be a blend
            else:
                # Masked area should be between img1 and img2 values
                assert 100 <= top_half_avg <= 200
                # Unmasked area should be img2
                assert bottom_half_avg == 200
