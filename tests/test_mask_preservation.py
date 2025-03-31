"""
Test module for structure preservation with masks.

BDD-style tests for applying segmentation masks to preserve room structure
following Semantic Seed Coding Standards.
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
import cv2

# Path manipulations for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import SegmentationHandler
from tests.test_data import create_test_interior_image, get_test_image_path


class TestMaskPreservation:
    """
    Tests for structure preservation through masking.
    
    Follows BDD-style testing to verify that architectural elements
    (walls, floors, ceilings) are preserved during transformations.
    """
    
    @pytest.fixture
    def sample_image_path(self):
        """Fixture that provides path to test image."""
        return get_test_image_path()
    
    @pytest.fixture
    def segmentation_handler(self):
        """Fixture that provides a segmentation handler instance."""
        return SegmentationHandler(model_type="vit_b")
    
    @pytest.fixture
    def sample_masks(self, segmentation_handler, sample_image_path):
        """Fixture that provides masks for a sample image."""
        img = np.array(Image.open(sample_image_path))
        return segmentation_handler.generate_masks(img)
    
    @pytest.fixture
    def stylized_image(self, sample_image_path):
        """
        Fixture that provides a fake 'stylized' image for testing.
        
        In a real scenario, this would come from Flux API.
        """
        # Load the test image
        original = np.array(Image.open(sample_image_path))
        
        # Create a 'stylized' version by applying some color changes
        # This simulates what might come back from a style transfer model
        stylized = original.copy()
        
        # Apply some color transformations to simulate style transfer
        # Increase saturation and adjust hue
        hsv = cv2.cvtColor(stylized, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
        hsv[:, :, 0] = (hsv[:, :, 0] + 20) % 180  # Shift hue
        stylized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return stylized
    
    def test_apply_structure_mask(self, segmentation_handler, sample_image_path, sample_masks, stylized_image):
        """
        GIVEN a segmentation handler, original image, structure masks, and a stylized image
        WHEN apply_structure_preservation is called
        THEN it should return an image where structure is from original but style from stylized
        """
        # Ensure the function exists
        assert hasattr(segmentation_handler, 'apply_structure_preservation'), \
            "SegmentationHandler missing method 'apply_structure_preservation'"
        
        # Load the original image
        original = np.array(Image.open(sample_image_path))
        
        # Apply structure preservation
        preserved = segmentation_handler.apply_structure_preservation(
            original_image=original,
            stylized_image=stylized_image,
            structure_mask=sample_masks["structure"]
        )
        
        # Verify the output
        assert isinstance(preserved, np.ndarray), "Result should be a numpy array"
        assert preserved.shape == original.shape, "Result should have the same shape as input"
        
        # Test that masked areas (structure) are preserved from the original
        # and unmasked areas take style from the stylized image
        structure_mask = sample_masks["structure"]
        
        # Get the structure region (where mask == 1)
        structure_pixels_original = original[structure_mask == 1]
        structure_pixels_result = preserved[structure_mask == 1]
        
        # Get the non-structure region (where mask == 0)
        non_structure_pixels_stylized = stylized_image[structure_mask == 0]
        non_structure_pixels_result = preserved[structure_mask == 0]
        
        # Calculate mean absolute difference
        structure_diff = np.mean(np.abs(structure_pixels_original - structure_pixels_result))
        non_structure_diff = np.mean(np.abs(non_structure_pixels_stylized - non_structure_pixels_result))
        
        # Structure pixels should be closer to original than to stylized
        assert structure_diff < 10, "Structure pixels should be preserved from original"
        
        # Non-structure pixels should be closer to stylized than to original
        assert non_structure_diff < 10, "Non-structure pixels should come from stylized image"
    
    def test_preserve_structure_in_styles(self, segmentation_handler, sample_image_path, sample_masks):
        """
        GIVEN a segmentation handler, original image, and structure masks
        WHEN preserve_structure_in_styles is called with different style variations
        THEN it should return a list of images where structure is preserved in all
        """
        # Ensure the function exists
        assert hasattr(segmentation_handler, 'preserve_structure_in_styles'), \
            "SegmentationHandler missing method 'preserve_structure_in_styles'"
        
        # Load the original image
        original = np.array(Image.open(sample_image_path))
        
        # Create fake style variations for testing
        style_variations = []
        for i in range(3):  # Create 3 different styles
            variation = original.copy()
            
            # Create different styles by adjusting colors
            hsv = cv2.cvtColor(variation, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.2 + i*0.3), 0, 255)  # Different saturations
            hsv[:, :, 0] = (hsv[:, :, 0] + 30*i) % 180  # Different hues
            variation = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            style_variations.append(variation)
        
        # Apply structure preservation to all style variations
        preserved_variations = segmentation_handler.preserve_structure_in_styles(
            original_image=original,
            style_variations=style_variations,
            structure_mask=sample_masks["structure"]
        )
        
        # Verify the output
        assert isinstance(preserved_variations, list), "Result should be a list"
        assert len(preserved_variations) == len(style_variations), "Should return same number of variations"
        
        for preserved in preserved_variations:
            assert isinstance(preserved, np.ndarray), "Each result should be a numpy array"
            assert preserved.shape == original.shape, "Each result should have the same shape as input"
            
            # Get the structure region (where mask == 1)
            structure_mask = sample_masks["structure"]
            structure_pixels_original = original[structure_mask == 1]
            structure_pixels_result = preserved[structure_mask == 1]
            
            # Calculate mean absolute difference for structure regions
            structure_diff = np.mean(np.abs(structure_pixels_original - structure_pixels_result))
            
            # Structure should be preserved in all variations
            assert structure_diff < 10, "Structure should be preserved in all style variations"
    
    def test_estimate_structure_preservation_score(self, segmentation_handler, sample_image_path, 
                                                 sample_masks, stylized_image):
        """
        GIVEN a segmentation handler, original image, structure masks, and a stylized image
        WHEN estimate_structure_preservation_score is called
        THEN it should return a score indicating how well structure is preserved
        """
        # Ensure the function exists
        assert hasattr(segmentation_handler, 'estimate_structure_preservation_score'), \
            "SegmentationHandler missing method 'estimate_structure_preservation_score'"
        
        # Load the original image
        original = np.array(Image.open(sample_image_path))
        
        # Get structure preservation score
        score = segmentation_handler.estimate_structure_preservation_score(
            original_image=original,
            stylized_image=stylized_image,
            structure_mask=sample_masks["structure"]
        )
        
        # Verify the output
        assert isinstance(score, float), "Score should be a float"
        assert 0 <= score <= 1, "Score should be between 0 and 1"
        
        # Create a badly preserved image where structure is not maintained
        bad_preservation = np.random.randint(0, 255, size=original.shape, dtype=np.uint8)
        
        # Get score for badly preserved image
        bad_score = segmentation_handler.estimate_structure_preservation_score(
            original_image=original,
            stylized_image=bad_preservation,
            structure_mask=sample_masks["structure"]
        )
        
        # The good preservation should have a higher score than the bad one
        assert score > bad_score, "Well-preserved structure should have higher score than poorly preserved"
