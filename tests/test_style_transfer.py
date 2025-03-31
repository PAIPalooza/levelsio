"""
Test module for style transfer functionality.

BDD-style tests for style-only transfer that preserves the original
layout and structure of interior images.
"""

import os
import pytest
import numpy as np
from unittest import mock
from PIL import Image

from src.style_transfer import StyleTransferService
from src.flux_integration import FluxClient
from src.segmentation import SegmentationHandler


class TestStyleTransfer:
    """
    Tests for style-only transfer functionality.
    
    BDD-style tests that verify applying style prompts while preserving
    the original layout and structure of interior images.
    """
    
    @pytest.fixture
    def style_transfer_service(self):
        """Fixture that provides a StyleTransferService instance."""
        # Create mocked flux client
        flux_client = mock.MagicMock(spec=FluxClient)
        
        # Configure the apply_style_transfer mock to return the input image
        def mock_apply_style(**kwargs):
            # If an image is provided, return it with slight modifications
            if 'image' in kwargs:
                modified = kwargs['image'].copy()
                # Add slight color tint to simulate style transfer
                modified = modified * 0.9 + 20
                return np.clip(modified, 0, 255).astype(np.uint8)
            return np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        flux_client.apply_style_transfer = mock.MagicMock(side_effect=mock_apply_style)
        
        # Configure the generate_style_variations mock to return variations
        def mock_generate_variations(**kwargs):
            # If an image is provided, return variations
            if 'image' in kwargs:
                base_image = kwargs['image'].copy()
                variations = []
                for i in range(kwargs.get('num_variations', 3)):
                    variant = base_image * (0.8 + i*0.05) + i*20
                    variations.append(np.clip(variant, 0, 255).astype(np.uint8))
                return variations
            return [np.ones((100, 100, 3), dtype=np.uint8) * (150 + i*30) for i in range(3)]
        
        flux_client.generate_style_variations = mock.MagicMock(side_effect=mock_generate_variations)
        
        # Create mocked segmentation handler
        segmentation_handler = mock.MagicMock(spec=SegmentationHandler)
        
        # Mock structure preservation methods
        def mock_apply_structure_preservation(original_image, stylized_image, structure_mask):
            return stylized_image
        segmentation_handler.apply_structure_preservation = mock.MagicMock(
            side_effect=mock_apply_structure_preservation
        )
        
        def mock_preserve_structure_in_styles(original_image, style_variations, structure_mask):
            return style_variations
        segmentation_handler.preserve_structure_in_styles = mock.MagicMock(
            side_effect=mock_preserve_structure_in_styles
        )
        
        # Create service with mocked dependencies
        service = StyleTransferService(
            flux_client=flux_client,
            segmentation_handler=segmentation_handler
        )
        
        return service
    
    @pytest.fixture
    def sample_image_path(self):
        """Fixture that provides a path to a sample image."""
        # Use an image from the test assets
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'assets', 'test_images', 'test_interior.jpg')
    
    @pytest.fixture
    def sample_masks(self):
        """Fixture that provides sample segmentation masks."""
        # Create simple test masks (100x100 for simplicity)
        size = (100, 100)
        
        wall_mask = np.zeros(size, dtype=np.uint8)
        wall_mask[:50, :] = 1  # Top half is wall
        
        floor_mask = np.zeros(size, dtype=np.uint8)
        floor_mask[50:, :] = 1  # Bottom half is floor
        
        ceiling_mask = np.ones(size, dtype=np.uint8)  # Whole image is ceiling for simplicity
        
        furniture_mask = np.zeros(size, dtype=np.uint8)
        furniture_mask[60:80, 30:70] = 1  # Rectangle in the middle-bottom is furniture
        
        # Combined mask for structure
        structure_mask = np.zeros(size, dtype=np.uint8)
        structure_mask[:50, :] = 1  # Walls
        structure_mask[50:, :] = 1  # Floors
        
        return {
            'wall': wall_mask,
            'floor': floor_mask,
            'ceiling': ceiling_mask,
            'furniture': furniture_mask,
            'structure': structure_mask
        }
    
    def test_style_transfer_service_initialization(self, style_transfer_service):
        """
        GIVEN a FluxClient instance
        WHEN initializing a StyleTransferService with the client
        THEN it should properly store the client
        """
        assert style_transfer_service is not None
        assert style_transfer_service.flux_client is not None
        assert style_transfer_service.segmentation_handler is not None
    
    def test_apply_style_only(self, style_transfer_service, sample_image_path, sample_masks):
        """
        GIVEN a StyleTransferService, an input image, and segmentation masks
        WHEN apply_style_only is called with a style prompt
        THEN it should apply the style while preserving layout and structure
        """
        if not style_transfer_service:
            pytest.skip("StyleTransferService not implemented yet")
        
        # Load test image
        original_image = np.array(Image.open(sample_image_path))
        
        # Define style prompt
        style_prompt = "A mid-century modern living room with earth tones"
        
        # Call the function being tested with preserve_structure=False to avoid prompt modification
        result = style_transfer_service.apply_style_only(
            original_image, 
            style_prompt=style_prompt,
            structure_mask=sample_masks["structure"],
            preserve_structure=False  # Don't modify the prompt for testing
        )
        
        # Assertions
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == original_image.shape
        
        # Verify Flux client was called with the correct parameters
        assert style_transfer_service.flux_client.apply_style_transfer.called
        call_kwargs = style_transfer_service.flux_client.apply_style_transfer.call_args[1]
        assert call_kwargs["prompt"] == style_prompt
        
        # Test with preserve_structure=True but check method calls instead of prompt
        result_with_preservation = style_transfer_service.apply_style_only(
            original_image, 
            style_prompt=style_prompt,
            structure_mask=sample_masks["structure"],
            preserve_structure=True
        )
        
        # Verify structure preservation logic was used
        assert style_transfer_service.segmentation_handler.apply_structure_preservation.called
        assert result_with_preservation is not None
        assert isinstance(result_with_preservation, np.ndarray)
    
    def test_generate_style_only_variations(self, style_transfer_service, sample_image_path, sample_masks):
        """
        GIVEN a StyleTransferService, an input image, and segmentation masks
        WHEN generate_style_only_variations is called with a style prompt
        THEN it should return multiple styled variations that preserve layout and structure
        """
        if not style_transfer_service:
            pytest.skip("StyleTransferService not implemented yet")
        
        # Load test image
        original_image = np.array(Image.open(sample_image_path))
        
        # Define style prompt
        style_prompt = "A Scandinavian style living room"
        
        # Call the function being tested with preserve_structure=False
        results = style_transfer_service.generate_style_only_variations(
            original_image,
            style_prompt=style_prompt,
            structure_mask=sample_masks["structure"],
            preserve_structure=False,  # Don't modify the prompt for testing
            num_variations=3
        )
        
        # Assertions
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == 3
        
        # Verify each variation is a valid image
        for variation in results:
            assert isinstance(variation, np.ndarray)
            assert variation.shape == original_image.shape
        
        # Verify Flux client was called with the correct parameters
        assert style_transfer_service.flux_client.generate_style_variations.called
        call_kwargs = style_transfer_service.flux_client.generate_style_variations.call_args[1]
        assert call_kwargs["prompt"] == style_prompt
        
        # Test with preserve_structure=True but check method calls instead of prompt
        results_with_preservation = style_transfer_service.generate_style_only_variations(
            original_image,
            style_prompt=style_prompt,
            structure_mask=sample_masks["structure"],
            preserve_structure=True,
            num_variations=3
        )
        
        # Verify structure preservation was used
        assert style_transfer_service.segmentation_handler.preserve_structure_in_styles.called
        assert results_with_preservation is not None
        assert isinstance(results_with_preservation, list)
    
    def test_style_prompt_formatting(self, style_transfer_service):
        """
        GIVEN a StyleTransferService
        WHEN format_style_prompt is called with a style name
        THEN it should return a properly formatted prompt for that style
        """
        if not style_transfer_service:
            pytest.skip("StyleTransferService not implemented yet")
        
        # Test with known style
        scandinavian_prompt = style_transfer_service.format_style_prompt(
            style_name="Scandinavian", 
            room_type="living room"
        )
        
        assert "Scandinavian" in scandinavian_prompt
        assert "living room" in scandinavian_prompt
        
        # Test with custom style
        custom_prompt = style_transfer_service.format_style_prompt(
            style_name="Japandi", 
            room_type="bedroom", 
            details="with natural materials and minimalist furniture"
        )
        
        assert "Japandi" in custom_prompt
        assert "bedroom" in custom_prompt
        assert "natural materials" in custom_prompt
