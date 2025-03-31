"""
Test module for style transfer functionality.

BDD-style tests for style-only transfer that preserves the original
layout and structure of interior images.
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
import tempfile
from unittest import mock

# Path manipulations for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_data import create_test_interior_image, get_test_image_path
from src.segmentation import SegmentationHandler


class TestStyleTransfer:
    """
    Tests for style-only transfer functionality.
    
    BDD-style tests that verify applying style prompts while preserving
    the original layout and structure of interior images.
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
    def style_transfer_service(self):
        """Fixture that provides a style transfer service instance."""
        try:
            from src.style_transfer import StyleTransferService
            from src.flux_integration import FluxClient
            
            # Use a mock API key for testing
            flux_client = FluxClient(api_key="test_key")
            return StyleTransferService(flux_client=flux_client)
        except ImportError:
            pytest.skip("StyleTransferService not implemented yet")
    
    @pytest.fixture
    def sample_masks(self, segmentation_handler, sample_image_path):
        """Fixture that provides masks for a sample image."""
        img = np.array(Image.open(sample_image_path))
        return segmentation_handler.generate_masks(img)
    
    def test_style_transfer_service_initialization(self):
        """
        GIVEN a FluxClient instance
        WHEN initializing a StyleTransferService with the client
        THEN it should properly store the client
        """
        try:
            from src.style_transfer import StyleTransferService
            from src.flux_integration import FluxClient
            
            # Test initialization with FluxClient
            flux_client = FluxClient(api_key="test_key")
            service = StyleTransferService(flux_client=flux_client)
            
            assert service.flux_client is not None
            assert service.flux_client == flux_client
            
        except ImportError:
            pytest.skip("StyleTransferService not implemented yet")
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_apply_style_only(self, mock_apply_style, style_transfer_service, sample_image_path, sample_masks):
        """
        GIVEN a StyleTransferService, an input image, and segmentation masks
        WHEN apply_style_only is called with a style prompt
        THEN it should apply the style while preserving layout and structure
        """
        if not style_transfer_service:
            pytest.skip("StyleTransferService not implemented yet")
        
        # Load test image
        original_image = np.array(Image.open(sample_image_path))
        
        # Create a fake styled result
        styled_result = original_image.copy()
        # Modify colors to simulate styling
        styled_result = styled_result * 0.8 + np.array([50, 30, 10])
        styled_result = np.clip(styled_result, 0, 255).astype(np.uint8)
        
        # Configure mock to return our fake styled result
        mock_apply_style.return_value = styled_result
        
        # Define style prompt
        style_prompt = "A mid-century modern living room with earth tones"
        
        # Call the function being tested
        result = style_transfer_service.apply_style_only(
            original_image, 
            style_prompt=style_prompt,
            structure_mask=sample_masks["structure"]
        )
        
        # Assertions
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == original_image.shape
        
        # Verify Flux client was called with the correct parameters
        mock_apply_style.assert_called_once()
        args, kwargs = mock_apply_style.call_args
        assert kwargs["prompt"] == style_prompt
        
        # Verify structure preservation
        from src.segmentation import SegmentationHandler
        handler = SegmentationHandler()
        preservation_score = handler.estimate_structure_preservation_score(
            original_image=original_image,
            stylized_image=result,
            structure_mask=sample_masks["structure"]
        )
        assert preservation_score > 0.8, "Structure should be well preserved"
    
    @mock.patch("src.flux_integration.FluxClient.generate_style_variations")
    def test_generate_style_only_variations(self, mock_generate_variations, 
                                          style_transfer_service, sample_image_path, sample_masks):
        """
        GIVEN a StyleTransferService, an input image, and segmentation masks
        WHEN generate_style_only_variations is called with a style prompt
        THEN it should return multiple styled variations that preserve layout and structure
        """
        if not style_transfer_service:
            pytest.skip("StyleTransferService not implemented yet")
        
        # Load test image
        original_image = np.array(Image.open(sample_image_path))
        
        # Create fake style variations
        variations = []
        for i in range(3):
            var = original_image.copy()
            # Different color modifications for each variant
            var = var * (0.7 + 0.1*i) + np.array([40+i*10, 30+i*5, 10+i*15])
            var = np.clip(var, 0, 255).astype(np.uint8)
            variations.append(var)
        
        # Configure mock to return our fake variations
        mock_generate_variations.return_value = variations
        
        # Define style prompt
        style_prompt = "A Scandinavian style living room"
        
        # Call the function being tested
        results = style_transfer_service.generate_style_only_variations(
            original_image, 
            style_prompt=style_prompt, 
            structure_mask=sample_masks["structure"],
            num_variations=3
        )
        
        # Assertions
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == 3
        
        # Verify Flux client was called with the correct parameters
        mock_generate_variations.assert_called_once()
        args, kwargs = mock_generate_variations.call_args
        assert kwargs["prompt"] == style_prompt
        assert kwargs["num_variations"] == 3
        
        # Verify structure preservation in all variations
        from src.segmentation import SegmentationHandler
        handler = SegmentationHandler()
        
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == original_image.shape
            
            preservation_score = handler.estimate_structure_preservation_score(
                original_image=original_image,
                stylized_image=result,
                structure_mask=sample_masks["structure"]
            )
            assert preservation_score > 0.8, "Structure should be well preserved in all variations"
    
    def test_style_prompt_formatting(self):
        """
        GIVEN a StyleTransferService
        WHEN format_style_prompt is called with a style name
        THEN it should return a properly formatted prompt for that style
        """
        try:
            from src.style_transfer import StyleTransferService
            
            # Should be able to create service without a FluxClient for this test
            service = StyleTransferService()
            
            # Test basic prompt formatting
            basic_prompt = service.format_style_prompt("Scandinavian")
            assert "Scandinavian" in basic_prompt or "scandinavian" in basic_prompt.lower()
            assert len(basic_prompt) > len("Scandinavian")
            
            # Test with room type
            room_prompt = service.format_style_prompt("Industrial", room_type="bedroom")
            assert "Industrial" in room_prompt or "industrial" in room_prompt.lower()
            assert "bedroom" in room_prompt.lower()
            
            # Test with additional details
            detailed_prompt = service.format_style_prompt(
                "Minimalist",
                room_type="living room",
                details="with clean lines and neutral colors"
            )
            assert "Minimalist" in detailed_prompt or "minimalist" in detailed_prompt.lower()
            assert "living room" in detailed_prompt.lower()
            assert "clean lines" in detailed_prompt.lower()
            
        except ImportError:
            pytest.skip("StyleTransferService not implemented yet")
