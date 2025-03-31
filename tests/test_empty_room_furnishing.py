"""
Test suite for empty room furnishing functionality.

Following Semantic Seed Coding Standards and BDD/TDD approach:
- Test cases are written before implementation
- Tests follow BDD "Given-When-Then" structure
- Each test has a docstring explaining its purpose
"""

import os
import pytest
import numpy as np
from unittest import mock
import cv2
from PIL import Image

class TestEmptyRoomFurnishing:
    """Test suite for the empty room furnishing functionality."""
    
    @pytest.fixture
    def empty_room_service(self):
        """Fixture that provides an empty room furnishing service instance."""
        from src.empty_room_furnishing import EmptyRoomFurnishingService
        from src.flux_integration import FluxClient
        
        # Use a mock API key for testing
        flux_client = FluxClient(api_key="test_key")
        return EmptyRoomFurnishingService(flux_client=flux_client)
    
    @pytest.fixture
    def sample_empty_room_image(self):
        """Fixture that provides a sample empty room image."""
        # Create a synthetic empty room image for testing
        height, width = 512, 512
        # Create a simple image with a floor and walls
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw floor (bottom half) in light beige
        image[height//2:, :] = [240, 230, 200]
        
        # Draw left wall in light gray
        image[:height//2, :width//2] = [230, 230, 230]
        
        # Draw right wall in slightly different gray
        image[:height//2, width//2:] = [210, 210, 210]
        
        return image
    
    @pytest.fixture
    def segmentation_handler(self):
        """Fixture that provides a segmentation handler."""
        from src.segmentation import SegmentationHandler
        
        # Return a mocked segmentation handler to avoid loading the model
        with mock.patch("src.segmentation.SegmentationHandler.load_model"):
            with mock.patch("src.segmentation.torch.cuda.is_available", return_value=False):
                # Create the handler with model_type and checkpoint_path
                handler = SegmentationHandler(model_type="vit_b", checkpoint_path="mock_path")
                
                # Mock the create_mask method to return plausible mask data
                def mock_create_mask(*args, **kwargs):
                    height, width = 512, 512
                    mask = np.zeros((height, width), dtype=np.uint8)
                    
                    # Make the bottom half (floor) part of the mask
                    mask[height//2:, :] = 1
                    
                    return mask
                
                handler.create_mask = mock.MagicMock(side_effect=mock_create_mask)
                return handler
        
    def test_empty_room_service_initialization(self):
        """
        GIVEN necessary components
        WHEN initializing an EmptyRoomFurnishingService
        THEN it should properly configure itself
        """
        from src.empty_room_furnishing import EmptyRoomFurnishingService
        from src.flux_integration import FluxClient
        
        # Test initialization with FluxClient
        flux_client = FluxClient(api_key="test_key")
        service = EmptyRoomFurnishingService(flux_client=flux_client)
        
        assert service.flux_client is not None
        assert service.flux_client == flux_client
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_furnish_empty_room(self, mock_apply_style, empty_room_service, sample_empty_room_image):
        """
        GIVEN an empty room image
        WHEN applying furnishing
        THEN it should return a furnished image
        """
        # Mock the response from the flux API
        mock_furnished_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        mock_apply_style.return_value = mock_furnished_image
        
        # Call the furnish method
        result = empty_room_service.furnish_room(
            sample_empty_room_image, 
            style="modern",
            furniture="couch, coffee table, TV stand",
            room_type="living room"
        )
        
        # Verify the result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_empty_room_image.shape
        
        # Verify the mock was called with appropriate parameters
        mock_apply_style.assert_called_once()
        
        # Check that the prompt contains furniture items and room type
        called_args = mock_apply_style.call_args[0]
        prompt = called_args[1]  # Second argument should be the prompt
        assert "modern" in prompt.lower()
        assert "couch" in prompt.lower()
        assert "coffee table" in prompt.lower()
        assert "tv stand" in prompt.lower()
        assert "living room" in prompt.lower()
    
    def test_furnish_prompt_formatting(self):
        """
        GIVEN furniture items and style
        WHEN format_furnishing_prompt is called
        THEN it should return a properly formatted prompt
        """
        from src.empty_room_furnishing import EmptyRoomFurnishingService
        
        service = EmptyRoomFurnishingService()
        
        # Test basic prompt formatting
        basic_prompt = service.format_furnishing_prompt(
            furniture="sofa and coffee table"
        )
        assert "sofa" in basic_prompt.lower()
        assert "coffee table" in basic_prompt.lower()
        assert "furnish" in basic_prompt.lower() or "furniture" in basic_prompt.lower()
        
        # Test with room type
        room_prompt = service.format_furnishing_prompt(
            furniture="bed, nightstands, dresser",
            room_type="bedroom"
        )
        assert "bed" in room_prompt.lower()
        assert "bedroom" in room_prompt.lower()
        
        # Test with style
        style_prompt = service.format_furnishing_prompt(
            furniture="dining table with six chairs, sideboard",
            room_type="dining room",
            style="Scandinavian"
        )
        assert "dining table" in style_prompt.lower()
        assert "dining room" in style_prompt.lower()
        assert "scandinavian" in style_prompt.lower()
        
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_structure_preservation(self, mock_apply_style, empty_room_service, 
                                   sample_empty_room_image, segmentation_handler):
        """
        GIVEN an empty room image and segmentation masks
        WHEN furnishing with structure preservation
        THEN it should maintain the architectural elements
        """
        # Mock the response from the flux API
        mock_furnished_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        mock_apply_style.return_value = mock_furnished_image
        
        # Create masks for structural elements
        empty_room_service.segmentation_handler = segmentation_handler
        
        # Call the furnish method with structure preservation
        result = empty_room_service.furnish_room(
            sample_empty_room_image, 
            style="modern",
            furniture="sectional sofa, coffee table, floor lamp",
            room_type="living room",
            preserve_structure=True
        )
        
        # Verify the result
        assert result is not None
        assert isinstance(result, np.ndarray)
        
        # Verify structure preservation logic was used
        mock_apply_style.assert_called_once()
        # Check the prompt includes structure preservation instructions
        called_args = mock_apply_style.call_args[0]
        prompt = called_args[1]  # Second argument should be the prompt
        assert "preserve" in prompt.lower() or "maintain" in prompt.lower()
        assert "structure" in prompt.lower() or "architecture" in prompt.lower()
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_generate_furnishing_variations(self, mock_apply_style, empty_room_service, 
                                           sample_empty_room_image):
        """
        GIVEN an empty room image
        WHEN generating multiple furnishing variations
        THEN it should return multiple unique designs
        """
        # Mock multiple responses for variations
        variation1 = np.ones((512, 512, 3), dtype=np.uint8) * 180
        variation2 = np.ones((512, 512, 3), dtype=np.uint8) * 200
        variation3 = np.ones((512, 512, 3), dtype=np.uint8) * 220
        
        mock_apply_style.side_effect = [variation1, variation2, variation3]
        
        # Generate 3 variations
        variations = empty_room_service.generate_furnishing_variations(
            sample_empty_room_image,
            style="bohemian",
            furniture="bookshelf, reading chair, small table",
            room_type="study",
            num_variations=3
        )
        
        # Verify we got the expected number of variations
        assert len(variations) == 3
        
        # Verify all variations are unique (in a real scenario)
        # Here we're just checking our mocked values are different
        assert not np.array_equal(variations[0], variations[1])
        assert not np.array_equal(variations[1], variations[2])
        
        # Verify the mock was called the correct number of times
        assert mock_apply_style.call_count == 3
