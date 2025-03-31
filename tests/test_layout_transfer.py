"""
Test module for style + layout transfer functionality.

BDD-style tests for style and layout transfer that changes furniture
arrangement while preserving architectural elements.
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


class TestLayoutTransfer:
    """
    Tests for style + layout transfer functionality.
    
    BDD-style tests that verify applying style and layout changes
    while preserving architectural structure.
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
    def layout_transfer_service(self):
        """Fixture that provides a layout transfer service instance."""
        try:
            from src.layout_transfer import LayoutTransferService
            from src.flux_integration import FluxClient
            
            # Use a mock API key for testing
            flux_client = FluxClient(api_key="test_key")
            return LayoutTransferService(flux_client=flux_client)
        except ImportError:
            pytest.skip("LayoutTransferService not implemented yet")
    
    @pytest.fixture
    def sample_masks(self, segmentation_handler, sample_image_path):
        """Fixture that provides masks for a sample image."""
        img = np.array(Image.open(sample_image_path))
        return segmentation_handler.generate_masks(img)
    
    def test_layout_transfer_service_initialization(self):
        """
        GIVEN the necessary components
        WHEN initializing a LayoutTransferService
        THEN it should properly configure itself
        """
        try:
            from src.layout_transfer import LayoutTransferService
            from src.flux_integration import FluxClient
            from src.style_transfer import StyleTransferService
            
            # Test initialization with FluxClient
            flux_client = FluxClient(api_key="test_key")
            service = LayoutTransferService(flux_client=flux_client)
            
            assert service.flux_client is not None
            assert service.flux_client == flux_client
            
            # Test initialization with style transfer service
            style_service = StyleTransferService(flux_client=flux_client)
            layout_service = LayoutTransferService(
                flux_client=flux_client,
                style_transfer_service=style_service
            )
            
            assert layout_service.style_transfer_service is not None
            assert layout_service.style_transfer_service == style_service
            
        except ImportError:
            pytest.skip("LayoutTransferService not implemented yet")
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_apply_layout_transfer(self, mock_apply_style, layout_transfer_service, 
                                  sample_image_path, sample_masks):
        """
        GIVEN a LayoutTransferService, an input image, and segmentation masks
        WHEN apply_layout_transfer is called with a layout prompt
        THEN it should change the furniture layout while preserving architecture
        """
        if not layout_transfer_service:
            pytest.skip("LayoutTransferService not implemented yet")
        
        # Load test image
        original_image = np.array(Image.open(sample_image_path))
        
        # Create a fake styled result
        styled_result = original_image.copy()
        # Modify colors and layout to simulate changes
        styled_result = styled_result * 0.8 + np.array([50, 30, 10])
        styled_result = np.clip(styled_result, 0, 255).astype(np.uint8)
        
        # Configure mock to return our fake styled result
        mock_apply_style.return_value = styled_result
        
        # Define layout prompt
        layout_prompt = "A living room with a sofa facing the window, two armchairs, and a coffee table in between"
        
        # Call the function being tested
        result = layout_transfer_service.apply_layout_transfer(
            original_image, 
            layout_prompt=layout_prompt,
            structure_mask=sample_masks["structure"]
        )
        
        # Assertions
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == original_image.shape
        
        # Verify Flux client was called with the correct parameters
        mock_apply_style.assert_called_once()
        args, kwargs = mock_apply_style.call_args
        assert "layout" in kwargs["prompt"].lower() or layout_prompt in kwargs["prompt"]
        
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
    def test_generate_layout_variations(self, mock_generate_variations, 
                                      layout_transfer_service, sample_image_path, sample_masks):
        """
        GIVEN a LayoutTransferService, an input image, and segmentation masks
        WHEN generate_layout_variations is called
        THEN it should return multiple layout variations while preserving architecture
        """
        if not layout_transfer_service:
            pytest.skip("LayoutTransferService not implemented yet")
        
        # Load test image
        original_image = np.array(Image.open(sample_image_path))
        
        # Create fake layout variations
        variations = []
        for i in range(3):
            var = original_image.copy()
            # Different color modifications for each variant
            var = var * (0.7 + 0.1*i) + np.array([40+i*10, 30+i*5, 10+i*15])
            var = np.clip(var, 0, 255).astype(np.uint8)
            variations.append(var)
        
        # Configure mock to return our fake variations
        mock_generate_variations.return_value = variations
        
        # Define layout prompt
        layout_prompt = "A living room with different furniture arrangements"
        
        # Call the function being tested
        results = layout_transfer_service.generate_layout_variations(
            original_image, 
            layout_prompt=layout_prompt, 
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
        assert "layout" in kwargs["prompt"].lower() or layout_prompt in kwargs["prompt"]
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
    
    def test_layout_prompt_formatting(self):
        """
        GIVEN a LayoutTransferService
        WHEN format_layout_prompt is called with a furniture arrangement
        THEN it should return a properly formatted prompt for layout transfer
        """
        try:
            from src.layout_transfer import LayoutTransferService
            
            service = LayoutTransferService()
            
            # Test basic layout prompt formatting
            basic_prompt = service.format_layout_prompt("sofa and coffee table")
            assert "sofa" in basic_prompt.lower()
            assert "coffee table" in basic_prompt.lower()
            assert "furniture" in basic_prompt.lower() or "layout" in basic_prompt.lower()
            
            # Test with room type
            room_prompt = service.format_layout_prompt("bed against wall, nightstands", room_type="bedroom")
            assert "bed" in room_prompt.lower()
            assert "bedroom" in room_prompt.lower()
            
            # Test with style
            style_prompt = service.format_layout_prompt(
                "dining table with six chairs",
                room_type="dining room",
                style="Scandinavian"
            )
            assert "dining table" in style_prompt.lower()
            assert "dining room" in style_prompt.lower()
            assert "scandinavian" in style_prompt.lower()
            
        except ImportError:
            pytest.skip("LayoutTransferService not implemented yet")
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_combined_style_layout_transfer(self, mock_apply_style, layout_transfer_service, 
                                          sample_image_path, sample_masks):
        """
        GIVEN a LayoutTransferService, an input image, and segmentation masks
        WHEN apply_style_and_layout is called with style and layout prompts
        THEN it should change both style and layout while preserving architecture
        """
        if not layout_transfer_service:
            pytest.skip("LayoutTransferService not implemented yet")
        
        # Load test image
        original_image = np.array(Image.open(sample_image_path))
        
        # Create a fake styled result
        styled_result = original_image.copy()
        # Modify colors and layout to simulate changes
        styled_result = styled_result * 0.8 + np.array([50, 30, 10])
        styled_result = np.clip(styled_result, 0, 255).astype(np.uint8)
        
        # Configure mock to return our fake styled result
        mock_apply_style.return_value = styled_result
        
        # Define prompts
        style_prompt = "Mid-Century Modern"
        layout_prompt = "A living room with a sofa facing the fireplace, two armchairs, and a coffee table"
        
        # Call the function being tested
        result = layout_transfer_service.apply_style_and_layout(
            original_image, 
            style_prompt=style_prompt,
            layout_prompt=layout_prompt,
            structure_mask=sample_masks["structure"]
        )
        
        # Assertions
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == original_image.shape
        
        # Verify Flux client was called with the correct parameters
        mock_apply_style.assert_called_once()
        args, kwargs = mock_apply_style.call_args
        assert style_prompt.lower() in kwargs["prompt"].lower() or "mid-century" in kwargs["prompt"].lower()
        assert "sofa" in kwargs["prompt"].lower() or "furniture" in kwargs["prompt"].lower()
        
        # Verify structure preservation
        from src.segmentation import SegmentationHandler
        handler = SegmentationHandler()
        preservation_score = handler.estimate_structure_preservation_score(
            original_image=original_image,
            stylized_image=result,
            structure_mask=sample_masks["structure"]
        )
        assert preservation_score > 0.8, "Structure should be well preserved"
