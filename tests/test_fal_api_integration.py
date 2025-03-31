"""
Integration tests for the FAL.ai API connections.

This module tests the actual API integration with FAL.ai, ensuring that
real API calls work correctly. It follows the BDD-style testing pattern
from Semantic Seed Coding Standards (SSCS).
"""

import os
import pytest
import numpy as np
from PIL import Image
import base64
import io
import logging

from src.flux_integration import FluxClient
from src.style_transfer import StyleTransferService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment variable or set directly for testing
FAL_KEY = os.getenv("FAL_KEY", "f51c8849-420f-4f98-ac43-9b98e5a58408:072cb70f9f6d9fc4f56f3930187d87cd")

# Test fixtures
@pytest.fixture
def sample_image():
    """Create a simple test image for API calls."""
    # Create a small sample image (50x50 pixels)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # Add some basic structure to make it look like a room
    img[10:40, 10:40, :] = 200  # Light gray box
    img[30:40, 20:30, :] = 100  # Darker rectangle for "furniture"
    return img

@pytest.fixture
def flux_client():
    """Initialize a real FluxClient with the actual API key."""
    if not FAL_KEY:
        pytest.skip("FAL_KEY environment variable not set, skipping real API tests")
    return FluxClient(api_key=FAL_KEY)

@pytest.fixture
def style_service(flux_client):
    """Initialize the StyleTransferService with a real client."""
    return StyleTransferService(flux_client=flux_client)

class TestFalApiIntegration:
    """Test real API integration with FAL.ai Flux service."""
    
    @pytest.mark.skipif(not FAL_KEY, reason="FAL_KEY environment variable not set")
    def test_flux_client_initialization(self, flux_client):
        """It should initialize the FluxClient with the actual API key."""
        assert flux_client is not None
        assert flux_client.api_key == FAL_KEY
        assert flux_client.model_key == "flux-1"  # Changed from model_id to model_key
    
    @pytest.mark.skipif(not FAL_KEY, reason="FAL_KEY environment variable not set")
    def test_real_style_transfer_call(self, flux_client, sample_image):
        """It should make a real API call to apply style transfer."""
        try:
            # Attempt to perform a real API call
            prompt = "Interior in minimalist style"
            styled_image = flux_client.apply_style_transfer(
                image=sample_image,
                prompt=prompt
            )
            
            # Verify we got an actual result (not a mock)
            assert styled_image is not None
            assert styled_image.shape == sample_image.shape
            
            # Verify the image has changed (style has been applied)
            # Images shouldn't be identical after style transfer
            assert not np.array_equal(styled_image, sample_image)
            
            logger.info("✅ Successfully called FAL.ai API with real credentials")
        except Exception as e:
            if "Failed to resolve 'api.fal.ai'" in str(e):
                pytest.skip(f"Network connectivity issue: {str(e)}")
            else:
                # For other errors, let the test fail
                logger.error(f"API call failed: {str(e)}")
                raise
    
    @pytest.mark.skipif(not FAL_KEY, reason="FAL_KEY environment variable not set")
    def test_style_preservation(self, style_service, sample_image):
        """It should make a real API call with structure preservation."""
        try:
            # Create a simple structure mask
            structure_mask = np.zeros((50, 50), dtype=np.uint8)
            structure_mask[5:45, 5:45] = 1  # Preserve a larger area
            
            # Attempt to perform a real API call with structure preservation
            prompt = "Interior in industrial style"
            # Using the correct method name from StyleTransferService
            styled_image = style_service.apply_style_only(
                image=sample_image,
                style_prompt=prompt,
                structure_mask=structure_mask,
                preserve_structure=True
            )
            
            # Verify we got an actual result
            assert styled_image is not None
            assert styled_image.shape == sample_image.shape
            
            logger.info("✅ Successfully applied style with structure preservation")
        except Exception as e:
            if "Failed to resolve 'api.fal.ai'" in str(e):
                pytest.skip(f"Network connectivity issue: {str(e)}")
            else:
                logger.error(f"API call failed: {str(e)}")
                raise

if __name__ == "__main__":
    # Run the tests with more detailed logs
    logging.basicConfig(level=logging.DEBUG)
    pytest.main(["-xvs", __file__])
