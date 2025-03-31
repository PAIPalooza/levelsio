"""
Test module for Flux integration with fal.ai.

BDD-style tests for the @fal API integration, including style transfer,
request handling, and error management.
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
import json
from unittest import mock
from io import BytesIO
import base64
import tempfile

# Path manipulations for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_data import create_test_interior_image, get_test_image_path
from src.segmentation import SegmentationHandler


class TestFluxIntegration:
    """
    Tests for Flux integration with fal.ai for style transfer.
    
    BDD-style tests that verify the API integration, style transfers,
    and error handling for the fal.ai Flux model.
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
    def flux_client(self):
        """Fixture that provides a Flux client instance."""
        # We'll need to import here to avoid issues when the module doesn't exist yet
        # during the Red testing phase
        try:
            from src.flux_integration import FluxClient
            return FluxClient(api_key="test_key")
        except ImportError:
            pytest.skip("FluxClient not implemented yet")
    
    @pytest.fixture
    def mock_response(self):
        """Fixture that provides a mock fal.ai API response."""
        # Create a temporary file for the test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_image_path = tmp_file.name
        
        # Create a base64 encoded test image to simulate API response
        test_img_path = create_test_interior_image(output_path=temp_image_path)
        test_img = Image.open(test_img_path)
        
        buffered = BytesIO()
        test_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Clean up the temporary file after creating the encoded response
        os.unlink(test_img_path)
        
        return {
            "images": [img_base64],
            "seed": 12345,
            "timing": {
                "prepared": 0.1,
                "processed": 2.5,
                "total": 2.6
            }
        }
    
    def test_flux_client_initialization(self):
        """
        GIVEN the FluxClient class
        WHEN initializing with an API key
        THEN it should properly store the key and create a client
        """
        try:
            from src.flux_integration import FluxClient
            
            # Test with actual API key
            client = FluxClient(api_key="test_key_123")
            assert client.api_key == "test_key_123"
            assert client.model_key == "flux-1"  # Default model
            
            # Test with environment variable
            with mock.patch.dict(os.environ, {"FAL_KEY": "env_key_abc"}):
                client = FluxClient()
                assert client.api_key == "env_key_abc"
                
            # Test with custom model key
            client = FluxClient(api_key="test_key_123", model_key="flux-2")
            assert client.model_key == "flux-2"
            
        except ImportError:
            pytest.skip("FluxClient not implemented yet")
    
    @mock.patch("requests.post")
    def test_apply_style_transfer(self, mock_post, flux_client, sample_image_path, mock_response):
        """
        GIVEN a FluxClient instance and an input image
        WHEN apply_style_transfer is called with a prompt
        THEN it should return the styled image
        """
        if not flux_client:
            pytest.skip("FluxClient not implemented yet")
        
        # Configure the mock response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        # Load test image
        input_image = np.array(Image.open(sample_image_path))
        
        # Define style prompt
        style_prompt = "A Scandinavian style living room with neutral colors"
        
        # Call the function being tested
        styled_image = flux_client.apply_style_transfer(input_image, style_prompt)
        
        # Assertions
        assert styled_image is not None
        assert isinstance(styled_image, np.ndarray)
        assert styled_image.shape == input_image.shape
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        
        # Extract the call arguments
        call_args = mock_post.call_args[1]["json"]
        
        # Verify key parameters
        assert "image" in call_args
        assert "prompt" in call_args
        assert call_args["prompt"] == style_prompt
    
    @mock.patch("requests.post")
    def test_generate_style_variations(self, mock_post, flux_client, sample_image_path, mock_response):
        """
        GIVEN a FluxClient instance and an input image
        WHEN generate_style_variations is called
        THEN it should return multiple styled variations
        """
        if not flux_client:
            pytest.skip("FluxClient not implemented yet")
        
        # Configure the mock to return multiple responses
        mock_variations = [mock_response.copy() for _ in range(3)]
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.side_effect = mock_variations
        
        # Load test image
        input_image = np.array(Image.open(sample_image_path))
        
        # Test parameters
        style_prompt = "A modern minimalist living room"
        num_variations = 3
        
        # Call the function being tested
        variations = flux_client.generate_style_variations(
            input_image, style_prompt, num_variations=num_variations
        )
        
        # Assertions
        assert variations is not None
        assert isinstance(variations, list)
        assert len(variations) == num_variations
        
        for variant in variations:
            assert isinstance(variant, np.ndarray)
            assert variant.shape == input_image.shape
    
    @mock.patch("requests.post")
    def test_api_error_handling(self, mock_post, flux_client):
        """
        GIVEN a FluxClient instance
        WHEN the API returns an error response
        THEN it should handle the error gracefully
        """
        if not flux_client:
            pytest.skip("FluxClient not implemented yet")
        
        # Configure mock to return an error
        mock_post.return_value.status_code = 400
        mock_post.return_value.json.return_value = {
            "error": "invalid_request",
            "message": "Prompt contains disallowed content"
        }
        
        # Create a dummy image
        dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Call with a prompt that would trigger content moderation
        with pytest.raises(Exception) as excinfo:
            flux_client.apply_style_transfer(
                dummy_image, 
                "A prompt that would violate content policy"
            )
        
        # Verify error handling
        assert "invalid_request" in str(excinfo.value) or "400" in str(excinfo.value)
    
    @mock.patch("requests.post")
    def test_network_error_retry(self, mock_post, flux_client, sample_image_path, mock_response):
        """
        GIVEN a FluxClient instance
        WHEN a network error occurs during API call
        THEN it should retry the request
        """
        if not flux_client:
            pytest.skip("FluxClient not implemented yet")
        
        # Configure mock to fail first time, succeed second time
        mock_post.side_effect = [
            Exception("Connection error"),  # First call fails
            mock.MagicMock(  # Second call succeeds
                status_code=200,
                json=mock.MagicMock(return_value=mock_response)
            )
        ]
        
        # Load test image
        input_image = np.array(Image.open(sample_image_path))
        
        # Call the function which should retry
        styled_image = flux_client.apply_style_transfer(
            input_image, 
            "Cozy farmhouse interior",
            max_retries=3
        )
        
        # Verify result was successful after retry
        assert styled_image is not None
        assert isinstance(styled_image, np.ndarray)
        
        # Verify API was called twice (initial failure + successful retry)
        assert mock_post.call_count == 2
