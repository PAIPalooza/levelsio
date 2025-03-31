"""
Test module for the generation features.

This module contains BDD-style tests for the FluxGenerator class,
following the Semantic Seed Coding Standards requirements.
"""

import os
import pytest
import numpy as np
from unittest import mock
from PIL import Image
from datetime import datetime

from src.generation import FluxGenerator


class TestFluxGenerator:
    """
    Test suite for the FluxGenerator class.
    
    BDD-style tests for verifying image generation functionality using
    the Flux API via fal.ai.
    """
    
    @pytest.fixture
    def generator(self):
        """Fixture providing a FluxGenerator instance with a mock API key."""
        with mock.patch.dict(os.environ, {"FAL_KEY": "mock-api-key"}):
            return FluxGenerator()
    
    @pytest.fixture
    def sample_image(self):
        """Fixture providing a sample test image as a NumPy array."""
        # Create a simple test image
        height, width = 128, 128
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add some color variation for testing
        image[:height//2, :width//2] = [180, 190, 200]  # top-left
        image[:height//2, width//2:] = [200, 180, 190]  # top-right
        image[height//2:, :width//2] = [190, 200, 180]  # bottom-left
        image[height//2:, width//2:] = [210, 220, 230]  # bottom-right
        
        return image
    
    @pytest.fixture
    def sample_image_path(self, sample_image, tmp_path):
        """Fixture providing a path to a sample image file."""
        # Save the sample image to a temporary file
        img_path = tmp_path / "test_interior.jpg"
        Image.fromarray(sample_image).save(img_path)
        return str(img_path)
    
    def test_initialization_with_key(self):
        """
        GIVEN an API key
        WHEN initializing a FluxGenerator with that key
        THEN it should properly store the key
        """
        # Test with explicit key
        explicit_key = "test-api-key-123"
        generator = FluxGenerator(api_key=explicit_key)
        assert generator.api_key == explicit_key
        assert generator.model_id == "fal-ai/flux"
    
    def test_initialization_with_env_var(self):
        """
        GIVEN an API key in environment variable
        WHEN initializing a FluxGenerator without explicit key
        THEN it should get the key from environment
        """
        # Test with environment variable
        env_key = "env-api-key-456"
        with mock.patch.dict(os.environ, {"FAL_KEY": env_key}):
            generator = FluxGenerator()
            assert generator.api_key == env_key
    
    def test_setup_client(self, generator):
        """
        GIVEN a FluxGenerator instance
        WHEN _setup_client is called
        THEN it should return True indicating successful setup
        """
        result = generator._setup_client()
        assert result is True
    
    def test_prepare_image(self, generator, sample_image, sample_image_path):
        """
        GIVEN image input in various formats
        WHEN _prepare_image is called
        THEN it should return a dictionary with prepared image data
        """
        # Test with placeholder implementation
        result = generator._prepare_image(sample_image)
        assert isinstance(result, dict)
        assert "image" in result
        
        # Test with string path
        result = generator._prepare_image(sample_image_path)
        assert isinstance(result, dict)
        assert "image" in result
        
        # Test with PIL image
        pil_image = Image.fromarray(sample_image)
        result = generator._prepare_image(pil_image)
        assert isinstance(result, dict)
        assert "image" in result
    
    def test_style_transfer_with_array(self, generator, sample_image):
        """
        GIVEN a NumPy array image
        WHEN style_transfer is called
        THEN it should return a dictionary with the processed image
        """
        prompt = "Modern Scandinavian living room"
        result = generator.style_transfer(sample_image, prompt)
        
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "image" in result
        assert np.array_equal(result["image"], sample_image)
        assert "prompt" in result
        assert result["prompt"] == prompt
        assert "timestamp" in result
    
    def test_style_transfer_with_path(self, generator, sample_image_path, sample_image):
        """
        GIVEN an image file path
        WHEN style_transfer is called
        THEN it should return a dictionary with the processed image
        """
        prompt = "Industrial style loft"
        result = generator.style_transfer(sample_image_path, prompt)
        
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "image" in result
        assert result["image"].shape == sample_image.shape
        assert "prompt" in result
        assert result["prompt"] == prompt
        assert "timestamp" in result
    
    def test_style_transfer_with_pil(self, generator, sample_image):
        """
        GIVEN a PIL Image
        WHEN style_transfer is called
        THEN it should return a dictionary with the processed image
        """
        prompt = "Minimalist Japanese interior"
        pil_image = Image.fromarray(sample_image)
        result = generator.style_transfer(pil_image, prompt)
        
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "image" in result
        assert result["image"].shape == sample_image.shape
        assert "prompt" in result
        assert result["prompt"] == prompt
        assert "timestamp" in result
    
    def test_style_layout_transfer(self, generator, sample_image):
        """
        GIVEN an image and prompt
        WHEN style_layout_transfer is called
        THEN it should call style_transfer with preserve_structure=True
        """
        prompt = "Mid-century modern living room with sectional sofa"
        
        # Spy on style_transfer
        with mock.patch.object(generator, 'style_transfer', wraps=generator.style_transfer) as spy:
            result = generator.style_layout_transfer(sample_image, prompt)
            
            # Verify style_transfer was called with the right parameters
            spy.assert_called_once_with(sample_image, prompt, None, preserve_structure=True)
            
            # Verify result format
            assert result is not None
            assert "status" in result
            assert result["status"] == "success"
            assert "image" in result
            assert "prompt" in result
            assert "timestamp" in result
    
    def test_furnish_empty_room(self, generator, sample_image):
        """
        GIVEN an empty room image and prompt
        WHEN furnish_empty_room is called
        THEN it should call style_transfer with furnish prompt
        """
        prompt = "modern sofa, coffee table, bookshelf"
        
        # Spy on style_transfer
        with mock.patch.object(generator, 'style_transfer', wraps=generator.style_transfer) as spy:
            result = generator.furnish_empty_room(sample_image, prompt)
            
            # Verify style_transfer was called with the right parameters
            spy.assert_called_once_with(sample_image, f"furnish with {prompt}", None, preserve_structure=True)
            
            # Verify result format
            assert result is not None
            assert "status" in result
            assert result["status"] == "success"
            assert "image" in result
            assert "prompt" in result
            assert "timestamp" in result
