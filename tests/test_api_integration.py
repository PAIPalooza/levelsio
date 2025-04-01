"""
Integration tests for the Interior Style Transfer API.

This module contains integration tests that make real API calls to external services,
following the Semantic Seed Coding Standards (SSCS) for BDD-style testing.
"""

import os
import base64
import pytest
import logging
from typing import Dict, Any
from fastapi.testclient import TestClient
from dotenv import load_dotenv

from src.api.main import create_app
from src.flux_integration import FluxAPI
from src.prompts.style_manager import StyleManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_integration")

# Initialize test client
app = create_app()
client = TestClient(app)

# API key for testing
API_KEY = os.getenv("FAL_KEY", "f51c8849-420f-4f98-ac43-9b98e5a58408:072cb70f9f6d9fc4f56f3930187d87cd")
API_KEY_HEADER = {"X-API-Key": API_KEY}

# Initialize style manager for test data
style_manager = StyleManager()

# Test image data - small 1x1 transparent PNG to minimize API usage during tests
TEST_IMAGE_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Larger sample image for visual tests when needed
SAMPLE_ROOM_BASE64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wgARCAAyADIDAREAAhEBAxEB/8QAGQAAAwEBAQAAAAAAAAAAAAAAAAMEBQIB/8QAFgEBAQEAAAAAAAAAAAAAAAAAAAEC/9oADAMBAAIQAxAAAAH6pJcNItQZEsV2WTG8WKiCwvdBxcMbxnYRXiRVj//EABsQAAMBAQEBAQAAAAAAAAAAAAIDBAEABRET/8QAFhEBAQEAAAAAAAAAAAAAAAAAAAER/9oADAMBAAIRAxAAAAH0+ZodDLUZ6jj0tgZ3oxFm9GKLURVnEyxei5c3piH/xAAiEAACAgICAgIDAAAAAAAAAAABAgADBBESIRMxFEEiUWH/2gAIAQEAAT8AKPFpFRW/sBqDx59xX1qGfx12zAQEHqLZkAazGfM1ZqKWP0fpiMjEgr1E3DpmYxXGYgQJY2xlvZGP3CbFQOw111GstysrwJ6MTyr5KvI+x9wHRnJtMH4fmz1s6MHJjF3SnDpIYTZYBqLnX51aAmh2IANRbzcMnGx67WAOzGprS7wHsgGPxYXfv+QiWD0OoQbLd1MJDAn7B6E1o9QgAgEzl9jfUNdS6LVrn3Kv5HXU+eWAEFi/Un//xAAbEQACAwEBAQAAAAAAAAAAAAACAwABBBEhEv/aAAgBAgEBPwB1MeXE3McriiKiqIxT7D5P/8QAGBEAAwEBAAAAAAAAAAAAAAAAAAERAhL/2gAIAQMBAT8APDdCLjwo2ZJ//9k="


@pytest.mark.integration
class TestAPIIntegration:
    """Test suite for API integration with external services."""
    
    def test_health_check(self):
        """Should return healthy status for health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_api_key_verification(self):
        """Should verify valid API key."""
        response = client.get(
            "/verify-key",
            headers=API_KEY_HEADER
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True
    
    def test_api_key_rejection(self):
        """Should reject invalid API key."""
        response = client.get(
            "/verify-key",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401
    
    def test_get_available_styles(self):
        """Should return list of available styles."""
        response = client.get(
            "/api/v1/prompts/list",
            headers=API_KEY_HEADER
        )
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert len(data["styles"]) > 0
    
    def test_get_style_categories(self):
        """Should return list of style categories."""
        response = client.get(
            "/api/v1/prompts/categories",
            headers=API_KEY_HEADER
        )
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) > 0
    
    def test_get_category_styles(self):
        """Should return styles for a specific category."""
        # First get categories
        categories_response = client.get(
            "/api/v1/prompts/categories",
            headers=API_KEY_HEADER
        )
        categories = categories_response.json()["categories"]
        
        # Test with first category
        first_category = categories[0]
        response = client.get(
            f"/api/v1/prompts/category/{first_category}",
            headers=API_KEY_HEADER
        )
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert len(data["styles"]) > 0
        assert "category" in data
        assert data["category"] == first_category


@pytest.mark.integration
@pytest.mark.flux
class TestFluxIntegration:
    """Test suite for Flux API integration."""
    
    def test_flux_api_connection(self):
        """Should connect to Flux API with provided credentials."""
        flux_api = FluxAPI()
        assert flux_api.api_key == API_KEY
    
    @pytest.mark.parametrize("style_name", ["minimalist", "modern"])
    def test_style_transfer_json_endpoint(self, style_name):
        """Should transfer style using JSON endpoint with real API call."""
        # Skip this test if FAL_DISABLE_REAL_CALLS is set
        if os.getenv("FAL_DISABLE_REAL_CALLS") == "1":
            pytest.skip("Real API calls disabled")
        
        response = client.post(
            "/api/v1/style/transfer-json",
            headers=API_KEY_HEADER,
            json={
                "image": TEST_IMAGE_BASE64,
                "style": style_name,
                "strength": 0.7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "styled_image" in data
        assert data["style"] == style_name
        assert "processing_time" in data
    
    def test_segmentation_json_endpoint(self):
        """Should segment interior using JSON endpoint with real API call."""
        # Skip this test if FAL_DISABLE_REAL_CALLS is set
        if os.getenv("FAL_DISABLE_REAL_CALLS") == "1":
            pytest.skip("Real API calls disabled")
        
        response = client.post(
            "/api/v1/segmentation/segment-json",
            headers=API_KEY_HEADER,
            json={
                "image": SAMPLE_ROOM_BASE64,
                "detail_level": "medium"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "regions" in data
        assert len(data["regions"]) > 0
        assert "processing_time" in data
    
    def test_style_evaluation_endpoint(self):
        """Should evaluate style transfer using real API call."""
        # Skip this test if FAL_DISABLE_REAL_CALLS is set
        if os.getenv("FAL_DISABLE_REAL_CALLS") == "1":
            pytest.skip("Real API calls disabled")
        
        response = client.post(
            "/api/v1/evaluation/style-score-json",
            headers=API_KEY_HEADER,
            json={
                "original_image": TEST_IMAGE_BASE64,
                "styled_image": TEST_IMAGE_BASE64,
                "metrics": ["ssim", "mse"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert len(data["metrics"]) > 0
        assert "overall_score" in data
    
    def test_custom_prompt_generation(self):
        """Should generate custom prompt for style transfer."""
        response = client.post(
            "/api/v1/prompts/generate-custom-prompt",
            headers=API_KEY_HEADER,
            json={
                "base_style": "minimalist",
                "colors": ["white", "beige"],
                "materials": ["wood", "steel"],
                "room_type": "living room",
                "mood": "calm"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert len(data["prompt"]) > 0
        
        # Prompt should contain specified elements
        prompt = data["prompt"].lower()
        assert "minimalist" in prompt
        assert "living room" in prompt
        assert any(color in prompt for color in ["white", "beige"])


@pytest.mark.integration
@pytest.mark.error_handling
class TestErrorHandling:
    """Test suite for API error handling."""
    
    def test_invalid_style_name(self):
        """Should return appropriate error for invalid style name."""
        response = client.post(
            "/api/v1/style/transfer-json",
            headers=API_KEY_HEADER,
            json={
                "image": TEST_IMAGE_BASE64,
                "style": "nonexistent_style",
                "strength": 0.7
            }
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "message" in data
        assert "nonexistent_style" in data["message"]
        assert "details" in data
        assert "available_styles" in data["details"]
    
    def test_invalid_image_data(self):
        """Should return appropriate error for invalid image data."""
        response = client.post(
            "/api/v1/style/transfer-json",
            headers=API_KEY_HEADER,
            json={
                "image": "invalid_base64_data",
                "style": "minimalist",
                "strength": 0.7
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "message" in data
        assert "image" in data["message"].lower()
    
    def test_missing_required_field(self):
        """Should return validation error for missing required field."""
        response = client.post(
            "/api/v1/style/transfer-json",
            headers=API_KEY_HEADER,
            json={
                # Missing required 'image' field
                "style": "minimalist",
                "strength": 0.7
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "message" in data
        # Check if errors is at the top level or in details
        assert "errors" in data or ("details" in data and "errors" in data["details"])
        
        # Get errors from wherever they are
        errors = data.get("errors", data.get("details", {}).get("errors", []))
        
        # Find the validation error for the missing 'image' field
        missing_image_error = None
        for err in errors:
            if isinstance(err, dict) and "loc" in err and "image" in str(err["loc"]):
                missing_image_error = err
                break
        
        assert missing_image_error is not None
        assert "required" in missing_image_error.get("msg", "").lower()

    def test_invalid_parameter_value(self):
        """Should return validation error for invalid parameter value."""
        response = client.post(
            "/api/v1/style/transfer-json",
            headers=API_KEY_HEADER,
            json={
                "image": TEST_IMAGE_BASE64,
                "style": "minimalist",
                "strength": 2.0  # Invalid: should be between 0.0 and 1.0
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "message" in data
        # Check if errors is at the top level or in details
        assert "errors" in data or ("details" in data and "errors" in data["details"])
        
        # Get errors from wherever they are
        errors = data.get("errors", data.get("details", {}).get("errors", []))
        
        # Find the validation error for the invalid 'strength' field
        strength_error = None
        for err in errors:
            if isinstance(err, dict) and "loc" in err and "strength" in str(err["loc"]):
                strength_error = err
                break
        
        assert strength_error is not None
        assert "less than or equal" in strength_error.get("msg", "").lower()
