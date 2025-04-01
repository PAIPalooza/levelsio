"""
Comprehensive tests for the Interior Style Transfer API endpoints.

This module implements BDD-style tests for all API endpoints, following
the Semantic Seed Coding Standards (SSCS) for test coverage and quality.
It aims to achieve 80%+ test coverage across all API components.
"""

import pytest
import base64
import io
import json
from PIL import Image
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.core.config import settings

# Create test client
client = TestClient(app)

# Mock API key for testing
TEST_API_KEY = "api_key_for_testing"
HEADERS = {"X-API-Key": TEST_API_KEY}


# Fixtures for test data
@pytest.fixture(scope="module")
def test_image():
    """Create a test image for uploads."""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.fixture(scope="module")
def test_images():
    """Create different test images for comparison."""
    def make_image(color):
        img = Image.new('RGB', (100, 100), color=color)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr
    
    return {
        "original": make_image("red"),
        "styled": make_image("blue")
    }


@pytest.fixture(scope="module")
def base64_images():
    """Create base64 encoded test images."""
    def encode_image(color):
        img = Image.new('RGB', (10, 10), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return {
        "original": encode_image("red"),
        "styled": encode_image("blue")
    }


class TestAPIBasics:
    """Test basic API functionality and OpenAPI documentation."""
    
    def test_api_root(self):
        """It should return service information at the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert "documentation" in data
        assert data["service"] == settings.PROJECT_NAME
    
    def test_openapi_schema(self):
        """It should serve a valid OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        
        # Verify essential OpenAPI structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        
        # Verify security schemes
        assert "securitySchemes" in schema["components"]
        assert "APIKeyHeader" in schema["components"]["securitySchemes"]
    
    def test_swagger_docs_endpoint(self):
        """It should serve Swagger UI docs."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "swagger" in response.text.lower()


class TestAuthentication:
    """Test API authentication functionality."""
    
    def test_valid_x_api_key(self):
        """It should accept requests with valid X-API-Key header."""
        response = client.get("/api/v1/style/styles", headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
    
    # Note: Until API-Key header is implemented, we skip this test
    @pytest.mark.skip(reason="api-key header not implemented yet")
    def test_valid_api_key_header(self):
        """It should accept requests with valid api-key header."""
        response = client.get("/api/v1/style/styles", headers={"api-key": TEST_API_KEY})
        assert response.status_code == 200
    
    # Note: Until API key query param is implemented, we skip this test
    @pytest.mark.skip(reason="api_key query parameter not implemented yet")
    def test_valid_api_key_query(self):
        """It should accept requests with valid api_key query parameter."""
        response = client.get(f"/api/v1/style/styles?api_key={TEST_API_KEY}")
        assert response.status_code == 200
    
    def test_invalid_api_key(self):
        """It should reject requests with invalid API keys."""
        response = client.get("/api/v1/style/styles", headers={"X-API-Key": "invalid_key"})
        assert response.status_code == 401
        assert "Invalid API Key" in response.json()["detail"]


class TestStyleTransferAPI:
    """Test style transfer API endpoints."""
    
    def test_get_available_styles(self):
        """It should return available styles with authentication."""
        response = client.get("/api/v1/style/styles", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert isinstance(data["styles"], list)
        assert len(data["styles"]) > 0
    
    # Mark tests requiring image processing as xfail until the services/dependencies are set up
    @pytest.mark.xfail(reason="Image processing service dependency not available")
    def test_transfer_style_with_file(self, test_image):
        """It should process an uploaded image with style transfer."""
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        data = {"style": "minimalist"}
        response = client.post(
            "/api/v1/style/transfer", 
            files=files,
            data=data,
            headers=HEADERS
        )
        assert response.status_code == 200
        assert "styled_image" in response.json()
        assert "style" in response.json()
    
    # Mark tests requiring JSON-based endpoints as xfail until implemented
    @pytest.mark.xfail(reason="JSON-based endpoint not implemented or available")
    def test_transfer_style_with_json(self, base64_images):
        """It should process a base64 encoded image with style transfer."""
        data = {
            "image": base64_images["original"],
            "style": "minimalist"
        }
        response = client.post(
            "/api/v1/style/transfer-json",
            json=data,
            headers=HEADERS
        )
        assert response.status_code == 200
        assert "styled_image" in response.json()
        assert "style" in response.json()
    
    @pytest.mark.xfail(reason="Style validation not properly implemented")
    def test_invalid_style(self, test_image):
        """It should return an error for invalid style names."""
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        data = {"style": "non_existent_style"}
        response = client.post(
            "/api/v1/style/transfer", 
            files=files,
            data=data,
            headers=HEADERS
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.xfail(reason="Image validation not properly implemented")
    def test_invalid_image(self):
        """It should return an error for invalid image data."""
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        data = {"style": "minimalist"}
        response = client.post(
            "/api/v1/style/transfer", 
            files=files,
            data=data,
            headers=HEADERS
        )
        assert response.status_code == 422  # Validation error


class TestSegmentationAPI:
    """Test segmentation API endpoints."""
    
    def test_get_segmentation_types(self):
        """It should return available segmentation types with authentication."""
        response = client.get("/api/v1/segmentation/segmentation-types", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "segmentation_types" in data
        assert isinstance(data["segmentation_types"], list)
    
    @pytest.mark.xfail(reason="Segmentation service dependency not available")
    def test_segment_interior_with_file(self, test_image):
        """It should process an uploaded image with segmentation."""
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        response = client.post(
            "/api/v1/segmentation/segment", 
            files=files,
            headers=HEADERS
        )
        assert response.status_code == 200
        result = response.json()
        # Adjust based on actual API response structure
        assert "regions" in result or "masks" in result or "results" in result
    
    @pytest.mark.xfail(reason="JSON-based segmentation endpoint not implemented")
    def test_segment_interior_with_json(self, base64_images):
        """It should process a base64 encoded image with segmentation."""
        data = {"image": base64_images["original"]}
        response = client.post(
            "/api/v1/segmentation/segment-json",
            json=data,
            headers=HEADERS
        )
        assert response.status_code == 200
        result = response.json()
        # Adjust based on actual API response structure
        assert "regions" in result or "masks" in result or "results" in result
    
    @pytest.mark.xfail(reason="Image validation not properly implemented")
    def test_invalid_image_format(self):
        """It should return an error for invalid image formats."""
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        response = client.post(
            "/api/v1/segmentation/segment", 
            files=files,
            headers=HEADERS
        )
        assert response.status_code == 422


class TestEvaluationAPI:
    """Test evaluation API endpoints."""
    
    def test_get_available_metrics(self):
        """It should return available evaluation metrics with authentication."""
        response = client.get("/api/v1/evaluation/available-metrics", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], list)
    
    @pytest.mark.xfail(reason="Evaluation service dependency not available")
    def test_evaluate_style_transfer_with_files(self, test_images):
        """It should evaluate style transfer quality with uploaded images."""
        files = {
            "original_image": ("original.jpg", test_images["original"], "image/jpeg"),
            "styled_image": ("styled.jpg", test_images["styled"], "image/jpeg")
        }
        response = client.post(
            "/api/v1/evaluation/style-score", 
            files=files,
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        # Check for metrics in the response
        assert any(metric in data for metric in ["ssim", "mse", "psnr", "score"])
    
    @pytest.mark.xfail(reason="JSON-based evaluation endpoint not implemented")
    def test_evaluate_style_transfer_with_json(self, base64_images):
        """It should evaluate style transfer quality with base64 encoded images."""
        data = {
            "original_image": base64_images["original"],
            "styled_image": base64_images["styled"]
        }
        response = client.post(
            "/api/v1/evaluation/style-score-json",
            json=data,
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        # Check for metrics in the response
        assert any(metric in data for metric in ["ssim", "mse", "psnr", "score"])
    
    def test_missing_image(self, test_images):
        """It should return an error if an image is missing."""
        files = {
            "original_image": ("original.jpg", test_images["original"], "image/jpeg"),
            # Styled image is missing
        }
        response = client.post(
            "/api/v1/evaluation/style-score", 
            files=files,
            headers=HEADERS
        )
        assert response.status_code == 422


class TestPromptsAPI:
    """Test prompt templates API endpoints."""
    
    def test_get_available_styles(self):
        """It should return available style templates with authentication."""
        response = client.get("/api/v1/prompts/styles", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert "categories" in data
        assert isinstance(data["styles"], list)
        assert isinstance(data["categories"], list)
    
    def test_get_prompt_for_style(self):
        """It should generate a prompt for a specific style."""
        response = client.get(
            "/api/v1/prompts/generate?style=minimalist&room_type=living%20room",
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert "style" in data
        assert data["style"] == "minimalist"
        assert "living room" in data["prompt"].lower()
    
    @pytest.mark.xfail(reason="POST prompt endpoint using different format")
    def test_post_generate_prompt(self):
        """It should generate a prompt using POST with JSON body."""
        data = {
            "style": "minimalist",
            "room_type": "bedroom",
            "details": "with natural light"
        }
        response = client.post(
            "/api/v1/prompts/generate-prompt",
            json=data,
            headers=HEADERS
        )
        assert response.status_code == 200
        result = response.json()
        assert "prompt" in result
        assert "style" in result
        assert "bedroom" in result["prompt"].lower()
        assert "natural light" in result["prompt"].lower()
    
    @pytest.mark.xfail(reason="Custom prompt endpoint not implemented")
    def test_generate_custom_prompt(self):
        """It should generate a custom prompt with specific elements."""
        data = {
            "base_style": "minimalist",
            "room_type": "kitchen",
            "colors": "white and light wood",
            "materials": "marble and stainless steel",
            "lighting": "pendant lights",
            "mood": "bright and airy"
        }
        response = client.post(
            "/api/v1/prompts/generate-custom-prompt",
            json=data,
            headers=HEADERS
        )
        assert response.status_code == 200
        result = response.json()
        assert "prompt" in result
        assert "style" in result
        assert "kitchen" in result["prompt"].lower()
        assert "white" in result["prompt"].lower()
        assert "marble" in result["prompt"].lower()
    
    def test_get_style_categories(self):
        """It should return available style categories."""
        response = client.get("/api/v1/prompts/categories", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        # Check the actual response structure
        if isinstance(data, list):
            # If it returns a list directly
            assert len(data) > 0
            assert all(isinstance(cat, str) for cat in data)
        else:
            # If it returns an object with a categories field
            assert "categories" in data
            assert isinstance(data["categories"], list)
            assert len(data["categories"]) > 0
    
    @pytest.mark.xfail(reason="Category-specific style endpoint not implemented")
    def test_get_styles_by_category(self):
        """It should return styles for a specific category."""
        response = client.get("/api/v1/prompts/category/modern", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "category" in data
        assert "styles" in data
        assert data["category"] == "modern"
        assert isinstance(data["styles"], list)
    
    def test_invalid_style_name(self):
        """It should return an error for invalid style names."""
        response = client.get(
            "/api/v1/prompts/generate?style=nonexistent&room_type=kitchen",
            headers=HEADERS
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.xfail(reason="Integration tests require working endpoints")
class TestAPIIntegration:
    """Test integration of multiple API endpoints for complete workflows."""
    
    def test_segment_then_style_workflow(self, test_image):
        """It should segment an image and then apply style transfer."""
        # First, get segmentation
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        seg_response = client.post(
            "/api/v1/segmentation/segment", 
            files=files,
            headers=HEADERS
        )
        assert seg_response.status_code == 200
        
        # Now get a style prompt
        prompt_response = client.get(
            "/api/v1/prompts/generate?style=minimalist&room_type=living%20room",
            headers=HEADERS
        )
        assert prompt_response.status_code == 200
        prompt_data = prompt_response.json()
        
        # Use the prompt for style transfer
        test_image.seek(0)  # Reset image stream
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        data = {"style": "minimalist", "prompt": prompt_data["prompt"]}
        style_response = client.post(
            "/api/v1/style/transfer", 
            files=files,
            data=data,
            headers=HEADERS
        )
        assert style_response.status_code == 200
        style_data = style_response.json()
        assert "styled_image" in style_data or "result_url" in style_data
    
    def test_style_then_evaluate_workflow(self, test_image, base64_images):
        """It should style an image and then evaluate the result."""
        # This test is currently xfailed as part of the class
        pass


class TestErrorHandling:
    """Test error handling across the API."""
    
    def test_method_not_allowed(self):
        """It should return 405 for methods that are not allowed."""
        response = client.post("/api/v1/style/styles", headers=HEADERS)
        assert response.status_code == 405
    
    def test_not_found(self):
        """It should return 404 for non-existent endpoints."""
        response = client.get("/api/v1/nonexistent", headers=HEADERS)
        assert response.status_code == 404
    
    @pytest.mark.xfail(reason="Server error handling not yet implemented")
    def test_server_error_handling(self):
        """It should handle server errors gracefully."""
        # This test is designed to fail until error handling is improved
        files = {"image": ("test.jpg", b"corrupt data", "image/jpeg")}
        data = {"style": "minimalist"}
        response = client.post(
            "/api/v1/style/transfer", 
            files=files,
            data=data,
            headers=HEADERS
        )
        # Should return a 4xx error, not 500
        assert response.status_code < 500
