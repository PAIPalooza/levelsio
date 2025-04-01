"""
Tests for the Interior Style Transfer API endpoints.

This module tests the API endpoints, request validation, and
Swagger/OpenAPI documentation functionality. It follows the BDD-style
testing pattern from Semantic Seed Coding Standards (SSCS).
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.core.config import settings

# Create test client
client = TestClient(app)

# Mock API key for testing
TEST_API_KEY = "api_key_for_testing"
HEADERS = {"X-API-Key": TEST_API_KEY}


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
        
        # Verify key Swagger UI elements in content
        assert "swagger-ui" in response.text.lower()
        assert "swagger" in response.text.lower()


class TestStyleTransferAPI:
    """Test style transfer API endpoints."""
    
    def test_get_available_styles(self):
        """It should return available styles with authentication."""
        response = client.get("/api/v1/style/styles", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert isinstance(data["styles"], list)
    
    def test_unauthorized_access(self):
        """It should reject requests without valid API key."""
        response = client.get("/api/v1/style/styles")
        assert response.status_code == 401  # Our auth implementation uses 401 Unauthorized for invalid/missing API keys
        assert "detail" in response.json()
        assert "Invalid API Key" in response.json()["detail"]


class TestSegmentationAPI:
    """Test segmentation API endpoints."""
    
    def test_get_segmentation_types(self):
        """It should return available segmentation types with authentication."""
        response = client.get("/api/v1/segmentation/segmentation-types", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "segmentation_types" in data
        assert isinstance(data["segmentation_types"], list)


class TestEvaluationAPI:
    """Test evaluation API endpoints."""
    
    def test_get_available_metrics(self):
        """It should return available evaluation metrics with authentication."""
        response = client.get("/api/v1/evaluation/available-metrics", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], list)


# Add tests for the prompts API endpoints
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
