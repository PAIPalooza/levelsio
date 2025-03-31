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
        """It should serve Swagger documentation UI."""
        response = client.get("/docs")
        assert response.status_code == 200
        # Response should be HTML containing Swagger UI
        assert "text/html" in response.headers["content-type"]
        assert "swagger" in response.text.lower()


class TestStyleTransferAPI:
    """Test style transfer API endpoints."""
    
    def test_get_available_styles(self):
        """It should return available styles with authentication."""
        response = client.get("/api/v1/style/styles", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert "examples" in data
        assert len(data["styles"]) > 0
        
    def test_unauthorized_access(self):
        """It should reject requests without valid API key."""
        response = client.get("/api/v1/style/styles")
        assert response.status_code == 403  # FastAPI security returns 403 Forbidden for missing security headers
        assert "Not authenticated" in response.json()["detail"]


class TestSegmentationAPI:
    """Test segmentation API endpoints."""
    
    def test_get_segmentation_types(self):
        """It should return available segmentation types with authentication."""
        response = client.get("/api/v1/segmentation/segmentation-types", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "segmentation_types" in data
        assert "descriptions" in data
        assert len(data["segmentation_types"]) > 0


class TestEvaluationAPI:
    """Test evaluation API endpoints."""
    
    def test_get_available_metrics(self):
        """It should return available evaluation metrics with authentication."""
        response = client.get("/api/v1/evaluation/available-metrics", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "descriptions" in data
        assert len(data["metrics"]) > 0
