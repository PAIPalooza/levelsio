"""
Test cases for the Prompt Templates API endpoints.

This module contains unit and integration tests for the prompt templates API
endpoints, following Behavior-Driven Development principles from the
Semantic Seed Coding Standards (SSCS).
"""

import os
import sys
import json
import pytest
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import FastAPI, status

# Configure path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API modules
from src.api.routes.prompts import router as prompts_router
from src.prompts import StylePromptManager, StyleCategory
from src.api.core.auth import verify_api_key


# Test API key for authentication
TEST_API_KEY = "test-api-key"


# Setup FastAPI test client
@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()
    
    # Override auth dependency for testing
    async def mock_verify_api_key():
        return TEST_API_KEY
    
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    app.include_router(prompts_router, prefix="/api/v1/prompts")
    return app


@pytest.fixture
def client(app):
    """Create a test client for the application."""
    return TestClient(app)


@pytest.fixture
def mock_style_manager():
    """Create a mock style manager with test data."""
    # Sample style templates for testing
    templates = {
        "Scandinavian": {
            "prompt": "A Scandinavian style {room_type} with clean lines, light colors, and minimal decoration",
            "category": "modern",
            "description": "Scandinavian design emphasizes clean lines, simplicity, and functionality.",
            "characteristics": ["Light colors", "Wood elements", "Minimal decoration"],
            "colors": ["White", "Light gray", "Light wood"],
            "materials": ["Light wood", "Wool", "Cotton"],
            "example_prompt": "A cozy Scandinavian living room"
        },
        "Industrial": {
            "prompt": "An industrial style {room_type} with exposed brick, metal fixtures, and raw surfaces",
            "category": "industrial",
            "description": "Industrial style features raw, unfinished elements with metal and exposed features.",
            "characteristics": ["Exposed brick", "Metal fixtures", "Raw surfaces"],
            "colors": ["Gray", "Rust", "Black"],
            "materials": ["Metal", "Wood", "Concrete"],
            "example_prompt": "An industrial style loft with exposed brick"
        }
    }
    
    # Create a patched manager
    with patch('src.prompts.style_manager.StylePromptManager') as mock_manager:
        manager_instance = mock_manager.return_value
        manager_instance.get_all_style_names.return_value = list(templates.keys())
        manager_instance.templates = templates
        
        # Mock style details
        def get_style_details(style_name):
            return templates.get(style_name, {}).copy()
            
        manager_instance.get_style_details.side_effect = get_style_details
        
        # Mock get_prompt_for_style
        def get_prompt(style, room_type="interior", details=None):
            if style not in templates:
                return f"A {style} style {room_type}"
                
            prompt = templates[style]["prompt"].format(room_type=room_type)
            if details:
                prompt += f" {details}"
            return prompt
            
        manager_instance.get_prompt_for_style.side_effect = get_prompt
        
        # Mock create_custom_prompt
        def create_custom(base_style, room_type="interior", **kwargs):
            base_prompt = get_prompt(base_style, room_type)
            extras = []
            
            if kwargs.get("colors"):
                extras.append(f"with {kwargs['colors']}")
                
            if kwargs.get("materials"):
                extras.append(f"featuring {kwargs['materials']}")
                
            if kwargs.get("lighting"):
                extras.append(f"with {kwargs['lighting']}")
                
            if kwargs.get("mood"):
                extras.append(f"creating a {kwargs['mood']} atmosphere")
                
            if extras:
                base_prompt += ", " + ", ".join(extras)
                
            return base_prompt
            
        manager_instance.create_custom_prompt.side_effect = create_custom
        
        # Mock get_style_preview_image_path
        def get_preview_path(style_name):
            return f"/fake/path/to/{style_name.lower()}_preview.jpg" if style_name in templates else None
            
        manager_instance.get_style_preview_image_path.side_effect = get_preview_path
        
        # Mock category methods
        manager_instance.get_styles_by_category.return_value = ["Scandinavian"]
        
        yield manager_instance


def test_get_all_styles(client, mock_style_manager):
    """Test getting all available style names."""
    with patch('src.api.routes.prompts.StylePromptManager', return_value=mock_style_manager):
        response = client.get("/api/v1/prompts/styles")
        
        # Check status code
        assert response.status_code == status.HTTP_200_OK
        
        # Check response body
        data = response.json()
        assert "styles" in data
        assert "categories" in data
        assert "total_styles" in data
        assert data["total_styles"] == len(data["styles"])
        
        # Verify style data
        styles = data["styles"]
        assert any(s["name"] == "Scandinavian" for s in styles)
        assert any(s["name"] == "Industrial" for s in styles)
        
        # Check that styles have the right structure
        for style in styles:
            assert "name" in style
            assert "category" in style
            assert "description" in style
            assert "preview_image_url" in style


def test_get_style_details(client, mock_style_manager):
    """Test getting details for a specific style."""
    with patch('src.api.routes.prompts.StylePromptManager', return_value=mock_style_manager):
        response = client.get("/api/v1/prompts/styles/Scandinavian")
        
        # Check status code
        assert response.status_code == status.HTTP_200_OK
        
        # Check response body
        data = response.json()
        assert data["name"] == "Scandinavian"
        assert data["category"] == "modern"
        assert "clean lines" in data["description"].lower()
        assert "Light colors" in data["characteristics"]
        assert "White" in data["colors"]
        assert "Light wood" in data["materials"]
        
        # Test non-existent style
        response = client.get("/api/v1/prompts/styles/NonExistentStyle")
        assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_style_categories(client):
    """Test getting all style categories."""
    with patch('src.api.routes.prompts.StyleCategory', spec=StyleCategory):
        # Mock the categories enum
        with patch('src.api.routes.prompts.StyleCategory.__iter__', return_value=iter([
            MagicMock(value="modern"),
            MagicMock(value="industrial")
        ])):
            response = client.get("/api/v1/prompts/categories")
            
            # Check status code
            assert response.status_code == status.HTTP_200_OK
            
            # Check response body
            data = response.json()
            assert "modern" in data
            assert "industrial" in data


def test_get_styles_by_category(client, mock_style_manager):
    """Test getting styles in a specific category."""
    with patch('src.api.routes.prompts.StylePromptManager', return_value=mock_style_manager):
        # Mock the StyleCategory enum
        with patch('src.api.routes.prompts.StyleCategory') as mock_category_class:
            # Create a mock for the enum instance
            mock_modern = MagicMock()
            mock_modern.value = "modern"
            
            # Setup the class method properly
            mock_category_class.return_value = mock_modern
            mock_category_class.__call__ = MagicMock(return_value=mock_modern)
            
            # Properly mock the classmethod get_description
            mock_category_class.get_description = MagicMock(return_value="Clean lines and simple design")
            
            response = client.get("/api/v1/prompts/categories/modern")
            
            # Check status code
            assert response.status_code == status.HTTP_200_OK
            
            # Check response body
            data = response.json()
            assert data["name"] == "modern"
            assert data["description"] == "Clean lines and simple design"
            assert "Scandinavian" in data["styles"]
            
            # Test non-existent category
            # Mock the enum constructor to raise ValueError
            mock_category_class.side_effect = ValueError("Invalid category")
            
            response = client.get("/api/v1/prompts/categories/nonexistent")
            assert response.status_code == status.HTTP_404_NOT_FOUND


def test_generate_style_prompt(client, mock_style_manager):
    """Test generating a prompt for a specific style."""
    with patch('src.api.routes.prompts.StylePromptManager', return_value=mock_style_manager):
        # Valid style request
        response = client.post(
            "/api/v1/prompts/generate-prompt",
            json={
                "style": "Scandinavian",
                "room_type": "living room",
                "details": "with large windows"
            }
        )
        
        # Check status code
        assert response.status_code == status.HTTP_200_OK
        
        # Check response body
        data = response.json()
        assert data["style"] == "Scandinavian"
        assert "living room" in data["prompt"]
        assert "large windows" in data["prompt"]
        
        # Test non-existent style
        response = client.post(
            "/api/v1/prompts/generate-prompt",
            json={
                "style": "NonExistentStyle",
                "room_type": "kitchen"
            }
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND


def test_generate_custom_prompt(client, mock_style_manager):
    """Test generating a custom prompt with specific elements."""
    with patch('src.api.routes.prompts.StylePromptManager', return_value=mock_style_manager):
        # Valid custom prompt request
        response = client.post(
            "/api/v1/prompts/generate-custom-prompt",
            json={
                "base_style": "Industrial",
                "room_type": "office",
                "colors": "dark gray and rust",
                "materials": "concrete and distressed wood",
                "lighting": "Edison bulb fixtures",
                "mood": "productive"
            }
        )
        
        # Check status code
        assert response.status_code == status.HTTP_200_OK
        
        # Check response body
        data = response.json()
        assert data["style"] == "Industrial"
        assert "office" in data["prompt"]
        assert "dark gray and rust" in data["prompt"]
        assert "concrete and distressed wood" in data["prompt"]
        assert "Edison bulb fixtures" in data["prompt"]
        assert "productive" in data["prompt"]
        
        # Test with minimal parameters
        response = client.post(
            "/api/v1/prompts/generate-custom-prompt",
            json={
                "base_style": "Scandinavian",
                "room_type": "bedroom"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["style"] == "Scandinavian"
        assert "bedroom" in data["prompt"]
        
        # Test non-existent base style
        response = client.post(
            "/api/v1/prompts/generate-custom-prompt",
            json={
                "base_style": "NonExistentStyle",
                "room_type": "kitchen"
            }
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
