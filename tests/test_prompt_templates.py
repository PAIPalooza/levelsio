"""
Tests for prompt templates in the Interior Style Transfer POC.

This module contains tests for the prompt templates functionality using pytest
and follows the Behavior-Driven Development principles from SSCS.
"""

import pytest
import os
import json
import numpy as np
from src.style_transfer import StyleTransferService
from src.prompts import StylePromptManager, StyleCategory

class TestPromptTemplates:
    """Test suite for prompt templates functionality."""
    
    def test_default_templates_exist(self):
        """Test that default style prompt templates exist and are properly formatted."""
        service = StyleTransferService()
        
        # Check common styles are present
        assert "Scandinavian" in service.STYLE_TEMPLATES
        assert "Industrial" in service.STYLE_TEMPLATES
        assert "Minimalist" in service.STYLE_TEMPLATES
        
        # Check template format contains room_type placeholder
        for style, template in service.STYLE_TEMPLATES.items():
            assert "{room_type}" in template
    
    def test_format_style_prompt(self):
        """Test that style prompts are correctly formatted."""
        service = StyleTransferService()
        
        # Test with known style
        prompt = service.format_style_prompt("Scandinavian", "living room")
        assert "Scandinavian style living room" in prompt
        
        # Test with unknown style
        prompt = service.format_style_prompt("Futuristic", "bedroom")
        assert "Futuristic style bedroom" in prompt
        
        # Test with additional details
        prompt = service.format_style_prompt("Industrial", "kitchen", "with copper fixtures")
        assert "Industrial style kitchen" in prompt
        assert "with copper fixtures" in prompt

    def test_prompt_manager_exists(self):
        """Test that the StylePromptManager class exists and can be instantiated."""
        manager = StylePromptManager()
        assert manager is not None
        
    def test_style_categories_defined(self):
        """Test that style categories are properly defined."""
        # Check that StyleCategory enum exists and has expected values
        assert StyleCategory.MODERN is not None
        assert StyleCategory.TRADITIONAL is not None
        assert StyleCategory.ECLECTIC is not None
        
    def test_get_styles_by_category(self):
        """Test getting styles filtered by category."""
        manager = StylePromptManager()
        
        # Get modern styles
        modern_styles = manager.get_styles_by_category(StyleCategory.MODERN)
        assert len(modern_styles) > 0
        assert "Minimalist" in modern_styles
        assert "Contemporary" in modern_styles
        
        # Get traditional styles
        traditional_styles = manager.get_styles_by_category(StyleCategory.TRADITIONAL)
        assert len(traditional_styles) > 0
        assert "Traditional" in traditional_styles
        assert "Rustic" in traditional_styles
        
    def test_get_all_style_names(self):
        """Test getting all available style names."""
        manager = StylePromptManager()
        all_styles = manager.get_all_style_names()
        
        # Check we have at least 10 styles
        assert len(all_styles) >= 10
        
        # Check some expected styles are present
        assert "Scandinavian" in all_styles
        assert "Industrial" in all_styles
        assert "Mid-Century" in all_styles
        
    def test_get_prompt_for_style(self):
        """Test retrieving prompt templates for specific styles."""
        manager = StylePromptManager()
        
        # Test with room type
        prompt = manager.get_prompt_for_style("Bohemian", "bedroom")
        assert "Bohemian" in prompt
        assert "bedroom" in prompt
        
        # Test with details
        prompt = manager.get_prompt_for_style("Contemporary", "living room", 
                                             "with large windows and city view")
        assert "Contemporary" in prompt
        assert "living room" in prompt
        assert "with large windows and city view" in prompt
        
    def test_get_style_preview_image(self):
        """Test retrieving preview images for styles."""
        manager = StylePromptManager()
        
        # Test preview image path exists (don't test actual image loading)
        path = manager.get_style_preview_image_path("Scandinavian")
        assert path is not None
        assert os.path.exists(path)
        
    def test_get_style_details(self):
        """Test retrieving detailed information about a style."""
        manager = StylePromptManager()
        
        # Get details for a specific style
        details = manager.get_style_details("Art Deco")
        assert details is not None
        assert "description" in details
        assert "characteristics" in details
        assert "colors" in details
        assert "example_prompt" in details
        
    def test_style_template_file_exists(self):
        """Test that the style template JSON file exists."""
        assert os.path.exists(os.path.join("src", "prompts", "style_templates.json"))
        
        # Check file can be parsed as JSON
        with open(os.path.join("src", "prompts", "style_templates.json"), "r") as f:
            templates = json.load(f)
            assert len(templates) > 0
            
    def test_custom_prompt_creation(self):
        """Test creating custom prompts."""
        manager = StylePromptManager()
        
        # Create custom prompt
        custom_prompt = manager.create_custom_prompt(
            base_style="Minimalist",
            room_type="office",
            colors="neutral whites and light grays",
            materials="wood and metal",
            lighting="bright natural lighting",
            mood="calm and productive"
        )
        
        # Check prompt contains expected elements
        assert "Minimalist" in custom_prompt
        assert "office" in custom_prompt
        assert "neutral whites and light grays" in custom_prompt
        assert "wood and metal" in custom_prompt
        assert "bright natural lighting" in custom_prompt
        assert "calm and productive" in custom_prompt
