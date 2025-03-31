"""
Extended test coverage module for evaluation functionality.

This module contains targeted BDD-style tests for the evaluation module
specifically focusing on previously uncovered lines to exceed
industry-standard coverage requirements (targeting >90%).
"""

import pytest
import numpy as np
from unittest import mock
from unittest.mock import MagicMock, patch
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.evaluation import VisualEvaluationService, ImageEvaluator


class TestImageEvaluatorExtended:
    """
    Extended test suite for ImageEvaluator.
    
    These tests specifically target uncovered lines 25-27, 41, 59, 80-102, 121, 130, 137,
    and 142 to improve code coverage beyond industry standards.
    """
    
    @pytest.fixture
    def sample_images(self):
        """Fixture providing sample images for testing."""
        # Create two similar but slightly different test images
        original = np.ones((100, 100, 3), dtype=np.uint8) * 100
        # Add some texture to original
        original[20:40, 20:40] = 150
        original[60:80, 60:80] = 50
        
        # Generated image is similar but with shifted features
        generated = np.ones((100, 100, 3), dtype=np.uint8) * 105  # Slightly brighter
        generated[25:45, 25:45] = 155  # Shifted by 5 pixels
        generated[65:85, 65:85] = 55   # Shifted by 5 pixels
        
        # Structure mask - covers the areas we want to preserve
        structure_mask = np.zeros((100, 100), dtype=np.uint8)
        structure_mask[15:45, 15:45] = 1  # Cover first feature with some margin
        structure_mask[55:85, 55:85] = 1  # Cover second feature with some margin
        
        return {
            "original": original, 
            "generated": generated, 
            "structure_mask": structure_mask,
            "background_mask": 1 - structure_mask
        }
    
    @pytest.fixture
    def image_evaluator(self):
        """Fixture providing an ImageEvaluator instance."""
        return ImageEvaluator()
    
    def test_evaluate_with_different_thresholds(self, image_evaluator, sample_images):
        """
        GIVEN an ImageEvaluator and sample images
        WHEN evaluate_structure_preservation is called with different thresholds
        THEN it should correctly apply these thresholds in evaluation
        """
        original = sample_images["original"]
        generated = sample_images["generated"]
        structure_mask = sample_images["structure_mask"]
        
        # Test with different thresholds
        for threshold in [0.4, 0.6, 0.8]:
            with patch.object(image_evaluator, 'threshold', threshold):
                result = image_evaluator.evaluate_structure_preservation(
                    original, generated, structure_mask
                )
                
                # Result should contain all expected keys
                assert "ssim_score" in result
                assert "mse_score" in result
                assert "is_structure_preserved" in result
                assert "threshold_used" in result
                
                # Threshold used should match what we set
                assert result["threshold_used"] == threshold
                
                # is_structure_preserved should be based on this threshold
                assert result["is_structure_preserved"] == (result["ssim_score"] > threshold)
    
    def test_calculate_ssim_with_mask(self, image_evaluator, sample_images):
        """
        GIVEN an ImageEvaluator and sample images
        WHEN calculate_ssim is called with a mask
        THEN it should compute SSIM only on the masked region
        """
        original = sample_images["original"]
        generated = sample_images["generated"]
        structure_mask = sample_images["structure_mask"]
        background_mask = sample_images["background_mask"]
        
        # Calculate SSIM on structure regions
        structure_ssim = image_evaluator.calculate_ssim(
            original, generated, structure_mask
        )
        
        # Calculate SSIM on background regions
        background_ssim = image_evaluator.calculate_ssim(
            original, generated, background_mask
        )
        
        # Both should return valid SSIM scores (between -1 and 1)
        assert -1.0 <= structure_ssim <= 1.0
        assert -1.0 <= background_ssim <= 1.0
        
        # Since we made the structure areas more different than the background,
        # structure SSIM should be lower (worse) than background SSIM
        assert structure_ssim <= background_ssim
    
    def test_calculate_mse_with_mask(self, image_evaluator, sample_images):
        """
        GIVEN an ImageEvaluator and sample images
        WHEN calculate_mse is called with a mask
        THEN it should compute MSE only on the masked region
        """
        original = sample_images["original"]
        generated = sample_images["generated"]
        structure_mask = sample_images["structure_mask"]
        background_mask = sample_images["background_mask"]
        
        # Calculate MSE on structure regions
        structure_mse = image_evaluator.calculate_mse(
            original, generated, structure_mask
        )
        
        # Calculate MSE on background regions
        background_mse = image_evaluator.calculate_mse(
            original, generated, background_mask
        )
        
        # Both should return valid MSE scores (>= 0)
        assert structure_mse >= 0
        assert background_mse >= 0
        
        # Since we made the structure areas more different than the background,
        # structure MSE should be higher (worse) than background MSE
        assert structure_mse >= background_mse
    
    def test_create_comparison_visualization(self, image_evaluator, sample_images, monkeypatch):
        """
        GIVEN an ImageEvaluator and sample images
        WHEN create_comparison_visualization is called
        THEN it should create and save a visualization
        """
        # Mock matplotlib to avoid actual plotting
        mock_plt = MagicMock()
        monkeypatch.setattr("matplotlib.pyplot", mock_plt)
        
        # Mock plt.savefig to simulate saving the file
        mock_plt.savefig = MagicMock()
        
        # Call the method with our sample images
        result = image_evaluator.create_comparison_visualization(
            sample_images["original"],
            sample_images["generated"],
            sample_images["structure_mask"],
            "test_visualization.png"
        )
        
        # Check if the method returns True
        assert result is True
        
        # Verify that savefig was called with the correct filename
        mock_plt.savefig.assert_called_once_with("test_visualization.png", dpi=150, bbox_inches="tight")
        
        # Verify figure was closed
        mock_plt.close.assert_called()
    
    def test_create_comparison_visualization_without_mask(self, image_evaluator, sample_images, monkeypatch):
        """
        GIVEN an ImageEvaluator and sample images
        WHEN create_comparison_visualization is called without a mask
        THEN it should still create a visualization without mask overlay
        """
        # Mock matplotlib to avoid actual plotting
        mock_plt = MagicMock()
        monkeypatch.setattr("matplotlib.pyplot", mock_plt)
        
        # Call the method without a structure mask
        result = image_evaluator.create_comparison_visualization(
            sample_images["original"],
            sample_images["generated"],
            None,  # No mask
            "test_visualization_no_mask.png"
        )
        
        # Check if the method returns True
        assert result is True
        
        # Verify that savefig was called
        mock_plt.savefig.assert_called_once()
        
        # Verify only 2 subplots were created (no mask overlay)
        assert mock_plt.subplot.call_count == 2


class TestVisualEvaluationServiceExtended:
    """
    Extended test suite for VisualEvaluationService.
    
    These tests specifically target uncovered lines 241, 262, 267, 269, 303, 311-312,
    and 318-324 to achieve comprehensive test coverage.
    """
    
    @pytest.fixture
    def visual_evaluation_service(self):
        """Fixture providing a VisualEvaluationService instance."""
        return VisualEvaluationService()
    
    @pytest.fixture
    def sample_images(self):
        """Fixture providing sample image paths for testing."""
        # Return paths that would be used for testing
        return {
            "original_path": "test_original.png",
            "generated_path": "test_generated.png",
            "structure_mask_path": "test_structure_mask.png"
        }
    
    def test_evaluate_single_image_style_transfer(self, visual_evaluation_service, sample_images, monkeypatch):
        """
        GIVEN a VisualEvaluationService and sample image paths
        WHEN evaluate_single_image_style_transfer is called
        THEN it should evaluate the style transfer quality
        """
        # Mock the image loading
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        # Mock PIL.Image.open
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.__array__ = MagicMock(return_value=mock_image)
        
        monkeypatch.setattr('PIL.Image.open', lambda path: mock_img)
        
        # Mock ImageEvaluator methods
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_structure_preservation.return_value = {
            "ssim_score": 0.85,
            "mse_score": 0.05,
            "is_structure_preserved": True,
            "threshold_used": 0.7
        }
        mock_evaluator.create_comparison_visualization.return_value = True
        
        monkeypatch.setattr(visual_evaluation_service, "evaluator", mock_evaluator)
        
        # Call the method
        result = visual_evaluation_service.evaluate_single_image_style_transfer(
            sample_images["original_path"],
            sample_images["generated_path"],
            sample_images["structure_mask_path"],
            "test_visualization.png"
        )
        
        # Check the result structure
        assert "ssim_score" in result
        assert "mse_score" in result
        assert "is_structure_preserved" in result
        assert "visualization_path" in result
        
        # Check that create_comparison_visualization was called
        mock_evaluator.create_comparison_visualization.assert_called_once()
    
    def test_evaluate_batch_style_transfer(self, visual_evaluation_service, monkeypatch):
        """
        GIVEN a VisualEvaluationService
        WHEN evaluate_batch_style_transfer is called
        THEN it should evaluate multiple style transfers and produce a summary
        """
        # Sample batch data
        batch_data = [
            {
                "original_path": "test_original1.png",
                "generated_path": "test_generated1.png",
                "structure_mask_path": "test_structure_mask1.png",
                "output_visualization_path": "test_vis1.png"
            },
            {
                "original_path": "test_original2.png",
                "generated_path": "test_generated2.png",
                "structure_mask_path": "test_structure_mask2.png",
                "output_visualization_path": "test_vis2.png"
            }
        ]
        
        # Mock evaluate_single_image_style_transfer to return predetermined results
        def mock_evaluate_single(*args, **kwargs):
            return {
                "ssim_score": 0.8,
                "mse_score": 0.1,
                "is_structure_preserved": True,
                "visualization_path": args[3]
            }
        
        monkeypatch.setattr(
            visual_evaluation_service,
            "evaluate_single_image_style_transfer",
            mock_evaluate_single
        )
        
        # Call the method
        result = visual_evaluation_service.evaluate_batch_style_transfer(batch_data)
        
        # Check the result structure
        assert "individual_results" in result
        assert "summary" in result
        
        # Check that individual_results has the right number of items
        assert len(result["individual_results"]) == len(batch_data)
        
        # Check summary includes aggregate metrics
        assert "average_ssim" in result["summary"]
        assert "average_mse" in result["summary"]
        assert "preservation_rate" in result["summary"]
    
    def test_generate_report(self, visual_evaluation_service, monkeypatch, tmp_path):
        """
        GIVEN a VisualEvaluationService and evaluation results
        WHEN generate_report is called
        THEN it should create an HTML report with the results
        """
        # Sample evaluation results
        evaluation_results = {
            "individual_results": [
                {
                    "ssim_score": 0.85,
                    "mse_score": 0.05,
                    "is_structure_preserved": True,
                    "visualization_path": "test_vis1.png"
                },
                {
                    "ssim_score": 0.75,
                    "mse_score": 0.15,
                    "is_structure_preserved": False,
                    "visualization_path": "test_vis2.png"
                }
            ],
            "summary": {
                "average_ssim": 0.8,
                "average_mse": 0.1,
                "preservation_rate": 0.5
            }
        }
        
        # Create a temporary output file
        output_file = os.path.join(tmp_path, "test_report.html")
        
        # Mock open to avoid actually writing the file
        mock_open = mock.mock_open()
        monkeypatch.setattr("builtins.open", mock_open)
        
        # Call the method
        result = visual_evaluation_service.generate_report(evaluation_results, output_file)
        
        # Check the result
        assert result is True
        
        # Verify that open was called with the right file
        mock_open.assert_called_once_with(output_file, "w")
        
        # Verify that HTML content was written to the file
        file_handle = mock_open()
        write_calls = file_handle.write.call_args_list
        
        # Check for key elements in the HTML
        html_written = ''.join(call[0][0] for call in write_calls)
        assert "html" in html_written.lower()
        assert "evaluation report" in html_written.lower()
        assert "average ssim" in html_written.lower()
    
    def test_color_mask_overlay(self, visual_evaluation_service):
        """
        GIVEN a VisualEvaluationService
        WHEN _color_mask_overlay is called
        THEN it should correctly overlay a colored mask on an image
        """
        # Create test image and mask
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 1  # Create a square mask
        
        # Call the method with red overlay
        result = visual_evaluation_service._color_mask_overlay(image, mask, color=[1.0, 0.0, 0.0])
        
        # Result should have same shape as input
        assert result.shape == image.shape
        
        # Masked area should be tinted with the color
        assert np.any(result[20, 20] != image[20, 20])  # Center of masked area should be different
        assert np.all(result[5, 5] == image[5, 5])      # Outside mask should be unchanged
    
    def test_normalize_array(self, visual_evaluation_service):
        """
        GIVEN a VisualEvaluationService
        WHEN _normalize_array is called
        THEN it should correctly normalize arrays
        """
        # Test with different arrays
        test_cases = [
            # Array with values 0-255 (common image range)
            (np.array([0, 127, 255]), np.array([0.0, 0.5, 1.0])),
            
            # Already normalized array (0-1)
            (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),
            
            # Negative and positive values
            (np.array([-10, 0, 10]), np.array([0.0, 0.5, 1.0])),
            
            # All identical values
            (np.array([5, 5, 5]), np.array([0.0, 0.0, 0.0]))
        ]
        
        for input_array, expected_output in test_cases:
            result = visual_evaluation_service._normalize_array(input_array)
            np.testing.assert_allclose(result, expected_output, rtol=1e-5)
    
    def test_format_percentage(self, visual_evaluation_service):
        """
        GIVEN a VisualEvaluationService
        WHEN _format_percentage is called
        THEN it should correctly format values as percentages
        """
        # Test with different values
        test_cases = [
            (0.0, "0.00%"),
            (0.123, "12.30%"),
            (1.0, "100.00%"),
            (0.5678, "56.78%"),
            (1.5, "150.00%")  # Should handle values > 1
        ]
        
        for value, expected_output in test_cases:
            result = visual_evaluation_service._format_percentage(value)
            assert result == expected_output
