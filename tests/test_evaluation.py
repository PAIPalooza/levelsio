"""
Test suite for visual evaluation metrics (SSIM, MSE).

Following Semantic Seed Coding Standards and BDD/TDD approach:
- Test cases are written before implementation
- Tests follow BDD "Given-When-Then" structure
- Each test has a docstring explaining its purpose
"""

import os
import pytest
import numpy as np
import cv2
from unittest import mock


class TestVisualEvaluation:
    """Test suite for visual evaluation metrics."""
    
    @pytest.fixture
    def sample_images(self):
        """Fixture that provides sample images for testing."""
        # Create two similar but slightly different 100x100 images
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 138  # Slightly lighter
        
        # Add some structures to both images for SSIM to detect
        img1[20:40, 20:40] = 200  # White square in both images
        img2[20:40, 20:40] = 200
        
        # Add different structure to second image
        img2[60:80, 60:80] = 50  # Dark square only in img2
        
        return img1, img2
    
    @pytest.fixture
    def identical_images(self):
        """Fixture that provides identical images for testing perfect scores."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return img, img.copy()
    
    @pytest.fixture
    def completely_different_images(self):
        """Fixture that provides completely different images for testing low scores."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        return img1, img2
    
    @pytest.fixture
    def evaluation_service(self):
        """Fixture that provides a visual evaluation service instance."""
        from src.evaluation import VisualEvaluationService
        return VisualEvaluationService()
    
    def test_ssim_calculation(self, evaluation_service, sample_images):
        """
        GIVEN two similar but not identical images
        WHEN calculating SSIM
        THEN the result should be between 0 and 1 with higher values for more similar images
        """
        img1, img2 = sample_images
        
        # Calculate SSIM between the two images
        ssim_value = evaluation_service.calculate_ssim(img1, img2)
        
        # SSIM should be a float between 0 and 1
        assert isinstance(ssim_value, float)
        assert 0 <= ssim_value <= 1
        
        # For these sample images, SSIM should be moderate (similar structure but differences)
        assert 0.5 <= ssim_value <= 0.95
    
    def test_ssim_identical_images(self, evaluation_service, identical_images):
        """
        GIVEN two identical images
        WHEN calculating SSIM
        THEN the result should be very close to 1
        """
        img1, img2 = identical_images
        
        # Calculate SSIM between identical images
        ssim_value = evaluation_service.calculate_ssim(img1, img2)
        
        # SSIM should be very close to 1 for identical images
        assert ssim_value > 0.99
    
    def test_ssim_different_images(self, evaluation_service, completely_different_images):
        """
        GIVEN two completely different images
        WHEN calculating SSIM
        THEN the result should be low
        """
        img1, img2 = completely_different_images
        
        # Calculate SSIM between very different images
        ssim_value = evaluation_service.calculate_ssim(img1, img2)
        
        # SSIM should be low for very different images
        assert ssim_value < 0.5
    
    def test_mse_calculation(self, evaluation_service, sample_images):
        """
        GIVEN two similar but not identical images
        WHEN calculating MSE
        THEN the result should be a positive number with lower values for more similar images
        """
        img1, img2 = sample_images
        
        # Calculate MSE between the two images
        mse_value = evaluation_service.calculate_mse(img1, img2)
        
        # MSE should be a positive float
        assert isinstance(mse_value, float)
        assert mse_value >= 0
        
        # For these sample images, MSE should be moderate
        # Given our sample images, we expect a moderate MSE value
        assert 50 <= mse_value <= 5000
    
    def test_mse_identical_images(self, evaluation_service, identical_images):
        """
        GIVEN two identical images
        WHEN calculating MSE
        THEN the result should be zero
        """
        img1, img2 = identical_images
        
        # Calculate MSE between identical images
        mse_value = evaluation_service.calculate_mse(img1, img2)
        
        # MSE should be 0 for identical images
        assert mse_value < 1e-10  # Use small epsilon to account for floating point precision
    
    def test_mse_different_images(self, evaluation_service, completely_different_images):
        """
        GIVEN two completely different images
        WHEN calculating MSE
        THEN the result should be high
        """
        img1, img2 = completely_different_images
        
        # Calculate MSE between very different images
        mse_value = evaluation_service.calculate_mse(img1, img2)
        
        # MSE should be high for very different images (black vs white)
        # RGB difference of (255,255,255) squared = 195075
        assert mse_value > 10000
    
    def test_combined_metrics(self, evaluation_service, sample_images):
        """
        GIVEN two images
        WHEN calculating multiple evaluation metrics
        THEN all metrics should be returned in a dictionary
        """
        img1, img2 = sample_images
        
        # Calculate multiple metrics
        metrics = evaluation_service.calculate_metrics(img1, img2)
        
        # Should return a dictionary with at least SSIM and MSE
        assert isinstance(metrics, dict)
        assert "ssim" in metrics
        assert "mse" in metrics
        
        # Values should match individual calculations
        assert metrics["ssim"] == evaluation_service.calculate_ssim(img1, img2)
        assert metrics["mse"] == evaluation_service.calculate_mse(img1, img2)
    
    def test_structure_preservation_score(self, evaluation_service, sample_images):
        """
        GIVEN original image and result image with masks
        WHEN calculating structure preservation score
        THEN it should return a score indicating how well structure was preserved
        """
        img1, img2 = sample_images
        
        # Create a simple binary mask (1 where structure should be preserved)
        structure_mask = np.zeros((100, 100), dtype=np.uint8)
        structure_mask[20:40, 20:40] = 1  # Mask the square that should be preserved
        
        # Calculate structure preservation score
        preservation_score = evaluation_service.calculate_preservation_score(
            original_image=img1,
            result_image=img2,
            structure_mask=structure_mask
        )
        
        # Score should be between 0 and 1
        assert isinstance(preservation_score, float)
        assert 0 <= preservation_score <= 1
        
        # In our test case, the structure (square at 20:40) is preserved, so score should be high
        assert preservation_score > 0.8
    
    def test_structure_preservation_report(self, evaluation_service, sample_images):
        """
        GIVEN original and result images with masks
        WHEN generating a structure preservation report
        THEN it should provide detailed metrics for different structural elements
        """
        img1, img2 = sample_images
        
        # Create masks for different structural elements
        wall_mask = np.zeros((100, 100), dtype=np.uint8)
        wall_mask[20:40, 20:40] = 1
        
        floor_mask = np.zeros((100, 100), dtype=np.uint8)
        floor_mask[60:80, 60:80] = 1
        
        # Generate preservation report
        report = evaluation_service.generate_structure_preservation_report(
            original_image=img1,
            result_image=img2,
            masks={
                "wall": wall_mask,
                "floor": floor_mask
            }
        )
        
        # Report should be a dictionary with scores for each element and overall
        assert isinstance(report, dict)
        assert "overall_score" in report
        assert "wall" in report
        assert "floor" in report
        
        # Overall score should be between 0 and 1
        assert 0 <= report["overall_score"] <= 1
        
        # Individual scores should be between 0 and 1
        assert 0 <= report["wall"] <= 1
        assert 0 <= report["floor"] <= 1
        
        # Wall should have higher preservation score than floor (based on our test images)
        assert report["wall"] > report["floor"]
        
    def test_visualization_creation(self, evaluation_service, sample_images):
        """
        GIVEN original and result images
        WHEN creating a comparison visualization
        THEN it should return an image showing the comparison
        """
        img1, img2 = sample_images
        
        # Create visualization
        visualization = evaluation_service.create_comparison_visualization(img1, img2)
        
        # Should return a valid numpy array image
        assert isinstance(visualization, np.ndarray)
        assert visualization.dtype == np.uint8
        
        # Visualization should be larger than input images to fit comparison
        assert visualization.shape[0] >= img1.shape[0]
        assert visualization.shape[1] >= img1.shape[1] * 2  # At least side by side
