"""
Evaluation module for Interior Style Transfer POC.

This module handles evaluation of generated images using metrics
like SSIM (Structural Similarity Index) and MSE (Mean Squared Error).
"""

import numpy as np
import cv2
import logging
from typing import Dict, Tuple, Union, Optional, List, Any
from PIL import Image
import matplotlib.pyplot as plt
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import skimage metrics for SSIM and MSE
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error as mse
    SKIMAGE_AVAILABLE = True
    logger.info("scikit-image metrics available")
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not installed. Install with: pip install scikit-image")


class VisualEvaluationService:
    """
    Service for evaluating visual quality and structure preservation of generated images.
    
    Provides methods for calculating similarity metrics (SSIM, MSE) and
    generating visualizations to assess the quality of style transfers.
    """
    
    def __init__(self):
        """Initialize the visual evaluation service."""
        if not SKIMAGE_AVAILABLE:
            logger.warning("Using fallback implementations for SSIM and MSE. For better results, install scikit-image.")
        logger.info("VisualEvaluationService initialized")
    
    def calculate_ssim(self, image1: np.ndarray, image2: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Structural Similarity Index between two images.
        
        GIVEN two images to compare
        WHEN the SSIM calculation is performed
        THEN a similarity score between 0 and 1 is returned (higher is better)
        
        Args:
            image1: First image as NumPy array (RGB)
            image2: Second image as NumPy array (RGB)
            mask: Optional mask to focus comparison on specific areas
            
        Returns:
            float: SSIM score (higher is better, max 1.0)
            
        Raises:
            ValueError: If images have different dimensions
        """
        # Ensure images are the same size
        if image1.shape != image2.shape:
            raise ValueError(f"Images must have the same dimensions. Got {image1.shape} and {image2.shape}")
        
        # Convert RGB to grayscale for SSIM calculation
        if len(image1.shape) == 3 and image1.shape[2] == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = image1
            gray2 = image2
        
        # Calculate SSIM
        if SKIMAGE_AVAILABLE:
            # Use scikit-image implementation (more accurate)
            if mask is not None:
                # Apply mask if provided (only calculate SSIM in masked areas)
                ssim_value = ssim(gray1, gray2, data_range=255, win_size=11, gaussian_weights=True, 
                                  sigma=1.5, use_sample_covariance=False, multichannel=False)
            else:
                ssim_value = ssim(gray1, gray2, data_range=255)
        else:
            # Fallback to OpenCV implementation
            try:
                # OpenCV 4.x uses SSIM method
                ssim_value = cv2.SSIM(gray1, gray2)[0]
            except AttributeError:
                # Simple fallback if neither scikit-image nor OpenCV SSIM is available
                # Compute a basic SSIM approximation
                # Luminance component
                mu1 = np.mean(gray1)
                mu2 = np.mean(gray2)
                lum_comp = (2 * mu1 * mu2 + 0.01) / (mu1**2 + mu2**2 + 0.01)
                
                # Contrast component
                sigma1 = np.std(gray1)
                sigma2 = np.std(gray2)
                contrast_comp = (2 * sigma1 * sigma2 + 0.03) / (sigma1**2 + sigma2**2 + 0.03)
                
                # Structure component (simplified)
                # Compute covariance
                sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
                struct_comp = (sigma12 + 0.03) / (sigma1 * sigma2 + 0.03)
                
                # Combine components
                ssim_value = lum_comp * contrast_comp * struct_comp
        
        return float(ssim_value)
    
    def calculate_mse(self, image1: np.ndarray, image2: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Mean Squared Error between two images.
        
        GIVEN two images to compare
        WHEN the MSE calculation is performed
        THEN an error value is returned (lower is better)
        
        Args:
            image1: First image as NumPy array (RGB)
            image2: Second image as NumPy array (RGB)
            mask: Optional mask to focus comparison on specific areas
            
        Returns:
            float: MSE value (lower is better, min 0.0)
            
        Raises:
            ValueError: If images have different dimensions
        """
        # Ensure images are the same size
        if image1.shape != image2.shape:
            raise ValueError(f"Images must have the same dimensions. Got {image1.shape} and {image2.shape}")
        
        # Convert to float for calculation
        img1 = image1.astype(np.float32)
        img2 = image2.astype(np.float32)
        
        if mask is not None:
            # If mask provided, only calculate MSE for masked pixels
            if mask.shape[:2] != image1.shape[:2]:
                raise ValueError(f"Mask dimensions {mask.shape} don't match image dimensions {image1.shape[:2]}")
            
            # Expand mask to match image dimensions if needed
            if len(mask.shape) == 2 and len(image1.shape) == 3:
                mask_expanded = np.expand_dims(mask, axis=2)
                mask_expanded = np.repeat(mask_expanded, image1.shape[2], axis=2)
            else:
                mask_expanded = mask
            
            # Apply mask and calculate MSE only for masked pixels
            mask_binary = mask_expanded > 0
            if np.sum(mask_binary) == 0:
                return 0.0  # No pixels to compare
            
            squared_diff = np.square(img1[mask_binary] - img2[mask_binary])
            mse_value = np.mean(squared_diff)
        else:
            # Calculate MSE for the entire image
            mse_value = np.mean(np.square(img1 - img2))
        
        return float(mse_value)
    
    def calculate_metrics(self, image1: np.ndarray, image2: np.ndarray, 
                         mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate multiple evaluation metrics between two images.
        
        Args:
            image1: First image as NumPy array
            image2: Second image as NumPy array
            mask: Optional mask to focus comparison on specific areas
            
        Returns:
            Dictionary with metric names and values
        """
        metrics = {
            "ssim": self.calculate_ssim(image1, image2, mask),
            "mse": self.calculate_mse(image1, image2, mask),
            # PSNR (Peak Signal-to-Noise Ratio)
            "psnr": cv2.PSNR(image1, image2) if np.max(image1) > 0 and np.max(image2) > 0 else 0.0
        }
        
        return metrics
    
    def generate_difference_visualization(self, original_image: np.ndarray, styled_image: np.ndarray) -> np.ndarray:
        """
        Generate a visualization comparing the original and styled images side by side with their difference.
        
        Args:
            original_image: Original image as NumPy array
            styled_image: Styled image as NumPy array
            
        Returns:
            Visualization image as NumPy array
        """
        # Ensure images are the same size
        if original_image.shape != styled_image.shape:
            # Resize styled image to match original dimensions
            h, w = original_image.shape[:2]
            styled_image_resized = cv2.resize(styled_image, (w, h))
        else:
            styled_image_resized = styled_image

        # Calculate absolute difference
        diff = cv2.absdiff(original_image, styled_image_resized)
        
        # Convert to heat map for better visualization
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        
        # Create a side-by-side visualization
        h, w = original_image.shape[:2]
        visualization = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Original image
        visualization[:, :w] = original_image
        
        # Styled image
        visualization[:, w:w*2] = styled_image_resized
        
        # Difference heatmap
        visualization[:, w*2:] = diff_heatmap
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(visualization, "Original", (10, 20), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(visualization, "Styled", (w + 10, 20), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(visualization, "Difference", (w*2 + 10, 20), font, font_scale, (255, 255, 255), font_thickness)
        
        # Add metrics text
        metrics = self.calculate_metrics(original_image, styled_image_resized)
        metric_text = f"SSIM: {metrics['ssim']:.3f}, MSE: {metrics['mse']:.1f}, PSNR: {metrics['psnr']:.1f} dB"
        cv2.putText(visualization, metric_text, (10, h - 10), font, font_scale, (255, 255, 255), font_thickness)
        
        return visualization
        
    def calculate_preservation_score(self, original_image: np.ndarray, result_image: np.ndarray,
                                    structure_mask: np.ndarray) -> float:
        """
        Calculate structure preservation score between original and result images.
        
        Args:
            original_image: Original image as NumPy array
            result_image: Result image as NumPy array
            structure_mask: Binary mask of structural elements to preserve
            
        Returns:
            Structure preservation score (higher is better, max 1.0)
        """
        # Calculate SSIM only for the masked areas
        ssim_value = self.calculate_ssim(original_image, result_image, structure_mask)
        
        # MSE for masked areas (lower is better)
        mse_value = self.calculate_mse(original_image, result_image, structure_mask)
        
        # Normalize MSE to [0, 1] range with exponential scaling for better sensitivity
        # Typical MSE values can be large, so we use exp(-mse/K) to map to [0, 1]
        K = 1000.0  # Scaling factor, adjust based on typical MSE values
        normalized_mse = np.exp(-mse_value / K)
        
        # Combine SSIM and normalized MSE for final score
        # Weight SSIM more heavily (0.7) as it's more perceptually relevant
        preservation_score = 0.7 * ssim_value + 0.3 * normalized_mse
        
        return preservation_score
    
    def create_comparison_visualization(self, original_image: np.ndarray, 
                                      result_image: np.ndarray) -> np.ndarray:
        """
        Create a visualization showing original and result images side by side.
        
        Args:
            original_image: Original image as NumPy array
            result_image: Result image as NumPy array
            
        Returns:
            Visualization image as NumPy array
        """
        # Ensure images are the same size for comparison
        if original_image.shape != result_image.shape:
            # Resize to match if needed
            result_image = cv2.resize(result_image, 
                                     (original_image.shape[1], original_image.shape[0]))
        
        # Calculate the absolute difference to visualize changes
        if original_image.dtype != np.uint8:
            original_image = original_image.astype(np.uint8)
        if result_image.dtype != np.uint8:
            result_image = result_image.astype(np.uint8)
            
        diff_image = cv2.absdiff(original_image, result_image)
        # Enhance the difference for better visibility
        diff_image = cv2.convertScaleAbs(diff_image, alpha=2.0)
        
        # Create a side-by-side comparison with the original, result, and diff
        h, w = original_image.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Place images side by side
        comparison[:, :w] = original_image
        comparison[:, w:2*w] = result_image
        comparison[:, 2*w:] = diff_image
        
        # Add labels (texts will be added by other visualization tools)
        
        return comparison
    
    def create_heatmap_visualization(self, original_image: np.ndarray, 
                                   result_image: np.ndarray) -> np.ndarray:
        """
        Create a heatmap visualization showing where changes are significant.
        
        Args:
            original_image: Original image as NumPy array
            result_image: Result image as NumPy array
            
        Returns:
            Heatmap visualization as NumPy array
        """
        # Ensure images are the same size for comparison
        if original_image.shape != result_image.shape:
            # Resize to match if needed
            result_image = cv2.resize(result_image, 
                                     (original_image.shape[1], original_image.shape[0]))
        
        # Convert to grayscale for SSIM
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            gray1 = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = original_image
            gray2 = result_image
        
        # Calculate SSIM and get the full SSIM image
        if SKIMAGE_AVAILABLE:
            _, ssim_map = ssim(gray1, gray2, data_range=255, full=True)
        else:
            try:
                # OpenCV 4.x uses SSIM method
                _, ssim_map = cv2.SSIM(gray1, gray2, full=True)
            except AttributeError:
                # Simple fallback if neither scikit-image nor OpenCV SSIM is available
                # Create a basic SSIM map (difference visualization)
                ssim_map = 1.0 - np.abs(gray1.astype(float) - gray2.astype(float)) / 255.0
        
        # Normalize SSIM map to [0, 1]
        ssim_map_normalized = (ssim_map - np.min(ssim_map)) / (np.max(ssim_map) - np.min(ssim_map))
        
        # Convert to heatmap (red = low similarity, blue = high similarity)
        heatmap = cv2.applyColorMap((1 - ssim_map_normalized) * 255, cv2.COLORMAP_JET)
        
        # Combine original image with heatmap
        alpha = 0.7  # Transparency of heatmap
        blended = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return blended


class ImageEvaluator:
    """
    Evaluates the quality and structural preservation of generated interior images.
    
    This class provides methods for calculating similarity metrics between input 
    and output images to ensure architectural elements are preserved during 
    style transfer, following SSCS principles for evaluation.
    """
    
    def __init__(self):
        """
        Initialize the image evaluator.
        
        GIVEN a need to evaluate image transformations
        WHEN an ImageEvaluator is created
        THEN it is initialized with the necessary evaluation services
        """
        self.evaluation_service = VisualEvaluationService()
        logger.info("ImageEvaluator initialized")
    
    def calculate_ssim(self, image1: np.ndarray, image2: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Structural Similarity Index between two images.
        
        GIVEN two images to compare
        WHEN the SSIM calculation is performed
        THEN a similarity score between 0 and 1 is returned (higher is better)
        
        Args:
            image1: First image as NumPy array (RGB)
            image2: Second image as NumPy array (RGB)
            mask: Optional mask to focus comparison on specific areas
            
        Returns:
            float: SSIM score (higher is better, max 1.0)
            
        Raises:
            ValueError: If images have different dimensions
        """
        return self.evaluation_service.calculate_ssim(image1, image2, mask)
    
    def calculate_mse(self, image1: np.ndarray, image2: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Mean Squared Error between two images.
        
        GIVEN two images to compare
        WHEN the MSE calculation is performed
        THEN an error value is returned (lower is better)
        
        Args:
            image1: First image as NumPy array (RGB)
            image2: Second image as NumPy array (RGB)
            mask: Optional mask to focus comparison on specific areas
            
        Returns:
            float: MSE value (lower is better, min 0.0)
            
        Raises:
            ValueError: If images have different dimensions
        """
        return self.evaluation_service.calculate_mse(image1, image2, mask)
    
    def evaluate_structure_preservation(self, 
                                       original: np.ndarray,
                                       generated: np.ndarray,
                                       structure_mask: np.ndarray,
                                       threshold: float = 0.8) -> Dict[str, Union[float, bool]]:
        """
        Evaluate if structural elements are preserved in the generated image.
        
        Args:
            original: Original image as NumPy array
            generated: Generated image as NumPy array
            structure_mask: Mask of structural elements
            threshold: SSIM threshold for structure preservation
            
        Returns:
            Dictionary with evaluation metrics and boolean result
        """
        # Calculate metrics
        ssim_score = self.calculate_ssim(original, generated, structure_mask)
        mse_score = self.calculate_mse(original, generated, structure_mask)
        
        # Determine if structure is preserved based on SSIM threshold
        is_preserved = ssim_score >= threshold
        
        return {
            "ssim_score": ssim_score,
            "mse_score": mse_score,
            "is_structure_preserved": is_preserved,
            "threshold_used": threshold
        }
    
    def visualize_differences(self, original: np.ndarray, generated: np.ndarray) -> np.ndarray:
        """
        Generate a visualization of differences between original and generated images.
        
        Args:
            original: Original image as NumPy array
            generated: Generated image as NumPy array
            
        Returns:
            Difference visualization as NumPy array
        """
        return self.evaluation_service.create_comparison_visualization(original, generated)
