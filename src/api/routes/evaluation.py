"""
Evaluation API endpoints.

This module defines the REST API endpoints for evaluating style transfer
quality and effectiveness. It follows Behavior-Driven Development principles
from the Semantic Seed Coding Standards (SSCS).
"""

import base64
import io
import traceback
from typing import Dict, List, Optional, Any, Union
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, status, Header, File, UploadFile, Form
from pydantic import ValidationError, BaseModel, Field

from src.api.models.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    GalleryEvaluationRequest,
    GalleryEvaluationResponse,
    MetricType,
    EvaluationJsonRequest
)
from src.api.core.config import settings
from src.api.core.auth import verify_api_key
from src.evaluation import ImageEvaluator, VisualEvaluationService

# Create router
router = APIRouter()


class StyleEvaluationResponse(BaseModel):
    """Response model for style transfer evaluation."""
    
    style_score: float = Field(..., description="Style adherence score (0-100)")
    quality_score: float = Field(..., description="Image quality score (0-100)")
    realism_score: float = Field(..., description="Realism score (0-100)")
    overall_score: float = Field(..., description="Overall quality score (0-100)")
    feedback: Dict[str, Any] = Field(..., description="Detailed feedback")
    
    class Config:
        schema_extra = {
            "example": {
                "style_score": 87.5,
                "quality_score": 92.3,
                "realism_score": 85.9,
                "overall_score": 88.6,
                "feedback": {
                    "strengths": ["Excellent color palette", "Good texture rendering"],
                    "weaknesses": ["Slightly inconsistent lighting"],
                    "suggestions": ["Consider adjusting the lighting parameters"]
                }
            }
        }


# Helper functions for image conversion
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 string to a numpy array image."""
    if base64_string is None:
        raise ValueError("Base64 image string cannot be None")
    
    # Remove data URI scheme if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Convert bytes to numpy array
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a numpy array image to a base64 string."""
    # Convert numpy array to PIL Image
    if not isinstance(image, (np.ndarray, Image.Image)):
        raise ValueError(f"Expected image as numpy array or PIL Image, got {type(image)}")
    
    if isinstance(image, np.ndarray):
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Convert PIL Image to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Encode bytes to base64
    base64_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_encoded}"


@router.post(
    "/style-score",
    response_model=StyleEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate style transfer result (File Upload)",
    description="""
    Evaluate the quality of a style transfer result.
    
    This endpoint accepts an original image and a styled image,
    and returns an evaluation of the style transfer quality.
    """
)
async def evaluate_style_transfer_upload(
    original_image: UploadFile = File(..., description="Original interior image"),
    styled_image: UploadFile = File(..., description="Style-transferred result image"),
    style_name: str = Form(..., description="Name of the style applied"),
    api_key: str = Depends(verify_api_key)
) -> StyleEvaluationResponse:
    """
    Evaluate the quality of a style transfer result.
    
    Args:
        original_image: Original interior image
        styled_image: Style-transferred result image
        style_name: Name of the style applied
        api_key: API key for authentication
        
    Returns:
        Style transfer evaluation scores and feedback
    """
    try:
        # Read image content
        original_content = await original_image.read()
        styled_content = await styled_image.read()
        
        # Initialize evaluator service
        evaluator = VisualEvaluationService()
        
        # Convert image contents to numpy arrays
        original_img = Image.open(io.BytesIO(original_content))
        styled_img = Image.open(io.BytesIO(styled_content))
        
        original_array = np.array(original_img)
        styled_array = np.array(styled_img)
        
        # Perform evaluation
        evaluation = evaluator.evaluate_style(
            original_array, 
            styled_array,
            style_name
        )
        
        # Return evaluation results
        return StyleEvaluationResponse(
            style_score=evaluation.get("style_score", 85.0),
            quality_score=evaluation.get("quality_score", 90.0),
            realism_score=evaluation.get("realism_score", 87.5),
            overall_score=evaluation.get("overall_score", 87.5),
            feedback={
                "strengths": evaluation.get("strengths", ["Good style transfer"]),
                "weaknesses": evaluation.get("weaknesses", []),
                "suggestions": evaluation.get("suggestions", [])
            }
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in style evaluation: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Style evaluation failed: {str(e)}"
        )


@router.post(
    "/compare-styles",
    status_code=status.HTTP_200_OK,
    summary="Compare multiple style transfer results (File Upload)",
    description="""
    Compare multiple style transfer results.
    
    This endpoint accepts an original image and multiple styled images,
    and returns a comparison analysis of the different styles.
    """
)
async def compare_style_transfers(
    original_image: UploadFile = File(..., description="Original interior image"),
    result_images: List[UploadFile] = File(..., description="Style-transferred result images"),
    style_names: str = Form(..., description="Comma-separated list of style names"),
    api_key: str = Depends(verify_api_key)
):
    """
    Compare multiple style transfer results.
    
    Args:
        original_image: Original interior image
        result_images: Style-transferred result images
        style_names: Comma-separated list of style names
        api_key: API key for authentication
        
    Returns:
        Comparison analysis
    """
    try:
        # Parse style names
        styles = [s.strip() for s in style_names.split(",")]
        
        # Check if number of styles matches number of images
        if len(styles) != len(result_images):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of style names must match number of result images"
            )
        
        # Read original image content
        original_content = await original_image.read()
        original_img = Image.open(io.BytesIO(original_content))
        original_array = np.array(original_img)
        
        # Initialize evaluator service
        evaluator = VisualEvaluationService()
        
        # Process each result image
        results = []
        
        for i, result_image in enumerate(result_images):
            # Read image content
            result_content = await result_image.read()
            result_img = Image.open(io.BytesIO(result_content))
            result_array = np.array(result_img)
            
            # Evaluate this style
            evaluation = evaluator.evaluate_style(
                original_array,
                result_array,
                styles[i]
            )
            
            # Add to results
            results.append({
                "style": styles[i],
                "style_score": evaluation.get("style_score", 85.0),
                "quality_score": evaluation.get("quality_score", 90.0),
                "realism_score": evaluation.get("realism_score", 87.5),
                "overall_score": evaluation.get("overall_score", 87.5)
            })
        
        # Rank styles by overall score
        ranked_styles = sorted(results, key=lambda x: x["overall_score"], reverse=True)
        
        # Return comparison results
        return {
            "ranked_styles": ranked_styles,
            "best_style": ranked_styles[0]["style"] if ranked_styles else None,
            "comparison_metrics": {
                "style_consistency": 92.5,  # Placeholder metric
                "quality_variation": 5.3    # Placeholder metric
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in style comparison: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Style comparison failed: {str(e)}"
        )


@router.post(
    "/metrics",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate style transfer quality",
    description="""
    Calculate metrics to evaluate the quality of style transfer.
    
    This endpoint compares the original image with the styled image to assess
    various quality metrics such as structural similarity (SSIM), mean squared
    error (MSE), and structure preservation.
    
    The response includes computed metrics and optional visualizations.
    """
)
@router.post(
    "/compare",  # Adding an alias route for compatibility with test_api_workflow.py
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare original and styled images",
    description="Alias for /metrics endpoint to maintain compatibility with existing clients."
)
async def evaluate_style_transfer(
    request: EvaluationRequest,
    api_key: str = Depends(verify_api_key)
) -> EvaluationResponse:
    """
    Evaluate style transfer quality between original and styled images.
    
    Given base64-encoded original and styled images, this endpoint calculates
    various quality metrics to assess the effectiveness of style transfer.
    
    Args:
        request: Evaluation request with original and styled images
        api_key: API key for authentication
        
    Returns:
        Evaluation results with metrics and optional visualization
    """
    try:
        # Initialize evaluator
        image_evaluator = ImageEvaluator()
        visual_evaluator = VisualEvaluationService()
        
        # Decode the images
        original_image = decode_base64_image(request.original_image)
        styled_image = decode_base64_image(request.styled_image)
        
        # Calculate metrics
        metrics = {}
        if request.metrics == MetricType.ALL or request.metrics == MetricType.SSIM:
            ssim = image_evaluator.calculate_ssim(original_image, styled_image)
            metrics["ssim"] = float(ssim)
            
        if request.metrics == MetricType.ALL or request.metrics == MetricType.MSE:
            mse = image_evaluator.calculate_mse(original_image, styled_image)
            metrics["mse"] = float(mse)
            
        if request.metrics == MetricType.ALL or request.metrics == MetricType.PSNR:
            psnr = image_evaluator.calculate_psnr(original_image, styled_image)
            metrics["psnr"] = float(psnr)
        
        # Calculate structure preservation if requested
        structure_score = None
        if request.calculate_structure_preservation:
            preservation = image_evaluator.calculate_structure_preservation(
                original_image, styled_image
            )
            structure_score = float(preservation["score"])
        
        # Generate visualization if requested
        visualization = None
        if request.include_visualization:
            viz = visual_evaluator.generate_difference_visualization(
                original_image, styled_image
            )
            visualization = encode_image_to_base64(viz)
        
        # Create response
        return EvaluationResponse(
            metrics=metrics,
            visualization=visualization,
            structure_preservation_score=structure_score
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in evaluation: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during evaluation: {str(e)}"
        )


@router.post(
    "/gallery-evaluation",
    response_model=GalleryEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a gallery of styled images",
    description="""
    Analyze multiple style transformations applied to the same image.
    
    This endpoint helps determine which interior design styles work best
    for a given interior image by comparing quality metrics across styles.
    """
)
async def evaluate_gallery(
    request: GalleryEvaluationRequest,
    api_key: str = Depends(verify_api_key)
) -> GalleryEvaluationResponse:
    """
    Evaluate and compare multiple styled versions of an interior image.
    
    This endpoint helps assess which styles work best for a particular interior
    by comparing metrics across different style applications.
    
    Args:
        request: Gallery evaluation request with original and multiple styled images
        api_key: API key for authentication
        
    Returns:
        Comparative evaluation with metrics and rankings
    """
    try:
        # Initialize evaluators
        image_evaluator = ImageEvaluator()
        visual_evaluator = VisualEvaluationService()
        
        # Decode the original image
        original_image = decode_base64_image(request.original_image)
        
        # Process each styled image
        results = []
        
        for idx, styled_image_b64 in enumerate(request.styled_images):
            try:
                # Decode the styled image
                styled_image = decode_base64_image(styled_image_b64)
                
                # Calculate metrics
                metrics = {}
                
                # Always calculate basic metrics
                ssim = image_evaluator.calculate_ssim(original_image, styled_image)
                mse = image_evaluator.calculate_mse(original_image, styled_image)
                psnr = image_evaluator.calculate_psnr(original_image, styled_image)
                
                metrics["ssim"] = float(ssim)
                metrics["mse"] = float(mse)
                metrics["psnr"] = float(psnr)
                
                # Calculate structure preservation
                structure_score = 0.0
                if request.calculate_structure_preservation:
                    preservation = image_evaluator.calculate_structure_preservation(
                        original_image, styled_image
                    )
                    structure_score = float(preservation["score"])
                
                # Generate difference visualization
                visualization = None
                if request.include_visualizations:
                    viz = visual_evaluator.generate_difference_visualization(
                        original_image, styled_image
                    )
                    visualization = encode_image_to_base64(viz)
                
                # Add result
                results.append({
                    "style_index": idx,
                    "metrics": metrics,
                    "structure_preservation_score": structure_score,
                    "visualization": visualization
                })
                
            except Exception as e:
                print(f"Error processing style {idx}: {str(e)}")
                # Continue with next style
        
        # Rank the styles based on metrics
        # Higher SSIM and PSNR are better, lower MSE is better
        ssim_ranking = sorted(
            range(len(results)), 
            key=lambda i: results[i]["metrics"]["ssim"], 
            reverse=True
        )
        
        mse_ranking = sorted(
            range(len(results)), 
            key=lambda i: results[i]["metrics"]["mse"]
        )
        
        psnr_ranking = sorted(
            range(len(results)), 
            key=lambda i: results[i]["metrics"]["psnr"], 
            reverse=True
        )
        
        # Create combined ranking (lower is better)
        combined_scores = []
        for i in range(len(results)):
            score = (
                ssim_ranking.index(i) + 
                mse_ranking.index(i) + 
                psnr_ranking.index(i)
            ) / 3.0
            combined_scores.append((i, score))
        
        combined_ranking = [idx for idx, _ in sorted(combined_scores, key=lambda x: x[1])]
        
        # Return response
        return GalleryEvaluationResponse(
            style_results=results,
            rankings={
                "ssim": ssim_ranking,
                "mse": mse_ranking,
                "psnr": psnr_ranking,
                "combined": combined_ranking
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in gallery evaluation: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gallery evaluation error: {str(e)}"
        )


@router.get(
    "/available-metrics",
    status_code=status.HTTP_200_OK,
    summary="Get available evaluation metrics",
    description="Retrieve the list of available evaluation metrics with descriptions."
)
async def get_available_metrics(
    api_key: str = Depends(verify_api_key)
) -> dict:
    """Get the list of available evaluation metrics with descriptions."""
    return {
        "metrics": [metric.value for metric in MetricType if metric != MetricType.ALL],
        "descriptions": {
            "ssim": "Structural Similarity Index (0-1, higher is better) - measures structural similarity between images",
            "mse": "Mean Squared Error (lower is better) - measures pixel-level differences",
            "psnr": "Peak Signal-to-Noise Ratio (higher is better) - measures image quality",
            "lpips": "Learned Perceptual Image Patch Similarity (lower is better) - measures perceptual similarity"
        },
        "recommended_for": {
            "style_evaluation": ["ssim", "lpips"],
            "structure_preservation": ["ssim", "mse"],
            "overall_quality": ["psnr", "ssim"]
        }
    }


@router.post(
    "/style-score-json",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate style transfer using JSON input",
    description="Evaluate the quality of a style transfer result using JSON input with base64-encoded images."
)
async def evaluate_style_transfer_json(
    request: EvaluationJsonRequest,
    api_key: str = Depends(verify_api_key)
) -> EvaluationResponse:
    """
    Evaluate the quality of a style transfer result using JSON input with base64-encoded images.
    
    Args:
        request: Evaluation request with JSON data
        api_key: API key for authentication
        
    Returns:
        EvaluationResponse with quality metrics
        
    Raises:
        HTTPException: If the request is invalid or evaluation fails
    """
    import logging
    import os
    
    # Validate base64 image data
    try:
        if not request.original_image.startswith('data:image/'):
            # Try to add prefix if not present
            request.original_image = f"data:image/jpeg;base64,{request.original_image}"
            
        if not request.styled_image.startswith('data:image/'):
            # Try to add prefix if not present  
            request.styled_image = f"data:image/jpeg;base64,{request.styled_image}"
    except Exception as e:
        logging.error(f"Error processing base64 images: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Invalid image data",
                "details": {"reason": "image_validation_error"}
            }
        )
    
    # Get metrics to calculate
    metrics = request.metrics or ["ssim", "mse", "psnr"]
    
    try:
        # Create a FluxAPI instance 
        from src.flux_integration import FluxAPI
        flux_api = FluxAPI()
        
        # Make the async API call properly
        result = await flux_api.evaluate_style_transfer(
            original_image=request.original_image,
            styled_image=request.styled_image,
            metrics=metrics,
            style_names=request.style_names
        )
        
        # Return the response
        return EvaluationResponse(
            metrics=result.get("metrics", []),
            overall_score=result.get("overall_score", 75.0),
            style_name=result.get("style_name"),
            processing_time=result.get("processing_time", 0.5)
        )
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": f"Evaluation failed: {str(e)}",
                "details": {"api_name": "Flux API"}
            }
        )
