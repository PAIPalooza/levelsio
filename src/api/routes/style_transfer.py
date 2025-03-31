"""
Style transfer API endpoints.

This module defines the REST API endpoints for applying style transfer
to interior images. It follows Behavior-Driven Development principles
from the Semantic Seed Coding Standards (SSCS).
"""

import base64
import io
import traceback
from typing import List, Optional
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, status, Header
from pydantic import ValidationError

from src.api.models.style_transfer import (
    StyleTransferRequest,
    StyleTransferResponse,
    BatchStyleRequest,
    BatchStyleResponse,
    StyleType
)
from src.api.core.config import settings
from src.api.core.auth import verify_api_key
from src.style_transfer import StyleTransferService
from src.flux_integration import FluxClient
from src.segmentation import SegmentationHandler

router = APIRouter()


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 string to a numpy array image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array
    """
    try:
        # Extract the base64 data if it's a data URI
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to numpy array
        return np.array(image)
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        raise


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode a numpy array image to a base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded image string with data URI prefix
    """
    try:
        # Ensure the image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert to PIL image
        pil_image = Image.fromarray(image)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        
        # Encode as base64 with data URI prefix
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image to base64: {str(e)}")
        raise


def resize_image_to_match(image_to_resize: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    """
    Resize an image to match the dimensions of a target image.
    
    Args:
        image_to_resize: Image to resize
        target_image: Image with the target dimensions
        
    Returns:
        Resized image matching target dimensions
    """
    target_height, target_width = target_image.shape[:2]
    
    # Convert to PIL image for resizing
    pil_image = Image.fromarray(image_to_resize)
    
    # Resize maintaining aspect ratio
    resized_pil = pil_image.resize((target_width, target_height), Image.LANCZOS)
    
    # Convert back to numpy array
    return np.array(resized_pil)


@router.post(
    "/apply",
    response_model=StyleTransferResponse,
    status_code=status.HTTP_200_OK,
    summary="Apply style transfer to an interior image",
    description="""
    Apply a specified style to an interior image with optional structure preservation.
    
    This endpoint processes the input image and applies the requested style, optionally
    preserving architectural elements like walls, windows, and other structural features.
    
    The response includes the styled image and information about whether structure
    preservation was applied.
    """,
    responses={
        200: {
            "description": "Style transfer successfully applied",
            "content": {
                "application/json": {
                    "example": {
                        "styled_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
                        "structure_preserved": True,
                        "metrics": {
                            "ssim": 0.92,
                            "mse": 105.34
                        }
                    }
                }
            }
        },
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Validation error in request data"},
        500: {"description": "Internal server error during style transfer"}
    }
)
async def apply_style(
    request: StyleTransferRequest,
    api_key: str = Depends(verify_api_key)
) -> StyleTransferResponse:
    """
    Apply style transfer to an interior image.
    
    Given a base64-encoded image and style specifications, this endpoint
    applies the requested style transfer and returns the styled image.
    """
    try:
        # Initialize the style transfer service
        flux_client = FluxClient(api_key=settings.FAL_KEY)
        style_service = StyleTransferService(flux_client)
        
        # Decode the input image
        image = decode_base64_image(request.image)
        
        # Process structure mask if needed
        structure_mask = None
        if request.preserve_structure:
            if request.structure_mask:
                structure_mask = decode_base64_image(request.structure_mask)
            else:
                # Generate structure mask using SegmentationHandler
                segmentation_handler = SegmentationHandler()
                # Call generate_masks to get all masks including structure
                masks = segmentation_handler.generate_masks(image)
                # Extract the structure mask from the result
                structure_mask = masks.get("structure", None)
                
                if structure_mask is None:
                    raise ValueError("Failed to generate structure mask")
        
        # Apply style transfer based on request parameters
        if isinstance(request.style, StyleType):
            style_prompt = f"Interior in {request.style.value} style"
        else:
            style_prompt = f"Interior in {request.style} style"
        
        # Apply style transfer with structure preservation if needed
        if request.preserve_structure and structure_mask is not None:
            # First apply style transfer to get styled image
            styled_image = style_service.apply_style_only(
                image=image, 
                style_prompt=style_prompt
            )
            
            # Resize styled image if dimensions don't match
            if styled_image.shape[:2] != image.shape[:2]:
                print(f"Resizing styled image from {styled_image.shape} to match original {image.shape}")
                styled_image = resize_image_to_match(styled_image, image)
            
            # Then apply structure preservation using segmentation handler
            segmentation_handler = segmentation_handler or SegmentationHandler()
            styled_image = segmentation_handler.apply_structure_preservation(
                original_image=image,
                stylized_image=styled_image,
                structure_mask=structure_mask
            )
        else:
            # Simple style transfer without structure preservation
            styled_image = style_service.apply_style_only(
                image=image, 
                style_prompt=style_prompt
            )
            
            # Resize styled image if dimensions don't match
            if styled_image.shape[:2] != image.shape[:2]:
                print(f"Resizing styled image from {styled_image.shape} to match original {image.shape}")
                styled_image = resize_image_to_match(styled_image, image)
        
        # Encode the styled image to base64
        styled_image_base64 = encode_image_to_base64(styled_image)
        
        # Create and return the response
        return StyleTransferResponse(
            styled_image=styled_image_base64,
            structure_preserved=request.preserve_structure,
            metrics=None  # Metrics could be added here if requested
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in style transfer: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during style transfer: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchStyleResponse,
    status_code=status.HTTP_200_OK,
    summary="Apply batch style transfer to multiple images",
    description="""
    Process multiple images with their respective styles in a single request.
    
    This endpoint is optimized for applying style transfer to multiple images
    in a single API call, which is useful for generating style variations or
    processing a collection of images efficiently.
    
    The response includes all styled images and any errors that occurred
    during processing.
    """,
    responses={
        200: {"description": "Batch style transfer successful"},
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Validation error in request data"},
        500: {"description": "Internal server error during batch processing"}
    }
)
async def batch_style(
    request: BatchStyleRequest,
    api_key: str = Depends(verify_api_key)
) -> BatchStyleResponse:
    """
    Apply batch style transfer to multiple interior images.
    
    Process multiple images with their corresponding styles in a single request,
    optimizing for efficiency when working with collections of images.
    """
    try:
        # Initialize services
        flux_client = FluxClient(api_key=settings.FAL_KEY)
        style_service = StyleTransferService(flux_client)
        
        # Initialize segmentation handler if structure preservation is requested
        segmentation_handler = None
        if request.preserve_structure:
            segmentation_handler = SegmentationHandler()
        
        # Process each image
        styled_images = []
        errors = {}
        
        for idx, (image_b64, style) in enumerate(zip(request.images, request.styles)):
            try:
                # Decode input image
                image = decode_base64_image(image_b64)
                
                # Generate structure mask if needed
                structure_mask = None
                if request.preserve_structure and segmentation_handler:
                    # Generate masks and extract structure mask
                    masks = segmentation_handler.generate_masks(image)
                    structure_mask = masks.get("structure", None)
                
                # Format style prompt
                if isinstance(style, StyleType):
                    style_prompt = f"Interior in {style.value} style"
                else:
                    style_prompt = f"Interior in {style} style"
                
                # Apply style transfer
                if request.preserve_structure and structure_mask is not None:
                    # First apply style transfer to get styled image
                    styled_image = style_service.apply_style_only(
                        image=image, 
                        style_prompt=style_prompt
                    )
                    
                    # Resize styled image if dimensions don't match
                    if styled_image.shape[:2] != image.shape[:2]:
                        print(f"Resizing styled image from {styled_image.shape} to match original {image.shape}")
                        styled_image = resize_image_to_match(styled_image, image)
                    
                    # Then apply structure preservation using segmentation handler
                    styled_image = segmentation_handler.apply_structure_preservation(
                        original_image=image,
                        stylized_image=styled_image,
                        structure_mask=structure_mask
                    )
                else:
                    # Simple style transfer without structure preservation
                    styled_image = style_service.apply_style_only(
                        image=image, 
                        style_prompt=style_prompt
                    )
                    
                    # Resize styled image if dimensions don't match
                    if styled_image.shape[:2] != image.shape[:2]:
                        print(f"Resizing styled image from {styled_image.shape} to match original {image.shape}")
                        styled_image = resize_image_to_match(styled_image, image)
                
                # Encode result
                styled_image_b64 = encode_image_to_base64(styled_image)
                styled_images.append(styled_image_b64)
                
            except Exception as e:
                # Record the error and continue with other images
                error_details = traceback.format_exc()
                print(f"Error in batch style transfer (image {idx+1}): {str(e)}\n{error_details}")
                errors[idx] = str(e)
                styled_images.append(None)  # Add placeholder for failed image
        
        # Filter out None values and return response
        styled_images = [img for img in styled_images if img is not None]
        
        return BatchStyleResponse(
            styled_images=styled_images,
            errors=errors if errors else None
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in batch style transfer: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during batch style transfer: {str(e)}"
        )


@router.get(
    "/styles",
    status_code=status.HTTP_200_OK,
    summary="Get available style types",
    description="Get the list of predefined style types available for transfer."
)
async def get_available_styles(
    api_key: str = Depends(verify_api_key)
) -> dict:
    """Get the list of predefined styles available for transfer."""
    try:
        return {
            "styles": [style.value for style in StyleType],
            "examples": {
                "modern": "Clean lines, neutral colors, and minimalist aesthetic",
                "rustic": "Natural materials, earthy tones, and vintage elements",
                "art_deco": "Bold geometric patterns, luxurious finishes, and vibrant colors",
                "minimalist": "Essential elements only, uncluttered spaces, and functionality",
                "industrial": "Raw materials, exposed structures, and utilitarian design",
                "scandinavian": "Light colors, natural elements, and functional simplicity",
                "bohemian": "Eclectic patterns, vibrant colors, and layered textures",
                "coastal": "Light blues, whites, and nautical elements"
            }
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in getting available styles: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving available styles: {str(e)}"
        )
