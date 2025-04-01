"""
Style transfer API endpoints.

This module defines the REST API endpoints for applying style transfer
to interior images. It follows Behavior-Driven Development principles
from the Semantic Seed Coding Standards (SSCS).
"""

import base64
import io
import traceback
import os
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, status, Header, File, UploadFile, Form
from pydantic import ValidationError

from src.api.models.style_transfer import (
    StyleTransferRequest,
    StyleTransferResponse,
    BatchStyleRequest,
    BatchStyleResponse,
    StyleType,
    StyleTransferJsonRequest
)
from src.api.core.config import settings
from src.api.core.auth import verify_api_key
from src.api.core.errors import (
    StyleAPIException,
    StyleNotFoundException,
    InvalidImageException,
    ExternalAPIException,
)
from src.style_transfer import StyleTransferService
from src.flux_integration import FluxClient
from src.segmentation import SegmentationHandler

# Create router
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
        raise InvalidImageException(f"Failed to process image: {str(e)}")


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
    "/transfer",
    status_code=status.HTTP_200_OK,
    summary="Apply style transfer to an interior image (File Upload)",
    description="""
    Transform an interior image according to a specific interior design style.
    
    This endpoint accepts an image file and a style name, and returns a
    transformed version of the image in the specified style.
    """
)
async def transfer_style(
    style: str = Form(..., description="Name of the style to apply"),
    room_type: str = Form("interior", description="Type of room in the image"),
    image: UploadFile = File(..., description="Interior image to transform"),
    details: Optional[str] = Form(None, description="Additional style details"),
    api_key: str = Depends(verify_api_key)
):
    """
    Apply style transfer to an interior image.
    
    Args:
        style: Name of the style to apply
        room_type: Type of room in the image
        image: Interior image to transform
        details: Additional style details
        api_key: API key for authentication
        
    Returns:
        Transformed image URL and details
    """
    try:
        # Read image file
        contents = await image.read()
        
        # Initialize style transfer service
        service = StyleTransferService()
        
        # Generate style prompt
        style_prompt = service.generate_style_prompt(
            style_name=style,
            room_type=room_type,
            details=details
        )
        
        # Apply style transfer
        result = service.apply_style(
            image_data=contents,
            style_prompt=style_prompt
        )
        
        return {
            "styled_image_url": result["image_url"],
            "style": style,
            "prompt_used": style_prompt,
            "processing_time_ms": result["processing_time_ms"]
        }
        
    except StyleAPIException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Style transfer failed: {str(e)}"
        )


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
    """
)
async def apply_style(
    request: StyleTransferRequest,
    api_key: str = Depends(verify_api_key)
) -> StyleTransferResponse:
    """
    Apply style transfer to an interior image.
    
    Given a base64-encoded image and style specifications, this endpoint
    applies the requested style transformation and returns the result.
    
    Args:
        request: Request containing image and style specifications
        api_key: API key for authentication
        
    Returns:
        Transformed image and processing details
    """
    try:
        # Initialize required services
        style_service = StyleTransferService()
        segmentation_handler = SegmentationHandler()
        
        # Decode the base64 image
        image = decode_base64_image(request.image)
        
        # Generate style prompt from the specified style or use custom prompt
        if request.custom_prompt:
            style_prompt = request.custom_prompt
        else:
            style_prompt = style_service.generate_style_prompt(
                style_name=request.style.value,
                room_type=request.room_type,
                details=request.details
            )
        
        # Check if we need to preserve structure
        if request.preserve_structure:
            # Generate structure mask
            structure_mask = segmentation_handler.generate_structure_mask(image)
            
            # Apply style transfer with structure preservation
            styled_image = style_service.apply_style_with_structure_preservation(
                image=image,
                structure_mask=structure_mask,
                style_prompt=style_prompt
            )
            
            # Resize styled image if dimensions don't match
            if styled_image.shape[:2] != image.shape[:2]:
                print(f"Resizing styled image from {styled_image.shape} to match original {image.shape}")
                styled_image = resize_image_to_match(styled_image, image)
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
        
        # Encode the result as base64
        styled_image_b64 = encode_image_to_base64(styled_image)
        
        # Return the response
        return StyleTransferResponse(
            styled_image=styled_image_b64,
            style=request.style.value if not request.custom_prompt else "custom",
            prompt_used=style_prompt,
            structure_preserved=request.preserve_structure
        )
        
    except StyleAPIException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
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
    summary="Apply style transfer to multiple interior images",
    description="""
    Process multiple images with the same style and settings in one request.
    
    This batch endpoint is useful for applying consistent styling across
    a set of interior images, with optional structure preservation.
    """
)
async def batch_style(
    request: BatchStyleRequest,
    api_key: str = Depends(verify_api_key)
) -> BatchStyleResponse:
    """
    Apply style transfer to multiple interior images in a batch.
    
    Args:
        request: Batch request containing multiple images and style specifications
        api_key: API key for authentication
        
    Returns:
        Batch response with styled images and any errors that occurred
    """
    try:
        # Initialize required services
        style_service = StyleTransferService()
        segmentation_handler = SegmentationHandler()
        
        # Generate style prompt
        if request.custom_prompt:
            style_prompt = request.custom_prompt
        else:
            style_prompt = style_service.generate_style_prompt(
                style_name=request.style.value,
                room_type=request.room_type,
                details=request.details
            )
        
        # Process each image
        styled_images = []
        errors = {}
        
        for idx, base64_image in enumerate(request.images):
            try:
                # Decode the base64 image
                image = decode_base64_image(base64_image)
                
                # Apply structure preservation if requested
                if request.preserve_structure:
                    # Generate structure mask
                    structure_mask = segmentation_handler.generate_structure_mask(image)
                    
                    # Apply style transfer with structure preservation
                    styled_image = style_service.apply_style_with_structure_preservation(
                        image=image,
                        structure_mask=structure_mask,
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
        
    except StyleAPIException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
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


@router.post(
    "/transfer-json",
    response_model=StyleTransferResponse,
    status_code=status.HTTP_200_OK,
    summary="Transfer style using JSON input",
    description="Apply interior design style to an image using JSON input with base64-encoded image."
)
async def transfer_style_json(
    request: StyleTransferJsonRequest,
    api_key: str = Depends(verify_api_key)
) -> StyleTransferResponse:
    """
    Transfer interior design style using JSON input with base64-encoded image.
    
    Args:
        request: Style transfer request with JSON data
        api_key: API key for authentication
        
    Returns:
        StyleTransferResponse with styled image
        
    Raises:
        HTTPException: If the request is invalid or transfer fails
    """
    import logging
    import base64
    
    # Get available styles
    available_styles = ["minimalist", "modern", "industrial", "scandinavian", 
                      "bohemian", "traditional", "farmhouse", "coastal"]
    
    # Validate base64 image data
    try:
        # Strip prefix if present
        image_data = request.image
        if image_data.startswith('data:image/'):
            # Extract base64 content after the comma
            image_data = image_data.split(',', 1)[1]
        
        # Try to decode to validate
        try:
            decoded_data = base64.b64decode(image_data)
            if len(decoded_data) < 10:  # Too small to be a valid image
                raise ValueError("Decoded data too small to be a valid image")
        except Exception as e:
            logging.error(f"Invalid base64 data: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Invalid image data",
                    "details": {"reason": "image_validation_error"}
                }
            )
            
        # Re-add prefix for processing
        if not request.image.startswith('data:image/'):
            request.image = f"data:image/jpeg;base64,{image_data}"
            
    except Exception as e:
        logging.error(f"Error processing base64 image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Invalid image data",
                "details": {"reason": "image_validation_error"}
            }
        )
    
    # Validate style
    if request.style not in available_styles:
        logging.error(f"Style '{request.style}' not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"Style '{request.style}' not found",
                "details": {"available_styles": available_styles}
            }
        )
    
    # Generate prompt if needed
    prompt = request.prompt
    if not prompt:
        prompt = f"A {request.style} style interior design with elegant furniture and decorations"
    
    # Process structure mask if provided
    structure_mask = request.structure_mask
    if structure_mask and not structure_mask.startswith('data:image/'):
        try:
            structure_mask = f"data:image/png;base64,{structure_mask}"
        except Exception as e:
            logging.error(f"Error processing structure mask: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Invalid structure mask data",
                    "details": {"reason": "mask_validation_error"}
                }
            )
    
    try:
        # Create a FluxAPI instance
        from src.flux_integration import FluxAPI
        flux_api = FluxAPI()
        
        # Make the async API call properly
        result = await flux_api.apply_style(
            image_base64=request.image,
            prompt=prompt,
            structure_mask=structure_mask,
            strength=request.strength,
            seed=request.seed
        )
        
        # Return the response
        return StyleTransferResponse(
            styled_image=result.get("styled_image", request.image),
            style=request.style,
            prompt=prompt,
            processing_time=result.get("processing_time", 1.0)
        )
        
    except Exception as e:
        logging.error(f"Error during style transfer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Style transfer failed: {str(e)}",
                "details": {"api_name": "Flux API"}
            }
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
    except StyleAPIException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in getting available styles: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving available styles: {str(e)}"
        )
