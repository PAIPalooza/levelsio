"""
Segmentation API endpoints.

This module defines the REST API endpoints for room segmentation and
structure extraction. It follows Behavior-Driven Development principles
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

from src.api.models.segmentation import (
    SegmentationRequest,
    SegmentationResponse,
    BatchSegmentationRequest,
    BatchSegmentationResponse,
    SegmentationType
)
from src.api.core.config import settings
from src.api.core.auth import verify_api_key
from src.segmentation import SegmentationHandler

router = APIRouter()

# Helper functions for image conversion
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 string to a numpy array image."""
    try:
        # Remove data URI scheme if present
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        raise

def encode_mask_to_base64(mask: np.ndarray) -> str:
    """
    Encode a binary mask as a base64 string with data URI.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Base64 encoded mask string with data URI prefix
    """
    try:
        # Ensure mask is binary and convert to uint8
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Convert to PIL image
        pil_mask = Image.fromarray(binary_mask)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_mask.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode as base64 with data URI prefix
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding mask to base64: {str(e)}")
        raise

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a numpy array image to a base64 string."""
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode bytes to base64
        base64_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_encoded}"
    except Exception as e:
        print(f"Error encoding image to base64: {str(e)}")
        raise

@router.post(
    "/segment",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment an interior image",
    description="""
    Generate segmentation masks for interior images.
    
    This endpoint processes the input image and creates segmentation masks based on
    the requested segmentation type. Options include structure segmentation (walls, 
    windows, etc.), furniture segmentation, or full room segmentation.
    
    The response includes the segmentation mask and optional visualization overlays.
    """,
    responses={
        200: {
            "description": "Segmentation successfully completed",
            "content": {
                "application/json": {
                    "example": {
                        "mask": "data:image/png;base64,iVBORw0KGgoA...",
                        "visualization": "data:image/png;base64,iVBORw0KGgoA...",
                        "segments": {
                            "wall": 0.42,
                            "furniture": 0.35,
                            "floor": 0.23
                        }
                    }
                }
            }
        },
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Validation error in request data"},
        500: {"description": "Internal server error during segmentation"}
    }
)
async def segment_image(
    request: SegmentationRequest,
    api_key: str = Depends(verify_api_key)
) -> SegmentationResponse:
    """
    Segment an interior image to identify different components.
    
    Given a base64-encoded image, this endpoint performs segmentation
    to identify room structure, furniture, or both, depending on the
    requested segmentation type.
    """
    try:
        # Initialize segmentation handler
        segmentation_handler = SegmentationHandler()
        
        # Decode the input image
        image = decode_base64_image(request.image)
        
        # Use generate_masks to get all masks
        masks = segmentation_handler.generate_masks(image)
        
        # Extract the individual masks
        structure_mask = masks.get("structure", np.zeros_like(image[:,:,0], dtype=np.uint8))
        furniture_mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        decor_mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        
        # Try to extract furniture mask from specific mask types
        if "furniture" in masks:
            furniture_mask = masks["furniture"]
        else:
            # Combine sofa, table, chair masks if available
            for item in ["sofa", "table", "chair", "bed"]:
                if item in masks:
                    furniture_mask = np.logical_or(furniture_mask, masks[item]).astype(np.uint8)
        
        # Try to extract decor mask from specific mask types
        for item in ["decor", "painting", "picture", "lamp", "plant"]:
            if item in masks:
                decor_mask = np.logical_or(decor_mask, masks[item]).astype(np.uint8)
        
        # Generate visualization if requested
        visualization = None
        if request.include_visualization:
            # Use visualize_masks instead of visualize_mask
            visualization_img = segmentation_handler.visualize_masks(
                image, 
                {
                    "structure": structure_mask,
                    "furniture": furniture_mask,
                    "decor": decor_mask,
                    "walls": masks.get("walls", np.zeros_like(structure_mask)),
                    "floor": masks.get("floor", np.zeros_like(structure_mask)),
                    "ceiling": masks.get("ceiling", np.zeros_like(structure_mask)),
                    "windows": masks.get("windows", np.zeros_like(structure_mask))
                }
            )
            visualization = encode_image_to_base64(visualization_img)
        
        # Encode the masks to base64
        structure_mask_base64 = encode_mask_to_base64(structure_mask)
        furniture_mask_base64 = encode_mask_to_base64(furniture_mask)
        decor_mask_base64 = encode_mask_to_base64(decor_mask)
        
        # Create and return the response
        return SegmentationResponse(
            structure_mask=structure_mask_base64,
            furniture_mask=furniture_mask_base64,
            decor_mask=decor_mask_base64,
            visualization=visualization,
            segments={
                "structure": np.mean(structure_mask) / 255,
                "furniture": np.mean(furniture_mask) / 255,
                "decor": np.mean(decor_mask) / 255
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in segmentation: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation error: {str(e)}"
        )

@router.post(
    "/batch-segment",
    response_model=BatchSegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment multiple interior images",
    description="""
    Generate segmentation masks for multiple interior images in a single request.
    
    This batch endpoint allows efficient processing of multiple images,
    generating segmentation masks for each according to the specified type.
    
    The response includes all segmentation masks and any errors encountered
    during processing.
    """,
    responses={
        200: {"description": "Batch segmentation completed"},
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Validation error in request data"},
        500: {"description": "Internal server error during batch processing"}
    }
)
async def batch_segment(
    request: BatchSegmentationRequest,
    api_key: str = Depends(verify_api_key)
) -> BatchSegmentationResponse:
    """
    Generate segmentation masks for multiple interior images.
    
    Process multiple images in a single request, generating segmentation
    masks according to the specified segmentation type.
    """
    try:
        # Initialize segmentation handler
        segmentation_handler = SegmentationHandler()
        
        # Process each image
        masks = []
        errors = {}
        
        for idx, image_base64 in enumerate(request.images):
            try:
                # Decode the input image
                image = decode_base64_image(image_base64)
                
                # Use generate_masks to get all masks
                masks_dict = segmentation_handler.generate_masks(image)
                
                # Extract the individual masks
                structure_mask = masks_dict.get("structure", np.zeros_like(image[:,:,0], dtype=np.uint8))
                furniture_mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
                decor_mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
                
                # Try to extract furniture mask from specific mask types
                if "furniture" in masks_dict:
                    furniture_mask = masks_dict["furniture"]
                else:
                    # Combine sofa, table, chair masks if available
                    for item in ["sofa", "table", "chair", "bed"]:
                        if item in masks_dict:
                            furniture_mask = np.logical_or(furniture_mask, masks_dict[item]).astype(np.uint8)
                
                # Try to extract decor mask from specific mask types
                for item in ["decor", "painting", "picture", "lamp", "plant"]:
                    if item in masks_dict:
                        decor_mask = np.logical_or(decor_mask, masks_dict[item]).astype(np.uint8)
                
                # Encode the masks to base64
                structure_mask_base64 = encode_mask_to_base64(structure_mask)
                furniture_mask_base64 = encode_mask_to_base64(furniture_mask)
                decor_mask_base64 = encode_mask_to_base64(decor_mask)
                
                # Append the masks to the list
                masks.append({
                    "structure_mask": structure_mask_base64,
                    "furniture_mask": furniture_mask_base64,
                    "decor_mask": decor_mask_base64
                })
                
            except Exception as e:
                # Record the error and continue with other images
                error_details = traceback.format_exc()
                print(f"Error in batch segmentation (image {idx}): {str(e)}\n{error_details}")
                errors[idx] = str(e)
                masks.append(None)  # Add placeholder for failed image
        
        # Return the batch response
        return BatchSegmentationResponse(
            masks=[mask for mask in masks if mask is not None],
            errors=errors if errors else None
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in batch segmentation: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch segmentation error: {str(e)}"
        )

@router.get(
    "/segmentation-types",
    status_code=status.HTTP_200_OK,
    summary="Get available segmentation types",
    description="Retrieve the list of available segmentation types with descriptions.",
    responses={
        200: {"description": "List of available segmentation types retrieved successfully"},
        401: {"description": "Invalid or missing API key"}
    }
)
async def get_segmentation_types(
    api_key: str = Depends(verify_api_key)
) -> dict:
    """Get the list of available segmentation types with descriptions."""
    return {
        "segmentation_types": [seg_type.value for seg_type in SegmentationType],
        "descriptions": {
            "structure": "Identifies structural elements such as walls, windows, doors, ceilings, and floors",
            "furniture": "Identifies furniture elements such as sofas, tables, chairs, and decorative items",
            "full": "Provides a complete segmentation of the room with all elements identified separately"
        }
    }
