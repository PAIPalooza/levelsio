"""
Segmentation API endpoints.

This module defines the REST API endpoints for room segmentation and
structure extraction. It follows Behavior-Driven Development principles
from the Semantic Seed Coding Standards (SSCS).
"""

import base64
import io
import traceback
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, status, Header, File, UploadFile, Form
from pydantic import ValidationError

from src.api.models.segmentation import (
    SegmentationRequest,
    SegmentationResponse,
    BatchSegmentationRequest,
    BatchSegmentationResponse,
    SegmentationType,
    SegmentationJsonRequest
)
from src.api.core.config import settings
from src.api.core.auth import verify_api_key
from src.segmentation import SegmentationHandler

# Create router
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
    status_code=status.HTTP_200_OK,
    summary="Segment an interior image (File Upload)",
    description="""
    Analyze an interior image and identify different regions or elements.
    
    This endpoint accepts an image file and returns segmentation data
    identifying different elements in the interior space.
    """
)
async def segment_interior(
    image: UploadFile = File(..., description="Interior image to segment"),
    detail_level: str = Form("medium", description="Level of segmentation detail"),
    api_key: str = Depends(verify_api_key)
):
    """
    Segment an interior image into different regions or elements.
    
    Args:
        image: Interior image to segment
        detail_level: Level of segmentation detail (low, medium, high)
        api_key: API key for authentication
        
    Returns:
        Segmentation data for the image
    """
    try:
        # Read image content
        image_content = await image.read()
        
        # Initialize segmentation handler
        segmentation_handler = SegmentationHandler()
        
        # Convert image content to numpy array
        image = Image.open(io.BytesIO(image_content))
        image_array = np.array(image)
        
        # Generate segmentation masks
        masks = segmentation_handler.generate_masks(image_array)
        
        # Convert masks to base64 for response
        mask_results = {}
        for mask_name, mask in masks.items():
            mask_results[mask_name] = encode_mask_to_base64(mask)
        
        # Return segmentation results
        return {
            "masks": mask_results,
            "detail_level": detail_level,
            "image_dimensions": {
                "width": image_array.shape[1],
                "height": image_array.shape[0]
            }
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in segmentation: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.post(
    "/segment-api",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment an interior image",
    description="""
    Generate segmentation masks for interior images.
    
    This endpoint processes the input image and creates segmentation masks based on
    the requested segmentation type. Options include structure segmentation (walls, 
    windows, etc.), furniture segmentation, or full room segmentation.
    
    The response includes the segmentation mask and optional visualization overlays.
    """
)
async def segment_interior_api(
    request: SegmentationRequest,
    api_key: str = Depends(verify_api_key)
) -> SegmentationResponse:
    """
    Segment an interior image to identify different components.
    
    Given a base64-encoded image, this endpoint performs segmentation
    to identify room structure and contents.
    
    Args:
        request: Segmentation request with image and parameters
        api_key: API key for authentication
        
    Returns:
        Segmentation results with masks and visualizations
    """
    try:
        # Initialize the segmentation handler
        segmentation_handler = SegmentationHandler()
        
        # Decode the input image
        image = decode_base64_image(request.image)
        
        # Generate segmentation based on the requested type
        if request.segmentation_type == SegmentationType.STRUCTURE:
            # Generate structure masks
            structure_mask = segmentation_handler.generate_structure_mask(image)
            
            # Create visualization if requested
            visualization = None
            if request.create_visualization:
                visualization = segmentation_handler.create_visualization(
                    image, 
                    structure_mask,
                    alpha=0.5
                )
                visualization_base64 = encode_image_to_base64(visualization)
            else:
                visualization_base64 = None
            
            # Encode the structure mask
            mask_base64 = encode_mask_to_base64(structure_mask)
            
            # Calculate mask statistics
            mask_stats = {
                "coverage": float(np.mean(structure_mask > 0)),
                "segments": 1
            }
            
            return SegmentationResponse(
                mask=mask_base64,
                visualization=visualization_base64,
                stats=mask_stats
            )
            
        elif request.segmentation_type == SegmentationType.FURNITURE:
            # Generate furniture masks
            masks_dict = segmentation_handler.generate_masks(image)
            
            # Extract furniture related masks
            furniture_mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
            for item in ["sofa", "table", "chair", "bed", "cabinet", "furniture"]:
                if item in masks_dict:
                    furniture_mask = np.logical_or(furniture_mask, masks_dict[item]).astype(np.uint8)
            
            # Create visualization if requested
            visualization = None
            if request.create_visualization:
                visualization = segmentation_handler.create_visualization(
                    image, 
                    furniture_mask,
                    alpha=0.5
                )
                visualization_base64 = encode_image_to_base64(visualization)
            else:
                visualization_base64 = None
            
            # Encode the furniture mask
            mask_base64 = encode_mask_to_base64(furniture_mask)
            
            # Calculate mask statistics
            mask_stats = {
                "coverage": float(np.mean(furniture_mask > 0)),
                "segments": 1
            }
            
            return SegmentationResponse(
                mask=mask_base64,
                visualization=visualization_base64,
                stats=mask_stats
            )
            
        else:  # Full segmentation
            # Generate all masks
            masks_dict = segmentation_handler.generate_masks(image)
            
            # Create full visualization if requested
            visualization = None
            if request.create_visualization:
                # Create a colored visualization for all masks
                visualization = segmentation_handler.create_multi_class_visualization(
                    image, 
                    masks_dict
                )
                visualization_base64 = encode_image_to_base64(visualization)
            else:
                visualization_base64 = None
            
            # Convert masks to a single labeled mask for response
            combined_mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
            mask_index = 1
            
            for mask_name, mask in masks_dict.items():
                # Add mask with unique index
                combined_mask[mask > 0] = mask_index
                mask_index += 1
            
            # Encode the combined mask
            mask_base64 = encode_mask_to_base64(combined_mask)
            
            # Calculate mask statistics
            mask_stats = {
                "coverage": float(np.mean(combined_mask > 0)),
                "segments": len(masks_dict),
                "segment_types": list(masks_dict.keys())
            }
            
            return SegmentationResponse(
                mask=mask_base64,
                visualization=visualization_base64,
                stats=mask_stats
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
            detail=f"An error occurred during segmentation: {str(e)}"
        )


@router.post(
    "/segment-json",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment room using JSON input",
    description="Segment an interior room image using JSON input with base64-encoded image."
)
async def segment_interior_json(
    request: SegmentationJsonRequest,
    api_key: str = Depends(verify_api_key)
) -> SegmentationResponse:
    """
    Segment an interior room image using JSON input with base64-encoded image.
    
    Args:
        request: Segmentation request with JSON data
        api_key: API key for authentication
        
    Returns:
        SegmentationResponse with segmentation results
        
    Raises:
        HTTPException: If the request is invalid or segmentation fails
    """
    import logging
    import os
    
    # Validate base64 image data
    try:
        if not request.image.startswith('data:image/'):
            # Try to add prefix if not present
            request.image = f"data:image/jpeg;base64,{request.image}"
    except Exception as e:
        logging.error(f"Error processing base64 image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Invalid image data",
                "details": {"reason": "image_validation_error"}
            }
        )
    
    # Process detail level
    detail_level = request.detail_level
    
    try:
        # Create a FluxAPI instance
        from src.flux_integration import FluxAPI
        flux_api = FluxAPI()
        
        # Make the async API call properly
        result = await flux_api.segment_interior(
            image_base64=request.image,
            detail_level=detail_level
        )
        
        # Return the response
        return SegmentationResponse(
            regions=result.get("regions", []),
            detail_level=detail_level,
            image_width=result.get("image_width", 1024),
            image_height=result.get("image_height", 768),
            processing_time=result.get("processing_time", 1.0)
        )
        
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Segmentation failed: {str(e)}",
                "details": {"api_name": "Flux API"}
            }
        )


@router.post(
    "/batch",
    response_model=BatchSegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch segment multiple interior images",
    description="""
    Process multiple images in a single request to generate segmentation masks.
    
    This endpoint is useful for efficient processing of multiple interior images,
    generating structure, furniture, and decor masks for each image.
    """
)
async def batch_segment(
    request: BatchSegmentationRequest,
    api_key: str = Depends(verify_api_key)
) -> BatchSegmentationResponse:
    """
    Generate segmentation masks for multiple interior images.
    
    Process multiple images in a single request, generating segmentation
    masks for each one based on the specified parameters.
    
    Args:
        request: Batch segmentation request with multiple images
        api_key: API key for authentication
        
    Returns:
        Batch response with masks for all images
    """
    try:
        # Initialize the segmentation handler
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
