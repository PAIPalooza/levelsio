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
import logging

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
def decodeBase64Image(base64String: str) -> np.ndarray:
    """Decode a base64 string to a numpy array image."""
    try:
        # Remove data URI scheme if present
        if ',' in base64String:
            base64String = base64String.split(',', 1)[1]
        
        # Decode base64 to bytes
        imageBytes = base64.b64decode(base64String)
        
        # Convert bytes to numpy array
        image = Image.open(io.BytesIO(imageBytes))
        return np.array(image)
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        raise

def encodeMaskToBase64(mask: np.ndarray) -> str:
    """
    Encode a binary mask as a base64 string with data URI.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Base64 encoded mask string with data URI prefix
    """
    try:
        # Ensure mask is binary and convert to uint8
        binaryMask = (mask > 0).astype(np.uint8) * 255
        
        # Convert to PIL image
        pilMask = Image.fromarray(binaryMask)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pilMask.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode as base64 with data URI prefix
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding mask to base64: {str(e)}")
        raise

def encodeImageToBase64(image: np.ndarray) -> str:
    """Encode a numpy array image to a base64 string."""
    try:
        # Convert numpy array to PIL Image
        pilImage = Image.fromarray(image)
        
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        pilImage.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode bytes to base64
        base64Encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64Encoded}"
    except Exception as e:
        print(f"Error encoding image to base64: {str(e)}")
        raise


@router.post(
    "/segment",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment an interior image (File Upload)",
    description="""
    Analyze an interior image and identify different regions or elements.
    
    This endpoint accepts an image file and returns segmentation data
    identifying different elements in the interior space.
    """
)
async def segmentImage(
    image: UploadFile = File(..., description="Interior image to segment"),
    detailLevel: str = Form("medium", description="Level of segmentation detail"),
    apiKey: str = Depends(verify_api_key)
):
    """
    Segment an interior image into different regions or elements.
    
    Args:
        image: Interior image to segment
        detailLevel: Level of segmentation detail (low, medium, high)
        apiKey: API key for authentication
        
    Returns:
        Segmentation data for the image
    """
    try:
        # Read image content
        imageContent = await image.read()
        
        # Initialize segmentation handler
        segmentationHandler = SegmentationHandler()
        
        # Convert image content to numpy array
        image = Image.open(io.BytesIO(imageContent))
        imageArray = np.array(image)
        
        # Generate segmentation masks
        masks = segmentationHandler.generateMasks(imageArray)
        
        # Convert masks to base64 for response
        maskResults = {}
        for maskName, mask in masks.items():
            maskResults[maskName] = encodeMaskToBase64(mask)
        
        # Return segmentation results
        return {
            "masks": maskResults,
            "detailLevel": detailLevel,
            "imageDimensions": {
                "width": imageArray.shape[1],
                "height": imageArray.shape[0]
            }
        }
    except Exception as e:
        errorDetails = traceback.format_exc()
        print(f"Error in segmentation: {str(e)}\n{errorDetails}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Segmentation failed",
                "details": {"reason": "segmentation_error", "error": str(e)}
            }
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
async def segmentImageApi(
    request: SegmentationRequest,
    apiKey: str = Depends(verify_api_key)
) -> SegmentationResponse:
    """
    Segment an interior image to identify different components.
    
    Given a base64-encoded image, this endpoint performs segmentation
    to identify room structure and contents.
    
    Args:
        request: Segmentation request with image and parameters
        apiKey: API key for authentication
        
    Returns:
        Segmentation results with masks and visualizations
    """
    try:
        # Initialize the segmentation handler
        segmentationHandler = SegmentationHandler()
        
        # Decode the input image
        image = decodeBase64Image(request.image)
        
        # Generate segmentation based on the requested type
        if request.segmentationType == SegmentationType.STRUCTURE:
            # Generate structure masks
            structureMask = segmentationHandler.generateStructureMask(image)
            
            # Create visualization if requested
            visualization = None
            if request.createVisualization:
                visualization = segmentationHandler.createVisualization(
                    image, 
                    structureMask,
                    alpha=0.5
                )
                visualizationBase64 = encodeImageToBase64(visualization)
            else:
                visualizationBase64 = None
            
            # Encode the structure mask
            maskBase64 = encodeMaskToBase64(structureMask)
            
            # Calculate mask statistics
            maskStats = {
                "coverage": float(np.mean(structureMask > 0)),
                "segments": 1
            }
            
            return SegmentationResponse(
                mask=maskBase64,
                visualization=visualizationBase64,
                stats=maskStats
            )
            
        elif request.segmentationType == SegmentationType.FURNITURE:
            # Generate furniture masks
            masksDict = segmentationHandler.generateMasks(image)
            
            # Extract furniture related masks
            furnitureMask = np.zeros_like(image[:,:,0], dtype=np.uint8)
            for item in ["sofa", "table", "chair", "bed", "cabinet", "furniture"]:
                if item in masksDict:
                    furnitureMask = np.logical_or(furnitureMask, masksDict[item]).astype(np.uint8)
            
            # Create visualization if requested
            visualization = None
            if request.createVisualization:
                visualization = segmentationHandler.createVisualization(
                    image, 
                    furnitureMask,
                    alpha=0.5
                )
                visualizationBase64 = encodeImageToBase64(visualization)
            else:
                visualizationBase64 = None
            
            # Encode the furniture mask
            maskBase64 = encodeMaskToBase64(furnitureMask)
            
            # Calculate mask statistics
            maskStats = {
                "coverage": float(np.mean(furnitureMask > 0)),
                "segments": 1
            }
            
            return SegmentationResponse(
                mask=maskBase64,
                visualization=visualizationBase64,
                stats=maskStats
            )
            
        else:  # Full segmentation
            # Generate all masks
            masksDict = segmentationHandler.generateMasks(image)
            
            # Create full visualization if requested
            visualization = None
            if request.createVisualization:
                # Create a colored visualization for all masks
                visualization = segmentationHandler.createMultiClassVisualization(
                    image, 
                    masksDict
                )
                visualizationBase64 = encodeImageToBase64(visualization)
            else:
                visualizationBase64 = None
            
            # Convert masks to a single labeled mask for response
            combinedMask = np.zeros_like(image[:,:,0], dtype=np.uint8)
            maskIndex = 1
            
            for maskName, mask in masksDict.items():
                # Add mask with unique index
                combinedMask[mask > 0] = maskIndex
                maskIndex += 1
            
            # Encode the combined mask
            maskBase64 = encodeMaskToBase64(combinedMask)
            
            # Calculate mask statistics
            maskStats = {
                "coverage": float(np.mean(combinedMask > 0)),
                "segments": len(masksDict),
                "segmentTypes": list(masksDict.keys())
            }
            
            return SegmentationResponse(
                mask=maskBase64,
                visualization=visualizationBase64,
                stats=maskStats
            )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Validation error",
                "details": {"reason": "validation_error", "error": str(e)}
            }
        )
    except Exception as e:
        errorDetails = traceback.format_exc()
        logging.error(f"Error in segmentation: {str(e)}\n{errorDetails}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Segmentation failed",
                "details": {"reason": "segmentation_error", "error": str(e)}
            }
        )


@router.post(
    "/segment-json",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment room using JSON input",
    description="Segment an interior room image using JSON input with base64-encoded image."
)
async def segmentInteriorJson(
    request: SegmentationJsonRequest,
    apiKey: str = Depends(verify_api_key)
) -> SegmentationResponse:
    """
    Segment an interior room image using JSON input with base64-encoded image.
    
    Args:
        request: Segmentation request with JSON data
        apiKey: API key for authentication
        
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
    detailLevel = request.detailLevel
    
    try:
        # Create a FluxAPI instance
        from src.flux_integration import FluxAPI
        fluxApi = FluxAPI()
        
        # Make the async API call properly
        result = await fluxApi.segmentInterior(
            imageBase64=request.image,
            detailLevel=detailLevel
        )
        
        # Return the response
        return SegmentationResponse(
            regions=result.get("regions", []),
            detailLevel=detailLevel,
            imageWidth=result.get("image_width", 1024),
            imageHeight=result.get("image_height", 768),
            processingTime=result.get("processing_time", 1.0)
        )
        
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Segmentation failed",
                "details": {"reason": "segmentation_error", "error": str(e)}
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
async def batchSegment(
    request: BatchSegmentationRequest,
    apiKey: str = Depends(verify_api_key)
) -> BatchSegmentationResponse:
    """
    Generate segmentation masks for multiple interior images.
    
    Process multiple images in a single request, generating segmentation
    masks for each one based on the specified parameters.
    
    Args:
        request: Batch segmentation request with multiple images
        apiKey: API key for authentication
        
    Returns:
        Batch response with masks for all images
    """
    try:
        # Initialize the segmentation handler
        segmentationHandler = SegmentationHandler()
        
        # Process each image
        masks = []
        errors = {}
        
        for idx, imageBase64 in enumerate(request.images):
            try:
                # Decode the input image
                image = decodeBase64Image(imageBase64)
                
                # Use generateMasks to get all masks
                masksDict = segmentationHandler.generateMasks(image)
                
                # Extract the individual masks
                structureMask = masksDict.get("structure", np.zeros_like(image[:,:,0], dtype=np.uint8))
                furnitureMask = np.zeros_like(image[:,:,0], dtype=np.uint8)
                decorMask = np.zeros_like(image[:,:,0], dtype=np.uint8)
                
                # Try to extract furniture mask from specific mask types
                if "furniture" in masksDict:
                    furnitureMask = masksDict["furniture"]
                else:
                    # Combine sofa, table, chair masks if available
                    for item in ["sofa", "table", "chair", "bed"]:
                        if item in masksDict:
                            furnitureMask = np.logical_or(furnitureMask, masksDict[item]).astype(np.uint8)
                
                # Try to extract decor mask from specific mask types
                for item in ["decor", "painting", "picture", "lamp", "plant"]:
                    if item in masksDict:
                        decorMask = np.logical_or(decorMask, masksDict[item]).astype(np.uint8)
                
                # Encode the masks to base64
                structureMaskBase64 = encodeMaskToBase64(structureMask)
                furnitureMaskBase64 = encodeMaskToBase64(furnitureMask)
                decorMaskBase64 = encodeMaskToBase64(decorMask)
                
                # Append the masks to the list
                masks.append({
                    "structureMask": structureMaskBase64,
                    "furnitureMask": furnitureMaskBase64,
                    "decorMask": decorMaskBase64
                })
                
            except Exception as e:
                # Record the error and continue with other images
                errorDetails = traceback.format_exc()
                print(f"Error in batch segmentation (image {idx}): {str(e)}\n{errorDetails}")
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
            detail={
                "message": "Validation error",
                "details": {"reason": "validation_error", "error": str(e)}
            }
        )
    except Exception as e:
        errorDetails = traceback.format_exc()
        logging.error(f"Error in batch segmentation: {str(e)}\n{errorDetails}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Batch segmentation error",
                "details": {"reason": "batch_segmentation_error", "error": str(e)}
            }
        )


@router.get(
    "/segmentation-types",
    status_code=status.HTTP_200_OK,
    summary="Get available segmentation types",
    description="Retrieve the list of available segmentation types with descriptions.",
)
async def getSegmentationTypes(
    apiKey: str = Depends(verify_api_key)
) -> dict:
    """Get the list of available segmentation types with descriptions."""
    return {
        "segmentationTypes": [segType.value for segType in SegmentationType],
        "descriptions": {
            "structure": "Identifies structural elements such as walls, windows, doors, ceilings, and floors",
            "furniture": "Identifies furniture elements such as sofas, tables, chairs, and decorative items",
            "full": "Provides a complete segmentation of the room with all elements identified separately"
        }
    }
