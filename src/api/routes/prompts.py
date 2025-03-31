"""
Prompt templates API endpoints.

This module defines the REST API endpoints for retrieving and managing
prompt templates for interior design styles. It follows Behavior-Driven
Development principles from the Semantic Seed Coding Standards (SSCS).
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import FileResponse

from src.prompts import StylePromptManager, StyleCategory
from src.api.core.auth import verify_api_key
from src.api.models.prompts import (
    StyleDetails,
    StylePromptRequest, 
    StylePromptResponse,
    CustomPromptRequest,
    CategoryInfo,
    StylePreview,
    StyleListResponse
)

router = APIRouter()


@router.get(
    "/styles",
    response_model=StyleListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get all available style names",
    description="""
    Retrieve a list of all available interior design style names with preview information.
    
    This endpoint returns the names and preview information for all styles available 
    in the system, which can be used for style selection in style transfer operations.
    """
)
async def get_available_styles(
    request: Request,
    category: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
) -> StyleListResponse:
    """
    Get all available style names with preview information.
    
    Args:
        request: FastAPI request object for URL generation
        category: Optional category filter
        
    Returns:
        List of styles with preview information
    """
    manager = StylePromptManager()
    styles_list = []
    
    # Get filtered styles if category is provided
    if category:
        try:
            category_enum = StyleCategory(category)
            style_names = manager.get_styles_by_category(category_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category '{category}' not found"
            )
    else:
        style_names = manager.get_all_style_names()
    
    # Generate base URL for preview images
    base_url = str(request.base_url).rstrip('/')
    
    # Create preview objects for each style
    for style_name in style_names:
        details = manager.get_style_details(style_name)
        if details:
            preview_path = manager.get_style_preview_image_path(style_name)
            preview_url = None
            
            if preview_path:
                # Get filename from path
                import os
                filename = os.path.basename(preview_path)
                preview_url = f"{base_url}/api/v1/prompts/previews/{filename}"
            
            styles_list.append(
                StylePreview(
                    name=style_name,
                    category=details.get("category", "uncategorized"),
                    description=details.get("description", "")[:100] + "..." if len(details.get("description", "")) > 100 else details.get("description", ""),
                    preview_image_url=preview_url
                )
            )
    
    # Get all categories
    categories = [c.value for c in StyleCategory]
    
    return StyleListResponse(
        styles=styles_list,
        categories=categories,
        total_styles=len(styles_list)
    )


@router.get(
    "/styles/{style_name}",
    response_model=StyleDetails,
    status_code=status.HTTP_200_OK,
    summary="Get details about a specific style",
    description="""
    Retrieve detailed information about a specific interior design style.
    
    This endpoint returns comprehensive details about the requested style,
    including description, category, characteristics, colors, and materials.
    """
)
async def get_style_details(
    style_name: str,
    api_key: str = Depends(verify_api_key)
) -> StyleDetails:
    """
    Get detailed information about a specific style.
    
    Args:
        style_name: Name of the style
        
    Returns:
        Detailed style information
    """
    manager = StylePromptManager()
    details = manager.get_style_details(style_name)
    
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Style '{style_name}' not found"
        )
    
    return StyleDetails(
        name=style_name,
        description=details.get("description", "No description available"),
        category=details.get("category", "uncategorized"),
        characteristics=details.get("characteristics", []),
        colors=details.get("colors", []),
        materials=details.get("materials", []),
        example_prompt=details.get("example_prompt", "")
    )


@router.get(
    "/categories",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="Get all style categories",
    description="""
    Retrieve a list of all available style categories.
    
    This endpoint returns the names of all style categories in the system,
    which can be used for filtering styles by category.
    """
)
async def get_style_categories(api_key: str = Depends(verify_api_key)) -> List[str]:
    """
    Get all available style categories.
    
    Returns:
        List of category names
    """
    return [category.value for category in StyleCategory]


@router.get(
    "/categories/{category_name}",
    response_model=CategoryInfo,
    status_code=status.HTTP_200_OK,
    summary="Get styles in a category",
    description="""
    Retrieve styles belonging to a specific category.
    
    This endpoint returns information about a style category and the styles
    that belong to it.
    """
)
async def get_styles_by_category(
    category_name: str,
    api_key: str = Depends(verify_api_key)
) -> CategoryInfo:
    """
    Get styles belonging to a specific category.
    
    Args:
        category_name: Name of the category
        
    Returns:
        Category information with styles
    """
    try:
        category = StyleCategory(category_name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{category_name}' not found"
        )
    
    manager = StylePromptManager()
    styles = manager.get_styles_by_category(category)
    
    return CategoryInfo(
        name=category.value,
        description=StyleCategory.get_description(category),
        styles=styles
    )


@router.post(
    "/generate-prompt",
    response_model=StylePromptResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a style prompt",
    description="""
    Generate a formatted prompt for a specific interior design style.
    
    This endpoint creates a prompt suitable for use with style transfer,
    combining the style name, room type, and optional details.
    """
)
async def generate_style_prompt(
    request: StylePromptRequest,
    api_key: str = Depends(verify_api_key)
) -> StylePromptResponse:
    """
    Generate a formatted prompt for a specific style.
    
    Args:
        request: Style prompt request
        
    Returns:
        Generated prompt
    """
    manager = StylePromptManager()
    
    # Check if style exists
    if request.style not in manager.get_all_style_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Style '{request.style}' not found"
        )
    
    prompt = manager.get_prompt_for_style(
        request.style,
        request.room_type,
        request.details
    )
    
    return StylePromptResponse(
        prompt=prompt,
        style=request.style
    )


@router.post(
    "/generate-custom-prompt",
    response_model=StylePromptResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a custom style prompt",
    description="""
    Generate a customized prompt with specific style elements.
    
    This endpoint creates a detailed prompt with customized elements such as
    colors, materials, lighting, and mood based on a specific style.
    """
)
async def generate_custom_prompt(
    request: CustomPromptRequest,
    api_key: str = Depends(verify_api_key)
) -> StylePromptResponse:
    """
    Generate a custom prompt with specific style elements.
    
    Args:
        request: Custom prompt request
        
    Returns:
        Generated custom prompt
    """
    manager = StylePromptManager()
    
    # Check if base style exists
    if request.base_style not in manager.get_all_style_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Base style '{request.base_style}' not found"
        )
    
    prompt = manager.create_custom_prompt(
        base_style=request.base_style,
        room_type=request.room_type,
        colors=request.colors,
        materials=request.materials,
        lighting=request.lighting,
        mood=request.mood
    )
    
    return StylePromptResponse(
        prompt=prompt,
        style=request.base_style
    )


@router.get(
    "/previews/{filename}",
    response_class=FileResponse,
    summary="Get style preview image",
    description="""
    Retrieve a preview image for a specific interior design style.
    
    This endpoint returns an image file representing the selected style.
    """
)
async def get_style_preview_image(
    filename: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get a preview image for a style.
    
    Args:
        filename: Name of the preview image file
        
    Returns:
        Image file
    """
    import os
    from src.prompts.style_manager import StylePromptManager
    
    manager = StylePromptManager()
    
    # Get style name from filename (remove extension and _preview suffix)
    style_name = filename.split('.')[0].replace('_preview', '')
    
    # Get path to preview image
    preview_path = manager.get_style_preview_image_path(style_name)
    
    if not preview_path or not os.path.exists(preview_path):
        # Return default preview if specific one doesn't exist
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static",
            "style_previews",
            "default_preview.jpg"
        )
        
        if os.path.exists(default_path):
            return FileResponse(default_path)
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preview image for style '{style_name}' not found"
        )
    
    return FileResponse(preview_path)
