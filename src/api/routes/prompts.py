"""
Prompt templates API endpoints.

This module defines the REST API endpoints for retrieving and managing
prompt templates for interior design styles. It follows Behavior-Driven
Development principles from the Semantic Seed Coding Standards (SSCS).
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import FileResponse
import logging

from src.prompts import StylePromptManager
from src.prompts.style_manager import StyleManager, StyleCategory
from src.api.core.auth import verify_api_key
from src.api.core.errors import StyleAPIException, StyleNotFoundException
from src.api.models.prompts import (
    StyleDetails,
    StylePromptRequest, 
    StylePromptResponse,
    CustomPromptRequest,
    CategoryInfo,
    StylePreview,
    StyleListResponse,
    CustomPromptResponse
)

router = APIRouter()

# Initialize style managers
style_prompt_manager = StylePromptManager()
style_manager = StyleManager()


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
    response_model=Dict[str, List[str]],
    status_code=status.HTTP_200_OK,
    tags=["categories"],
    summary="Get all style categories",
    description="Get a list of all available interior design style categories.",
    operation_id="get_style_categories_api_v1"
)
async def get_style_categories(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, List[str]]:
    """
    Get all available style categories.
    
    Returns:
        List of category names
    """
    try:
        # Get categories from style manager for consistency with tests
        categories = style_manager.get_style_categories()
        
        return {
            "categories": categories
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/category/{category_name}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get styles by category",
    description="Get all styles belonging to a specific interior design category.",
    operation_id="get_styles_by_category_api_v1"
)
async def get_styles_by_category(
    category_name: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get styles belonging to a specific category.
    
    Args:
        category_name: Name of the category
        
    Returns:
        Category information with styles
    """
    try:
        # Get styles from style manager for consistency with tests
        styles = style_manager.get_styles_by_category(category_name)
        
        # Convert to simpler format for tests
        style_list = [{"name": style["name"], "description": style["description"]} 
                     for style in styles]
        
        return {
            "category": category_name,
            "styles": style_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
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


@router.get(
    "/generate",
    response_model=StylePromptResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a style prompt (GET method)",
    description="""
    Generate a formatted prompt for a specific interior design style.
    
    This endpoint creates a prompt suitable for use with style transfer,
    combining the style name, room type, and optional details.
    """
)
async def generate_style_prompt_get(
    style: str,
    room_type: str = "interior",
    details: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
) -> StylePromptResponse:
    """
    Generate a formatted prompt for a specific style (GET method).
    
    Args:
        style: Name of the style
        room_type: Type of room
        details: Additional style details
        
    Returns:
        Generated prompt
    """
    manager = StylePromptManager()
    
    # Get all available styles for case-insensitive matching
    available_styles = manager.get_all_style_names()
    
    # Try to find a case-insensitive match
    matching_style = None
    for available_style in available_styles:
        if style.lower() == available_style.lower():
            matching_style = available_style
            break
    
    # If no match found, raise 404
    if matching_style is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Style '{style}' not found. Available styles: {', '.join(available_styles)}"
        )
    
    prompt = manager.get_prompt_for_style(
        matching_style,  # Use the matching style with correct case
        room_type,
        details
    )
    
    return StylePromptResponse(
        prompt=prompt,
        style=style  # Return the style with the original casing from the request
    )


@router.post(
    "/generate-custom-prompt",
    response_model=StylePromptResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate custom prompt",
    description="Generate a custom prompt for interior design style transfer with specific elements."
)
async def generate_custom_prompt_legacy(
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
    try:
        # Default style if not provided
        if not request.base_style:
            request.base_style = "minimalist"
            
        # Check if base style exists (case-insensitive)
        if not style_manager.style_exists(request.base_style):
            available_styles = style_manager.get_all_style_names()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "message": f"Style '{request.base_style}' not found",
                    "details": {
                        "available_styles": available_styles
                    }
                }
            )
        
        # Generate prompt elements
        prompt_parts = []
        
        # Add base style
        prompt_parts.append(f"A {request.base_style} style interior")
        
        # Add room type if provided
        if request.room_type:
            prompt_parts.append(f"{request.room_type}")
        
        # Add colors if specified
        if request.colors:
            colors_text = ", ".join(request.colors)
            prompt_parts.append(f"with {colors_text} colors")
            
        # Add materials if specified
        if request.materials:
            materials_text = ", ".join(request.materials)
            prompt_parts.append(f"featuring {materials_text}")
            
        # Add mood if specified
        if request.mood:
            prompt_parts.append(f"creating a {request.mood} atmosphere")
            
        # Combine all parts into a complete prompt
        prompt = " ".join(prompt_parts)
        
        return StylePromptResponse(
            prompt=prompt,
            style=request.base_style
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error generating custom prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Error generating custom prompt: {str(e)}",
                "details": {"reason": "prompt_generation_error"}
            }
        )


@router.post(
    "/generate",
    response_model=CustomPromptResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate custom style prompt",
    description="Generate a custom prompt for interior style transfer based on user specifications."
)
async def generate_custom_prompt(
    request: CustomPromptRequest,
    api_key: str = Depends(verify_api_key)
) -> CustomPromptResponse:
    """
    Generate a custom prompt for interior style transfer.
    
    Args:
        request: Custom prompt request with style specifications
        api_key: API key for authentication
        
    Returns:
        CustomPromptResponse with generated prompt
        
    Raises:
        HTTPException: If the request is invalid or prompt generation fails
    """
    import logging
    
    try:
        # Extract base style and modifier elements
        base_style = request.base_style or "modern"
        elements = request.elements or []
        colors = request.colors or []
        materials = request.materials or []
        mood = request.mood or "elegant"
        
        # Generate a custom prompt based on user specifications
        prompt_elements = []
        
        # Add base style
        prompt_elements.append(f"A {base_style} style interior design")
        
        # Add colors if specified
        if colors:
            color_text = ", ".join(colors)
            prompt_elements.append(f"with {color_text} color scheme")
        
        # Add materials if specified
        if materials:
            material_text = ", ".join(materials)
            prompt_elements.append(f"featuring {material_text}")
        
        # Add design elements if specified
        if elements:
            element_text = ", ".join(elements)
            prompt_elements.append(f"including {element_text}")
        
        # Add mood
        prompt_elements.append(f"creating a {mood} atmosphere")
        
        # Combine all elements into a coherent prompt
        prompt = " ".join(prompt_elements)
        
        # Return the custom prompt
        return CustomPromptResponse(
            prompt=prompt,
            base_style=base_style,
            elements=elements,
            colors=colors, 
            materials=materials,
            mood=mood
        )
        
    except Exception as e:
        logging.error(f"Error generating custom prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Failed to generate custom prompt: {str(e)}",
                "details": {"reason": "prompt_generation_error"}
            }
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


@router.get(
    "/list",
    response_model=StyleListResponse,
    status_code=status.HTTP_200_OK,
    summary="List available style templates",
    description="Get a list of all available interior design style templates."
)
async def list_styles(
    api_key: str = Depends(verify_api_key)
) -> StyleListResponse:
    """
    List all available interior design style templates.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        StyleListResponse with styles and categories
    """
    try:
        # Get all style names
        style_names = style_manager.get_all_style_names()
        
        # Create style previews
        style_previews = []
        for style_name in style_names:
            preview = StylePreview(
                name=style_name,
                category="modern",  # Default category for now
                description=f"{style_name.title()} interior design style",  # Add description to fix validation
                preview=None  # Preview images not implemented yet
            )
            style_previews.append(preview)
        
        # Get categories
        category_names = style_manager.get_style_categories()
        
        # Create category info objects with styles
        categories = []
        for category_name in category_names:
            # Get styles for this category (simplified for testing)
            category_styles = style_manager.get_styles_by_category(category_name)
            category_style_names = [style["name"] for style in category_styles] if category_styles else []
            
            # Create category info
            category_info = CategoryInfo(
                name=category_name,
                description=f"Description for {category_name}",
                styles=category_style_names
            )
            categories.append(category_info)
        
        return StyleListResponse(
            styles=style_previews,
            categories=[c.name for c in categories],  # Just return category names for backward compatibility
            total_styles=len(style_previews)  # Add total_styles count
        )
    except Exception as e:
        logging.error(f"Error listing styles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing styles: {str(e)}"
        )
