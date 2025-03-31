"""
Style transfer module for Interior Style Transfer POC.

This module provides services for applying style transfer to interior images
while preserving structural elements. It coordinates between segmentation
and the Flux API to generate styled interiors with intact architecture.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import prompt management
from src.prompts import StylePromptManager, StyleCategory


class StyleTransferService:
    """
    Service for style transfer operations on interior images.
    
    Coordinates between segmentation and Flux API to apply styles
    while preserving architectural elements and structure.
    """
    
    # Style prompt templates for common interior styles - kept for backward compatibility
    STYLE_TEMPLATES = {
        "Scandinavian": "A Scandinavian style {room_type} with clean lines, light colors, wood elements, and minimal decoration",
        "Industrial": "An industrial style {room_type} with exposed brick, metal fixtures, and rustic elements",
        "Minimalist": "A minimalist {room_type} with clean lines, neutral colors, and uncluttered space",
        "Mid-Century": "A mid-century modern {room_type} with clean lines, organic shapes, and bold colors",
        "Bohemian": "A bohemian style {room_type} with vibrant colors, mixed patterns, plants, and eclectic decor",
        "Coastal": "A coastal style {room_type} with light blues, whites, natural textures, and beach-inspired elements",
        "Traditional": "A traditional style {room_type} with classic furniture, rich colors, and elegant details",
        "Contemporary": "A contemporary {room_type} with sleek furniture, neutral colors, and minimal decoration",
        "Rustic": "A rustic {room_type} with warm colors, wooden elements, and natural textures",
        "Art Deco": "An Art Deco style {room_type} with bold geometric patterns, luxurious materials, and vibrant colors"
    }
    
    def __init__(self, flux_client=None, segmentation_handler=None, prompt_manager=None):
        """
        Initialize the style transfer service.
        
        Args:
            flux_client: FluxClient instance for API calls
            segmentation_handler: SegmentationHandler instance for structure preservation
            prompt_manager: StylePromptManager instance for style prompts
        """
        self.flux_client = flux_client
        self.segmentation_handler = segmentation_handler
        
        # Initialize prompt manager if not provided
        self.prompt_manager = prompt_manager
        if self.prompt_manager is None:
            self.prompt_manager = StylePromptManager()
            
        logger.info("StyleTransferService initialized")
    
    def format_style_prompt(self, style_name: str, room_type: str = "interior", details: Optional[str] = None) -> str:
        """
        Format a style prompt using templates.
        
        Args:
            style_name: Name of the style (e.g., 'Scandinavian', 'Industrial')
            room_type: Type of room (e.g., 'living room', 'bedroom')
            details: Additional style details to append
            
        Returns:
            Formatted style prompt
        """
        # Use the prompt manager to format the prompt
        return self.prompt_manager.get_prompt_for_style(style_name, room_type, details)
    
    def get_available_styles(self) -> List[str]:
        """
        Get list of available style names.
        
        Returns:
            List of available style names
        """
        return self.prompt_manager.get_all_style_names()
    
    def get_styles_by_category(self, category: StyleCategory) -> List[str]:
        """
        Get styles filtered by category.
        
        Args:
            category: Style category to filter by
            
        Returns:
            List of style names in the category
        """
        return self.prompt_manager.get_styles_by_category(category)
    
    def get_style_details(self, style_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a style.
        
        Args:
            style_name: Name of the style
            
        Returns:
            Dictionary with style details or None if not found
        """
        return self.prompt_manager.get_style_details(style_name)
    
    def get_style_preview_path(self, style_name: str) -> Optional[str]:
        """
        Get path to preview image for a style.
        
        Args:
            style_name: Name of the style
            
        Returns:
            Path to preview image or None if not found
        """
        return self.prompt_manager.get_style_preview_image_path(style_name)
    
    def create_custom_prompt(self, base_style: str, room_type: str, **kwargs) -> str:
        """
        Create a custom prompt with specific elements.
        
        Args:
            base_style: Base style to start from
            room_type: Type of room
            **kwargs: Additional elements (colors, materials, lighting, mood)
            
        Returns:
            Custom prompt with specified elements
        """
        return self.prompt_manager.create_custom_prompt(
            base_style=base_style,
            room_type=room_type,
            colors=kwargs.get('colors'),
            materials=kwargs.get('materials'),
            lighting=kwargs.get('lighting'),
            mood=kwargs.get('mood')
        )
    
    def apply_style_only(
        self, 
        image: np.ndarray, 
        style_prompt: str,
        structure_mask: Optional[np.ndarray] = None,
        preserve_structure: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        negative_prompt: str = "blurry, distorted architecture, unrealistic",
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Apply style-only transfer while preserving structure.
        
        Args:
            image: Input image as NumPy array
            style_prompt: Style description
            structure_mask: Binary mask of structural elements to preserve
            preserve_structure: Whether to preserve architectural structure
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Styled image with preserved structure
        """
        if not self.flux_client:
            raise ValueError("FluxClient is required for style transfer")
            
        logger.info(f"Applying style-only transfer with prompt: '{style_prompt}'")
        
        # If preserve_structure is True, augment the prompt
        if preserve_structure:
            if not style_prompt.endswith('.'):
                style_prompt += '.'
            style_prompt += " Maintain all architectural elements and structural features of the room precisely. Preserve all walls, ceilings, floors, windows, doors, and built-in features exactly as they are positioned in the original image."
        
        # If structure preservation is requested but no mask is provided, generate one
        if preserve_structure and structure_mask is None and self.segmentation_handler is not None:
            logger.info("Generating structure mask for preservation")
            # Generate wall mask
            wall_mask = self.segmentation_handler.create_mask(
                image, 
                label="wall",
                use_sam=True
            )
            
            # Generate floor mask
            floor_mask = self.segmentation_handler.create_mask(
                image, 
                label="floor",
                use_sam=True
            )
            
            # Combine masks for structural elements
            structure_mask = np.logical_or(wall_mask, floor_mask).astype(np.uint8)
            logger.info("Structure mask generated")
        
        # Apply style transfer using Flux API
        styled_image = self.flux_client.apply_style_transfer(
            image=image,
            prompt=style_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            max_retries=max_retries
        )
        
        # If structure mask is provided and preservation is enabled, apply it
        if structure_mask is not None and preserve_structure:
            logger.info("Preserving structure using mask")
            
            # Use provided segmentation handler or import on-demand
            handler = self.segmentation_handler
            if handler is None:
                # Import segmentation handler if not provided
                from src.segmentation import SegmentationHandler
                handler = SegmentationHandler()
            
            # Apply structure preservation
            preserved_image = handler.apply_structure_preservation(
                original_image=image,
                stylized_image=styled_image,
                structure_mask=structure_mask
            )
            
            return preserved_image
        else:
            logger.info("No structure preservation applied, returning styled image as-is")
            return styled_image
    
    def generate_style_only_variations(
        self, 
        image: np.ndarray, 
        style_prompt: str,
        structure_mask: Optional[np.ndarray] = None,
        preserve_structure: bool = True,
        num_variations: int = 3,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        negative_prompt: str = "blurry, distorted architecture, unrealistic",
        max_retries: int = 3
    ) -> List[np.ndarray]:
        """
        Generate multiple style-only variations while preserving structure.
        
        Args:
            image: Input image as NumPy array
            style_prompt: Style description
            structure_mask: Binary mask of structural elements to preserve
            preserve_structure: Whether to preserve architectural structure
            num_variations: Number of variations to generate
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of styled images with preserved structure
        """
        variations = []
        
        # Use incremental seeds if a base seed is provided
        seeds = None
        if seed is not None:
            seeds = [seed + i for i in range(num_variations)]
        
        # Generate variations with different seeds
        for i in range(num_variations):
            variation_seed = seeds[i] if seeds else None
            logger.info(f"Generating variation {i+1}/{num_variations}" + 
                       (f" with seed {variation_seed}" if variation_seed else ""))
            
            styled_image = self.apply_style_only(
                image=image,
                style_prompt=style_prompt,
                structure_mask=structure_mask,
                preserve_structure=preserve_structure,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=variation_seed,
                negative_prompt=negative_prompt,
                max_retries=max_retries
            )
            
            variations.append(styled_image)
        
        return variations
    
    def apply_style_to_furniture(
        self, 
        image: np.ndarray, 
        style_prompt: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        negative_prompt: str = "blurry, distorted furniture, unrealistic",
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Apply style transfer to furniture while preserving room structure.
        
        Args:
            image: Input image as NumPy array
            style_prompt: Style description
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Image with styled furniture
        """
        if not self.flux_client or not self.segmentation_handler:
            raise ValueError("FluxClient and SegmentationHandler are required for furniture styling")
            
        logger.info(f"Applying style to furniture with prompt: '{style_prompt}'")
        
        # Generate furniture mask
        furniture_mask = self.segmentation_handler.create_mask(
            image, 
            label="furniture",
            use_sam=True
        )
        
        # Generate structure mask (inverse of furniture mask)
        structure_mask = 1 - furniture_mask
        
        # Apply style transfer
        styled_image = self.flux_client.apply_style_transfer(
            image=image,
            prompt=style_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            max_retries=max_retries
        )
        
        # Apply structure preservation (preserve everything except furniture)
        preserved_image = self.segmentation_handler.apply_structure_preservation(
            original_image=image,
            stylized_image=styled_image,
            structure_mask=structure_mask
        )
        
        return preserved_image
    
    def apply_style_by_name(
        self,
        image: np.ndarray,
        style_name: str,
        room_type: str = "interior",
        preserve_structure: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        details: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply a style by name to an interior image.
        
        Args:
            image: Input image as NumPy array
            style_name: Name of the style to apply
            room_type: Type of room
            preserve_structure: Whether to preserve architectural structure
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            details: Additional details for the style prompt
            
        Returns:
            Styled image
        """
        # Generate style prompt using the prompt manager
        style_prompt = self.format_style_prompt(style_name, room_type, details)
        
        # Apply style
        return self.apply_style_only(
            image=image,
            style_prompt=style_prompt,
            preserve_structure=preserve_structure,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed
        )
