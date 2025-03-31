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


class StyleTransferService:
    """
    Service for style transfer operations on interior images.
    
    Coordinates between segmentation and Flux API to apply styles
    while preserving architectural elements and structure.
    """
    
    # Style prompt templates for common interior styles
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
    
    def __init__(self, flux_client=None, segmentation_handler=None):
        """
        Initialize the style transfer service.
        
        Args:
            flux_client: FluxClient instance for API calls
            segmentation_handler: SegmentationHandler instance for structure preservation
        """
        self.flux_client = flux_client
        self.segmentation_handler = segmentation_handler
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
        # Check if we have a template for this style
        if style_name in self.STYLE_TEMPLATES:
            prompt = self.STYLE_TEMPLATES[style_name].format(room_type=room_type)
        else:
            # Generic format for unknown styles
            prompt = f"A {style_name} style {room_type}"
        
        # Add additional details if provided
        if details:
            prompt += f" {details}"
        
        logger.info(f"Formatted style prompt: '{prompt}'")
        return prompt
    
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
        if not self.flux_client:
            raise ValueError("FluxClient is required for style transfer")
            
        logger.info(f"Generating {num_variations} style-only variations with prompt: '{style_prompt}'")
        
        # If preserve_structure is True, augment the prompt
        if preserve_structure:
            if not style_prompt.endswith('.'):
                style_prompt += '.'
            style_prompt += " Maintain all architectural elements and structural features of the room precisely. Preserve all walls, ceilings, floors, windows, doors, and built-in features exactly as they are positioned in the original image."
        
        # If structure preservation is requested but no mask is provided, generate one
        if preserve_structure and structure_mask is None and self.segmentation_handler is not None:
            logger.info("Generating structure mask for variations")
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
            logger.info("Structure mask generated for variations")
        
        # Generate style variations using Flux API
        styled_variations = self.flux_client.generate_style_variations(
            image=image,
            prompt=style_prompt,
            num_variations=num_variations,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            max_retries=max_retries
        )
        
        # If structure mask is provided and preservation is enabled, apply to all variations
        if structure_mask is not None and preserve_structure and styled_variations:
            logger.info("Preserving structure in all variations")
            
            # Use provided segmentation handler or import on-demand
            handler = self.segmentation_handler
            if handler is None:
                # Import segmentation handler if not provided
                from src.segmentation import SegmentationHandler
                handler = SegmentationHandler()
            
            # Apply structure preservation to all variations
            preserved_variations = handler.preserve_structure_in_styles(
                original_image=image,
                style_variations=styled_variations,
                structure_mask=structure_mask
            )
            
            return preserved_variations
        else:
            logger.info("No structure preservation applied, returning styled variations as-is")
            return styled_variations
