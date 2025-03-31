"""
Layout transfer module for Interior Style Transfer POC.

This module provides services for applying style and layout transfer to interior images
while preserving architectural structure. It enables changing furniture arrangement
while maintaining walls, floors, and other structural elements.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LayoutTransferService:
    """
    Service for style and layout transfer operations on interior images.
    
    Enables changing furniture arrangement and style while preserving
    architectural structure, coordinating between segmentation and Flux API.
    """
    
    # Layout prompt templates for common furniture arrangements
    LAYOUT_TEMPLATES = {
        "living_room": "A {style} living room with furniture layout: {furniture_arrangement}. Preserve the room's architecture including walls, floors, and ceiling.",
        "bedroom": "A {style} bedroom with furniture layout: {furniture_arrangement}. Maintain the room's structure and architectural elements.",
        "dining_room": "A {style} dining room with furniture layout: {furniture_arrangement}. Keep the room's architectural integrity.",
        "kitchen": "A {style} kitchen with furniture layout: {furniture_arrangement}. Preserve the room's structure and layout of fixed elements.",
        "bathroom": "A {style} bathroom with furniture layout: {furniture_arrangement}. Maintain all structural elements of the room.",
        "home_office": "A {style} home office with furniture layout: {furniture_arrangement}. Keep the room's architectural details intact."
    }
    
    def __init__(self, flux_client=None, style_transfer_service=None, segmentation_handler=None):
        """
        Initialize the layout transfer service.
        
        Args:
            flux_client: FluxClient instance for API calls
            style_transfer_service: Optional StyleTransferService instance
            segmentation_handler: Optional SegmentationHandler instance
        """
        self.flux_client = flux_client
        self.style_transfer_service = style_transfer_service
        self.segmentation_handler = segmentation_handler
        
        # Import style transfer service if not provided but needed
        if not self.style_transfer_service and flux_client:
            try:
                from src.style_transfer import StyleTransferService
                self.style_transfer_service = StyleTransferService(
                    flux_client=flux_client,
                    segmentation_handler=segmentation_handler
                )
            except ImportError:
                logger.warning("StyleTransferService not available, some functionality may be limited")
                
        logger.info("LayoutTransferService initialized")
    
    def format_layout_prompt(
        self, 
        furniture_arrangement: str, 
        room_type: str = "living_room", 
        style: Optional[str] = None
    ) -> str:
        """
        Format a layout prompt using templates.
        
        Args:
            furniture_arrangement: Description of furniture placement
            room_type: Type of room (e.g., 'living_room', 'bedroom')
            style: Optional style to apply (e.g., 'Scandinavian', 'Industrial')
            
        Returns:
            Formatted layout prompt
        """
        # Normalize room type to match template keys
        normalized_room_type = room_type.lower().replace(" ", "_")
        
        # Default to living room if template not found
        if normalized_room_type not in self.LAYOUT_TEMPLATES:
            normalized_room_type = "living_room"
            
        # Default style if none provided
        style_text = style if style else "modern"
        
        # Format the prompt using template
        prompt = self.LAYOUT_TEMPLATES[normalized_room_type].format(
            style=style_text,
            furniture_arrangement=furniture_arrangement
        )
        
        logger.info(f"Formatted layout prompt: '{prompt}'")
        return prompt
    
    def apply_layout_transfer(
        self, 
        image: np.ndarray, 
        layout_prompt: str,
        structure_mask: Optional[np.ndarray] = None,
        preserve_structure: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        negative_prompt: str = "blurry, distorted architecture, unrealistic proportions, warped walls",
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Apply layout transfer while preserving architectural structure.
        
        Args:
            image: Input image as NumPy array
            layout_prompt: Layout description with furniture arrangement
            structure_mask: Binary mask of structural elements to preserve
            preserve_structure: Whether to preserve architectural elements
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Image with new layout and preserved structure
        """
        if not self.flux_client:
            raise ValueError("FluxClient is required for layout transfer")
            
        logger.info(f"Applying layout transfer with prompt: '{layout_prompt}'")
        
        # Enhance prompt with structure preservation emphasis if requested
        enhanced_prompt = layout_prompt
        if preserve_structure:
            if not enhanced_prompt.endswith('.'):
                enhanced_prompt += '.'
            enhanced_prompt += " Maintain the room's architectural structure exactly as in the original image. Preserve all walls, ceilings, floors, windows, doors, and built-in features in their exact original positions."
        
        # If structure preservation is requested but no mask is provided, generate one
        if preserve_structure and structure_mask is None and self.segmentation_handler is not None:
            logger.info("Generating structure mask for layout transfer")
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
            logger.info("Structure mask generated for layout transfer")
        
        # Apply layout transfer using Flux API
        layout_image = self.flux_client.apply_style_transfer(
            image=image,
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            max_retries=max_retries
        )
        
        # If structure mask is provided and structure preservation is enabled, apply it
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
                stylized_image=layout_image,
                structure_mask=structure_mask
            )
            
            return preserved_image
        else:
            logger.info("No structure preservation applied, returning layout image as-is")
            return layout_image
    
    def generate_layout_variations(
        self, 
        image: np.ndarray, 
        layout_prompt: str,
        structure_mask: Optional[np.ndarray] = None,
        preserve_structure: bool = True,
        num_variations: int = 3,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        negative_prompt: str = "blurry, distorted architecture, unrealistic proportions, warped walls",
        max_retries: int = 3
    ) -> List[np.ndarray]:
        """
        Generate multiple layout variations while preserving architecture.
        
        Args:
            image: Input image as NumPy array
            layout_prompt: Layout description with furniture arrangement
            structure_mask: Binary mask of structural elements to preserve
            preserve_structure: Whether to preserve architectural elements
            num_variations: Number of variations to generate
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of images with different layouts and preserved structure
        """
        if not self.flux_client:
            raise ValueError("FluxClient is required for layout transfer")
            
        logger.info(f"Generating {num_variations} layout variations with prompt: '{layout_prompt}'")
        
        # Enhance prompt with structure preservation emphasis if requested
        enhanced_prompt = layout_prompt
        if preserve_structure:
            if not enhanced_prompt.endswith('.'):
                enhanced_prompt += '.'
            enhanced_prompt += " Maintain the room's architectural structure exactly as in the original image. Preserve all walls, ceilings, floors, windows, doors, and built-in features in their exact original positions."
        
        # If structure preservation is requested but no mask is provided, generate one
        if preserve_structure and structure_mask is None and self.segmentation_handler is not None:
            logger.info("Generating structure mask for layout variations")
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
            logger.info("Structure mask generated for layout variations")
        
        # Generate layout variations using Flux API
        layout_variations = self.flux_client.generate_style_variations(
            image=image,
            prompt=enhanced_prompt,
            num_variations=num_variations,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            max_retries=max_retries
        )
        
        # If structure mask is provided and preservation is enabled, apply to all variations
        if structure_mask is not None and preserve_structure and layout_variations:
            logger.info("Preserving structure in all layout variations")
            
            # Use provided segmentation handler or import on-demand
            handler = self.segmentation_handler
            if handler is None:
                # Import segmentation handler if not provided
                from src.segmentation import SegmentationHandler
                handler = SegmentationHandler()
            
            # Apply structure preservation to all variations
            preserved_variations = handler.preserve_structure_in_styles(
                original_image=image,
                style_variations=layout_variations,
                structure_mask=structure_mask
            )
            
            return preserved_variations
        else:
            logger.info("No structure preservation applied, returning layout variations as-is")
            return layout_variations
    
    def apply_style_and_layout(
        self, 
        image: np.ndarray, 
        style_prompt: str,
        layout_prompt: str,
        structure_mask: Optional[np.ndarray] = None,
        preserve_structure: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        negative_prompt: str = "blurry, distorted architecture, unrealistic proportions",
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Apply both style and layout changes while preserving architecture.
        
        Args:
            image: Input image as NumPy array
            style_prompt: Style description
            layout_prompt: Layout description with furniture arrangement
            structure_mask: Binary mask of structural elements to preserve
            preserve_structure: Whether to preserve architectural elements
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Image with new style, layout and preserved structure
        """
        if not self.flux_client:
            raise ValueError("FluxClient is required for style and layout transfer")
            
        logger.info(f"Applying style and layout transfer with prompts: Style='{style_prompt}', Layout='{layout_prompt}'")
        
        # Combine style and layout prompts
        combined_prompt = f"A {style_prompt} style room with {layout_prompt}."
        if preserve_structure:
            if not combined_prompt.endswith('.'):
                combined_prompt += '.'
            combined_prompt += " Maintain the room's architectural structure exactly as in the original image. Preserve all walls, ceilings, floors, windows, doors, and built-in features in their exact original positions."
        
        # If structure preservation is requested but no mask is provided, generate one
        if preserve_structure and structure_mask is None and self.segmentation_handler is not None:
            logger.info("Generating structure mask for style and layout transfer")
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
            logger.info("Structure mask generated for style and layout transfer")
        
        # Apply combined style and layout transfer using Flux API
        transformed_image = self.flux_client.apply_style_transfer(
            image=image,
            prompt=combined_prompt,
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
                stylized_image=transformed_image,
                structure_mask=structure_mask
            )
            
            return preserved_image
        else:
            logger.info("No structure preservation applied, returning transformed image as-is")
            return transformed_image
