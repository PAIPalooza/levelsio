"""
Empty Room Furnishing Service for Interior Style Transfer.

This module provides functionality to furnish empty rooms with AI-generated furniture in various styles.
It integrates with the Flux API through the FluxClient to generate furnished rooms based on prompt templates.

Following the Semantic Seed Coding Standards:
- Comprehensive docstrings
- Type annotations
- Error handling
- Appropriate layer of abstraction
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import cv2
from PIL import Image

from src.flux_integration import FluxClient
from src.segmentation import SegmentationHandler


class EmptyRoomFurnishingService:
    """
    Service for furnishing empty rooms with style-consistent furniture.
    
    This service generates appealing furniture arrangements for empty rooms
    based on provided style, room type, and furniture descriptions.
    """
    
    # Default furnishing prompt template
    DEFAULT_FURNISHING_TEMPLATE = "Furnish this empty room with {furniture} in a {style} style."
    
    # Room-specific furnishing prompt templates
    FURNISHING_TEMPLATES = {
        "living_room": "Furnish this empty living room with {furniture} in a {style} style. Preserve the room's architecture including walls, floors, and ceiling.",
        "bedroom": "Furnish this empty bedroom with {furniture} in a {style} style. Maintain the room's structure and architectural elements.",
        "dining_room": "Furnish this empty dining room with {furniture} in a {style} style. Keep the room's architectural integrity.",
        "kitchen": "Furnish this empty kitchen with {furniture} in a {style} style. Preserve the room's structure and fixed elements.",
        "bathroom": "Furnish this empty bathroom with {furniture} in a {style} style. Maintain all structural elements of the room.",
        "home_office": "Furnish this empty home office with {furniture} in a {style} style. Keep the room's architectural details intact.",
        "study": "Furnish this empty study with {furniture} in a {style} style. Preserve the room's architecture and structural elements."
    }
    
    def __init__(self, flux_client: Optional[FluxClient] = None, segmentation_handler: Optional[SegmentationHandler] = None):
        """
        Initialize the empty room furnishing service.
        
        Args:
            flux_client: Client for interacting with the Flux API
            segmentation_handler: Handler for segmenting images and creating masks
        """
        if flux_client is None:
            self.flux_client = FluxClient()
        else:
            self.flux_client = flux_client
            
        if segmentation_handler is None:
            self.segmentation_handler = SegmentationHandler()
        else:
            self.segmentation_handler = segmentation_handler
    
    def format_furnishing_prompt(self, furniture: str, room_type: str = "", style: str = "modern") -> str:
        """
        Format a prompt for furnishing an empty room.
        
        Args:
            furniture: Description of furniture to add
            room_type: Type of room (living room, bedroom, etc.)
            style: Design style to apply (e.g., modern, rustic, industrial)
            
        Returns:
            Formatted prompt string
        """
        # Normalize room type for template lookup
        normalized_room_type = room_type.lower().replace(" ", "_") if room_type else ""
        
        # Select the appropriate template based on room type
        if normalized_room_type in self.FURNISHING_TEMPLATES:
            template = self.FURNISHING_TEMPLATES[normalized_room_type]
        else:
            template = self.DEFAULT_FURNISHING_TEMPLATE
            
            # If room type was provided but doesn't match our templates, add it to the prompt
            if room_type:
                template = f"Furnish this empty {room_type} with {{furniture}} in a {{style}} style."
        
        # Format the template with the provided values
        return template.format(furniture=furniture, style=style)
    
    def furnish_room(self, 
                     empty_room_image: np.ndarray, 
                     furniture: str, 
                     style: str = "modern", 
                     room_type: str = "", 
                     preserve_structure: bool = False) -> np.ndarray:
        """
        Generate a furnished version of an empty room.
        
        Args:
            empty_room_image: Image of an empty room
            furniture: Description of furniture to add
            style: Design style to apply
            room_type: Type of room (living room, bedroom, etc.)
            preserve_structure: Whether to strictly preserve architectural elements
            
        Returns:
            Image with the room furnished according to specifications
        """
        # Format the furnishing prompt
        prompt = self.format_furnishing_prompt(furniture, room_type, style)
        
        # If structure preservation is enabled, create segmentation masks
        # and add explicit preservation instructions
        if preserve_structure:
            # Generate structure mask (walls, floor, ceiling)
            walls_mask = self.segmentation_handler.create_mask(
                empty_room_image, 
                label="wall",
                use_sam=True
            )
            
            floors_mask = self.segmentation_handler.create_mask(
                empty_room_image, 
                label="floor",
                use_sam=True
            )
            
            # Combine masks for structural elements
            structure_mask = np.logical_or(walls_mask, floors_mask).astype(np.uint8)
            
            # Add structure preservation to the prompt
            if "preserve" not in prompt.lower() and "maintain" not in prompt.lower():
                prompt += " Preserve the architectural structure including walls and floors."
        
        # Apply the furnishing transformation
        furnished_image = self.flux_client.apply_style_transfer(
            empty_room_image,
            prompt
        )
        
        return furnished_image
    
    def generate_furnishing_variations(self, 
                                      empty_room_image: np.ndarray, 
                                      furniture: str, 
                                      style: str = "modern", 
                                      room_type: str = "",
                                      num_variations: int = 3,
                                      preserve_structure: bool = False) -> List[np.ndarray]:
        """
        Generate multiple furnished variations of an empty room.
        
        Args:
            empty_room_image: Image of an empty room
            furniture: Description of furniture to add
            style: Design style to apply
            room_type: Type of room (living room, bedroom, etc.)
            num_variations: Number of variations to generate
            preserve_structure: Whether to strictly preserve architectural elements
            
        Returns:
            List of furnished room variations
        """
        variations = []
        
        # Base prompt
        base_prompt = self.format_furnishing_prompt(furniture, room_type, style)
        
        # Generate multiple variations with slight prompt modifications
        for i in range(num_variations):
            # Create variation-specific prompts
            if i == 0:
                variation_prompt = base_prompt
            elif i == 1:
                variation_prompt = f"{base_prompt} Alternative arrangement #{i+1}."
            else:
                variation_prompt = f"{base_prompt} Different furniture arrangement, variation #{i+1}."
                
            # Add structure preservation if requested
            if preserve_structure and "preserve" not in variation_prompt.lower():
                variation_prompt += " Preserve the architectural structure including walls and floors."
            
            # Generate the furnished image
            furnished_variation = self.flux_client.apply_style_transfer(
                empty_room_image,
                variation_prompt
            )
            
            variations.append(furnished_variation)
            
        return variations
