"""
Style prompt management for Interior Style Transfer POC.

This module provides a comprehensive management system for style prompts,
categories, and template organization following the Semantic Seed Coding
Standards (SSCS) for clean, maintainable code.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StyleCategory(str, Enum):
    """Categories for interior design styles."""
    
    MODERN = "modern"
    TRADITIONAL = "traditional"
    ECLECTIC = "eclectic"
    MINIMAL = "minimal"
    INDUSTRIAL = "industrial"
    NATURAL = "natural"
    RETRO = "retro"
    LUXURY = "luxury"
    COASTAL = "coastal"
    
    @classmethod
    def get_description(cls, category: 'StyleCategory') -> str:
        """
        Get a description for a style category.
        
        Args:
            category: Style category enum value
            
        Returns:
            Description of the category
        """
        descriptions = {
            cls.MODERN: "Clean lines, simple color palettes, and emphasis on functionality",
            cls.TRADITIONAL: "Classic, timeless design with rich colors and ornate details",
            cls.ECLECTIC: "Mix of styles, textures, and time periods with personal expression",
            cls.MINIMAL: "Simplicity, clean lines, and a monochromatic palette with minimal decoration",
            cls.INDUSTRIAL: "Raw, unfinished elements with metal, wood, and exposed features",
            cls.NATURAL: "Emphasis on natural materials, textures, and connection to environment",
            cls.RETRO: "Nostalgic designs from the 20th century with characteristic elements",
            cls.LUXURY: "Opulent design with rich materials, ornate details, and elegance",
            cls.COASTAL: "Beach and ocean-inspired with light colors, natural textures, and airy feel"
        }
        return descriptions.get(category, "No description available")


# Type for custom formatters
StyleFormatterFunction = TypeVar('StyleFormatterFunction', bound=Callable[[str, str, Optional[str]], str])


class StylePromptManager:
    """
    Manager for interior design style prompts and templates.
    
    This class manages style prompt templates, categories, and related information
    for the Interior Style Transfer application.
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the style prompt manager.
        
        Args:
            templates_path: Path to JSON file with style templates
        """
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Default templates path
        if templates_path is None:
            templates_path = os.path.join(self.base_path, "style_templates.json")
            
        self.templates_path = templates_path
        self._load_templates()
        
        # Preview images directory
        self.preview_images_dir = os.path.join(
            os.path.dirname(self.base_path),  # src directory
            "static",
            "style_previews"
        )
        
        # Ensure the preview images directory exists
        os.makedirs(self.preview_images_dir, exist_ok=True)
        
        # Dictionary to store custom formatters
        self.custom_formatters: Dict[str, StyleFormatterFunction] = {}
        
        logger.info(f"StylePromptManager initialized with {len(self.templates)} templates")
        
    def register_custom_formatter(self, style_name: str, formatter_func: StyleFormatterFunction) -> bool:
        """
        Register a custom formatter function for a specific style.
        
        This allows for custom handling of prompt formatting beyond the template-based approach.
        
        Args:
            style_name: Name of the style
            formatter_func: Function that takes (style, room_type, details) and returns a formatted prompt
            
        Returns:
            True if registered successfully, False otherwise
        """
        if not callable(formatter_func):
            logger.error(f"Failed to register formatter for '{style_name}': Not a callable function")
            return False
            
        self.custom_formatters[style_name] = formatter_func
        logger.info(f"Registered custom formatter for '{style_name}'")
        return True
        
    def get_prompt_for_style(self, style_name: str, room_type: str = "interior", details: Optional[str] = None) -> str:
        """
        Get a formatted prompt for a specific style.
        
        Args:
            style_name: Name of the style
            room_type: Type of room (e.g., living room, bedroom)
            details: Additional style details
            
        Returns:
            Formatted prompt
        """
        # Check for custom formatter first
        if style_name in self.custom_formatters:
            logger.info(f"Using custom formatter for '{style_name}'")
            return self.custom_formatters[style_name](style_name, room_type, details)
            
        # Check if style exists
        if style_name in self.templates:
            # Get the prompt template
            template = self.templates[style_name]["prompt"]
            
            # Format the template with room type
            prompt = template.format(room_type=room_type)
            
            # Add details if provided
            if details:
                prompt += f" {details}"
                
            logger.info(f"Generated prompt for '{style_name}': '{prompt}'")
            return prompt
            
        # Style not found, return a generic prompt
        logger.warning(f"Style '{style_name}' not found. Using a generic prompt.")
        prompt = f"A {style_name} style {room_type}"
        
        if details:
            prompt += f" {details}"
            
        return prompt
        
    def is_valid_style(self, style_name: str) -> bool:
        """
        Check if a style name is valid.
        
        Args:
            style_name: Name of the style to check
            
        Returns:
            True if style exists, False otherwise
        """
        return style_name in self.templates
        
    def get_styles_by_category(self, category: StyleCategory) -> List[str]:
        """
        Get styles belonging to a specific category.
        
        Args:
            category: Style category
            
        Returns:
            List of style names in the category
        """
        return [
            style_name for style_name, style_data in self.templates.items()
            if style_data.get("category") == category.value
        ]
        
    def _load_templates(self) -> None:
        """Load style templates from JSON file."""
        try:
            if os.path.exists(self.templates_path):
                with open(self.templates_path, 'r') as f:
                    self.templates = json.load(f)
            else:
                # Use default templates if file doesn't exist
                self.templates = self._get_default_templates()
                # Save default templates
                self._save_templates()
                
            logger.info(f"Loaded {len(self.templates)} style templates")
            
        except Exception as e:
            logger.error(f"Error loading style templates: {str(e)}")
            # Fallback to default templates
            self.templates = self._get_default_templates()
    
    def _save_templates(self) -> None:
        """Save style templates to JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.templates_path), exist_ok=True)
            
            with open(self.templates_path, 'w') as f:
                json.dump(self.templates, f, indent=2)
                
            logger.info(f"Saved {len(self.templates)} style templates to {self.templates_path}")
            
        except Exception as e:
            logger.error(f"Error saving style templates: {str(e)}")
    
    def _get_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default style templates.
        
        Returns:
            Dictionary of style templates
        """
        return {
            "Scandinavian": {
                "prompt": "A Scandinavian style {room_type} with clean lines, light colors, wood elements, and minimal decoration",
                "category": StyleCategory.MODERN,
                "description": "Scandinavian design emphasizes clean lines, simplicity, and functionality combined with light colors and natural materials.",
                "characteristics": ["Light color palette", "Wood elements", "Clean lines", "Minimal decoration", "Functional furniture"],
                "colors": ["White", "Light gray", "Light wood tones", "Black accents", "Muted blues"],
                "materials": ["Light wood", "Wool", "Cotton", "Natural textiles"],
                "example_prompt": "A cozy Scandinavian living room with light wooden floors, white walls, simple furniture, and minimal decoration with a few green plants for color accent"
            },
            "Industrial": {
                "prompt": "An industrial style {room_type} with exposed brick, metal fixtures, raw surfaces, and rustic elements",
                "category": StyleCategory.INDUSTRIAL,
                "description": "Industrial design draws inspiration from factories and industrial spaces, featuring raw, unfinished elements and materials.",
                "characteristics": ["Exposed brick", "Metal fixtures", "Raw surfaces", "Open spaces", "Utilitarian design"],
                "colors": ["Gray", "Rust", "Black", "Brown", "Neutrals"],
                "materials": ["Metal", "Wood", "Concrete", "Exposed brick", "Leather"],
                "example_prompt": "An industrial style loft apartment with exposed brick walls, concrete floors, metal light fixtures, and reclaimed wood furniture"
            },
            "Minimalist": {
                "prompt": "A minimalist {room_type} with clean lines, neutral colors, uncluttered space, and essential furniture",
                "category": StyleCategory.MINIMAL,
                "description": "Minimalist design focuses on simplicity, using only essential elements to create uncluttered, serene spaces.",
                "characteristics": ["Clean lines", "Neutral colors", "Uncluttered space", "Essential furniture only", "Functional elements"],
                "colors": ["White", "Black", "Gray", "Beige", "Muted tones"],
                "materials": ["Glass", "Steel", "Wood", "Concrete"],
                "example_prompt": "A minimalist bedroom with a low platform bed, white walls, polished concrete floor, and precisely arranged essential furniture with no decoration"
            },
            "Mid-Century Modern": {
                "prompt": "A mid-century modern {room_type} with clean lines, organic shapes, bold colors, and iconic furniture designs from the 1950s-60s",
                "category": StyleCategory.RETRO,
                "description": "Mid-Century Modern refers to the design period of the 1950s and 60s, known for clean lines, organic forms, and bold colors.",
                "characteristics": ["Clean lines", "Organic shapes", "Bold colors", "Iconic furniture", "Geometric patterns"],
                "colors": ["Teal", "Orange", "Brown", "Mustard yellow", "Olive green"],
                "materials": ["Teak wood", "Molded plastic", "Glass", "Brass"],
                "example_prompt": "A mid-century modern living room with an Eames lounge chair, teak credenza, geometric area rug, and floor-to-ceiling windows with mustard accent pillows"
            },
            "Bohemian": {
                "prompt": "A bohemian style {room_type} with vibrant colors, layered textiles, mixed patterns, plants, and eclectic decor",
                "category": StyleCategory.ECLECTIC,
                "description": "Bohemian or 'Boho' style celebrates freedom and non-conformity through layered patterns, textures, and global influences.",
                "characteristics": ["Vibrant colors", "Mixed patterns", "Plants", "Eclectic decor", "Layered textiles"],
                "colors": ["Jewel tones", "Earth tones", "Purples", "Oranges", "Pinks"],
                "materials": ["Natural fibers", "Rattan", "Macramé", "Houseplants", "Worldly artifacts"],
                "example_prompt": "A bohemian living room with a colorful Persian rug, floor cushions, macramé wall hangings, plenty of houseplants, and globally-inspired accessories"
            },
            "Coastal": {
                "prompt": "A coastal style {room_type} with light blues, whites, sandy beiges, natural textures, and beach-inspired elements",
                "category": StyleCategory.COASTAL,
                "description": "Coastal design draws inspiration from beaches and oceans, creating bright, airy spaces with nautical elements.",
                "characteristics": ["Light colors", "Natural textures", "Beach-inspired elements", "Airy spaces", "Nautical accents"],
                "colors": ["White", "Light blue", "Sandy beige", "Sea foam green", "Navy accents"],
                "materials": ["Light wood", "Jute", "Cotton", "Linen", "Seagrass"],
                "example_prompt": "A coastal style bedroom with white shiplap walls, blue and white bedding, natural fiber rug, bamboo furniture, and subtle beach-themed accessories"
            },
            "Traditional": {
                "prompt": "A traditional style {room_type} with classic furniture, rich colors, elegant details, symmetrical arrangements, and timeless appeal",
                "category": StyleCategory.TRADITIONAL,
                "description": "Traditional design features classic, timeless elements inspired by European decor of the 18th and 19th centuries.",
                "characteristics": ["Classic furniture", "Rich colors", "Elegant details", "Symmetrical arrangements", "Fine fabrics"],
                "colors": ["Deep red", "Navy blue", "Hunter green", "Warm neutrals", "Gold accents"],
                "materials": ["Cherry wood", "Mahogany", "Silk", "Velvet", "Crystal"],
                "example_prompt": "A traditional living room with a classic Chesterfield sofa, matching wingback chairs, mahogany coffee table, elegant drapes, and a Persian area rug"
            },
            "Contemporary": {
                "prompt": "A contemporary {room_type} with sleek furniture, neutral colors, smooth lines, minimal decoration, and up-to-date elements",
                "category": StyleCategory.MODERN,
                "description": "Contemporary design reflects current trends and aesthetics, with sleek, updated elements and a focus on simplicity.",
                "characteristics": ["Sleek furniture", "Neutral colors", "Smooth lines", "Minimal decoration", "Current trends"],
                "colors": ["Black", "White", "Gray", "Taupe", "Subtle accent colors"],
                "materials": ["Glass", "Metal", "Concrete", "Engineered wood"],
                "example_prompt": "A contemporary kitchen with flat-panel cabinets, quartz countertops, stainless steel appliances, minimal hardware, and sleek bar stools at an island"
            },
            "Rustic": {
                "prompt": "A rustic {room_type} with warm colors, rough wooden elements, natural materials, cozy atmosphere, and earthy textures",
                "category": StyleCategory.NATURAL,
                "description": "Rustic design celebrates natural beauty with rough textures, warm colors, and organic elements from the countryside.",
                "characteristics": ["Rough wooden elements", "Warm colors", "Natural materials", "Cozy atmosphere", "Earthy textures"],
                "colors": ["Brown", "Terracotta", "Forest green", "Beige", "Warm neutrals"],
                "materials": ["Reclaimed wood", "Stone", "Leather", "Iron", "Natural textiles"],
                "example_prompt": "A rustic mountain cabin living room with a stone fireplace, exposed wooden beams, leather furniture, wool throws, and a cowhide rug"
            },
            "Art Deco": {
                "prompt": "An Art Deco style {room_type} with bold geometric patterns, luxurious materials, vibrant colors, symmetrical designs, and decorative details",
                "category": StyleCategory.LUXURY,
                "description": "Art Deco emerged in the 1920s and 30s, known for its bold geometric patterns, vibrant colors, and glamorous aesthetic.",
                "characteristics": ["Bold geometric patterns", "Luxurious materials", "Vibrant colors", "Symmetrical designs", "Decorative detailing"],
                "colors": ["Black", "Gold", "Jade green", "Royal blue", "Deep red"],
                "materials": ["Chrome", "Brass", "Velvet", "Lacquered wood", "Marble"],
                "example_prompt": "An Art Deco living room with bold geometric wallpaper, a velvet sofa, lacquered furniture with brass inlays, a sunburst mirror, and a black and white marble floor"
            },
            "Farmhouse": {
                "prompt": "A farmhouse style {room_type} with rustic charm, white shiplap, vintage accessories, practical fixtures, and comfortable furniture",
                "category": StyleCategory.TRADITIONAL,
                "description": "Farmhouse style combines rustic charm with practical comfort, inspired by traditional American farmhouses.",
                "characteristics": ["White shiplap", "Rustic wood elements", "Vintage accessories", "Practical fixtures", "Comfortable furniture"],
                "colors": ["White", "Cream", "Black accents", "Natural wood tones", "Muted blues"],
                "materials": ["Distressed wood", "Galvanized metal", "Cotton", "Ceramic", "Cast iron"],
                "example_prompt": "A farmhouse kitchen with white shiplap walls, open shelving with vintage dishware, a farmhouse sink, wooden countertops, and a large harvest table"
            },
            "Japanese": {
                "prompt": "A Japanese style {room_type} with minimalist aesthetics, natural elements, clean lines, neutral colors, and zen-like tranquility",
                "category": StyleCategory.MINIMAL,
                "description": "Japanese design emphasizes minimalism, natural elements, and balance to create peaceful, harmonious spaces.",
                "characteristics": ["Minimalist aesthetics", "Natural elements", "Clean lines", "Neutral colors", "Zen-like tranquility"],
                "colors": ["Natural wood tones", "White", "Black", "Beige", "Soft greens"],
                "materials": ["Wood", "Paper", "Bamboo", "Stone", "Cotton", "Tatami"],
                "example_prompt": "A Japanese-inspired living room with low wooden furniture, tatami mats, shoji screens, a single ikebana flower arrangement, and minimalist décor"
            },
            "Mediterranean": {
                "prompt": "A Mediterranean style {room_type} with terracotta colors, ornate tiles, wrought iron details, arched doorways, and warm textures",
                "category": StyleCategory.TRADITIONAL,
                "description": "Mediterranean design draws inspiration from countries bordering the Mediterranean Sea, featuring warm colors and ornate details.",
                "characteristics": ["Terracotta colors", "Ornate tiles", "Wrought iron details", "Arched doorways", "Warm textures"],
                "colors": ["Terracotta", "Azure blue", "Ochre yellow", "Olive green", "Warm neutrals"],
                "materials": ["Terracotta", "Ceramic tile", "Wrought iron", "Stucco", "Natural stone"],
                "example_prompt": "A Mediterranean-style dining room with terracotta floors, hand-painted ceramic tiles, a wrought iron chandelier, stucco walls, and an olive wood dining table"
            },
            "Modern Luxury": {
                "prompt": "A modern luxury {room_type} with high-end finishes, sleek surfaces, statement lighting, sophisticated color palette, and curated art pieces",
                "category": StyleCategory.LUXURY,
                "description": "Modern Luxury combines contemporary clean lines with high-end materials and elegant details for a sophisticated aesthetic.",
                "characteristics": ["High-end finishes", "Sleek surfaces", "Statement lighting", "Sophisticated color palette", "Curated art"],
                "colors": ["Gray", "Cream", "Black", "Gold accents", "Rich jewel tones"],
                "materials": ["Marble", "Polished metals", "Crystal", "Exotic woods", "High-end fabrics"],
                "example_prompt": "A modern luxury penthouse living room with floor-to-ceiling windows, white marble floors, a designer sofa, statement chandelier, and curated modern art pieces"
            },
            "Shabby Chic": {
                "prompt": "A shabby chic {room_type} with vintage furniture, distressed finishes, floral patterns, pastel colors, and romantic details",
                "category": StyleCategory.ECLECTIC,
                "description": "Shabby Chic combines vintage elements and soft aesthetics for a romantic, lived-in elegance with feminine touches.",
                "characteristics": ["Vintage furniture", "Distressed finishes", "Floral patterns", "Pastel colors", "Romantic details"],
                "colors": ["White", "Cream", "Pastel pink", "Pastel blue", "Mint green"],
                "materials": ["Distressed wood", "Linen", "Cotton", "Lace", "Vintage items"],
                "example_prompt": "A shabby chic bedroom with a white wrought iron bed, floral bedding, distressed painted furniture, vintage accessories, and soft pastel colors"
            },
            "Hollywood Regency": {
                "prompt": "A Hollywood Regency style {room_type} with glamorous details, bold colors, lacquered surfaces, metallic finishes, and dramatic contrast",
                "category": StyleCategory.LUXURY,
                "description": "Hollywood Regency epitomizes glamour and opulence from the golden age of cinema, with bold contrasts and statement pieces.",
                "characteristics": ["Glamorous details", "Bold colors", "Lacquered surfaces", "Metallic finishes", "Dramatic contrast"],
                "colors": ["Black", "White", "Bold jewel tones", "Gold", "Silver"],
                "materials": ["Lacquered finishes", "Mirrored surfaces", "Velvet", "Metallics", "Lucite"],
                "example_prompt": "A Hollywood Regency living room with black and white checkered floors, a lacquered cabinet, velvet seating in emerald green, gold accents, and dramatic lighting"
            },
            "French Country": {
                "prompt": "A French Country style {room_type} with rustic elegance, toile patterns, weathered furniture, warm colors, and provincial charm",
                "category": StyleCategory.TRADITIONAL,
                "description": "French Country combines rustic charm with elegant sophistication, inspired by the homes of Provence and rural France.",
                "characteristics": ["Rustic elegance", "Toile patterns", "Weathered furniture", "Warm colors", "Provincial charm"],
                "colors": ["Soft gold", "Terra cotta", "Lavender", "Cornflower blue", "Warm neutrals"],
                "materials": ["Weathered wood", "Stone", "Wrought iron", "Linen", "Ceramic"],
                "example_prompt": "A French Country kitchen with a stone floor, weathered wood beams, blue and white pottery, toile curtains, and a large farmhouse table with mismatched chairs"
            }
        }
        
    def get_all_style_names(self) -> List[str]:
        """
        Get all available style names.
        
        Returns:
            List of style names
        """
        return list(self.templates.keys())
    
    def get_style_preview_image_path(self, style_name: str) -> Optional[str]:
        """
        Get the path to the preview image for a style.
        
        Args:
            style_name: Name of the style
            
        Returns:
            Path to the preview image file or None if not found
        """
        if style_name not in self.templates:
            logger.warning(f"Style '{style_name}' not found when looking for preview image")
            return None
        
        # Try different image formats
        for ext in ['jpg', 'jpeg', 'png']:
            # Standard naming: style_name_preview.ext (lowercase with underscores)
            normalized_name = style_name.lower().replace(' ', '_')
            image_path = os.path.join(
                self.preview_images_dir,
                f"{normalized_name}_preview.{ext}"
            )
            
            if os.path.exists(image_path):
                logger.info(f"Found preview image for '{style_name}': {image_path}")
                return image_path
            
        # Return default preview if no specific one exists
        default_path = os.path.join(self.preview_images_dir, "default_preview.jpg")
        if os.path.exists(default_path):
            logger.info(f"Using default preview image for '{style_name}'")
            return default_path
        
        logger.warning(f"No preview image found for '{style_name}'")
        return None
        
    def create_custom_prompt(
        self,
        base_style: str,
        room_type: str = "interior",
        colors: Optional[str] = None,
        materials: Optional[str] = None,
        lighting: Optional[str] = None,
        mood: Optional[str] = None
    ) -> str:
        """
        Create a customized prompt based on a specific style.
        
        Args:
            base_style: Name of the base style
            room_type: Type of room
            colors: Color palette description
            materials: Materials description
            lighting: Lighting description
            mood: Mood/atmosphere description
            
        Returns:
            Customized prompt
        """
        # Check if base style exists
        if base_style not in self.templates:
            logger.warning(f"Base style '{base_style}' not found. Using a generic prompt.")
            prompt = f"A {base_style} style {room_type}"
        else:
            # Start with base prompt
            prompt = self.get_prompt_for_style(base_style, room_type)
        
        # Add custom details
        details = []
        if colors:
            details.append(f"with {colors}")
        if materials:
            details.append(f"featuring {materials}")
        if lighting:
            details.append(f"with {lighting}")
        if mood:
            details.append(f"creating a {mood} atmosphere")
            
        if details:
            prompt += ", " + ", ".join(details)
        
        logger.info(f"Created custom prompt: {prompt}")
        return prompt
    
    def get_style_details(self, style_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a style.
        
        Args:
            style_name: Name of the style
            
        Returns:
            Dictionary with style details or None if not found
        """
        if style_name in self.templates:
            style_data = self.templates[style_name].copy()
            # Remove the prompt template from the returned data
            style_data.pop("prompt", None)
            return style_data
        return None
    
    def add_style_template(self, style_name: str, template: Dict[str, Any]) -> bool:
        """
        Add a new style template.
        
        Args:
            style_name: Name of the style
            template: Style template dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if "prompt" not in template:
                logger.error("Template must contain a 'prompt' field")
                return False
                
            self.templates[style_name] = template
            self._save_templates()
            logger.info(f"Added new style template: {style_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding style template: {str(e)}")
            return False
            
    def update_style_template(self, style_name: str, template: Dict[str, Any]) -> bool:
        """
        Update an existing style template.
        
        Args:
            style_name: Name of the style to update
            template: Updated style template dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if style_name not in self.templates:
            logger.error(f"Style '{style_name}' not found")
            return False
            
        try:
            if "prompt" not in template:
                logger.error("Template must contain a 'prompt' field")
                return False
                
            self.templates[style_name] = template
            self._save_templates()
            logger.info(f"Updated style template: {style_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating style template: {str(e)}")
            return False
            
    def delete_style_template(self, style_name: str) -> bool:
        """
        Delete a style template.
        
        Args:
            style_name: Name of the style to delete
            
        Returns:
            True if successful, False otherwise
        """
        if style_name not in self.templates:
            logger.error(f"Style '{style_name}' not found")
            return False
            
        try:
            del self.templates[style_name]
            self._save_templates()
            logger.info(f"Deleted style template: {style_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting style template: {str(e)}")
            return False
