#!/usr/bin/env python
"""
Example script demonstrating the Prompt Templates API.

This script shows how to use the Interior Style Transfer API for retrieving
and using prompt templates, following the Semantic Seed Coding Standards (SSCS)
for clean, maintainable code.
"""

import os
import sys
import json
import requests
import argparse
from typing import Dict, List, Optional, Any
from pprint import pprint

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import StylePromptManager, StyleCategory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default API URL - updated to use port 8080
DEFAULT_API_URL = "http://localhost:8080/api/v1"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prompt Templates API Example")
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("API_URL", DEFAULT_API_URL),
        help=f"API URL (default: {DEFAULT_API_URL})"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("INTERIOR_STYLE_API_KEY", "development-key"),
        help="API key for authentication"
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default="Scandinavian",
        help="Style name for generating prompts"
    )
    
    parser.add_argument(
        "--room-type",
        type=str,
        default="living room",
        help="Room type for the prompt"
    )
    
    parser.add_argument(
        "--details",
        type=str,
        help="Additional details for the prompt"
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local StylePromptManager instead of API"
    )
    
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List all available styles"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        help="Get styles for a specific category"
    )
    
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Generate a custom prompt with more parameters"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="API request timeout in seconds"
    )
    
    return parser.parse_args()


def demo_local_usage(args):
    """
    Demonstrate using the StylePromptManager directly.
    
    This shows how to use the library locally without the API.
    """
    print("\n=== Using Local StylePromptManager ===\n")
    
    # Create StylePromptManager instance
    manager = StylePromptManager()
    
    # List all styles
    if args.list_styles:
        styles = manager.get_all_style_names()
        print(f"Available styles ({len(styles)}):")
        for style in sorted(styles):
            print(f"- {style}")
        print()
    
    # List all categories
    if args.list_categories:
        categories = [c.value for c in StyleCategory]
        print(f"Available categories ({len(categories)}):")
        for category in categories:
            desc = StyleCategory.get_description(StyleCategory(category))
            print(f"- {category}: {desc}")
        print()
    
    # Get styles for a specific category
    if args.category:
        try:
            category = StyleCategory(args.category)
            styles = manager.get_styles_by_category(category)
            print(f"Styles in '{args.category}' category ({len(styles)}):")
            for style in sorted(styles):
                print(f"- {style}")
            print()
        except ValueError:
            print(f"Error: Category '{args.category}' not found")
    
    # Get style details
    style_details = manager.get_style_details(args.style)
    if style_details:
        print(f"Details for '{args.style}':")
        pprint(style_details)
        print()
    
    # Register a custom formatter for demonstration
    def custom_modern_formatter(style: str, room_type: str, details: Optional[str]) -> str:
        """Example of a custom prompt formatter that adds modern design elements."""
        prompt = f"A cutting-edge {style} {room_type} with innovative design elements"
        if details:
            prompt += f", {details}"
        prompt += ", featuring clean lines, sustainable materials, and smart home integration"
        return prompt
    
    if args.style == "Modern":
        print("Registering custom formatter for 'Modern' style...")
        manager.register_custom_formatter("Modern", custom_modern_formatter)
    
    # Generate prompt
    prompt = manager.get_prompt_for_style(args.style, args.room_type, args.details)
    print(f"Generated prompt for '{args.style}':")
    print(f"  {prompt}")


def demo_api_usage(args):
    """
    Demonstrate using the Prompt Templates API.
    
    This shows how to interact with the API endpoints.
    """
    print("\n=== Using Prompt Templates API ===\n")
    
    api_url = args.api_url
    headers = {"Content-Type": "application/json"}
    params = {"api_key": args.api_key}
    timeout = args.timeout
    
    # List all styles
    if args.list_styles:
        try:
            response = requests.get(
                f"{api_url}/prompts/styles",
                params=params,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "styles" in data:
                print(f"Available styles ({data.get('total_styles', len(data['styles']))}):")
                for style in data["styles"]:
                    print(f"- {style['name']} ({style.get('category', 'unknown')}): {style.get('description', '')[:50]}...")
            else:
                # Handle legacy API format
                print(f"Available styles ({len(data)}):")
                for style in data:
                    print(f"- {style}")
            print()
            
        except requests.RequestException as e:
            print(f"Error listing styles: {str(e)}")
    
    # List all categories
    if args.list_categories:
        try:
            response = requests.get(
                f"{api_url}/prompts/categories",
                params=params,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            categories = response.json()
            
            print(f"Available categories ({len(categories)}):")
            for category in categories:
                print(f"- {category}")
            print()
            
        except requests.RequestException as e:
            print(f"Error listing categories: {str(e)}")
    
    # Get styles for a specific category
    if args.category:
        try:
            response = requests.get(
                f"{api_url}/prompts/categories/{args.category}",
                params=params,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"Styles in '{args.category}' category ({len(data['styles'])}):")
            for style in data["styles"]:
                print(f"- {style}")
            print()
            
        except requests.RequestException as e:
            print(f"Error getting styles by category: {str(e)}")
    
    # Get style details
    try:
        response = requests.get(
            f"{api_url}/prompts/styles/{args.style}",
            params=params,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        style_details = response.json()
        
        print(f"Details for '{args.style}':")
        pprint(style_details)
        print()
        
    except requests.RequestException as e:
        print(f"Error getting style details: {str(e)}")
    
    # Generate prompt
    try:
        if args.custom:
            # Generate a custom prompt with more parameters
            payload = {
                "base_style": args.style,
                "room_type": args.room_type,
                "colors": "warm earth tones with accent colors",
                "materials": "natural wood and stone",
                "lighting": "soft ambient lighting",
                "mood": "relaxing and inviting"
            }
            
            response = requests.post(
                f"{api_url}/prompts/generate-custom-prompt",
                json=payload,
                params=params,
                headers=headers,
                timeout=timeout
            )
        else:
            # Generate a standard prompt
            payload = {
                "style": args.style,
                "room_type": args.room_type,
                "details": args.details
            }
            
            response = requests.post(
                f"{api_url}/prompts/generate-prompt",
                json=payload,
                params=params,
                headers=headers,
                timeout=timeout
            )
        
        response.raise_for_status()
        prompt_data = response.json()
        
        print(f"Generated prompt for '{args.style}':")
        print(f"  {prompt_data['prompt']}")
        
    except requests.RequestException as e:
        print(f"Error generating prompt: {str(e)}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.local:
        demo_local_usage(args)
    else:
        demo_api_usage(args)


if __name__ == "__main__":
    main()
