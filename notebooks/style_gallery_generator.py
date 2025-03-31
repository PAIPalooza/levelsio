"""
Style Gallery Generator for Interior Style Transfer POC.

This module generates a comprehensive gallery of style variations for
interior images, following the Semantic Seed Coding Standards with
proper BDD documentation and robust error handling.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import glob
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

# Define constants for better maintainability
SAMPLE_DIR = 'sample_images'
OUTPUT_DIR = 'gallery_outputs'
RESULTS_JSON = os.path.join(OUTPUT_DIR, 'gallery_results.json')

# Style definitions for varied appearance
STYLE_VARIATIONS = {
    "rustic": {
        "color": 0.9,      # Less colorful
        "contrast": 1.4,    # Higher contrast
        "brightness": 0.8,  # Darker
        "warmth": 20        # Warm tones
    },
    "minimalist": {
        "color": 0.7,       # Desaturated
        "contrast": 1.1,    # Slightly higher contrast
        "brightness": 1.1,  # Brighter
        "warmth": -10       # Cool tones
    },
    "art_deco": {
        "color": 1.5,       # More saturated
        "contrast": 1.3,    # Higher contrast
        "brightness": 1.0,  # Normal brightness
        "warmth": 0         # Neutral tones
    },
    "scandinavian": {
        "color": 0.8,       # Less colorful
        "contrast": 0.9,    # Lower contrast
        "brightness": 1.3,  # Brighter
        "warmth": -20       # Very cool tones
    },
    "industrial": {
        "color": 0.6,       # Less colorful
        "contrast": 1.5,    # High contrast
        "brightness": 0.7,  # Darker
        "warmth": -5        # Slightly cool tones
    }
}

def load_sample_images(sample_dir: str = SAMPLE_DIR) -> Dict[str, np.ndarray]:
    """
    Load sample interior images from a directory.
    
    Args:
        sample_dir: Directory containing sample images
        
    Returns:
        Dictionary mapping image names to numpy arrays
    """
    images = {}
    
    # Get the absolute path to the sample images
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(current_dir, sample_dir)
    
    logger.info(f"Looking for sample images in: {sample_dir}")
    
    sample_files = (
        glob.glob(os.path.join(sample_dir, '*.jpg')) + 
        glob.glob(os.path.join(sample_dir, '*.jpeg')) + 
        glob.glob(os.path.join(sample_dir, '*.png'))
    )
    
    if not sample_files:
        logger.warning(f"No sample images found in {sample_dir}")
        return images
    
    for file_path in sample_files:
        try:
            # Get the filename without extension as the key
            name = os.path.splitext(os.path.basename(file_path))[0]
            # Load the image
            img = Image.open(file_path)
            # Convert to numpy array
            img_array = np.array(img)
            # Add to dictionary
            images[name] = img_array
            logger.info(f"Loaded {name} from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            
    return images

def apply_mock_style(
    content_image: np.ndarray, 
    style_name: str
) -> np.ndarray:
    """
    Apply a mock style transfer effect to an image.
    
    This function simulates style transfer for demo purposes when
    API access is not available, following BDD principles with
    clear documentation and predictable outputs.
    
    Args:
        content_image: Source image to be styled
        style_name: Name of the style to apply
        
    Returns:
        Styled version of the content image
    """
    logger.info(f"Applying mock {style_name} style")
    
    # Create a copy of the original image
    result = content_image.copy()
    
    # Convert to PIL Image for processing
    pil_image = Image.fromarray(result)
    
    # Get style parameters or use defaults for unknown styles
    style_params = STYLE_VARIATIONS.get(style_name.lower(), {
        "color": 1.0,
        "contrast": 1.0,
        "brightness": 1.0,
        "warmth": 0
    })
    
    # Apply the style adjustments
    # 1. Color saturation
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(style_params["color"])
    
    # 2. Contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(style_params["contrast"])
    
    # 3. Brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(style_params["brightness"])
    
    # 4. Warmth adjustment (using split channels)
    if style_params["warmth"] != 0:
        r, g, b = pil_image.split()
        
        # Adjust red channel for warmth
        r_enhancer = ImageEnhance.Brightness(r)
        r = r_enhancer.enhance(1 + style_params["warmth"]/100)
        
        # Adjust blue channel for coolness (opposite of warmth)
        b_enhancer = ImageEnhance.Brightness(b)
        b = b_enhancer.enhance(1 - style_params["warmth"]/100)
        
        # Recombine channels
        pil_image = Image.merge("RGB", (r, g, b))
    
    # Convert back to numpy array
    result = np.array(pil_image)
    
    return result

def apply_structure_preservation(
    original: np.ndarray, 
    styled: np.ndarray,
    structure_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Preserve structural elements from the original image in the styled version.
    
    Args:
        original: The original image
        styled: The styled image
        structure_mask: Binary mask indicating structural elements (1=structure)
        
    Returns:
        Styled image with preserved structural elements
    """
    logger.info("Applying structure preservation")
    
    # If no structure mask provided, create a simple one
    if structure_mask is None:
        # Create a simple mask based on edge detection
        from scipy import ndimage
        
        # Convert to grayscale
        if len(original.shape) == 3:
            gray = np.mean(original, axis=2).astype(np.uint8)
        else:
            gray = original.astype(np.uint8)
        
        # Detect edges
        edges = ndimage.sobel(gray)
        
        # Threshold to create binary mask
        threshold = np.percentile(edges, 90)  # Top 10% of edges
        structure_mask = (edges > threshold).astype(np.uint8)
        
        # Dilate the mask to include surrounding areas
        structure_mask = ndimage.binary_dilation(structure_mask, iterations=2).astype(np.uint8)
    
    # Ensure mask has proper dimensions
    if len(structure_mask.shape) == 2 and len(original.shape) == 3:
        structure_mask = np.expand_dims(structure_mask, axis=2)
        structure_mask = np.repeat(structure_mask, 3, axis=2)
    
    # Blend original and styled images based on structure mask
    blended = np.where(structure_mask > 0, original, styled)
    
    # Apply a slight gaussian blur at the boundaries for smoother transitions
    from scipy.ndimage import gaussian_filter
    
    # Create a gradient around the edges of the mask
    gradient_mask = gaussian_filter(structure_mask.astype(float), sigma=2) - gaussian_filter(structure_mask.astype(float), sigma=1)
    gradient_mask = np.clip(gradient_mask * 5, 0, 1)
    
    # Apply the gradient blending
    final_result = original * gradient_mask + blended * (1 - gradient_mask)
    
    return final_result.astype(np.uint8)

def generate_style_gallery(
    content_names: Optional[List[str]] = None,
    style_names: Optional[List[str]] = None,
    create_thumbnails: bool = True
) -> Dict[str, Any]:
    """
    Generate a gallery of styled interiors with different combinations.
    
    Args:
        content_names: List of content image names to use (or None for all)
        style_names: List of style names to apply (or None for defaults)
        create_thumbnails: Whether to create thumbnail images for the gallery
        
    Returns:
        Dictionary with gallery metadata and results
    """
    # Load sample images
    sample_images = load_sample_images()
    
    if not sample_images:
        logger.error("No sample images found. Cannot generate gallery.")
        return {"success": False, "error": "No sample images found"}
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Select content images to use
    if content_names is None:
        # Select interior images by default
        interior_keywords = ['kitchen', 'living', 'bedroom', 'bathroom', 'dining']
        content_names = [name for name in sample_images.keys() 
                        if any(keyword in name.lower() for keyword in interior_keywords)]
        
        if not content_names:
            # If no interiors identified, use all images
            content_names = list(sample_images.keys())
    else:
        # Filter to only include available images
        content_names = [name for name in content_names if name in sample_images]
    
    # Select styles to apply
    if style_names is None:
        style_names = list(STYLE_VARIATIONS.keys())
    
    # Track results
    gallery_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "content_images": content_names,
        "styles_applied": style_names,
        "variations": []
    }
    
    # Generate style variations for each content image
    for content_name in content_names:
        content_image = sample_images[content_name]
        
        logger.info(f"Generating style variations for {content_name}")
        
        content_variations = {
            "content_name": content_name,
            "styles": []
        }
        
        # Create a row of variations for display
        row_images = [content_image]  # First image is original
        row_titles = ["Original"]
        
        # Apply each style
        for style_name in style_names:
            logger.info(f"Applying {style_name} style to {content_name}")
            
            # Apply the style
            try:
                styled_image = apply_mock_style(content_image, style_name)
                
                # Apply structure preservation
                preserved_image = apply_structure_preservation(content_image, styled_image)
                
                # Save the output image
                output_filename = f"{content_name}_{style_name}_style.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                Image.fromarray(preserved_image).save(output_path)
                logger.info(f"Saved styled image to {output_path}")
                
                # Add to the row for display
                row_images.append(preserved_image)
                row_titles.append(style_name.capitalize())
                
                # Add to results
                content_variations["styles"].append({
                    "style_name": style_name,
                    "output_file": output_filename
                })
                
                # Create thumbnail if requested
                if create_thumbnails:
                    thumb_dir = os.path.join(OUTPUT_DIR, "thumbnails")
                    os.makedirs(thumb_dir, exist_ok=True)
                    
                    thumb_path = os.path.join(thumb_dir, f"{content_name}_{style_name}_thumb.jpg")
                    thumb_size = (256, 256)
                    
                    # Create and save thumbnail
                    thumb = Image.fromarray(preserved_image)
                    thumb.thumbnail(thumb_size)
                    thumb.save(thumb_path)
                    
            except Exception as e:
                logger.error(f"Error applying {style_name} to {content_name}: {str(e)}")
                # Continue with next style
        
        # Display the row of variations
        plt.figure(figsize=(5 * len(row_images), 4))
        
        for i, (img, title) in enumerate(zip(row_images, row_titles)):
            plt.subplot(1, len(row_images), i + 1)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        
        plt.suptitle(f"Style Variations for {content_name}", fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Save the comparison image
        comparison_filename = f"{content_name}_style_comparison.jpg"
        comparison_path = os.path.join(OUTPUT_DIR, comparison_filename)
        
        # Create a new larger figure for saving
        fig = plt.figure(figsize=(5 * len(row_images), 4))
        
        for i, (img, title) in enumerate(zip(row_images, row_titles)):
            ax = fig.add_subplot(1, len(row_images), i + 1)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        fig.suptitle(f"Style Variations for {content_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150)
        plt.close(fig)
        
        # Add the comparison image to the results
        content_variations["comparison_image"] = comparison_filename
        gallery_results["variations"].append(content_variations)
    
    # Save gallery results to JSON
    with open(RESULTS_JSON, 'w') as f:
        json.dump(gallery_results, f, indent=2)
    
    logger.info(f"Gallery generation complete. Results saved to {RESULTS_JSON}")
    
    return gallery_results

def create_gallery_html(gallery_results: Dict[str, Any]) -> str:
    """
    Create an HTML gallery page from the gallery results.
    
    Args:
        gallery_results: Gallery results dictionary
        
    Returns:
        Path to the generated HTML file
    """
    html_path = os.path.join(OUTPUT_DIR, "interior_style_gallery.html")
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interior Style Transfer Gallery</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2 {{ color: #333; }}
        .gallery-container {{ margin-bottom: 40px; }}
        .gallery-image {{ width: 100%; max-width: 1000px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .timestamp {{ color: #666; font-style: italic; margin-bottom: 20px; }}
        .style-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }}
        .style-item {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; transition: transform 0.3s; }}
        .style-item:hover {{ transform: scale(1.05); }}
        .style-item img {{ width: 100%; }}
        .style-item .caption {{ padding: 10px; text-align: center; background: #f5f5f5; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <h1>Interior Style Transfer Gallery</h1>
    <p class="timestamp">Generated on: {gallery_results.get('timestamp', 'Unknown')}</p>
    
    <p>This gallery showcases various interior design style transformations applied to different spaces.
    The original images are shown alongside stylized versions that maintain structural integrity.</p>
"""
    
    # Add each content image and its variations
    for variation in gallery_results.get('variations', []):
        content_name = variation.get('content_name', 'Unknown')
        comparison_image = variation.get('comparison_image', '')
        
        html_content += f"""
    <div class="gallery-container">
        <h2>{content_name}</h2>
        <img src="{comparison_image}" alt="Style comparison for {content_name}" class="gallery-image">
        
        <h3>Individual Styles</h3>
        <div class="style-grid">
"""
        
        # Add individual style items
        for style in variation.get('styles', []):
            style_name = style.get('style_name', 'Unknown')
            output_file = style.get('output_file', '')
            
            html_content += f"""
            <div class="style-item">
                <img src="{output_file}" alt="{content_name} with {style_name} style">
                <div class="caption">{style_name.capitalize()}</div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    # Add footer
    html_content += """
    <div class="footer">
        <p>Created with Interior Style Transfer POC - Semantic Seed Venture Studio</p>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Gallery HTML created at {html_path}")
    return html_path

def run_gallery_generation():
    """
    Run the full gallery generation process.
    """
    print("=== Style Gallery Generation ===")
    print("Generating comprehensive style variations...")
    
    # Generate the gallery
    gallery_results = generate_style_gallery()
    
    if gallery_results.get("success") == False:
        print(f"⚠️ Gallery generation failed: {gallery_results.get('error', 'Unknown error')}")
        return
    
    # Create HTML gallery
    html_path = create_gallery_html(gallery_results)
    
    print(f"\n✅ Gallery generation complete!")
    print(f"Results saved to {OUTPUT_DIR} directory")
    print(f"HTML gallery created at {html_path}")
    print("\nSuggested next steps:")
    print("1. Open the HTML gallery in a browser to view all variations")
    print("2. Use the comparison images for client presentations")
    print("3. Incorporate preferred styles into your design workflow")

if __name__ == "__main__":
    run_gallery_generation()
