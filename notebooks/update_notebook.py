#!/usr/bin/env python3
"""
Update the Interior Style Transfer Demo notebook.

This script updates the existing Jupyter notebook to use local sample images and
utility functions, making it more robust and user-friendly.
"""

import os
import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell

def update_notebook():
    """Update the Interior Style Transfer Demo notebook."""
    notebook_path = 'notebooks/Interior_Style_Transfer_Demo.ipynb'
    output_path = 'notebooks/Interior_Style_Transfer_Demo_Updated.ipynb'
    
    # Read the existing notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    # Create a new notebook
    new_nb = nbf.v4.new_notebook()
    new_nb['metadata'] = nb['metadata']
    
    # Title and Introduction cell
    new_nb['cells'] = [new_markdown_cell("""
# Interior Style Transfer - Interactive Demo

This notebook demonstrates the capabilities of the Interior Style Transfer POC. It allows users to:
1. Transfer styles between interior design images
2. Segment room elements (walls, floors, ceiling)
3. Evaluate the quality of style transfers
4. Visualize and save the results

Follow along with the steps to see how the system works!
""")]
    
    # Setup and Imports (updated to use utility functions)
    new_nb['cells'].append(new_markdown_cell("""
## Setup and Imports

First, let's set up our environment and import the necessary modules.
"""))
    
    new_nb['cells'].append(new_code_cell("""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Import utils and project modules
from notebook_utils import load_sample_images, display_images, check_environment_variables, save_image_result

# Import our custom modules
from src.style_transfer import StyleTransferService
from src.segmentation import SegmentationHandler
from src.evaluation import VisualEvaluationService, ImageEvaluator
from src.flux_integration import FluxAPIClient

# Check environment variables
if check_environment_variables():
    print("✅ Environment is properly configured!")
else:
    print("⚠️ Some environment variables are missing. Results may be affected.")

print("Environment setup complete!")
"""))
    
    # Loading Sample Images - updated to use local files
    new_nb['cells'].append(new_markdown_cell("""
## Loading Sample Images

Let's load the sample interior images from our local directory.
"""))
    
    new_nb['cells'].append(new_code_cell("""
# Load sample images from local directory
sample_images = load_sample_images("notebooks/sample_images")

if sample_images:
    # Display the loaded images
    display_images(
        list(sample_images.values()), 
        titles=list(sample_images.keys()), 
        figsize=(20, 10),
        rows=2
    )
    print(f"✅ Successfully loaded {len(sample_images)} sample images!")
else:
    print("⚠️ No sample images were found. Please check the sample_images directory.")
"""))
    
    # Room Segmentation
    new_nb['cells'].append(new_markdown_cell("""
## Room Segmentation

Now, let's use the SegmentationHandler to identify key elements in our interior images:
- Walls
- Floors
- Ceiling
- Windows
- Structural elements

This will help us preserve these elements during style transfer.
"""))
    
    new_nb['cells'].append(new_code_cell("""
# Initialize the segmentation handler
segmentation_handler = SegmentationHandler()

# Choose an image to segment
if sample_images:
    # Select an image for demonstration (modern_living_room if available)
    image_key = 'modern_living_room' if 'modern_living_room' in sample_images else list(sample_images.keys())[0]
    original_image = sample_images[image_key]
    
    print(f"Generating segmentation masks for {image_key}...")
    
    # Generate segmentation masks
    masks = segmentation_handler.generate_masks(original_image)
    
    # Display the original image and masks
    plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title(f"Original: {image_key}")
    plt.axis("off")
    
    # Display masks
    mask_types = ["walls", "floor", "ceiling", "windows", "structure"]
    for i, mask_type in enumerate(mask_types, 2):
        plt.subplot(2, 3, i)
        plt.imshow(original_image)
        mask = masks.get(mask_type, np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8))
        plt.imshow(mask, alpha=0.7, cmap="cool")
        plt.title(f"{mask_type.capitalize()} Mask")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Save the structure mask for later use
    structure_mask = masks.get("structure", np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8))
    
    print("✅ Segmentation complete!")
else:
    print("⚠️ No images available for segmentation.")
"""))
    
    # Style Transfer
    new_nb['cells'].append(new_markdown_cell("""
## Style Transfer

Now, let's apply style transfer to our interior images using the StyleTransferService.
We'll transfer styles between different interior design styles.
"""))
    
    new_nb['cells'].append(new_code_cell("""
# Initialize the style transfer service
style_transfer_service = StyleTransferService()

def perform_style_transfer(content_image, style_image, content_name="Content", style_name="Style"):
    \"\"\"Perform style transfer between two images.\"\"\"
    print(f"Applying style from {style_name} to {content_name}...")
    
    # Perform style transfer
    result = style_transfer_service.transfer_style(content_image, style_image)
    
    # Display the results
    display_images(
        [content_image, style_image, result],
        titles=[f"Content: {content_name}", f"Style: {style_name}", "Style Transfer Result"],
        figsize=(18, 6)
    )
    
    # Save the result
    save_image_result(result, f"{content_name}_styled_as_{style_name}")
    
    return result

# Perform style transfer if we have at least two images
if len(sample_images) >= 2:
    # Get two distinct images for content and style
    content_key = 'modern_living_room' if 'modern_living_room' in sample_images else list(sample_images.keys())[0]
    
    # Choose a different image for style
    style_keys = [k for k in sample_images.keys() if k != content_key]
    style_key = style_keys[0] if style_keys else content_key
    
    content_image = sample_images[content_key]
    style_image = sample_images[style_key]
    
    # Perform the style transfer
    result_image = perform_style_transfer(content_image, style_image, content_key, style_key)
else:
    print("⚠️ Need at least two images for style transfer demonstration.")
"""))
    
    # Structure Preservation
    new_nb['cells'].append(new_markdown_cell("""
## Structure Preservation

Now, let's demonstrate how we can preserve the structural elements of the original room while applying style transfer.
This is important for maintaining the architectural integrity of the space.
"""))
    
    new_nb['cells'].append(new_code_cell("""
# Only proceed if we have generated masks and performed style transfer
if all(var in locals() for var in ['structure_mask', 'result_image', 'content_image']):
    print("Applying structure preservation...")
    
    # Apply structure preservation
    preserved_result = segmentation_handler.apply_structure_preservation(
        content_image, result_image, structure_mask
    )
    
    # Display the results
    display_images(
        [content_image, result_image, preserved_result],
        titles=["Original Image", "Direct Style Transfer", "Structure-Preserved Result"],
        figsize=(18, 6)
    )
    
    # Save the structure-preserved result
    save_image_result(preserved_result, f"{content_key}_preserved_style_{style_key}")
    
    print("✅ Structure preservation complete!")
else:
    print("⚠️ Structure preservation requires segmentation masks and style transfer results.")
"""))
    
    # Evaluation
    new_nb['cells'].append(new_markdown_cell("""
## Evaluation

Let's evaluate the quality of our style transfer results using the evaluation metrics.
We'll look at:
1. Structural Similarity Index (SSIM)
2. Mean Squared Error (MSE)
3. Structure preservation score
"""))
    
    new_nb['cells'].append(new_code_cell("""
# Initialize the evaluation service
evaluation_service = VisualEvaluationService()
image_evaluator = ImageEvaluator()

# Only proceed if we have the original image, result image, and structure-preserved result
if all(var in locals() for var in ['content_image', 'result_image', 'preserved_result', 'structure_mask']):
    print("Evaluating style transfer results...")
    
    # Evaluate direct style transfer
    direct_metrics = evaluation_service.calculate_metrics(content_image, result_image)
    print("Direct Style Transfer Metrics:")
    print(f"SSIM: {direct_metrics['ssim']:.4f} (higher is better)")
    print(f"MSE: {direct_metrics['mse']:.4f} (lower is better)")
    
    # Evaluate structure-preserved result
    preserved_metrics = evaluation_service.calculate_metrics(content_image, preserved_result)
    print("\nStructure-Preserved Style Transfer Metrics:")
    print(f"SSIM: {preserved_metrics['ssim']:.4f} (higher is better)")
    print(f"MSE: {preserved_metrics['mse']:.4f} (lower is better)")
    
    # Calculate structure preservation score
    preservation_score = evaluation_service.calculate_preservation_score(
        content_image, preserved_result, structure_mask
    )
    print(f"\nStructure Preservation Score: {preservation_score:.4f} (higher is better)")
    
    # Evaluate structure preservation with ImageEvaluator
    structure_eval = image_evaluator.evaluate_structure_preservation(
        content_image, preserved_result, structure_mask
    )
    print("\nDetailed Structure Preservation Evaluation:")
    for key, value in structure_eval.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Create visualization of the differences
    print("\nGenerating difference visualization...")
    diff_vis = image_evaluator.visualize_differences(content_image, preserved_result)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(diff_vis)
    plt.title("Difference Visualization")
    plt.axis("off")
    plt.show()
    
    # Save the difference visualization
    save_image_result(diff_vis, f"{content_key}_diff_visualization")
    
    print("✅ Evaluation complete!")
else:
    print("⚠️ Evaluation requires original image, style transfer result, and structure-preserved result.")
"""))
    
    # Additional Style Variations
    new_nb['cells'].append(new_markdown_cell("""
## Style Variations

Let's generate multiple style variations to provide options for the user.
We'll apply different styles to the same original room.
"""))
    
    new_nb['cells'].append(new_code_cell("""
# Generate style variations if we have at least 3 images
if len(sample_images) >= 3 and 'content_key' in locals():
    # Use the same content image as before
    content_image = sample_images[content_key]
    
    # Use different images as style references
    style_keys = [k for k in sample_images.keys() if k != content_key]
    
    # Generate variations
    print("Generating style variations...")
    variations = []
    variation_titles = []
    
    for style_key in style_keys:
        style_img = sample_images[style_key]
        
        # Apply style transfer
        print(f"Applying style from {style_key}...")
        result = style_transfer_service.transfer_style(content_image, style_img)
        
        # If structure mask is available, apply structure preservation
        if 'structure_mask' in locals():
            result = segmentation_handler.apply_structure_preservation(
                content_image, result, structure_mask
            )
        
        variations.append(result)
        variation_titles.append(f"Style: {style_key}")
        
        # Save the variation
        save_image_result(result, f"{content_key}_variation_{style_key}")
    
    # Display all variations
    if variations:
        # First show the original
        plt.figure(figsize=(10, 8))
        plt.imshow(content_image)
        plt.title(f"Original: {content_key}")
        plt.axis("off")
        plt.show()
        
        # Then show all variations
        display_images(
            variations,
            titles=variation_titles,
            figsize=(20, 5 * ((len(variations) - 1) // 2 + 1)),
            rows=(len(variations) - 1) // 2 + 1
        )
        
        print(f"✅ Generated {len(variations)} style variations!")
    else:
        print("No style variations were generated.")
else:
    print("⚠️ Need at least 3 images for style variation demonstration.")
"""))
    
    # Conclusion
    new_nb['cells'].append(new_markdown_cell("""
## Conclusion

In this notebook, we've demonstrated the key features of the Interior Style Transfer POC:

1. **Image Segmentation**: We identified key structural elements in interior images.
2. **Style Transfer**: We transferred design styles between different interior images.
3. **Structure Preservation**: We maintained architectural integrity during style transfer.
4. **Quality Evaluation**: We evaluated the results using objective metrics.
5. **Style Variations**: We generated multiple style options for user selection.

This interactive demo showcases the potential of AI-powered interior design tools to quickly generate design concepts while preserving architectural constraints.

All generated outputs have been saved to the `notebooks/outputs` directory for your reference.

### Next Steps

- Try with your own interior photos
- Experiment with different style references
- Adjust segmentation parameters for better structure identification
- Use the API for integration with design software
"""))
    
    # Write the updated notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(new_nb, f)
    
    print(f"Updated notebook created at {output_path}")

if __name__ == "__main__":
    update_notebook()
