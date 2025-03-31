#!/usr/bin/env python3
"""
Create the Interior Style Transfer Demo notebook.

This script creates a new interactive Jupyter notebook for the Interior Style Transfer POC
demonstration, following Semantic Seed Coding Standards and BDD principles.
"""

import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell
import os

def create_demo_notebook():
    """
    Create an interactive demo notebook for the Interior Style Transfer POC.
    
    The notebook showcases segmentation, style transfer, evaluation, and visualization
    capabilities in an interactive format.
    
    Returns:
        str: Path to the created notebook
    """
    notebook_path = 'notebooks/Interior_Style_Transfer_Demo.ipynb'
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Title and Introduction
    cells = [
        new_markdown_cell("""
# Interior Style Transfer - Interactive Demo

This notebook demonstrates the capabilities of the Interior Style Transfer POC. It allows users to:
1. Transfer styles between interior design images
2. Segment room elements (walls, floors, ceiling)
3. Evaluate the quality of style transfers
4. Visualize and save the results

Follow along with the steps to see how the system works!
"""),
        
        # Setup and Imports
        new_markdown_cell("""
## Setup and Imports

First, let's set up our environment and import the necessary modules.
"""),
        
        new_code_cell("""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Add the parent directory to the path
sys.path.append('..')

# Helper functions for displaying images
def display_images(images, titles=None, figsize=(15, 10), rows=1):
    """Display multiple images in a row."""
    cols = len(images) // rows
    if len(images) % rows != 0:
        cols += 1
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    for i, (img, ax) in enumerate(zip(images, axes.flatten())):
        if i < len(images):
            if img.max() <= 1.0 and img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Helper function to load sample images
def load_sample_images(sample_dir='notebooks/sample_images'):
    """Load sample interior images from a directory."""
    import glob
    
    images = {}
    sample_files = (
        glob.glob(os.path.join(sample_dir, '*.jpg')) + 
        glob.glob(os.path.join(sample_dir, '*.jpeg')) + 
        glob.glob(os.path.join(sample_dir, '*.png'))
    )
    
    if not sample_files:
        print(f'No sample images found in {sample_dir}')
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
            print(f'Loaded {name} from {file_path}')
        except Exception as e:
            print(f'Error loading {file_path}: {str(e)}')
            
    return images

# Import our custom modules
try:
    from src.style_transfer import StyleTransferService
    from src.segmentation import SegmentationHandler
    from src.evaluation import VisualEvaluationService, ImageEvaluator
    from src.flux_integration import FluxAPIClient
    print("✅ Successfully imported project modules")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Some functions may not work without the required modules.")

# Check environment variables
fal_key = os.getenv('FAL_KEY')
if fal_key:
    print('✅ FAL_KEY environment variable is set')
else:
    print('⚠️ FAL_KEY environment variable is not set. Style transfer may not work.')

print('Environment setup complete!')
"""),
        
        # Loading Sample Images
        new_markdown_cell("""
## Loading Sample Images

Let's load the sample interior images from our local directory.
"""),
        
        new_code_cell("""
# Load sample images from local directory
sample_images = load_sample_images('notebooks/sample_images')

if sample_images:
    # Display the loaded images
    display_images(
        list(sample_images.values()), 
        titles=list(sample_images.keys()), 
        figsize=(20, 10),
        rows=2
    )
    print(f'✅ Successfully loaded {len(sample_images)} sample images!')
else:
    print('⚠️ No sample images were found. Please check the sample_images directory.')
"""),
        
        # Room Segmentation
        new_markdown_cell("""
## Room Segmentation

Now, let's use the SegmentationHandler to identify key elements in our interior images:
- Walls
- Floors
- Ceiling
- Windows
- Structural elements

This will help us preserve these elements during style transfer.
"""),
        
        new_code_cell("""
# Initialize the segmentation handler
try:
    segmentation_handler = SegmentationHandler()

    # Choose an image to segment
    if sample_images:
        # Select an image for demonstration (modern_living_room if available)
        image_key = 'modern_living_room' if 'modern_living_room' in sample_images else list(sample_images.keys())[0]
        original_image = sample_images[image_key]
        
        print(f'Generating segmentation masks for {image_key}...')
        
        # Generate segmentation masks
        masks = segmentation_handler.generate_masks(original_image)
        
        # Display the original image and masks
        plt.figure(figsize=(20, 15))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.title(f'Original: {image_key}')
        plt.axis('off')
        
        # Display masks
        mask_types = ['walls', 'floor', 'ceiling', 'windows', 'structure']
        for i, mask_type in enumerate(mask_types, 2):
            plt.subplot(2, 3, i)
            plt.imshow(original_image)
            mask = masks.get(mask_type, np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8))
            plt.imshow(mask, alpha=0.7, cmap='cool')
            plt.title(f'{mask_type.capitalize()} Mask')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the structure mask for later use
        structure_mask = masks.get('structure', np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8))
        
        print('✅ Segmentation complete!')
    else:
        print('⚠️ No images available for segmentation.')
except Exception as e:
    print(f'⚠️ Error during segmentation: {str(e)}')
"""),
        
        # Style Transfer
        new_markdown_cell("""
## Style Transfer

Now, let's apply style transfer to our interior images using the StyleTransferService.
We'll transfer styles between different interior design styles.
"""),
        
        new_code_cell("""
try:
    # Initialize the style transfer service
    style_transfer_service = StyleTransferService()

    def perform_style_transfer(content_image, style_image, content_name='Content', style_name='Style'):
        """Perform style transfer between two images."""
        print(f'Applying style from {style_name} to {content_name}...')
        
        # Perform style transfer
        result = style_transfer_service.transfer_style(content_image, style_image)
        
        # Display the results
        display_images(
            [content_image, style_image, result],
            titles=[f'Content: {content_name}', f'Style: {style_name}', 'Style Transfer Result'],
            figsize=(18, 6)
        )
        
        # Save the result if needed
        output_dir = 'notebooks/outputs'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{content_name}_styled_as_{style_name}.jpg')
        Image.fromarray(result.astype(np.uint8)).save(output_path)
        print(f'Saved result to {output_path}')
        
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
        print('⚠️ Need at least two images for style transfer demonstration.')
except Exception as e:
    print(f'⚠️ Error during style transfer: {str(e)}')
"""),
        
        # Structure Preservation
        new_markdown_cell("""
## Structure Preservation

Now, let's demonstrate how we can preserve the structural elements of the original room while applying style transfer.
This is important for maintaining the architectural integrity of the space.
"""),
        
        new_code_cell("""
try:
    # Only proceed if we have generated masks and performed style transfer
    if all(var in locals() for var in ['structure_mask', 'result_image', 'content_image']):
        print('Applying structure preservation...')
        
        # Apply structure preservation
        preserved_result = segmentation_handler.apply_structure_preservation(
            content_image, result_image, structure_mask
        )
        
        # Display the results
        display_images(
            [content_image, result_image, preserved_result],
            titles=['Original Image', 'Direct Style Transfer', 'Structure-Preserved Result'],
            figsize=(18, 6)
        )
        
        # Save the structure-preserved result
        output_dir = 'notebooks/outputs'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{content_key}_preserved_style_{style_key}.jpg')
        Image.fromarray(preserved_result.astype(np.uint8)).save(output_path)
        print(f'Saved result to {output_path}')
        
        print('✅ Structure preservation complete!')
    else:
        print('⚠️ Structure preservation requires segmentation masks and style transfer results.')
except Exception as e:
    print(f'⚠️ Error during structure preservation: {str(e)}')
"""),
        
        # Evaluation
        new_markdown_cell("""
## Evaluation

Let's evaluate the quality of our style transfer results using the evaluation metrics.
We'll look at:
1. Structural Similarity Index (SSIM)
2. Mean Squared Error (MSE)
3. Structure preservation score
"""),
        
        new_code_cell("""
try:
    # Initialize the evaluation service
    evaluation_service = VisualEvaluationService()
    image_evaluator = ImageEvaluator()

    # Only proceed if we have the original image, result image, and structure-preserved result
    if all(var in locals() for var in ['content_image', 'result_image', 'preserved_result', 'structure_mask']):
        print('Evaluating style transfer results...')
        
        # Evaluate direct style transfer
        direct_metrics = evaluation_service.calculate_metrics(content_image, result_image)
        print('Direct Style Transfer Metrics:')
        print(f'SSIM: {direct_metrics["ssim"]:.4f} (higher is better)')
        print(f'MSE: {direct_metrics["mse"]:.4f} (lower is better)')
        
        # Evaluate structure-preserved result
        preserved_metrics = evaluation_service.calculate_metrics(content_image, preserved_result)
        print('\\nStructure-Preserved Style Transfer Metrics:')
        print(f'SSIM: {preserved_metrics["ssim"]:.4f} (higher is better)')
        print(f'MSE: {preserved_metrics["mse"]:.4f} (lower is better)')
        
        # Calculate structure preservation score
        preservation_score = evaluation_service.calculate_preservation_score(
            content_image, preserved_result, structure_mask
        )
        print(f'\\nStructure Preservation Score: {preservation_score:.4f} (higher is better)')
        
        # Evaluate structure preservation with ImageEvaluator
        structure_eval = image_evaluator.evaluate_structure_preservation(
            content_image, preserved_result, structure_mask
        )
        print('\\nDetailed Structure Preservation Evaluation:')
        for key, value in structure_eval.items():
            if isinstance(value, float):
                print(f'{key}: {value:.4f}')
            else:
                print(f'{key}: {value}')
        
        # Create visualization of the differences
        print('\\nGenerating difference visualization...')
        diff_vis = image_evaluator.visualize_differences(content_image, preserved_result)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(diff_vis)
        plt.title('Difference Visualization')
        plt.axis('off')
        plt.show()
        
        # Save the difference visualization
        output_dir = 'notebooks/outputs'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{content_key}_diff_visualization.jpg')
        Image.fromarray(diff_vis.astype(np.uint8)).save(output_path)
        print(f'Saved visualization to {output_path}')
        
        print('✅ Evaluation complete!')
    else:
        print('⚠️ Evaluation requires original image, style transfer result, and structure-preserved result.')
except Exception as e:
    print(f'⚠️ Error during evaluation: {str(e)}')
"""),
        
        # Style Variations
        new_markdown_cell("""
## Style Variations

Let's generate multiple style variations to provide options for the user.
We'll apply different styles to the same original room.
"""),
        
        new_code_cell("""
try:
    # Generate style variations if we have at least 3 images
    if len(sample_images) >= 3 and 'content_key' in locals():
        # Use the same content image as before
        content_image = sample_images[content_key]
        
        # Use different images as style references
        style_keys = [k for k in sample_images.keys() if k != content_key]
        
        # Generate variations
        print('Generating style variations...')
        variations = []
        variation_titles = []
        
        for style_key in style_keys:
            style_img = sample_images[style_key]
            
            # Apply style transfer
            print(f'Applying style from {style_key}...')
            result = style_transfer_service.transfer_style(content_image, style_img)
            
            # If structure mask is available, apply structure preservation
            if 'structure_mask' in locals():
                result = segmentation_handler.apply_structure_preservation(
                    content_image, result, structure_mask
                )
            
            variations.append(result)
            variation_titles.append(f'Style: {style_key}')
            
            # Save the variation
            output_dir = 'notebooks/outputs'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{content_key}_variation_{style_key}.jpg')
            Image.fromarray(result.astype(np.uint8)).save(output_path)
            print(f'Saved variation to {output_path}')
        
        # Display all variations
        if variations:
            # First show the original
            plt.figure(figsize=(10, 8))
            plt.imshow(content_image)
            plt.title(f'Original: {content_key}')
            plt.axis('off')
            plt.show()
            
            # Then show all variations
            display_images(
                variations,
                titles=variation_titles,
                figsize=(20, 5 * ((len(variations) - 1) // 2 + 1)),
                rows=(len(variations) - 1) // 2 + 1
            )
            
            print(f'✅ Generated {len(variations)} style variations!')
        else:
            print('No style variations were generated.')
    else:
        print('⚠️ Need at least 3 images for style variation demonstration.')
except Exception as e:
    print(f'⚠️ Error during style variations: {str(e)}')
"""),
        
        # Conclusion
        new_markdown_cell("""
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
""")
    ]
    
    # Add cells to the notebook
    nb['cells'] = cells
    
    # Write the notebook to a file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Created notebook at {notebook_path}")
    return notebook_path

if __name__ == '__main__':
    create_demo_notebook()
