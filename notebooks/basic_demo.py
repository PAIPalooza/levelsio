"""
Basic interactive demo script for Interior Style Transfer.

This script provides a simplified demonstration of the core functionality
of the Interior Style Transfer POC, following Semantic Seed Coding Standards.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

# Configure output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Helper functions
def load_sample_images(sample_dir='sample_images'):
    """Load sample interior images from a directory."""
    images = {}
    
    # Get the absolute path to the sample images
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(current_dir, sample_dir)
    
    print(f"Looking for sample images in: {sample_dir}")
    
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

def save_output(image, filename):
    """Save an image output."""
    output_path = os.path.join(output_dir, filename)
    if not output_path.endswith(('.jpg', '.jpeg', '.png')):
        output_path += '.jpg'
    
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0 and image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image
    
    pil_image.save(output_path)
    print(f'Saved output to {output_path}')
    return output_path

def run_demo():
    """Run the Interior Style Transfer demo."""
    # Check environment variables
    fal_key = os.getenv('FAL_KEY')
    if fal_key:
        print('✅ FAL_KEY environment variable is set')
    else:
        print('⚠️ FAL_KEY environment variable is not set. Style transfer may not work.')
    
    # Set offline mode for demo - change to False to attempt actual API calls
    offline_mode = True
    if offline_mode:
        print('🔄 Running in OFFLINE mode: API calls will be simulated for demo purposes')
    else:
        print('🌐 Running in ONLINE mode: Using Fal.ai API for real style transfer')
    
    # Step 1: Load sample images
    print("\n=== STEP 1: Loading Sample Images ===")
    sample_images = load_sample_images()
    
    if sample_images:
        print(f'✅ Successfully loaded {len(sample_images)} sample images!')
        # Display the loaded images
        display_images(
            list(sample_images.values()), 
            titles=list(sample_images.keys()), 
            figsize=(20, 10),
            rows=2
        )
    else:
        print('⚠️ No sample images were found. Exiting demo.')
        return
    
    # Step 2: Segmentation
    print("\n=== STEP 2: Room Segmentation ===")
    try:
        from src.segmentation import SegmentationHandler
        
        # Initialize the segmentation handler
        segmentation_handler = SegmentationHandler()
        
        # Choose an image for demonstration
        image_key = 'modern_living_room' if 'modern_living_room' in sample_images else list(sample_images.keys())[0]
        original_image = sample_images[image_key]
        
        print(f'Generating segmentation masks for {image_key}...')
        masks = segmentation_handler.generate_masks(original_image)
        
        # Display original image and masks
        plt.figure(figsize=(18, 12))
        
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
        save_output(structure_mask, f'{image_key}_structure_mask')
        
        print('✅ Segmentation complete!')
    except Exception as e:
        print(f'⚠️ Error during segmentation: {str(e)}')
        print('Skipping segmentation step...')
    
    # Step 3: Style Transfer
    print("\n=== STEP 3: Style Transfer ===")
    try:
        from src.style_transfer import StyleTransferService
        from src.flux_integration import FluxClient
        
        # Mock style transfer function for offline demo
        def mock_style_transfer(content_image, style_image=None, style_prompt=None):
            """
            Simulate style transfer for offline demo purposes.
            
            This follows BDD principles by providing a testable mock implementation
            that maintains the same interface as the real implementation.
            
            Args:
                content_image: Original image
                style_image: Style reference image (optional)
                style_prompt: Style description prompt (optional)
                
            Returns:
                Modified version of content image as style transfer simulation
            """
            print("📝 Using mock style transfer (offline mode)")
            
            # Create a modified version of the original image to simulate style transfer
            # This is just for demonstration purposes
            result = content_image.copy()
            
            # Apply some basic color modifications to simulate style
            # Adjust hue and saturation
            from PIL import Image, ImageEnhance
            
            # Convert to PIL Image for processing
            pil_image = Image.fromarray(result)
            
            # Apply a series of enhancements
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(1.5)  # Increase color saturation
            
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.2)  # Increase contrast
            
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(0.9)  # Slightly darker
            
            # Convert back to numpy array
            result = np.array(pil_image)
            
            return result
        
        # Initialize style transfer service - use real or mock based on mode
        if offline_mode:
            # In offline mode, we'll use our mock implementation
            flux_client = None
            style_transfer_service = StyleTransferService()
            style_transfer_service.apply_style_only = mock_style_transfer
        else:
            # In online mode, use the real API
            flux_client = FluxClient()
            style_transfer_service = StyleTransferService(flux_client=flux_client)
        
        # Get two distinct images for content and style
        if len(sample_images) >= 2:
            # Select content image - make sure to use a proper interior image
            if 'minimalist_kitchen' in sample_images:
                content_key = 'minimalist_kitchen'  # Explicitly select a kitchen interior
            elif 'art_deco_living' in sample_images:
                content_key = 'art_deco_living'  # Fallback to living room interior
            else:
                content_key = list(sample_images.keys())[0]
            
            # Choose a different image for style
            style_keys = [k for k in sample_images.keys() if k != content_key]
            
            # Prefer rustic_bedroom for style if available
            if 'rustic_bedroom' in style_keys:
                style_key = 'rustic_bedroom'
            else:
                style_key = style_keys[0] if style_keys else content_key
            
            content_image = sample_images[content_key]
            style_image = sample_images[style_key]
            
            # Perform style transfer
            print(f'Applying style from {style_key} to {content_key}...')
            result_image = style_transfer_service.apply_style_only(content_image, f"Interior in the style of {style_key}")
            
            # Display results
            display_images(
                [content_image, style_image, result_image],
                titles=[f'Content: {content_key}', f'Style: {style_key}', 'Style Transfer Result'],
                figsize=(18, 6)
            )
            
            # Save the result
            save_output(result_image, f'{content_key}_styled_as_{style_key}')
            
            # Structure preservation if segmentation was successful
            if 'segmentation_handler' in locals() and 'structure_mask' in locals():
                print("\n=== STEP 4: Structure Preservation ===")
                preserved_result = segmentation_handler.apply_structure_preservation(
                    content_image, result_image, structure_mask
                )
                
                # Display results
                display_images(
                    [content_image, result_image, preserved_result],
                    titles=['Original Image', 'Direct Style Transfer', 'Structure-Preserved Result'],
                    figsize=(18, 6)
                )
                
                # Save the structure-preserved result
                save_output(preserved_result, f'{content_key}_preserved_style_{style_key}')
                
                print('✅ Structure preservation complete!')
                
                # Evaluation
                try:
                    from src.evaluation import VisualEvaluationService, ImageEvaluator
                    
                    print("\n=== STEP 5: Evaluation ===")
                    evaluation_service = VisualEvaluationService()
                    image_evaluator = ImageEvaluator()
                    
                    # Evaluate direct style transfer
                    direct_metrics = evaluation_service.calculate_metrics(content_image, result_image)
                    print('Direct Style Transfer Metrics:')
                    print(f'SSIM: {direct_metrics["ssim"]:.4f} (higher is better)')
                    print(f'MSE: {direct_metrics["mse"]:.4f} (lower is better)')
                    
                    # Evaluate structure-preserved result
                    preserved_metrics = evaluation_service.calculate_metrics(content_image, preserved_result)
                    print('\nStructure-Preserved Style Transfer Metrics:')
                    print(f'SSIM: {preserved_metrics["ssim"]:.4f} (higher is better)')
                    print(f'MSE: {preserved_metrics["mse"]:.4f} (lower is better)')
                    
                    # Calculate structure preservation score
                    preservation_score = evaluation_service.calculate_preservation_score(
                        content_image, preserved_result, structure_mask
                    )
                    print(f'\nStructure Preservation Score: {preservation_score:.4f} (higher is better)')
                    
                    # Detailed evaluation
                    structure_eval = image_evaluator.evaluate_structure_preservation(
                        content_image, preserved_result, structure_mask
                    )
                    print('\nDetailed Structure Preservation Evaluation:')
                    for key, value in structure_eval.items():
                        if isinstance(value, float):
                            print(f'{key}: {value:.4f}')
                        else:
                            print(f'{key}: {value}')
                    
                    # Create visualization of the differences
                    print('\nGenerating difference visualization...')
                    diff_vis = image_evaluator.visualize_differences(content_image, preserved_result)
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(diff_vis)
                    plt.title('Difference Visualization')
                    plt.axis('off')
                    plt.show()
                    
                    # Save the visualization
                    save_output(diff_vis, f'{content_key}_diff_visualization')
                    
                    print('✅ Evaluation complete!')
                except Exception as e:
                    print(f'⚠️ Error during evaluation: {str(e)}')
            else:
                print('⚠️ Structure preservation requires successful segmentation.')
        else:
            print('⚠️ Need at least two images for style transfer demonstration.')
    except Exception as e:
        print(f'⚠️ Error during style transfer: {str(e)}')
    
    print("\n=== Demo Complete ===")
    print(f"All outputs saved to '{output_dir}' directory")

if __name__ == "__main__":
    run_demo()
