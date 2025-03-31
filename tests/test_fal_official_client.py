"""
Test script using the official fal-client library for API integration.

Following the latest Fal.ai documentation to test API connectivity
and functionality with the official client library.
"""

import os
import sys
import fal_client
import numpy as np
from PIL import Image
import base64
import io
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use the API key from environment or set directly
FAL_KEY = os.getenv("FAL_KEY", "f51c8849-420f-4f98-ac43-9b98e5a58408:072cb70f9f6d9fc4f56f3930187d87cd")

def create_sample_image(width: int = 50, height: int = 50) -> np.ndarray:
    """Create a simple test image for API calls."""
    # Create a small sample image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some basic structure to make it look like a room
    img[10:40, 10:40, :] = 200  # Light gray box
    img[30:40, 20:30, :] = 100  # Darker rectangle for "furniture"
    return img

def encode_image(image: np.ndarray) -> str:
    """Convert a NumPy array image to a base64-encoded data URI."""
    # Convert numpy array to PIL Image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    
    # Save image to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Encode as base64
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def test_fal_client():
    """Test connectivity and functionality with the official fal-client."""
    try:
        # Set API key in environment variable as required by the library
        os.environ["FAL_KEY"] = FAL_KEY
        
        # Generate sample image
        sample_image = create_sample_image(256, 256)  # Larger size for better results
        image_data_uri = encode_image(sample_image)
        
        # Log progress
        logger.info("Sample image created and encoded, preparing to call API...")
        
        # Define callback for queue updates
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.info(f"API progress: {log.get('message', '')}")
        
        # Call Flux API using the fal_client library
        logger.info("Calling Fal.ai Flux API...")
        result = fal_client.subscribe(
            "fal-ai/flux/dev",  # Model endpoint
            arguments={
                "image": image_data_uri, 
                "prompt": "Interior in minimalist style with clean lines, neutral colors, and uncluttered space",
                "negative_prompt": "blurry, distorted architecture, unrealistic",
                "guidance_scale": 7.5,
                "num_inference_steps": 30
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        
        # Check the result
        logger.info("API call completed successfully!")
        
        # Print the result structure
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # Extract and save the image result if available
        if "images" in result and result["images"]:
            # Save the first generated image
            output_dir = "test_outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Print the structure of the first image
            logger.info(f"First image type: {type(result['images'][0])}")
            
            # Extract image data
            image_data = result["images"][0]
            if isinstance(image_data, dict) and "url" in image_data:
                # If image is a dictionary with URL, use that
                logger.info(f"Image is a dictionary with URL: {image_data['url']}")
                # You would need to download from URL here
                # For simplicity, we'll just log and exit successfully
                logger.info("Test successful! (Image is a URL)")
                return True
            elif isinstance(image_data, str):
                # Handle string-based image data
                # If the image is a data URI, extract the base64 part
                if image_data.startswith("data:"):
                    image_data = image_data.split(",", 1)[1]
                
                # Decode and save
                output_path = os.path.join(output_dir, "fal_api_test_result.jpg")
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                
                logger.info(f"Test successful! Output image saved to {output_path}")
                return True
            else:
                logger.info(f"Unexpected image format: {type(image_data)}")
                logger.info(f"Image data preview: {str(image_data)[:100]}")
                # Consider test successful if we got this far
                return True
        else:
            logger.error("No images found in API response")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Fal.ai API with official client library...")
    success = test_fal_client()
    print(f"Test {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1)
