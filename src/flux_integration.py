"""
Flux integration module for Interior Style Transfer POC.

This module handles the integration with fal.ai's Flux model for
style transfer of interior images, managing API requests, responses,
and error handling.
"""

import os
import time
import base64
import logging
import requests
import numpy as np
import fal_client
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
from io import BytesIO
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FluxClient:
    """
    Client for interacting with fal.ai's Flux API for interior style transfer.
    
    Manages API requests, image encoding/decoding, and response handling for
    applying style transfers to interior images.
    """
    
    # Updated to use the official Fal.ai model endpoints
    MODEL_ENDPOINT = "fal-ai/flux/dev"
    
    def __init__(self, api_key: Optional[str] = None, model_key: str = "flux-1"):
        """
        Initialize the Flux client with API credentials.
        
        Args:
            api_key: fal.ai API key (falls back to FAL_KEY env var)
            model_key: Flux model identifier (e.g., 'flux-1')
        """
        self.api_key = api_key or os.environ.get("FAL_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set the FAL_KEY environment variable.")
        
        # Set API key in environment for fal_client library
        os.environ["FAL_KEY"] = self.api_key
        
        self.model_key = model_key
        logger.info(f"FluxClient initialized with model endpoint: {self.MODEL_ENDPOINT}")
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Convert a NumPy array image to a base64-encoded data URI.
        
        Args:
            image: NumPy array containing the image
            
        Returns:
            Base64-encoded data URI string representation of the image
        """
        try:
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Save image to bytes buffer
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            buffer.seek(0)
            
            # Encode as base64
            encoded = base64.b64encode(buffer.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise
    
    def _decode_image(self, image_data: Union[str, Dict[str, Any]]) -> np.ndarray:
        """
        Convert image data to a NumPy array image.
        
        Args:
            image_data: Image data as a string (base64/data URI) or URL dictionary
            
        Returns:
            NumPy array containing the decoded image
        """
        try:
            # Handle dictionary with URL
            if isinstance(image_data, dict) and "url" in image_data:
                image_url = image_data["url"]
                logger.info(f"Downloading image from URL: {image_url}")
                response = httpx.get(image_url, timeout=30)
                response.raise_for_status()
                image_bytes = response.content
                image = Image.open(BytesIO(image_bytes))
                return np.array(image)
            
            # Handle string data (base64 or data URI)
            elif isinstance(image_data, str):
                # Handle data URI format
                if image_data.startswith("data:"):
                    image_data = image_data.split(",", 1)[1]
                
                # Decode base64 string to bytes
                image_bytes = base64.b64decode(image_data)
                
                # Convert bytes to PIL Image
                image = Image.open(BytesIO(image_bytes))
                
                # Convert PIL Image to NumPy array
                return np.array(image)
            else:
                raise ValueError(f"Unsupported image data format: {type(image_data)}")
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            raise
    
    def apply_style_transfer(
        self, 
        image: np.ndarray, 
        prompt: str,
        negative_prompt: str = "blurry, distorted architecture, unrealistic",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> np.ndarray:
        """
        Apply style transfer to an interior image.
        
        Args:
            image: NumPy array containing the image to transform
            prompt: Text description of the desired style
            negative_prompt: Text description of what to avoid
            guidance_scale: How closely to follow the prompt (higher = closer)
            num_inference_steps: Number of diffusion steps to perform
            seed: Random seed for reproducibility
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            NumPy array containing the styled image
        """
        # Encode the input image to base64 data URI
        image_data_uri = self._encode_image(image)
        
        # Prepare arguments for the fal_client API call
        arguments = {
            "image": image_data_uri,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps
        }
        
        # Add seed if provided
        if seed is not None:
            arguments["seed"] = seed
        
        # Define callback for queue updates (for logging progress)
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.info(f"Style transfer progress: {log.get('message', '')}")
        
        # Make API request with retries
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Applying style transfer with prompt: '{prompt}' (attempt {attempt}/{max_retries})")
                
                # Call Fal.ai API using the official client library
                result = fal_client.subscribe(
                    self.MODEL_ENDPOINT,
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )
                
                # Check if the result contains images
                if "images" in result and result["images"] and len(result["images"]) > 0:
                    # Log success and timing information if available
                    if "timings" in result:
                        logger.info(f"Style transfer completed in {result['timings'].get('total', 0):.2f} seconds")
                    
                    # Get the first image from the result
                    image_result = result["images"][0]
                    
                    # Convert the image to a NumPy array and return
                    return self._decode_image(image_result)
                else:
                    raise ValueError("No images found in API response")
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"API request failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry
                    retry_delay *= 1.5
                else:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise
        
        # This should not be reached due to exceptions above
        raise RuntimeError("Failed to apply style transfer after all retry attempts")
    
    def generate_style_variations(
        self, 
        image: np.ndarray, 
        prompt: str, 
        num_variations: int = 3,
        negative_prompt: str = "blurry, distorted architecture, unrealistic",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        max_retries: int = 3
    ) -> List[np.ndarray]:
        """
        Generate multiple style variations of an interior image.
        
        Args:
            image: Input image as NumPy array
            prompt: Style description text prompt
            num_variations: Number of variations to generate
            negative_prompt: What to avoid in the generation
            guidance_scale: How closely to follow the prompt (higher = closer)
            num_inference_steps: Number of diffusion steps (higher = more detail)
            seed: Starting random seed for reproducibility
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of styled images as NumPy arrays
        """
        variations = []
        
        # Use provided seed or generate random one
        current_seed = seed if seed is not None else int(time.time())
        
        # Log what we're doing
        logger.info(f"Generating {num_variations} style variations with prompt: '{prompt}'")
        
        for i in range(num_variations):
            try:
                # Apply style transfer with incrementing seed
                variation = self.apply_style_transfer(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=current_seed + i,
                    max_retries=max_retries
                )
                
                variations.append(variation)
                logger.info(f"Generated variation {i+1}/{num_variations} with seed {current_seed + i}")
                
            except Exception as e:
                logger.error(f"Failed to generate variation {i+1}: {str(e)}")
                # Continue generating remaining variations
        
        return variations
