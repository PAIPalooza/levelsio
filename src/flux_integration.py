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
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FluxClient:
    """
    Client for interacting with fal.ai's Flux API for interior style transfer.
    
    Manages API requests, image encoding/decoding, and response handling for
    applying style transfers to interior images.
    """
    
    # API endpoints
    FAL_API_URL = "https://api.fal.ai/v1/inference"
    
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
        
        self.model_key = model_key
        self.headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"FluxClient initialized with model: {model_key}")
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Convert a NumPy image array to base64-encoded string.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Base64-encoded image string
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
            
        # Save to BytesIO buffer and encode
        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _decode_image(self, base64_str: str) -> np.ndarray:
        """
        Convert a base64-encoded image string to NumPy array.
        
        Args:
            base64_str: Base64-encoded image string
            
        Returns:
            Image as NumPy array
        """
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_str)
        
        # Convert to PIL Image and then to NumPy array
        image = Image.open(BytesIO(image_bytes))
        return np.array(image)
    
    def _make_request(
        self, 
        payload: Dict[str, Any], 
        max_retries: int = 3, 
        retry_delay: float = 2.0
    ) -> Dict[str, Any]:
        """
        Make an API request to fal.ai with retry logic.
        
        Args:
            payload: Request payload
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If all retry attempts fail
        """
        endpoint = f"{self.FAL_API_URL}/{self.model_key}"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request to Flux (attempt {attempt + 1}/{max_retries})")
                response = requests.post(endpoint, headers=self.headers, json=payload)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_info = response.json() if response.text else {"status_code": response.status_code}
                    logger.warning(f"API request failed: {error_info}")
                    
                    # Check if we should retry based on error type
                    if response.status_code in [429, 500, 502, 503, 504]:
                        # Throttling or server errors - retry
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        # Increase delay for next attempt
                        retry_delay *= 1.5
                        continue
                    else:
                        # Client errors - don't retry
                        error_message = f"API error: {response.status_code} - {error_info}"
                        raise Exception(error_message)
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next attempt
                    retry_delay *= 1.5
                else:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise
        
        # This should not be reached due to the exception above, but just in case
        raise Exception("All retry attempts failed")
    
    def apply_style_transfer(
        self, 
        image: np.ndarray, 
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Apply style transfer to an interior image using Flux.
        
        Args:
            image: Input image as NumPy array
            prompt: Style description text prompt
            negative_prompt: What to avoid in the generation
            guidance_scale: How closely to follow the prompt (higher = closer)
            num_inference_steps: Number of diffusion steps (higher = more detail)
            seed: Random seed for reproducibility
            max_retries: Maximum number of retry attempts
            
        Returns:
            Styled image as NumPy array
        """
        # Encode image to base64
        image_base64 = self._encode_image(image)
        
        # Build API payload
        payload = {
            "image": image_base64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        # Make the API request
        try:
            logger.info(f"Applying style transfer with prompt: {prompt}")
            response = self._make_request(payload, max_retries=max_retries)
            
            # Extract the first image from response
            if "images" in response and response["images"]:
                styled_image_base64 = response["images"][0]
                styled_image = self._decode_image(styled_image_base64)
                
                # Log timing information if available
                if "timing" in response:
                    logger.info(f"Style transfer completed in {response['timing'].get('total', 0):.2f} seconds")
                
                return styled_image
            else:
                raise ValueError("No images found in API response")
                
        except Exception as e:
            logger.error(f"Style transfer failed: {str(e)}")
            raise
    
    def generate_style_variations(
        self, 
        image: np.ndarray, 
        prompt: str, 
        num_variations: int = 3,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        max_retries: int = 3
    ) -> List[np.ndarray]:
        """
        Generate multiple style variations for an interior image.
        
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
        
        logger.info(f"Generating {num_variations} style variations with prompt: {prompt}")
        
        for i in range(num_variations):
            # Use different seeds for each variation if a seed is provided
            current_seed = None if seed is None else seed + i
            
            try:
                # Generate a variation
                logger.info(f"Generating variation {i+1}/{num_variations}")
                variation = self.apply_style_transfer(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=current_seed,
                    max_retries=max_retries
                )
                
                variations.append(variation)
                
            except Exception as e:
                logger.error(f"Failed to generate variation {i+1}: {str(e)}")
                # Continue with next variation instead of failing completely
        
        return variations
