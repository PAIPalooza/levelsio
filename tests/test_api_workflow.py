"""
Test script to verify the complete workflow of the Interior Style Transfer API.

This script tests the full API workflow including:
1. Style transfer endpoint
2. Segmentation endpoint
3. Evaluation endpoint

It uses the actual API running locally to ensure everything works properly.
"""

import os
import sys
import requests
import numpy as np
from PIL import Image
import base64
import io
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
API_BASE_URL = "http://127.0.0.1:8000"
API_KEY = "api_key_for_testing"  # Using the correct API key from auth.py
SAMPLE_IMAGE_PATH = "tests/fixtures/sample_room.jpg"  # Path to a test image
OUTPUT_DIR = "test_outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 with data URI."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Get the file extension to determine MIME type
    extension = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg"  # Default to JPEG
    if extension == ".png":
        mime_type = "image/png"
    elif extension == ".gif":
        mime_type = "image/gif"
    
    # Return as data URI
    return f"data:{mime_type};base64,{encoded_string}"

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    # Extract the base64 data if it's a data URI
    if base64_string.startswith('data:image/'):
        base64_string = base64_string.split(',', 1)[1]
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Convert to PIL image
    return Image.open(io.BytesIO(image_bytes))

def save_base64_image(base64_string: str, output_path: str):
    """Save a base64 encoded image to a file."""
    image = decode_base64_to_image(base64_string)
    image.save(output_path)
    logger.info(f"Saved image to {output_path}")

def test_style_transfer_endpoint() -> Dict[str, Any]:
    """Test the style transfer endpoint."""
    logger.info("Calling style transfer endpoint...")
    
    # Prepare the request payload
    payload = {
        "image": encode_image_to_base64(SAMPLE_IMAGE_PATH),
        "style": "modern",
        "preserve_structure": True
    }
    
    # Set headers with API key
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    # Send the request
    url = f"{API_BASE_URL}/api/v1/style/apply"
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Process the response
        result = response.json()
        
        # Save the styled image
        if "styled_image" in result:
            save_base64_image(
                result["styled_image"], 
                os.path.join(OUTPUT_DIR, "styled_image.jpg")
            )
        
        logger.info("Style transfer successful!")
        return result
    except Exception as e:
        logger.error(f"Style transfer test failed: {str(e)}")
        return {}

def test_segmentation_endpoint() -> Dict[str, Any]:
    """Test the segmentation endpoint."""
    logger.info("Calling segmentation endpoint...")
    
    # Prepare the request payload
    payload = {
        "image": encode_image_to_base64(SAMPLE_IMAGE_PATH),
        "include_visualization": True
    }
    
    # Set headers with API key
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    # Send the request
    url = f"{API_BASE_URL}/api/v1/segmentation/segment"
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Process the response
        result = response.json()
        logger.info(f"Segmentation successful! Status code: {response.status_code}")
        
        # Check if the response contains the expected fields
        if "structure_mask" in result:
            save_base64_image(
                result["structure_mask"], 
                os.path.join(OUTPUT_DIR, "structure_mask.png")
            )
        else:
            logger.error("Response does not contain a structure_mask field")
            logger.info(f"Response content: {result}")
        
        # Save visualization if present
        if "visualization" in result and result["visualization"]:
            save_base64_image(
                result["visualization"], 
                os.path.join(OUTPUT_DIR, "segmentation_visualization.jpg")
            )
        
        return result
    except Exception as e:
        logger.error(f"Segmentation test failed: {str(e)}")
        return {}

def test_evaluation_endpoint(original_image: str, styled_image: str) -> Dict[str, Any]:
    """Test the evaluation endpoint."""
    logger.info("Calling evaluation endpoint...")
    
    # Prepare the request payload
    payload = {
        "original_image": original_image,
        "stylized_image": styled_image,
        "include_visualization": True
    }
    
    # Set headers with API key
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    # Send the request
    url = f"{API_BASE_URL}/api/v1/evaluation/compare"
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Process the response
        result = response.json()
        
        # Save the visualization if present
        if "visualization" in result and result["visualization"]:
            save_base64_image(
                result["visualization"], 
                os.path.join(OUTPUT_DIR, "evaluation_visualization.jpg")
            )
        
        logger.info(f"Evaluation successful! Metrics: {json.dumps(result['metrics'], indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Evaluation test failed: {str(e)}")
        return {}

def run_complete_workflow_test():
    """Run a complete workflow test through all API endpoints."""
    logger.info("Starting API workflow tests...")
    
    # Step 1: Test style transfer
    logger.info("\n===== Testing Style Transfer Endpoint =====")
    style_result = test_style_transfer_endpoint()
    style_test_passed = "styled_image" in style_result
    
    # Step 2: Test segmentation
    logger.info("\n===== Testing Segmentation Endpoint =====")
    seg_result = test_segmentation_endpoint()
    seg_test_passed = "structure_mask" in seg_result
    
    # Step 3: Test evaluation (only if style transfer worked)
    logger.info("\n===== Testing Evaluation Endpoint =====")
    eval_test_passed = False
    
    if style_test_passed:
        original_image = encode_image_to_base64(SAMPLE_IMAGE_PATH)
        styled_image = style_result["styled_image"]
        
        eval_result = test_evaluation_endpoint(original_image, styled_image)
        eval_test_passed = "metrics" in eval_result
    else:
        logger.warning("Skipping evaluation test because style transfer failed")
    
    # Print summary
    logger.info("\n===== Test Results =====")
    logger.info(f"Style Transfer: {'✅ PASSED' if style_test_passed else '❌ FAILED'}")
    logger.info(f"Segmentation:    {'✅ PASSED' if seg_test_passed else '❌ FAILED'}")
    logger.info(f"Evaluation:      {'✅ PASSED' if eval_test_passed else '❌ FAILED'}")
    
    if not (style_test_passed and seg_test_passed and eval_test_passed):
        logger.warning("\n⚠️ Some tests failed. Please check the logs for details.")
    else:
        logger.info("\n🎉 All tests passed! The API workflow is functioning correctly.")

if __name__ == "__main__":
    run_complete_workflow_test()
