# Interior Style Transfer API Integration Documentation

## Overview

This document provides detailed information about the integration of the Interior Style Transfer API with the Fal.ai service. The API provides endpoints for style transfer, segmentation, and evaluation of styled interior images.

## API Endpoints

### Style Transfer

**Endpoint**: `/api/v1/style/apply`

**Description**: Applies a style to an interior image based on the specified style prompt.

**Request**:
```json
{
  "image": "base64_encoded_image",
  "style": "modern",
  "preserve_structure": true,
  "structure_threshold": 0.8
}
```

**Response**:
```json
{
  "styled_image": "base64_encoded_styled_image",
  "processing_time": 2.5
}
```

### Segmentation

**Endpoint**: `/api/v1/segmentation/segment`

**Description**: Segments an interior image into structural elements (walls, floors, etc.) and furniture.

**Request**:
```json
{
  "image": "base64_encoded_image",
  "segmentation_type": "structure",
  "include_visualization": true
}
```

**Response**:
```json
{
  "structure_mask": "base64_encoded_structure_mask",
  "furniture_mask": "base64_encoded_furniture_mask",
  "decor_mask": "base64_encoded_decor_mask",
  "visualization": "base64_encoded_visualization",
  "segments": {
    "walls": 0.45,
    "floor": 0.25,
    "furniture": 0.3
  }
}
```

### Evaluation

**Endpoint**: `/api/v1/evaluation/metrics` or `/api/v1/evaluation/compare`

**Description**: Evaluates the quality of style transfer by comparing the original and styled images.

**Request**:
```json
{
  "original_image": "base64_encoded_original_image",
  "styled_image": "base64_encoded_styled_image",
  "metrics": ["ssim", "mse", "psnr"],
  "include_visualization": true
}
```

**Response**:
```json
{
  "metrics": {
    "ssim": 0.85,
    "mse": 120.5,
    "psnr": 28.3
  },
  "visualization": "base64_encoded_comparison_visualization",
  "structure_preservation_score": 0.92
}
```

## Integration with Fal.ai

The API integrates with the Fal.ai service through the `fal_client` library. This integration allows for high-quality style transfer using state-of-the-art models.

### Authentication

The API uses an API key for authentication with both the client and the Fal.ai service. The Fal.ai API key is stored in the `.env` file as `FAL_KEY`.

### Error Handling

The API provides detailed error messages for various scenarios, including:
- Invalid input format
- Missing image data
- Network connectivity issues
- Fal.ai service errors

## Workflow Testing

A comprehensive test workflow is available in `tests/test_api_workflow.py`. This script tests all three main API endpoints in sequence:

1. Style Transfer: Applies a style to a test image
2. Segmentation: Generates structure masks for the original image
3. Evaluation: Compares the original and styled images

## Implementation Details

### Style Transfer Service

The style transfer service uses Fal.ai's Flux API to apply styles to interior images. It supports:
- Various interior design styles (modern, minimalist, industrial, etc.)
- Structure preservation to maintain architectural elements
- Customizable style strength and guidance

### Segmentation Handler

The segmentation handler uses the Segment Anything Model (SAM) to identify structural elements in interior images. It provides:
- Structure masks (walls, floors, ceilings)
- Furniture masks
- Decor masks
- Visualizations of segmented elements

### Evaluation Service

The evaluation service compares original and styled images using various metrics:
- Structural Similarity Index (SSIM)
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structure preservation assessment

## Future Improvements

Potential improvements for the API include:
- Batch processing for multiple images
- Additional style options and presets
- Enhanced structure preservation techniques
- More detailed segmentation for specific interior elements
- Real-time style preview capabilities

## Conclusion

The Interior Style Transfer API provides a robust solution for applying professional interior design styles to images while maintaining structural integrity. The integration with Fal.ai ensures high-quality results with minimal latency.
