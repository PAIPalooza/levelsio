# Interior Style Transfer POC

Interior Style Transfer POC using Flux on @fal - A proof-of-concept for AI-driven interior design transformations that preserves architectural structure.

## Project Overview

This project demonstrates an AI-driven interior design system that uses Flux via @fal to perform style transfer on input images of interiors. It enables three key transformations while preserving the room's architectural structure:

1. **Style-only transfer**: Change the style/theme of a room while maintaining layout
2. **Style+layout transfer**: Change both style and furniture arrangement
3. **Empty room furnishing**: Add furnishings to an empty interior in a specified style

The system uses auto-segmentation and masking to ensure structural elements (walls, floors, windows) remain intact.

## Features

### Image Segmentation
The application uses the Segment Anything Model (SAM) to identify different elements in interior images:
- Walls
- Floors
- Ceilings
- Windows
- Combined structure masks

### Structure Preservation
When applying style transfer to interior images, the application preserves architectural elements:
- Maintains structural integrity of rooms (walls, floors, ceilings)
- Applies style changes primarily to furniture and decorative elements
- Supports multiple style variations while maintaining the same structure
- Provides scoring mechanism to evaluate structure preservation quality

### Style Transfer via Flux
The application integrates with [fal.ai's Flux API](https://fal.ai/) for state-of-the-art style transfer:
- Generate stunning interior transformations while preserving architectural structure
- Support for customizable style prompts
- Multiple style variations for the same input image
- Automatic error handling and retry mechanisms for API calls
- Built-in network resilience with exponential backoff

### Style-Only Transfer
The application provides specialized support for style-only transfer:
- Apply style changes while preserving original layout and structure
- Pre-built templates for common interior design styles (Scandinavian, Industrial, Mid-Century, etc.)
- Customizable style prompts with room type and additional details
- Structure preservation with segmentation masks
- Support for generating multiple style variations

### Style + Layout Transfer
The application enables comprehensive interior transformations:
- Change both style and furniture arrangement in a single operation
- Rearrange furniture while preserving architectural elements (walls, floors, ceilings)
- Pre-built templates for common room layouts
- Combine style preferences with furniture arrangement preferences
- Generate multiple layout variations to explore design alternatives
- Structure preservation ensures architectural integrity through transformations

### Empty Room Furnishing
The application can transform empty rooms into fully furnished spaces:
- Add furniture and decorative elements to bare rooms
- Apply specified design styles to empty spaces
- Create multiple furnishing variations for the same room
- Preserve architectural elements during the furnishing process
- Pre-built templates for common room types (living room, bedroom, etc.)
- Customizable furniture specifications for precise results

### Data Model & Project Tracking
The application includes a comprehensive data model for tracking and managing interior design projects:
- Track input images, segmentation masks, and generated results
- Maintain detailed metadata for each image and transformation
- Preserve full transformation history and lineage
- Serialize and persist design projects for later review
- Support for project organization with multiple related transformations
- Robust data structures for image, mask, prompt, and result tracking

### Visual Evaluation
The application provides tools for evaluating the quality and structural preservation of generated images:
- SSIM (Structural Similarity Index) metrics to measure structure preservation
- MSE (Mean Squared Error) calculation for pixel-level comparison
- Structure preservation scoring for architectural elements
- Multi-element preservation reports for detailed analysis
- Side-by-side visualizations of original and generated images
- Heatmap visualization of structural differences between images

## API Documentation

The Interior Style Transfer API (v1.0.0) provides a comprehensive set of endpoints for applying interior design transformations, segmenting interior images, and evaluating results. The API follows OpenAPI 3.1 specification and can be accessed at `/openapi.json`.

### Authentication

All API endpoints require authentication using an API key, which should be provided in one of the following ways:
- As an `X-API-Key` header
- As an `api_key` query parameter

```bash
curl -X GET "https://api.example.com/health" -H "X-API-Key: your_api_key_here"
```

### API Categories

The API is organized into four main categories:

#### 1. Style Transfer

Endpoints for applying interior design styles to room images:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/style/transfer` | Apply style transfer to an interior image (File Upload) |
| POST | `/style/apply` | Apply style transfer to an interior image (Base64) |
| POST | `/style/batch` | Apply style transfer to multiple interior images |
| POST | `/style/transfer-json` | Transfer style using JSON input |
| GET | `/style/styles` | Get available style types |

Example request (using JSON):
```json
{
  "image": "base64_encoded_image",
  "style": "scandinavian",
  "room_type": "living_room",
  "preserve_structure": true
}
```

#### 2. Segmentation

Endpoints for segmenting interior images into structural components:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/segmentation/segment` | Segment an interior image (File Upload) |
| POST | `/segmentation/segment-api` | Segment an interior image (Base64) |
| POST | `/segmentation/segment-json` | Segment room using JSON input |
| POST | `/segmentation/batch` | Batch segment multiple interior images |
| GET | `/segmentation/segmentation-types` | Get available segmentation types |

Example request (using JSON):
```json
{
  "image": "base64_encoded_image",
  "segmentation_type": "structure",
  "create_visualization": true
}
```

#### 3. Evaluation

Endpoints for evaluating the quality of style transfer results:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/evaluation/style-score` | Evaluate style transfer result (File Upload) |
| POST | `/evaluation/compare-styles` | Compare multiple style transfer results |
| POST | `/evaluation/compare` | Compare original and styled images |
| POST | `/evaluation/metrics` | Evaluate style transfer quality |
| POST | `/evaluation/gallery-evaluation` | Evaluate a gallery of styled images |
| GET | `/evaluation/available-metrics` | Get available evaluation metrics |
| POST | `/evaluation/style-score-json` | Evaluate style transfer using JSON input |

Example request (using JSON):
```json
{
  "original_image": "base64_encoded_original_image",
  "styled_image": "base64_encoded_styled_image",
  "metrics": ["ssim", "mse", "psnr"],
  "include_visualization": true
}
```

#### 4. Prompts

Endpoints for managing and generating style prompt templates:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/prompts/styles` | Get all available style names |
| GET | `/prompts/styles/{style_name}` | Get details about a specific style |
| GET | `/prompts/categories` | Get all style categories |
| GET | `/prompts/category/{category_name}` | Get styles by category |
| POST | `/prompts/generate-prompt` | Generate a style prompt |
| GET | `/prompts/generate` | Generate a style prompt (GET method) |
| POST | `/prompts/generate` | Generate custom style prompt |
| POST | `/prompts/generate-custom-prompt` | Generate custom prompt |
| GET | `/prompts/previews/{filename}` | Get style preview image |
| GET | `/prompts/list` | List available style templates |

Example request (for generating custom prompt):
```json
{
  "style": "scandinavian",
  "room_type": "bedroom",
  "descriptors": ["bright", "minimal", "cozy"],
  "materials": ["wood", "cotton"]
}
```

#### 5. Utility

Service-related endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API information |
| GET | `/health` | Health check endpoint |
| GET | `/verify-key` | Verify API key validity |

### Alternative Base Paths

All endpoints can also be accessed using the `/api/v1` prefix for better versioning support:

```
/api/v1/style/transfer
/api/v1/segmentation/segment
/api/v1/evaluation/metrics
/api/v1/prompts/styles
```

### Error Handling

All API endpoints return standardized error responses:

```json
{
  "message": "Error description",
  "details": {
    "reason": "error_type",
    "error": "Detailed error information"
  }
}
```

Common HTTP status codes:
- `200 OK`: Request successful
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing or invalid API key
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server-side error

### Rate Limiting

The API implements rate limiting to ensure fair usage:
- 100 requests per minute per API key
- 1000 requests per day per API key

Exceeded rate limits will return a `429 Too Many Requests` response.

## Installation

### Prerequisites
- Python 3.11+
- pip or conda

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/PAIPalooza/levelsio.git
   cd levelsio
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up @fal API key (required for Flux model access):
   - Create an account at [fal.ai](https://fal.ai)
   - Generate an API key
   - Set as environment variable: `export FAL_KEY=your_api_key`

## Usage

1. Prepare an interior image and place it in `assets/test_images/`
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/interior-style-transfer.ipynb
   ```
3. Follow the instructions in the notebook to:
   - Load and segment your image
   - Apply style transformations
   - View and save results

## Project Structure

```
project/
├── assets/             # Image assets
│   ├── test_images/    # Input images
│   ├── outputs/        # Generated outputs
│   └── masks/          # Generated masks
├── src/                # Core Python modules
├── notebooks/          # Jupyter notebooks
├── models/             # Pre-trained model weights
└── tests/              # BDD-style tests
```

## Technologies

- **Segment Anything (SAM)**: For automatic segmentation
- **Flux (via @fal)**: For style transfer generation
- **OpenCV & PIL**: For image processing
- **PyTorch**: Deep learning framework
- **Jupyter**: Interactive notebooks

## Development

This project follows Test-Driven Development (TDD) and Behavior-Driven Development (BDD) methodologies aligned with Semantic Seed Coding Standards. To contribute:

1. Check out the relevant issue
2. Create a feature branch: `git checkout -b feature/issue-id-description`
3. Write tests before implementation
4. Implement your solution
5. Run tests: `pytest`
6. Submit a pull request

## License

[Add license information]

## Acknowledgments

- @fal.ai for API access
- Meta AI for Segment Anything Model
