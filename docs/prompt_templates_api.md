# Interior Style Transfer API: Prompt Templates

## Overview

The Prompt Templates API provides endpoints for retrieving, managing, and generating prompt templates for various interior design styles. This feature enables clients to access predefined style templates and create custom prompts for style transfer operations.

## API Endpoints

### List Available Styles

```http
GET /api/v1/prompts/styles
```

Retrieves a list of all available interior design styles with preview information.

**Query Parameters:**
- `category` (optional): Filter styles by category (e.g., "modern", "traditional")
- `api_key`: Required API key for authentication

**Response:**
```json
{
  "styles": [
    {
      "name": "Scandinavian",
      "category": "modern",
      "description": "Clean lines, light colors, and natural materials",
      "preview_image_url": "/api/v1/prompts/previews/scandinavian.jpg"
    },
    {
      "name": "Industrial",
      "category": "industrial",
      "description": "Raw, unfinished elements with metal and exposed features",
      "preview_image_url": "/api/v1/prompts/previews/industrial.jpg"
    }
  ],
  "categories": ["modern", "industrial", "traditional", "eclectic"],
  "total_styles": 15
}
```

### Get Style Details

```http
GET /api/v1/prompts/styles/{style_name}
```

Retrieves detailed information about a specific interior design style.

**Path Parameters:**
- `style_name`: Name of the style (e.g., "Scandinavian")

**Query Parameters:**
- `api_key`: Required API key for authentication

**Response:**
```json
{
  "name": "Scandinavian",
  "description": "Scandinavian design emphasizes clean lines, simplicity, and functionality combined with light colors and natural materials.",
  "category": "modern",
  "characteristics": [
    "Light color palette",
    "Wood elements", 
    "Clean lines", 
    "Minimal decoration", 
    "Functional furniture"
  ],
  "colors": [
    "White", 
    "Light gray", 
    "Light wood tones", 
    "Black accents", 
    "Muted blues"
  ],
  "materials": [
    "Light wood", 
    "Wool", 
    "Cotton", 
    "Natural textiles"
  ],
  "example_prompt": "A cozy Scandinavian living room with light wooden floors, white walls, simple furniture, and minimal decoration with a few green plants for color accent"
}
```

### List Style Categories

```http
GET /api/v1/prompts/categories
```

Retrieves a list of all available style categories.

**Query Parameters:**
- `api_key`: Required API key for authentication

**Response:**
```json
[
  "modern",
  "traditional",
  "eclectic",
  "minimal",
  "industrial",
  "natural",
  "retro",
  "luxury",
  "coastal"
]
```

### Get Styles by Category

```http
GET /api/v1/prompts/categories/{category_name}
```

Retrieves styles belonging to a specific category.

**Path Parameters:**
- `category_name`: Name of the category (e.g., "modern")

**Query Parameters:**
- `api_key`: Required API key for authentication

**Response:**
```json
{
  "name": "modern",
  "description": "Clean lines, simple color palettes, and emphasis on functionality",
  "styles": [
    "Scandinavian",
    "Contemporary",
    "Minimalist"
  ]
}
```

### Generate Style Prompt

```http
POST /api/v1/prompts/generate-prompt
```

Generates a formatted prompt for a specific interior design style.

**Request Body:**
```json
{
  "style": "Scandinavian",
  "room_type": "living room",
  "details": "with large windows and a view of nature"
}
```

**Query Parameters:**
- `api_key`: Required API key for authentication

**Response:**
```json
{
  "prompt": "A Scandinavian style living room with clean lines, light colors, wood elements, and minimal decoration with large windows and a view of nature",
  "style": "Scandinavian"
}
```

### Generate Custom Prompt

```http
POST /api/v1/prompts/generate-custom-prompt
```

Generates a customized prompt with specific style elements.

**Request Body:**
```json
{
  "base_style": "Minimalist",
  "room_type": "office",
  "colors": "neutral whites and light grays",
  "materials": "wood and metal",
  "lighting": "bright natural lighting",
  "mood": "calm and productive"
}
```

**Query Parameters:**
- `api_key`: Required API key for authentication

**Response:**
```json
{
  "prompt": "A minimalist office with clean lines, neutral colors, uncluttered space, and essential furniture, with neutral whites and light grays, featuring wood and metal, with bright natural lighting, creating a calm and productive atmosphere",
  "style": "Minimalist"
}
```

### Get Style Preview Image

```http
GET /api/v1/prompts/previews/{filename}
```

Retrieves a preview image for a specific interior design style.

**Path Parameters:**
- `filename`: Name of the preview image file (e.g., "scandinavian_preview.jpg")

**Query Parameters:**
- `api_key`: Required API key for authentication

**Response:**
- Image file (JPEG or PNG)

## Error Responses

All endpoints return standard HTTP error codes:

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Invalid or missing API key
- `404 Not Found`: Requested resource not found
- `500 Internal Server Error`: Server-side error

Example error response:
```json
{
  "detail": "Style 'NonExistentStyle' not found"
}
```

## Usage Examples

### Retrieve available styles and generate a prompt

```python
import requests

# API credentials
API_URL = "http://localhost:8000/api/v1"
API_KEY = "your-api-key"

# Get available styles
response = requests.get(
    f"{API_URL}/prompts/styles",
    params={"api_key": API_KEY}
)

if response.status_code == 200:
    styles = response.json()
    print(f"Available styles: {len(styles['styles'])}")
    
    # Generate a prompt for the first style
    if styles['styles']:
        first_style = styles['styles'][0]['name']
        prompt_response = requests.post(
            f"{API_URL}/prompts/generate-prompt",
            json={
                "style": first_style,
                "room_type": "living room",
                "details": "with large windows"
            },
            params={"api_key": API_KEY}
        )
        
        if prompt_response.status_code == 200:
            prompt_data = prompt_response.json()
            print(f"Generated prompt: {prompt_data['prompt']}")
```

## Implementation Details

The Prompt Templates API is implemented using FastAPI and follows the Semantic Seed Coding Standards (SSCS) with Behavior-Driven Development (BDD) principles. The API is designed to be:

- **Testable**: Comprehensive test coverage with pytest
- **Maintainable**: Well-documented code with clear responsibilities
- **Secure**: API key authentication for all endpoints
- **Extensible**: Easy to add new styles and categories

## Related Components

- `StylePromptManager`: Core class that manages style templates
- `StyleCategory`: Enum defining available style categories
- `style_templates.json`: JSON file containing style prompt templates
