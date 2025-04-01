# Interior Style Transfer - Prompt Templates API

This directory contains the Prompt Templates API implementation for the Interior Style Transfer project. The API provides access to style prompts, templates, and generation capabilities for interior design styles.

## Features

- **Style Templates Management**: Access to predefined style templates
- **Category-based Style Organization**: Styles organized by design categories
- **Prompt Generation**: Generate style prompts for style transfer operations
- **Custom Prompt Creation**: Create custom prompts with specific design elements
- **Preview Images**: Access to preview images for each style

## Running the API Server

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python run_api_server.py
```

3. Optional command-line arguments:
```bash
python run_api_server.py --host 0.0.0.0 --port 8000 --log-level info --reload
```

## API Documentation

The API documentation is available at:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

For detailed endpoint information, see [docs/prompt_templates_api.md](docs/prompt_templates_api.md).

## Example Usage

An example script is provided in the `examples` directory to demonstrate how to use the API:

```bash
# Using the API endpoints
python examples/use_prompt_api.py --list-styles --style "Scandinavian" --room-type "living room"

# Using the local StylePromptManager directly
python examples/use_prompt_api.py --local --list-styles --category "modern"
```

## Testing

Run the tests to ensure everything is working correctly:

```bash
pytest -xvs tests/test_prompt_api.py
```

## Implementation Architecture

- `src/prompts/style_manager.py`: Core class for managing style prompts and templates
- `src/prompts/style_templates.json`: JSON file with style templates data
- `src/api/routes/prompts.py`: API endpoints for the prompt templates feature
- `src/api/models/prompts.py`: Pydantic models for API request/response validation

## Extending the API

To add new styles:
1. Update the `style_templates.json` file with new style definitions
2. Restart the API server

To add custom prompt formatters:
```python
from src.prompts import StylePromptManager

# Create a custom formatter
def custom_formatter(style: str, room_type: str, details: str) -> str:
    return f"A custom {style} {room_type} with special features, {details}"

# Register the formatter
manager = StylePromptManager()
manager.register_custom_formatter("CustomStyle", custom_formatter)
```

## Development Status

This feature is implemented following the Semantic Seed Coding Standards (SSCS) with Test-Driven Development approach and comprehensive documentation.
