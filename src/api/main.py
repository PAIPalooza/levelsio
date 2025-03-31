"""
Main API server for Interior Style Transfer POC.

This module implements the FastAPI server for the Interior Style Transfer
API, following Behavior-Driven Development principles from the
Semantic Seed Coding Standards (SSCS) for enterprise-quality code.
"""

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from src.api.core.config import settings
from src.api.routes import api_router
from src.api.routes import style_transfer, segmentation, evaluation
from src.api.core.auth import verify_api_key, verify_x_api_key
from src.api.models.prompts import (
    StyleDetails,
    StylePromptRequest,
    StylePromptResponse,
    CustomPromptRequest,
    CategoryInfo,
    StylePreview,
    StyleListResponse
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="""
        Interior Style Transfer API provides endpoints for applying various interior design 
        styles to room images, with options for structure preservation and style evaluation.
        
        ## Features
        
        * **Style Transfer**: Apply design styles to interior images
        * **Segmentation**: Extract room structure for preservation
        * **Evaluation**: Measure quality and effectiveness of style transfer
        * **Gallery Generation**: Create collections of styled interiors
        * **Prompt Templates**: Generate style-specific prompts for interior design
        
        ## Authentication
        
        API endpoints require authentication via API key header.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files directory if exists
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("Static files directory mounted")
    except Exception as e:
        logger.warning(f"Static files directory not mounted: {e}")
    
    # Include API router for all endpoints
    app.include_router(api_router)
    
    # Also include individual routers for standard endpoints
    app.include_router(
        style_transfer.router,
        prefix=f"{settings.API_V1_STR}/style",
        tags=["style-transfer"]
    )

    app.include_router(
        segmentation.router,
        prefix=f"{settings.API_V1_STR}/segmentation",
        tags=["segmentation"]
    )

    app.include_router(
        evaluation.router,
        prefix=f"{settings.API_V1_STR}/evaluation",
        tags=["evaluation"]
    )
    
    # Root endpoint for API status
    @app.get("/", tags=["status"])
    def read_root():
        """API root endpoint returns basic service information."""
        return {
            "service": settings.PROJECT_NAME,
            "version": "1.0.0",
            "status": "operational",
            "documentation": "/docs",
            "redoc": "/redoc"
        }
    
    return app


# Create app instance
app = create_app()


# Custom OpenAPI schema with enhanced documentation
def custom_openapi():
    """Generate a custom OpenAPI schema with enhanced metadata."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Interior Style Transfer API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add components section if it doesn't exist
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    # Add schemas section if it doesn't exist
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    # Add API key security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication. Request from administrator."
        },
        "ApiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "api-key",
            "description": "API key for authentication."
        },
        "ApiKeyQuery": {
            "type": "apiKey",
            "in": "query",
            "name": "api_key",
            "description": "API key for authentication as a query parameter."
        }
    }
    
    # Apply security globally to all endpoints
    openapi_schema["security"] = [
        {"APIKeyHeader": []}, 
        {"ApiKey": []},
        {"ApiKeyQuery": []}
    ]
    
    # Manually add models that are defined but might not be detected
    model_schemas = {
        "StyleDetails": {
            "title": "StyleDetails",
            "type": "object",
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "description": {"title": "Description", "type": "string"},
                "category": {"title": "Category", "type": "string"}
            },
            "required": ["name", "description", "category"]
        },
        "StylePromptRequest": {
            "title": "StylePromptRequest",
            "type": "object",
            "properties": {
                "style": {"title": "Style", "type": "string"},
                "room_type": {"title": "Room Type", "type": "string", "default": "interior"},
                "details": {"title": "Details", "type": "string", "nullable": True}
            },
            "required": ["style"]
        },
        "StylePromptResponse": {
            "title": "StylePromptResponse",
            "type": "object",
            "properties": {
                "prompt": {"title": "Prompt", "type": "string"},
                "style": {"title": "Style", "type": "string"}
            },
            "required": ["prompt", "style"]
        },
        "CustomPromptRequest": {
            "title": "CustomPromptRequest",
            "type": "object",
            "properties": {
                "prompt_template": {"title": "Prompt Template", "type": "string"},
                "style": {"title": "Style", "type": "string"},
                "room_type": {"title": "Room Type", "type": "string", "default": "interior"},
                "details": {"title": "Details", "type": "string", "nullable": True}
            },
            "required": ["prompt_template", "style"]
        },
        "CategoryInfo": {
            "title": "CategoryInfo",
            "type": "object",
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "description": {"title": "Description", "type": "string"}
            },
            "required": ["name", "description"]
        },
        "StylePreview": {
            "title": "StylePreview",
            "type": "object",
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "category": {"title": "Category", "type": "string"},
                "preview": {"title": "Preview", "type": "string", "nullable": True}
            },
            "required": ["name", "category"]
        },
        "StyleListResponse": {
            "title": "StyleListResponse",
            "type": "object",
            "properties": {
                "styles": {
                    "title": "Styles",
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/StylePreview"}
                },
                "categories": {
                    "title": "Categories",
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/CategoryInfo"}
                }
            },
            "required": ["styles", "categories"]
        }
    }
    
    # Add the schemas to the OpenAPI components
    for schema_name, schema in model_schemas.items():
        openapi_schema["components"]["schemas"][schema_name] = schema
    
    # Add examples, servers, etc.
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Set custom OpenAPI schema
app.openapi = custom_openapi


# Use API environment variables if provided, otherwise use defaults
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))


if __name__ == "__main__":
    """Run the API server directly if this module is executed."""
    import uvicorn
    
    logger.info(f"Starting API server at {API_HOST}:{API_PORT}")
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
