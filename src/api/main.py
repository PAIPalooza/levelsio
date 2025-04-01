"""
Main FastAPI application for Interior Style Transfer API.

This module sets up the FastAPI application and includes all routes.
"""

import os
import logging
from fastapi import FastAPI, Depends, Header, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv

from src.api.routes import api_router
from src.api.routes import style_transfer, segmentation, evaluation, prompts
from src.api.core.auth import verify_api_key, verify_x_api_key
from src.api.core.error_handlers import register_exception_handlers
from src.api.models.prompts import (
    StyleDetails,
    StylePromptRequest,
    StylePromptResponse
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api.main")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Interior Style Transfer API",
        description="API for transferring interior design styles to room images",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    
    # Register exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "message": "Validation error",
                "errors": exc.errors(),
                "details": {"body": exc.body}
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        # If detail is already a dict with our structure, ensure it has consistent format
        if isinstance(exc.detail, dict):
            content = exc.detail.copy()
            
            # If it already has a 'message' key, but no top-level 'errors'
            if "message" in content and "errors" not in content:
                # Extract errors from details if available
                if "details" in content and "errors" in content["details"]:
                    content["errors"] = content["details"]["errors"]
                else:
                    content["errors"] = []
                    
            # If it doesn't have a 'message' key but has 'details'
            elif "message" not in content and "details" in content:
                content["message"] = "Error occurred"
                content["errors"] = []
                
            # Make sure both fields are present
            if "details" not in content:
                content["details"] = {}
            if "errors" not in content:
                content["errors"] = []
                
            return JSONResponse(
                status_code=exc.status_code,
                content=content
            )
        
        # Otherwise format into our standard structure
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "message": str(exc.detail),
                "errors": [],
                "details": {}
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Internal server error",
                "errors": [str(exc)],
                "details": {"type": str(type(exc).__name__)}
            }
        )
    
    # Mount static files for serving images
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception as e:
        logger.warning(f"Static files directory not mounted: {e}")
    
    # Register error handlers
    register_exception_handlers(app)
    
    # Include API router for all endpoints
    app.include_router(api_router)
    
    # Define root endpoint
    @app.get("/")
    async def root():
        """Root endpoint that returns API information."""
        return {
            "name": "Interior Style Transfer API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "endpoints": {
                "style_transfer": "/api/v1/style/transfer",
                "segmentation": "/api/v1/segmentation/segment",
                "evaluation": "/api/v1/evaluation/style-score",
                "prompts": "/api/v1/prompts/list",
            },
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        return {"status": "healthy", "service": "interior-style-api"}
    
    # API Key verification endpoint
    @app.get("/verify-key")
    async def verify_key(api_key: str = Depends(verify_api_key)):
        """Verify that an API key is valid."""
        return {"valid": True, "message": "API key is valid"}
    
    # Include all route modules
    app.include_router(style_transfer.router, prefix="/api/v1/style", tags=["Style Transfer"])
    app.include_router(segmentation.router, prefix="/api/v1/segmentation", tags=["Segmentation"])
    app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["Evaluation"])
    app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["Prompts"])
    
    # Add simple root-level routes for convenience
    app.include_router(style_transfer.router, prefix="/style", tags=["Style Transfer"])
    app.include_router(segmentation.router, prefix="/segmentation", tags=["Segmentation"])
    app.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
    app.include_router(prompts.router, prefix="/prompts", tags=["Prompts"])
    
    # Customize OpenAPI schema
    def custom_openapi():
        """Customize OpenAPI schema with security information and tag descriptions."""
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add API key security scheme
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyHeader": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication",
            }
        }
        
        # Add tag descriptions for better organization
        openapi_schema["tags"] = [
            {
                "name": "Style Transfer",
                "description": "Endpoints for applying interior design styles to room images"
            },
            {
                "name": "Segmentation",
                "description": "Endpoints for segmenting interior images into structural components"
            },
            {
                "name": "Evaluation",
                "description": "Endpoints for evaluating the quality of style transfer results"
            },
            {
                "name": "Prompts",
                "description": "Endpoints for managing and generating style prompt templates"
            }
        ]
        
        # Apply security to all operations
        for path in openapi_schema["paths"]:
            for method in openapi_schema["paths"][path]:
                if method != "options":  # Skip options for CORS
                    openapi_schema["paths"][path][method]["security"] = [
                        {"ApiKeyHeader": []}
                    ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app


# Create the application instance
app = create_app()
