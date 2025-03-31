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
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from src.api.routes import api_router
from src.api.core.auth import verify_api_key
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
        title="Interior Style Transfer API",
        description="""
        API for transforming interior photos using style transfer techniques.
        This API provides endpoints for style transfer, segmentation, evaluation,
        and prompt templates for various interior design styles.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify your frontend domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # Include API router
    app.include_router(api_router, prefix="/api/v1")
    
    # Add health check endpoint
    @app.get(
        "/health", 
        status_code=status.HTTP_200_OK,
        summary="API health check",
        description="Check if the API is up and running"
    )
    async def health_check():
        """
        Perform a health check on the API.
        
        Returns:
            Health status
        """
        return {"status": "healthy", "version": "1.0.0"}
    
    # Customize OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="Interior Style Transfer API",
            version="1.0.0",
            description="""
            API for transforming interior photos using style transfer techniques.
            
            This API provides endpoints for:
            - Style transfer: Transform interior photos into different styles
            - Segmentation: Segment interior photos into regions
            - Evaluation: Evaluate style transfer results
            - Prompt Templates: Access and use interior design style templates
            """,
            routes=app.routes,
        )
        
        # Add model schemas explicitly to avoid reference errors
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
            
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}
            
        # Add model schemas explicitly
        model_schemas = {
            "StyleDetails": StyleDetails.schema(),
            "StylePromptRequest": StylePromptRequest.schema(),
            "StylePromptResponse": StylePromptResponse.schema(),
            "CustomPromptRequest": CustomPromptRequest.schema(),
            "CategoryInfo": CategoryInfo.schema(),
            "StylePreview": StylePreview.schema(),
            "StyleListResponse": StyleListResponse.schema()
        }
        
        for name, schema in model_schemas.items():
            openapi_schema["components"]["schemas"][name] = schema
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    logger.info("API server initialized successfully")
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
