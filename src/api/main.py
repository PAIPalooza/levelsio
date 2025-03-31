"""
Main FastAPI application module for Interior Style Transfer API.

This module configures and launches the FastAPI application with
OpenAPI/Swagger documentation. It follows the Semantic Seed Coding
Standards (SSCS) for maintainability, security, and documentation.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.api.core.config import settings
from src.api.routes import style_transfer, segmentation, evaluation

# Initialize FastAPI app with metadata for Swagger/OpenAPI
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
    
    ## Authentication
    
    API endpoints require authentication via API key header.
    """,
    version="1.0.0",
    docs_url="/docs",  # Use standard /docs URL for Swagger UI
    redoc_url="/redoc",  # Enable ReDoc at standard /redoc URL
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API route modules
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

# Custom OpenAPI schema with enhanced documentation
def custom_openapi():
    """Generate a custom OpenAPI schema with enhanced metadata."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add API key security scheme
    openapi_schema["components"] = {
        "securitySchemes": {
            "APIKeyHeader": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication. Request from administrator."
            }
        }
    }
    
    # Apply security globally to all endpoints
    openapi_schema["security"] = [{"APIKeyHeader": []}]
    
    # Add examples, servers, etc.
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Set custom OpenAPI schema
app.openapi = custom_openapi

# Root endpoint
@app.get("/", tags=["status"])
def read_root():
    """API root endpoint returns basic service information."""
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",  # Update to point to standard docs URL
        "redoc": "/redoc"
    }
