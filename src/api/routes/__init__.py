"""
API routes package for Interior Style Transfer POC.

This package contains all the API routes for the Interior Style Transfer
application. It follows Behavior-Driven Development principles from the
Semantic Seed Coding Standards (SSCS).
"""

from fastapi import APIRouter
from src.api.routes.style_transfer import router as style_transfer_router
from src.api.routes.segmentation import router as segmentation_router
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.prompts import router as prompts_router

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(
    style_transfer_router,
    prefix="/style",
    tags=["style-transfer"]
)

api_router.include_router(
    segmentation_router,
    prefix="/segmentation",
    tags=["segmentation"]
)

api_router.include_router(
    evaluation_router,
    prefix="/evaluation",
    tags=["evaluation"]
)

# Add our new prompts router
api_router.include_router(
    prompts_router,
    prefix="/prompts",
    tags=["prompts"]
)

__all__ = ["api_router"]
