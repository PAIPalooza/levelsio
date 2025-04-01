"""
Error handlers for the FastAPI application.

This module registers exception handlers for the FastAPI application
to provide consistent error responses following SSCS.
"""

import traceback
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from src.api.core.errors import (
    StyleAPIException, 
    handle_validation_error,
    handle_style_api_exception
)
import logging

# Configure logger
logger = logging.getLogger("api.error_handlers")

def register_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers for the FastAPI application.
    
    Args:
        app: The FastAPI application instance
    """
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Handle validation errors from FastAPI's request validation system.
        
        Args:
            request: The HTTP request
            exc: The validation exception
            
        Returns:
            JSONResponse with formatted error details
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "message": "Validation error",
                "details": {
                    "errors": exc.errors(),
                    "body": exc.body if hasattr(exc, "body") else None
                }
            }
        )

    @app.exception_handler(StyleAPIException)
    async def style_api_exception_handler(request: Request, exc: StyleAPIException):
        """
        Handle custom StyleAPIException errors.
        
        Args:
            request: The HTTP request
            exc: The custom exception
            
        Returns:
            JSONResponse with formatted error details
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "message": exc.message,
                "details": exc.details
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """
        Handle any unhandled exceptions.
        """
        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc()
            }
        )
        
        # In production, you might want to hide the actual error details
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "An unexpected error occurred",
                "details": {
                    "type": type(exc).__name__,
                    "error": str(exc)
                }
            }
        )
