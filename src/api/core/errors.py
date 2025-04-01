"""
Error handling utilities for the Interior Style Transfer API.

This module defines custom exceptions and error handling functions
following the Semantic Seed Coding Standards (SSCS).
"""

from typing import Dict, Any, Optional, List, Union
from fastapi import HTTPException, status
import logging
from pydantic import ValidationError

# Configure logger
logger = logging.getLogger("api.errors")


class StyleAPIException(Exception):
    """Base exception for style API errors."""
    
    def __init__(
        self, 
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class StyleNotFoundException(StyleAPIException):
    """Exception raised when a requested style is not found."""
    
    def __init__(self, style_name: str, available_styles: Optional[List[str]] = None):
        self.style_name = style_name
        self.available_styles = available_styles
        message = f"Style '{style_name}' not found"
        if available_styles:
            message += f". Available styles: {', '.join(available_styles)}"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details={"style_name": style_name, "available_styles": available_styles}
        )


class InvalidImageException(StyleAPIException):
    """Exception raised when an image is invalid or corrupted."""
    
    def __init__(self, message: str = "Invalid image data"):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"reason": "image_validation_error"}
        )


class ExternalAPIException(StyleAPIException):
    """Exception raised when an external API call fails."""
    
    def __init__(
        self,
        message: str = "External API request failed",
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        api_name: str = "external",
        original_error: Optional[Exception] = None
    ):
        details = {"api_name": api_name}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            message=message,
            status_code=status_code,
            details=details
        )


class AuthenticationException(StyleAPIException):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={"reason": "authentication_error"}
        )


class RateLimitException(StyleAPIException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        details = {"reason": "rate_limit_exceeded"}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


def handle_validation_error(error: ValidationError) -> HTTPException:
    """
    Convert a Pydantic validation error to an HTTPException.
    
    Args:
        error: The Pydantic validation error
        
    Returns:
        HTTPException with formatted error details
    """
    logger.warning(f"Validation error: {error}")
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "message": "Validation error",
            "errors": [
                {
                    "loc": list(err["loc"]),
                    "msg": err["msg"],
                    "type": err["type"]
                }
                for err in error.errors()
            ]
        }
    )


def handle_style_api_exception(error: StyleAPIException) -> HTTPException:
    """
    Convert a StyleAPIException to an HTTPException.
    
    Args:
        error: The StyleAPIException
        
    Returns:
        HTTPException with formatted error details
    """
    logger.error(f"API error ({error.status_code}): {error.message}")
    return HTTPException(
        status_code=error.status_code,
        detail={
            "message": error.message,
            "details": error.details
        }
    )


def log_external_api_error(
    api_name: str,
    error: Exception,
    endpoint: str,
    request_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log detailed information about an external API error.
    
    Args:
        api_name: Name of the external API
        error: The exception that occurred
        endpoint: The endpoint that was called
        request_data: Optional request data that was sent
    """
    logger.error(
        f"Error calling {api_name} API at {endpoint}: {str(error)}",
        extra={
            "api_name": api_name,
            "endpoint": endpoint,
            "error_type": type(error).__name__,
            "request_data": request_data
        }
    )
