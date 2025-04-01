"""
Authentication and authorization utilities for the API.

This module implements authentication mechanisms for the Interior Style
Transfer API, including API key validation and header verification following SSCS.
"""

import os
import logging
from typing import Optional
from fastapi import Header, HTTPException, Depends, status, Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.auth")

# Read API key from environment variables
API_KEY = os.getenv("INTERIOR_STYLE_API_KEY")
FAL_KEY = os.getenv("FAL_KEY", "f51c8849-420f-4f98-ac43-9b98e5a58408:072cb70f9f6d9fc4f56f3930187d87cd")

# Set dummy key for testing if not provided
if not API_KEY:
    logger.warning("INTERIOR_STYLE_API_KEY environment variable not set, using test key")
    API_KEY = "test-api-key-123"

# The FAL key should also be valid for testing
VALID_KEYS = [API_KEY, FAL_KEY]


def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Verify that a valid API key is provided in the request header.
    
    Args:
        api_key: API key from request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        logger.warning("API key header missing in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Use X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Check if the provided key is valid
    if api_key not in VALID_KEYS:
        logger.warning("Unauthorized API access attempt with invalid key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    logger.info("API key validated successfully")
    return api_key


def verify_x_api_key(request: Request) -> str:
    """
    Alternative dependency for API key validation from header.
    
    This is useful for direct header access in middleware scenarios.
    
    Args:
        request: FastAPI request object
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Try to get key from X-API-Key header
    api_key = request.headers.get("X-API-Key")
    
    # If not found, try api-key header (alternative format)
    if not api_key:
        api_key = request.headers.get("api-key")
    
    # If still not found, try API_KEY query parameter
    if not api_key and "api_key" in request.query_params:
        api_key = request.query_params.get("api_key")
    
    # Validate the key
    if not api_key or api_key not in VALID_KEYS:
        logger.warning("Unauthorized API access attempt via alternative headers")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key
