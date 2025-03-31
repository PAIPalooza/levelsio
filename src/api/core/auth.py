"""
Authentication utilities for Interior Style Transfer API.

This module implements API key authentication and validation following
the Semantic Seed Coding Standards (SSCS) for secure, testable code.
"""

import os
import logging
from typing import Optional, List, Dict
from fastapi import Depends, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API key names
API_KEY_NAME = "api-key"
API_KEY_QUERY = "api_key"

# Get API key from environment
API_KEY = os.getenv("INTERIOR_STYLE_API_KEY", "development-key")

# Security schemes
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY, auto_error=False)


async def verify_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query)
) -> str:
    """
    Verify the API key provided in header or query parameter.
    
    Args:
        api_key_header: API key from header
        api_key_query: API key from query parameter
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    # Check if valid API key in header
    if api_key_header == API_KEY:
        return api_key_header
        
    # Check if valid API key in query
    if api_key_query == API_KEY:
        return api_key_query
        
    # Log unauthorized access attempt
    logger.warning(f"Unauthorized API access attempt with invalid key")
    
    # Raise exception for invalid API key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
        headers={"WWW-Authenticate": "ApiKey"}
    )


def get_allowed_operations(api_key: str = Depends(verify_api_key)) -> List[str]:
    """
    Get the list of operations allowed for the API key.
    
    Args:
        api_key: Validated API key
        
    Returns:
        List of allowed operations
    """
    # Default to all operations
    return [
        "style_transfer",
        "segmentation",
        "evaluation",
        "prompts"
    ]
