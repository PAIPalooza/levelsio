"""
Authentication utilities for Interior Style Transfer API.

This module implements API key authentication and validation following
the Semantic Seed Coding Standards (SSCS) for secure, testable code.
It provides multiple authentication mechanisms for the API endpoints.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import Depends, HTTPException, status, Security, Header
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from dotenv import load_dotenv
from src.api.core.config import settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API key names and security schemes
API_KEY_NAME = "api-key"
API_KEY_QUERY = "api_key"
API_KEY_HEADER_NAME = "X-API-Key"

# Get API key from environment
API_KEY = os.getenv("INTERIOR_STYLE_API_KEY", "development-key")

# Security schemes
simple_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
simple_api_key_query = APIKeyQuery(name=API_KEY_QUERY, auto_error=False)
x_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=True)

# In a production environment, this would validate against a secure database
# For the POC, we'll use a simple in-memory set of valid API keys
VALID_API_KEYS = {
    "api_key_for_testing": {
        "owner": "demo_user",
        "scopes": ["style", "segmentation", "evaluation", "prompts"]
    }
}


async def verify_api_key(
    api_key_header: str = Security(simple_api_key_header),
    api_key_query: str = Security(simple_api_key_query)
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


async def verify_x_api_key(api_key: str = Security(x_api_key_header)) -> Dict[str, Any]:
    """
    Verify the API key provided in X-API-Key header.
    
    This is a dependency that can be used in route definitions to
    enforce more structured API key authentication.
    
    Args:
        api_key: The API key provided in the X-API-Key header
        
    Returns:
        dict: Information about the API key owner if valid
        
    Raises:
        HTTPException: If the API key is invalid or missing
    """
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return VALID_API_KEYS[api_key]


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
