"""
Authentication middleware for the Interior Style Transfer API.

This module provides authentication mechanisms for the API endpoints,
following security best practices from the Semantic Seed Coding
Standards (SSCS).
"""

from fastapi import Depends, HTTPException, status, Header, Security
from fastapi.security import APIKeyHeader
from src.api.core.config import settings

# Define API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# In a production environment, this would validate against a secure database
# For the POC, we'll use a simple in-memory set of valid API keys
VALID_API_KEYS = {
    "api_key_for_testing": {
        "owner": "demo_user",
        "scopes": ["style", "segmentation", "evaluation"]
    }
}

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the API key provided in request headers.
    
    This is a dependency that can be used in route definitions to
    enforce API key authentication.
    
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
