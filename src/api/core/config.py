"""
Configuration settings for the Interior Style Transfer API.

This module handles configuration settings, environment variables, and
constants used throughout the API. It follows the Semantic Seed Coding
Standards (SSCS) for maintainability and security.
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APISettings(BaseSettings):
    """API configuration settings based on environment variables."""
    
    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Interior Style Transfer API"
    DEBUG: bool = Field(default=False)
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Security
    SECRET_KEY: str = Field(
        default=os.getenv("SECRET_KEY", "development_secret_key"), 
        description="Secret key for JWT token generation and validation"
    )
    
    # Access token settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # FAL.ai settings
    FAL_KEY: Optional[str] = Field(
        default=os.getenv("FAL_KEY"),
        description="API key for the FAL.ai Flux API"
    )
    
    # Model settings
    DEFAULT_MODEL: str = "flux-1"
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields not defined in the model


# Initialize settings object
settings = APISettings()
