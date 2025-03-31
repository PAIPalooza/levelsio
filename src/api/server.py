"""
Server module for running the Interior Style Transfer API.

This script configures and launches the FastAPI server with
Swagger/OpenAPI documentation. It follows the Semantic Seed Coding
Standards (SSCS) for maintainability and deployment.
"""

import uvicorn
import os
import argparse
from dotenv import load_dotenv
from src.api.main import app
from src.api.core.config import settings

# Load environment variables
load_dotenv()

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the Interior Style Transfer API server."""
    print(f"🚀 Starting {settings.PROJECT_NAME}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"📘 ReDoc Documentation: http://{host}:{port}/redoc")
    print(f"🔧 Environment: {'Development' if settings.DEBUG else 'Production'}")
    
    # Run the API with uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info" if settings.DEBUG else "warning",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Interior Style Transfer API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Override debug setting if specified
    if args.debug:
        os.environ["DEBUG"] = "True"
    
    # Run the server
    run_server(args.host, args.port, args.reload)
