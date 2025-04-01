#!/usr/bin/env python
"""
CLI script to run the Interior Style Transfer API server.

This script runs the FastAPI server for the Interior Style Transfer API
with configurable host, port, and log level options. It follows the
Semantic Seed Coding Standards (SSCS) for command-line tools.
"""

import os
import sys
import argparse
import logging
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Run the Interior Style Transfer API server'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=os.getenv('API_HOST', '0.0.0.0'),
        help='Host to run the server on (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('API_PORT', '8000')),
        help='Port to run the server on (default: 8000)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default=os.getenv('LOG_LEVEL', 'info'),
        help='Log level (default: info)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    return parser.parse_args()


def main():
    """Run the API server with the specified options."""
    args = parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Log level: {args.log_level}")
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
