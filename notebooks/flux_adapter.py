"""
Adapter module for the Interior Style Transfer Notebook.

This module provides the necessary adapters and aliases to make
the notebook compatible with the existing codebase.
"""

import sys
import os

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FluxClient
from src.flux_integration import FluxClient

# Create an alias for the notebook
FluxAPIClient = FluxClient
