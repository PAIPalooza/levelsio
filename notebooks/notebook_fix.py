#!/usr/bin/env python3
"""
Fix and update the Interior Style Transfer Demo Notebook.

This script modifies the notebook to properly handle imports and file paths
for the interactive demo, following Semantic Seed Coding Standards.
"""

import json
import os
import shutil

def fix_notebook():
    """
    Fix the Interior Style Transfer Demo notebook.
    
    Resolves import issues and file path problems that prevent the notebook
    from running properly.
    
    Returns:
        str: Path to the fixed notebook
    """
    source_path = 'notebooks/Interior_Style_Transfer_Demo.ipynb'
    output_path = 'notebooks/Interior_Style_Transfer_Demo_Fixed.ipynb'
    
    # Read the existing notebook
    with open(source_path, 'r') as f:
        nb = json.load(f)
    
    # Find the setup cell and update it
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                source = ''.join(source)
            
            # This is the setup/import cell
            if "import sys" in source and "from src.style_transfer import" in source:
                # Replace the import line for FluxAPIClient
                updated_source = source.replace(
                    "from src.flux_integration import FluxAPIClient",
                    "# Import with adapter\nfrom flux_adapter import FluxAPIClient"
                )
                
                # Fix the path for sample_images
                updated_source = updated_source.replace(
                    "def load_sample_images(sample_dir='notebooks/sample_images'):",
                    "def load_sample_images(sample_dir='sample_images'):"
                )
                
                # Update the cell source
                cell['source'] = updated_source
                break
    
    # Write the updated notebook
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=2)
    
    # Make a backup of the original
    backup_path = f"{source_path}.bak"
    shutil.copy2(source_path, backup_path)
    
    # Replace the original with the fixed version
    shutil.copy2(output_path, source_path)
    
    print(f"Fixed notebook saved to {source_path}")
    print(f"Original notebook backed up to {backup_path}")

if __name__ == '__main__':
    fix_notebook()
