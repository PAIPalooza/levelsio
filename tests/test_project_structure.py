"""
Test module to verify project structure follows SSCS requirements.

This follows BDD style testing to validate the project has the required
directories and files according to Semantic Seed Coding Standards.
"""

import os
import pytest


class TestProjectStructure:
    """
    Test the project structure follows SSCS requirements.
    
    Uses BDD-style tests to verify all directories and base files exist.
    """

    def test_required_directories_exist(self):
        """
        GIVEN a project following SSCS
        WHEN the project is set up
        THEN all required directories should exist
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Required directories according to SSCS
        required_dirs = [
            os.path.join(base_dir, "src"),
            os.path.join(base_dir, "notebooks"),
            os.path.join(base_dir, "assets"),
            os.path.join(base_dir, "assets", "test_images"),
            os.path.join(base_dir, "assets", "outputs"),
            os.path.join(base_dir, "assets", "masks"),
            os.path.join(base_dir, "models"),
            os.path.join(base_dir, "tests"),
        ]
        
        # Assert all required directories exist
        for dir_path in required_dirs:
            assert os.path.isdir(dir_path), f"Directory does not exist: {dir_path}"

    def test_readme_exists_and_has_content(self):
        """
        GIVEN a project following SSCS 
        WHEN the project is set up
        THEN README.md should exist and have meaningful content
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        readme_path = os.path.join(base_dir, "README.md")
        
        # Assert README exists and has content
        assert os.path.isfile(readme_path), "README.md does not exist"
        
        with open(readme_path, 'r') as f:
            content = f.read()
            
        # Check for meaningful content (more than just the project name)
        assert len(content) > 100, "README.md has insufficient content"
        
        # Check for required sections
        required_sections = [
            "# Interior Style Transfer POC", 
            "## Installation", 
            "## Usage"
        ]
        
        for section in required_sections:
            assert section in content, f"README missing required section: {section}"
