"""
Test suite for data models used for image, mask, prompt, and result tracking.

Following Semantic Seed Coding Standards and BDD/TDD approach:
- Test cases are written before implementation
- Tests follow BDD "Given-When-Then" structure
- Each test has a docstring explaining its purpose
"""

import os
import pytest
import numpy as np
import json
from datetime import datetime
import uuid
from pathlib import Path


class TestDataModel:
    """Test suite for data models for tracking images, masks, prompts, and results."""
    
    @pytest.fixture
    def sample_model_data(self):
        """Fixture that provides sample data for testing models."""
        return {
            "input_image_path": "/path/to/image.jpg",
            "input_prompt": "Modern living room with blue sofa",
            "style": "Scandinavian",
            "room_type": "living room",
            "creation_date": datetime.now().isoformat(),
            "user_id": "user123",
            "project_id": "project456"
        }
    
    @pytest.fixture
    def sample_image_data(self):
        """Fixture that provides a small test image as numpy array."""
        # Create a simple 10x10 RGB test image
        return np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask_data(self):
        """Fixture that provides a small test binary mask."""
        # Create a simple 10x10 binary mask
        return np.random.randint(0, 2, (10, 10), dtype=np.uint8)
    
    def test_model_initialization(self, sample_model_data):
        """
        GIVEN model data with required fields
        WHEN a BaseModel is initialized
        THEN it should correctly store and provide access to the data
        """
        from src.data_model import BaseModel
        
        # Initialize the base model
        model = BaseModel(**sample_model_data)
        
        # Verify that all properties were correctly set
        for key, value in sample_model_data.items():
            assert getattr(model, key) == value
        
        # Verify default fields were set
        assert model.id is not None
        assert model.model_version is not None
        assert model.creation_timestamp is not None
    
    def test_image_model(self, sample_model_data, sample_image_data):
        """
        GIVEN image data and metadata
        WHEN an ImageModel is initialized
        THEN it should store both the image and metadata
        """
        from src.data_model import ImageModel
        
        # Initialize with image data
        model = ImageModel(
            image_data=sample_image_data,
            **sample_model_data
        )
        
        # Verify properties
        assert model.image_data is not None
        assert isinstance(model.image_data, np.ndarray)
        assert model.image_data.shape == sample_image_data.shape
        assert np.array_equal(model.image_data, sample_image_data)
        
        # Verify metadata
        assert model.input_image_path == sample_model_data["input_image_path"]
        assert model.input_prompt == sample_model_data["input_prompt"]
    
    def test_mask_model(self, sample_model_data, sample_mask_data):
        """
        GIVEN mask data and metadata
        WHEN a MaskModel is initialized
        THEN it should store both the mask and metadata
        """
        from src.data_model import MaskModel
        
        # Initialize with mask data
        model = MaskModel(
            mask_data=sample_mask_data,
            mask_type="wall",
            **sample_model_data
        )
        
        # Verify properties
        assert model.mask_data is not None
        assert isinstance(model.mask_data, np.ndarray)
        assert model.mask_data.shape == sample_mask_data.shape
        assert np.array_equal(model.mask_data, sample_mask_data)
        
        # Verify mask-specific fields
        assert model.mask_type == "wall"
    
    def test_result_model(self, sample_model_data, sample_image_data):
        """
        GIVEN result image data and metadata
        WHEN a ResultModel is initialized
        THEN it should store the result and its relationship to input
        """
        from src.data_model import ResultModel
        
        # Initialize with result data
        model = ResultModel(
            result_image_data=sample_image_data,
            input_model_id="input123",
            generation_method="Flux",
            generation_params={"strength": 0.7, "guidance_scale": 7.5},
            **sample_model_data
        )
        
        # Verify properties
        assert model.result_image_data is not None
        assert isinstance(model.result_image_data, np.ndarray)
        assert np.array_equal(model.result_image_data, sample_image_data)
        
        # Verify result-specific fields
        assert model.input_model_id == "input123"
        assert model.generation_method == "Flux"
        assert model.generation_params["strength"] == 0.7
    
    def test_project_model(self, sample_model_data):
        """
        GIVEN project data
        WHEN a ProjectModel is initialized
        THEN it should track multiple related models
        """
        from src.data_model import ProjectModel, ImageModel
        
        # Create a project
        project = ProjectModel(
            project_name="Interior Redesign",
            **sample_model_data
        )
        
        # Verify project properties
        assert project.project_name == "Interior Redesign"
        assert project.project_id == sample_model_data["project_id"]
        assert project.image_models == []
        assert project.result_models == []
        
        # Add image models to project
        project.add_image_model(ImageModel(
            image_data=np.zeros((10, 10, 3), dtype=np.uint8),
            **sample_model_data
        ))
        
        # Verify image was added
        assert len(project.image_models) == 1
    
    def test_model_serialization(self, sample_model_data, sample_image_data):
        """
        GIVEN a populated model
        WHEN to_json() is called
        THEN it should return a valid JSON representation
        """
        from src.data_model import ImageModel
        
        # Create a model
        model = ImageModel(
            image_data=sample_image_data,
            **sample_model_data
        )
        
        # Serialize to JSON
        json_data = model.to_json()
        
        # Verify it's valid JSON
        parsed_data = json.loads(json_data)
        
        # Verify key fields are included
        assert parsed_data["id"] == model.id
        assert parsed_data["input_prompt"] == model.input_prompt
        
        # Image data should be stored as path or metadata, not in JSON
        assert "image_data" not in parsed_data
    
    def test_model_saving_and_loading(self, sample_model_data, sample_image_data, tmp_path):
        """
        GIVEN a model with image data
        WHEN save() and load() methods are called
        THEN the model should be correctly persisted and restored
        """
        from src.data_model import ImageModel
        
        # Create model
        model = ImageModel(
            image_data=sample_image_data,
            **sample_model_data
        )
        
        # Set up test directory
        data_dir = tmp_path / "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save model
        save_path = model.save(data_dir)
        
        # Verify file exists
        assert os.path.exists(save_path)
        
        # Load model back
        loaded_model = ImageModel.load(save_path)
        
        # Verify key data was preserved
        assert loaded_model.id == model.id
        assert loaded_model.input_prompt == model.input_prompt
        assert np.array_equal(loaded_model.image_data, model.image_data)
    
    def test_model_history_tracking(self, sample_model_data, sample_image_data):
        """
        GIVEN a series of transformations on an image
        WHEN each step is tracked in the model
        THEN the full history should be accessible
        """
        from src.data_model import ImageModel, ResultModel, TransformationHistory
        
        # Create initial image model
        input_model = ImageModel(
            image_data=sample_image_data,
            **sample_model_data
        )
        
        # Create a result from the input
        result1 = ResultModel(
            result_image_data=sample_image_data,
            input_model_id=input_model.id,
            generation_method="Flux",
            **sample_model_data
        )
        
        # Create a history tracker
        history = TransformationHistory()
        
        # Add models to history
        history.add_model(input_model)
        history.add_model(result1)
        
        # Create a second result from the first result
        result2 = ResultModel(
            result_image_data=sample_image_data,
            input_model_id=result1.id,
            generation_method="Refinement",
            **sample_model_data
        )
        
        # Add to history
        history.add_model(result2)
        
        # Verify history tracking
        assert len(history.get_all_models()) == 3
        
        # Verify we can trace lineage
        lineage = history.get_model_lineage(result2.id)
        assert len(lineage) == 3
        assert lineage[0].id == input_model.id
        assert lineage[1].id == result1.id
        assert lineage[2].id == result2.id
