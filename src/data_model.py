"""
Data Models for tracking images, masks, prompts, and results.

This module provides the data models and persistence mechanisms for tracking:
- Input images
- Segmentation masks
- Style and layout prompts
- Generated results
- Transformation history

Following the Semantic Seed Coding Standards:
- Comprehensive docstrings
- Type annotations
- Error handling
- Appropriate abstraction layers
"""

import os
import json
import uuid
import pickle
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseModel:
    """
    Base class for all data models with common functionality.
    
    Provides core attributes and methods for identification, serialization,
    persistence, and versioning.
    """
    
    # Current model version for compatibility tracking
    MODEL_VERSION = "1.0.0"
    
    def __init__(self, **kwargs):
        """
        Initialize a base model with common properties.
        
        Args:
            **kwargs: Any model-specific properties
        """
        # Set a unique ID for each model instance
        self.id = kwargs.get('id', str(uuid.uuid4()))
        
        # Track model version for compatibility
        self.model_version = self.MODEL_VERSION
        
        # Add creation timestamp for tracking
        self.creation_timestamp = datetime.now().timestamp()
        
        # Set all passed properties
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to a dictionary representation.
        
        Returns:
            Dictionary containing model properties
        """
        # Get all attributes that don't start with underscore
        result = {}
        for key, value in self.__dict__.items():
            # Skip numpy arrays and other non-serializable objects
            if not key.startswith('_') and not isinstance(value, np.ndarray):
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """
        Convert model to a JSON string representation.
        
        Returns:
            JSON string representation of the model
        """
        return json.dumps(self.to_dict(), default=str)
    
    def save(self, directory: Union[str, Path]) -> str:
        """
        Save the model to disk.
        
        Args:
            directory: Path to the directory where model should be saved
            
        Returns:
            Path to the saved model file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save the model using pickle for objects like numpy arrays
        path = os.path.join(directory, f"{self.id}.pickle")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Also save a JSON metadata file for easy inspection
        json_path = os.path.join(directory, f"{self.id}.json")
        with open(json_path, 'w') as f:
            f.write(self.to_json())
        
        logger.info(f"Saved model {self.id} to {path}")
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]):
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model file
            
        Returns:
            Loaded model instance
        """
        # Load the model
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model {model.id} from {path}")
        return model


class ImageModel(BaseModel):
    """
    Model for tracking input images and their metadata.
    """
    
    def __init__(self, image_data: Optional[np.ndarray] = None, **kwargs):
        """
        Initialize an image model.
        
        Args:
            image_data: The image as a numpy array
            **kwargs: Additional image metadata
        """
        super().__init__(**kwargs)
        self.image_data = image_data


class MaskModel(BaseModel):
    """
    Model for tracking segmentation masks and their metadata.
    """
    
    def __init__(self, mask_data: Optional[np.ndarray] = None, mask_type: str = "", **kwargs):
        """
        Initialize a mask model.
        
        Args:
            mask_data: The binary mask as a numpy array
            mask_type: Type of mask (e.g., "wall", "floor", "ceiling")
            **kwargs: Additional mask metadata
        """
        super().__init__(**kwargs)
        self.mask_data = mask_data
        self.mask_type = mask_type


class ResultModel(BaseModel):
    """
    Model for tracking generated results and their metadata.
    """
    
    def __init__(self, result_image_data: Optional[np.ndarray] = None, 
                 input_model_id: str = "", generation_method: str = "",
                 generation_params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize a result model.
        
        Args:
            result_image_data: The generated image as a numpy array
            input_model_id: ID of the input model used for generation
            generation_method: Method used for generation (e.g., "Flux")
            generation_params: Parameters used for generation
            **kwargs: Additional result metadata
        """
        super().__init__(**kwargs)
        self.result_image_data = result_image_data
        self.input_model_id = input_model_id
        self.generation_method = generation_method
        self.generation_params = generation_params or {}


class ProjectModel(BaseModel):
    """
    Model for tracking a project containing multiple related models.
    """
    
    def __init__(self, project_name: str = "", **kwargs):
        """
        Initialize a project model.
        
        Args:
            project_name: Name of the project
            **kwargs: Additional project metadata
        """
        super().__init__(**kwargs)
        self.project_name = project_name
        self.image_models = []
        self.mask_models = []
        self.result_models = []
    
    def add_image_model(self, model: ImageModel):
        """
        Add an image model to the project.
        
        Args:
            model: ImageModel to add
        """
        self.image_models.append(model)
    
    def add_mask_model(self, model: MaskModel):
        """
        Add a mask model to the project.
        
        Args:
            model: MaskModel to add
        """
        self.mask_models.append(model)
    
    def add_result_model(self, model: ResultModel):
        """
        Add a result model to the project.
        
        Args:
            model: ResultModel to add
        """
        self.result_models.append(model)
    
    def get_all_models(self) -> List[BaseModel]:
        """
        Get all models in the project.
        
        Returns:
            List of all models in the project
        """
        return self.image_models + self.mask_models + self.result_models


class TransformationHistory:
    """
    Track the history and lineage of transformations.
    """
    
    def __init__(self):
        """Initialize the transformation history tracker."""
        self.models = {}  # id -> model
        self.edges = {}   # child_id -> parent_id
    
    def add_model(self, model: BaseModel):
        """
        Add a model to the history.
        
        Args:
            model: Model to add to the history
        """
        self.models[model.id] = model
        
        # If it's a result model, add an edge to its input
        if isinstance(model, ResultModel) and model.input_model_id:
            self.edges[model.id] = model.input_model_id
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """
        Get a model by its ID.
        
        Args:
            model_id: ID of the model to get
            
        Returns:
            The model, or None if not found
        """
        return self.models.get(model_id)
    
    def get_all_models(self) -> List[BaseModel]:
        """
        Get all models in the history.
        
        Returns:
            List of all models
        """
        return list(self.models.values())
    
    def get_model_lineage(self, model_id: str) -> List[BaseModel]:
        """
        Get the lineage of a model (the chain of models leading to it).
        
        Args:
            model_id: ID of the model to get lineage for
            
        Returns:
            List of models in the lineage, from oldest to newest
        """
        lineage = []
        current_id = model_id
        
        # Follow the chain of edges back to the root
        while current_id:
            model = self.get_model(current_id)
            if model:
                lineage.append(model)
            
            # Move to parent
            current_id = self.edges.get(current_id)
            
            # Prevent infinite loops
            if current_id and current_id in [m.id for m in lineage]:
                break
        
        # Reverse to get oldest -> newest
        return list(reversed(lineage))
    
    def save(self, directory: Union[str, Path]) -> str:
        """
        Save the transformation history.
        
        Args:
            directory: Directory to save to
            
        Returns:
            Path to the saved history file
        """
        os.makedirs(directory, exist_ok=True)
        
        path = os.path.join(directory, f"history_{uuid.uuid4()}.pickle")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
        logger.info(f"Saved transformation history to {path}")
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TransformationHistory':
        """
        Load a transformation history from disk.
        
        Args:
            path: Path to the saved history file
            
        Returns:
            Loaded history
        """
        with open(path, 'rb') as f:
            history = pickle.load(f)
            
        logger.info(f"Loaded transformation history from {path}")
        return history


class DataModelManager:
    """
    Manager for working with data models and coordinating persistence.
    """
    
    def __init__(self, base_directory: Union[str, Path] = None):
        """
        Initialize the model manager.
        
        Args:
            base_directory: Base directory for storing models and history
        """
        self.base_directory = base_directory or os.path.join(os.getcwd(), "data")
        self.history = TransformationHistory()
        
        # Create directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the directory structure for storing models."""
        os.makedirs(self.base_directory, exist_ok=True)
        os.makedirs(os.path.join(self.base_directory, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.base_directory, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.base_directory, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.base_directory, "projects"), exist_ok=True)
    
    def save_image_model(self, model: ImageModel) -> str:
        """
        Save an image model.
        
        Args:
            model: Image model to save
            
        Returns:
            Path to the saved model
        """
        # Add to history
        self.history.add_model(model)
        
        # Save model
        return model.save(os.path.join(self.base_directory, "images"))
    
    def save_mask_model(self, model: MaskModel) -> str:
        """
        Save a mask model.
        
        Args:
            model: Mask model to save
            
        Returns:
            Path to the saved model
        """
        # Add to history
        self.history.add_model(model)
        
        # Save model
        return model.save(os.path.join(self.base_directory, "masks"))
    
    def save_result_model(self, model: ResultModel) -> str:
        """
        Save a result model.
        
        Args:
            model: Result model to save
            
        Returns:
            Path to the saved model
        """
        # Add to history
        self.history.add_model(model)
        
        # Save model
        return model.save(os.path.join(self.base_directory, "results"))
    
    def save_project_model(self, model: ProjectModel) -> str:
        """
        Save a project model.
        
        Args:
            model: Project model to save
            
        Returns:
            Path to the saved model
        """
        # Add all associated models to history
        for image_model in model.image_models:
            self.history.add_model(image_model)
        
        for mask_model in model.mask_models:
            self.history.add_model(mask_model)
            
        for result_model in model.result_models:
            self.history.add_model(result_model)
        
        # Save model
        return model.save(os.path.join(self.base_directory, "projects"))
    
    def save_history(self) -> str:
        """
        Save the transformation history.
        
        Returns:
            Path to the saved history
        """
        return self.history.save(self.base_directory)
    
    def load_model(self, model_id: str) -> Optional[BaseModel]:
        """
        Load a model by its ID.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded model, or None if not found
        """
        # Check if model is already in history
        model = self.history.get_model(model_id)
        if model:
            return model
        
        # Look for model file in all directories
        for subdir in ["images", "masks", "results", "projects"]:
            path = os.path.join(self.base_directory, subdir, f"{model_id}.pickle")
            if os.path.exists(path):
                model = BaseModel.load(path)
                self.history.add_model(model)
                return model
        
        return None
