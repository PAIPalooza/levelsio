"""
Test module for data models.

This module contains BDD-style tests for the data model classes,
following the Semantic Seed Coding Standards requirements.
"""

import pytest
import numpy as np
from datetime import datetime
import uuid

from src.models import (
    InteriorImage,
    SegmentationMask,
    StylePrompt,
    GenerationRequest,
    GenerationResult,
    EvaluationResult
)


class TestInteriorImage:
    """Test suite for the InteriorImage class."""
    
    @pytest.fixture
    def sample_image_data(self):
        """Fixture providing sample data for an InteriorImage."""
        return {
            "id": str(uuid.uuid4()),
            "file_path": "/path/to/interior.jpg",
            "image_array": np.ones((100, 100, 3), dtype=np.uint8),
            "uploaded_at": datetime.now(),
            "source_type": "upload"
        }
    
    def test_interior_image_initialization(self, sample_image_data):
        """
        GIVEN valid image data
        WHEN initializing an InteriorImage with that data
        THEN it should correctly store all attributes
        """
        image = InteriorImage(**sample_image_data)
        
        assert image.id == sample_image_data["id"]
        assert image.file_path == sample_image_data["file_path"]
        assert np.array_equal(image.image_array, sample_image_data["image_array"])
        assert image.uploaded_at == sample_image_data["uploaded_at"]
        assert image.source_type == sample_image_data["source_type"]


class TestSegmentationMask:
    """Test suite for the SegmentationMask class."""
    
    @pytest.fixture
    def sample_mask_data(self):
        """Fixture providing sample data for a SegmentationMask."""
        return {
            "id": str(uuid.uuid4()),
            "source_image_id": str(uuid.uuid4()),
            "mask_array": np.zeros((100, 100), dtype=np.uint8),
            "segmentation_method": "SAM",
            "classes_detected": ["wall", "floor", "ceiling"],
            "created_at": datetime.now()
        }
    
    def test_segmentation_mask_initialization(self, sample_mask_data):
        """
        GIVEN valid mask data
        WHEN initializing a SegmentationMask with that data
        THEN it should correctly store all attributes
        """
        mask = SegmentationMask(**sample_mask_data)
        
        assert mask.id == sample_mask_data["id"]
        assert mask.source_image_id == sample_mask_data["source_image_id"]
        assert np.array_equal(mask.mask_array, sample_mask_data["mask_array"])
        assert mask.segmentation_method == sample_mask_data["segmentation_method"]
        assert mask.classes_detected == sample_mask_data["classes_detected"]
        assert mask.created_at == sample_mask_data["created_at"]


class TestStylePrompt:
    """Test suite for the StylePrompt class."""
    
    @pytest.fixture
    def sample_prompt_data(self):
        """Fixture providing sample data for a StylePrompt."""
        return {
            "id": str(uuid.uuid4()),
            "title": "Scandinavian Modern",
            "prompt_text": "A bright, minimalist Scandinavian modern living room",
            "allow_layout_change": False,
            "applied_at": datetime.now()
        }
    
    def test_style_prompt_initialization(self, sample_prompt_data):
        """
        GIVEN valid prompt data
        WHEN initializing a StylePrompt with that data
        THEN it should correctly store all attributes
        """
        prompt = StylePrompt(**sample_prompt_data)
        
        assert prompt.id == sample_prompt_data["id"]
        assert prompt.title == sample_prompt_data["title"]
        assert prompt.prompt_text == sample_prompt_data["prompt_text"]
        assert prompt.allow_layout_change == sample_prompt_data["allow_layout_change"]
        assert prompt.applied_at == sample_prompt_data["applied_at"]
    
    def test_style_prompt_without_applied_at(self, sample_prompt_data):
        """
        GIVEN valid prompt data without applied_at
        WHEN initializing a StylePrompt with that data
        THEN it should set applied_at to None
        """
        # Remove applied_at from the data
        data = {k: v for k, v in sample_prompt_data.items() if k != "applied_at"}
        
        prompt = StylePrompt(**data)
        
        assert prompt.id == sample_prompt_data["id"]
        assert prompt.title == sample_prompt_data["title"]
        assert prompt.prompt_text == sample_prompt_data["prompt_text"]
        assert prompt.allow_layout_change == sample_prompt_data["allow_layout_change"]
        assert prompt.applied_at is None


class TestGenerationRequest:
    """Test suite for the GenerationRequest class."""
    
    @pytest.fixture
    def sample_request_data(self):
        """Fixture providing sample data for a GenerationRequest."""
        return {
            "id": str(uuid.uuid4()),
            "input_image_id": str(uuid.uuid4()),
            "segmentation_mask_id": str(uuid.uuid4()),
            "prompt_id": str(uuid.uuid4()),
            "model_used": "flux",
            "fal_model_id": "fal-ai/flux",
            "allow_layout_change": False,
            "run_mode": "style_only",
            "created_at": datetime.now(),
            "status": "pending"
        }
    
    def test_generation_request_initialization(self, sample_request_data):
        """
        GIVEN valid request data
        WHEN initializing a GenerationRequest with that data
        THEN it should correctly store all attributes
        """
        request = GenerationRequest(**sample_request_data)
        
        assert request.id == sample_request_data["id"]
        assert request.input_image_id == sample_request_data["input_image_id"]
        assert request.segmentation_mask_id == sample_request_data["segmentation_mask_id"]
        assert request.prompt_id == sample_request_data["prompt_id"]
        assert request.model_used == sample_request_data["model_used"]
        assert request.fal_model_id == sample_request_data["fal_model_id"]
        assert request.allow_layout_change == sample_request_data["allow_layout_change"]
        assert request.run_mode == sample_request_data["run_mode"]
        assert request.created_at == sample_request_data["created_at"]
        assert request.status == sample_request_data["status"]
    
    def test_generation_request_with_defaults(self):
        """
        GIVEN only required fields for a request
        WHEN initializing a GenerationRequest with that data
        THEN it should use default values for optional fields
        """
        required_data = {
            "id": str(uuid.uuid4()),
            "input_image_id": str(uuid.uuid4()),
            "prompt_id": str(uuid.uuid4()),
            "created_at": datetime.now()
        }
        
        request = GenerationRequest(**required_data)
        
        assert request.id == required_data["id"]
        assert request.input_image_id == required_data["input_image_id"]
        assert request.segmentation_mask_id is None
        assert request.prompt_id == required_data["prompt_id"]
        assert request.model_used == "flux"
        assert request.fal_model_id == "fal-ai/flux"
        assert request.allow_layout_change is False
        assert request.run_mode == "style_only"
        assert request.created_at == required_data["created_at"]
        assert request.status == "pending"


class TestGenerationResult:
    """Test suite for the GenerationResult class."""
    
    @pytest.fixture
    def sample_result_data(self):
        """Fixture providing sample data for a GenerationResult."""
        return {
            "id": str(uuid.uuid4()),
            "generation_request_id": str(uuid.uuid4()),
            "output_image_array": np.ones((100, 100, 3), dtype=np.uint8) * 200,
            "output_path": "/path/to/output.jpg",
            "similarity_score": 0.85,
            "completed_at": datetime.now()
        }
    
    def test_generation_result_initialization(self, sample_result_data):
        """
        GIVEN valid result data
        WHEN initializing a GenerationResult with that data
        THEN it should correctly store all attributes
        """
        result = GenerationResult(**sample_result_data)
        
        assert result.id == sample_result_data["id"]
        assert result.generation_request_id == sample_result_data["generation_request_id"]
        assert np.array_equal(result.output_image_array, sample_result_data["output_image_array"])
        assert result.output_path == sample_result_data["output_path"]
        assert result.similarity_score == sample_result_data["similarity_score"]
        assert result.completed_at == sample_result_data["completed_at"]
    
    def test_generation_result_with_minimal_data(self):
        """
        GIVEN only required fields for a result
        WHEN initializing a GenerationResult with that data
        THEN it should set optional fields to None
        """
        required_data = {
            "id": str(uuid.uuid4()),
            "generation_request_id": str(uuid.uuid4()),
            "output_image_array": np.ones((100, 100, 3), dtype=np.uint8) * 200,
            "completed_at": datetime.now()
        }
        
        result = GenerationResult(**required_data)
        
        assert result.id == required_data["id"]
        assert result.generation_request_id == required_data["generation_request_id"]
        assert np.array_equal(result.output_image_array, required_data["output_image_array"])
        assert result.output_path is None
        assert result.similarity_score is None
        assert result.completed_at == required_data["completed_at"]


class TestEvaluationResult:
    """Test suite for the EvaluationResult class."""
    
    @pytest.fixture
    def sample_evaluation_data(self):
        """Fixture providing sample data for an EvaluationResult."""
        return {
            "id": str(uuid.uuid4()),
            "input_image_id": str(uuid.uuid4()),
            "output_image_id": str(uuid.uuid4()),
            "ssim_score": 0.87,
            "mse_score": 0.12,
            "is_structure_preserved": True,
            "human_rating": 4,
            "evaluated_at": datetime.now()
        }
    
    def test_evaluation_result_initialization(self, sample_evaluation_data):
        """
        GIVEN valid evaluation data
        WHEN initializing an EvaluationResult with that data
        THEN it should correctly store all attributes
        """
        evaluation = EvaluationResult(**sample_evaluation_data)
        
        assert evaluation.id == sample_evaluation_data["id"]
        assert evaluation.input_image_id == sample_evaluation_data["input_image_id"]
        assert evaluation.output_image_id == sample_evaluation_data["output_image_id"]
        assert evaluation.ssim_score == sample_evaluation_data["ssim_score"]
        assert evaluation.mse_score == sample_evaluation_data["mse_score"]
        assert evaluation.is_structure_preserved == sample_evaluation_data["is_structure_preserved"]
        assert evaluation.human_rating == sample_evaluation_data["human_rating"]
        assert evaluation.evaluated_at == sample_evaluation_data["evaluated_at"]
    
    def test_evaluation_result_without_human_rating(self, sample_evaluation_data):
        """
        GIVEN evaluation data without human_rating
        WHEN initializing an EvaluationResult with that data
        THEN it should set human_rating to None
        """
        # Remove human_rating from the data
        data = {k: v for k, v in sample_evaluation_data.items() if k != "human_rating"}
        
        evaluation = EvaluationResult(**data)
        
        assert evaluation.id == sample_evaluation_data["id"]
        assert evaluation.input_image_id == sample_evaluation_data["input_image_id"]
        assert evaluation.output_image_id == sample_evaluation_data["output_image_id"]
        assert evaluation.ssim_score == sample_evaluation_data["ssim_score"]
        assert evaluation.mse_score == sample_evaluation_data["mse_score"]
        assert evaluation.is_structure_preserved == sample_evaluation_data["is_structure_preserved"]
        assert evaluation.human_rating is None
        assert evaluation.evaluated_at == sample_evaluation_data["evaluated_at"]
