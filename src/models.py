"""
Data models for Interior Style Transfer POC.

This module defines the core data structures for tracking images, masks,
prompts, generation requests, and results following the Semantic Seed Coding Standards.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any

import numpy as np


@dataclass
class InteriorImage:
    """
    Represents an input image of an interior space.
    
    Attributes:
        id: Unique identifier
        file_path: Local path or URL
        image_array: Raw image as NumPy array
        uploaded_at: Timestamp
        source_type: Origin of the image ('upload', 'url', 'test_sample')
    """
    id: str
    file_path: str
    image_array: np.ndarray
    uploaded_at: datetime
    source_type: str  # 'upload' | 'url' | 'test_sample'


@dataclass
class SegmentationMask:
    """
    Represents the mask used to preserve room construction (walls, ceiling, floor).
    
    Attributes:
        id: Unique identifier
        source_image_id: Foreign key to InteriorImage
        mask_array: Binary mask as NumPy array
        segmentation_method: Method used to create mask
        classes_detected: List of detected object classes
        created_at: Timestamp
    """
    id: str
    source_image_id: str
    mask_array: np.ndarray
    segmentation_method: str  # e.g., 'SAM', 'GroundingDINO+SAM'
    classes_detected: List[str]  # Optional: ['wall', 'floor']
    created_at: datetime


@dataclass
class StylePrompt:
    """
    Encapsulates the user-provided or pre-set style description.
    
    Attributes:
        id: Unique identifier
        title: Short title for the style
        prompt_text: Full text prompt for the style
        allow_layout_change: Whether to allow furniture layout changes
        applied_at: When the prompt was used (Optional)
    """
    id: str
    title: str
    prompt_text: str
    allow_layout_change: bool
    applied_at: Optional[datetime] = None


@dataclass
class GenerationRequest:
    """
    Tracks each call made to Flux with all relevant data.
    
    Attributes:
        id: Unique identifier
        input_image_id: ID of source image
        segmentation_mask_id: ID of mask (Optional)
        prompt_id: ID of style prompt
        model_used: Model name
        fal_model_id: FAL-specific model identifier
        allow_layout_change: Whether layout changes are allowed
        run_mode: Type of transformation
        created_at: Timestamp
        status: Current status of request
    """
    id: str
    input_image_id: str
    segmentation_mask_id: Optional[str]
    prompt_id: str
    model_used: str = "flux"
    fal_model_id: str = "fal-ai/flux"
    allow_layout_change: bool = False
    run_mode: str = "style_only"  # style_only | style_and_layout | furnish_empty
    created_at: datetime
    status: str = "pending"  # pending | complete | error


@dataclass
class GenerationResult:
    """
    Stores the results and metadata for each transformation.
    
    Attributes:
        id: Unique identifier
        generation_request_id: ID of associated request
        output_image_array: Generated image as NumPy array
        output_path: Path where output is saved (Optional)
        similarity_score: Structural similarity score (Optional)
        completed_at: Timestamp
    """
    id: str
    generation_request_id: str
    output_image_array: np.ndarray
    output_path: Optional[str] = None
    similarity_score: Optional[float] = None  # (e.g., SSIM vs input)
    completed_at: datetime


@dataclass
class EvaluationResult:
    """
    For testing structural similarity or prompt effectiveness.
    
    Attributes:
        id: Unique identifier
        input_image_id: ID of source image
        output_image_id: ID of generated image
        ssim_score: Structural similarity score
        mse_score: Mean squared error
        is_structure_preserved: Whether structure was maintained
        human_rating: Optional human rating (1-5)
        evaluated_at: Timestamp
    """
    id: str
    input_image_id: str
    output_image_id: str
    ssim_score: float
    mse_score: float
    is_structure_preserved: bool
    human_rating: Optional[int] = None  # 1–5 scale for realism
    evaluated_at: datetime
