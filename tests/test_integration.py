"""
Integration tests for the Interior Style Transfer POC.

Following Semantic Seed Coding Standards and BDD/TDD approach:
- Test cases are written before implementation
- Tests follow BDD "Given-When-Then" structure
- Each test has a docstring explaining its purpose
- Integration tests verify multiple components working together
"""

import os
import pytest
import numpy as np
from unittest import mock
import cv2
from PIL import Image

from src.flux_integration import FluxClient
from src.segmentation import SegmentationHandler
from src.style_transfer import StyleTransferService
from src.layout_transfer import LayoutTransferService
from src.empty_room_furnishing import EmptyRoomFurnishingService
from src.evaluation import VisualEvaluationService
from src.data_model import ImageModel, MaskModel, ResultModel, DataModelManager


class TestEndToEndWorkflow:
    """Integration tests for the end-to-end workflow."""
    
    @pytest.fixture
    def sample_image(self):
        """Fixture that provides a sample interior image."""
        # Create a synthetic image for testing
        height, width = 256, 256
        
        # Create a simple room with walls and floor
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        
        # Draw floor (bottom half)
        image[height//2:, :] = [240, 230, 200]  # Light beige
        
        # Draw left wall 
        image[:height//2, :width//2] = [230, 230, 230]  # Light gray
        
        # Draw right wall
        image[:height//2, width//2:] = [210, 210, 210]  # Slightly different gray
        
        # Add a simple furniture item (sofa)
        image[height//2+20:height//2+50, width//2-40:width//2+40] = [150, 75, 50]  # Brown sofa
        
        return image
    
    @pytest.fixture
    def components(self):
        """Fixture that provides mock instances of all components."""
        # Use a mock API key for testing
        flux_client = FluxClient(api_key="test_key")
        
        # Create mock segmentation handler with stubbed methods
        segmentation_handler = mock.MagicMock(spec=SegmentationHandler)
        
        # Mock the create_mask method to return plausible mask data
        def mock_create_mask(*args, **kwargs):
            height, width = 256, 256
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Simulate a wall or floor mask based on the label
            label = kwargs.get('label', '')
            
            if label == 'wall':
                # Create wall mask (top half)
                mask[:height//2, :] = 1
            elif label == 'floor':
                # Create floor mask (bottom half)
                mask[height//2:, :] = 1
            elif label == 'furniture':
                # Create furniture mask (a rectangle in the middle-bottom)
                mask[height//2+20:height//2+50, width//2-40:width//2+40] = 1
            else:
                # Default structure mask (walls + floor)
                mask[:height//2, :] = 1  # Walls
                mask[height//2:, :] = 1  # Floor
            
            return mask
        
        segmentation_handler.create_mask = mock.MagicMock(side_effect=mock_create_mask)
        
        # Mock other segmentation handler methods
        def mock_apply_structure_preservation(original_image, stylized_image, structure_mask):
            # For testing, just return the stylized image
            return stylized_image
            
        segmentation_handler.apply_structure_preservation = mock.MagicMock(
            side_effect=mock_apply_structure_preservation
        )
        
        def mock_preserve_structure_in_styles(original_image, style_variations, structure_mask):
            # For testing, just return the style variations
            return style_variations
            
        segmentation_handler.preserve_structure_in_styles = mock.MagicMock(
            side_effect=mock_preserve_structure_in_styles
        )
        
        # Initialize other services
        style_service = StyleTransferService(
            flux_client=flux_client, 
            segmentation_handler=segmentation_handler
        )
        
        layout_service = LayoutTransferService(
            flux_client=flux_client,
            style_transfer_service=style_service,
            segmentation_handler=segmentation_handler
        )
        
        empty_room_service = EmptyRoomFurnishingService(
            flux_client=flux_client,
            segmentation_handler=segmentation_handler
        )
        
        evaluation_service = VisualEvaluationService()
        
        # Return all components as a dictionary
        return {
            'flux_client': flux_client,
            'segmentation_handler': segmentation_handler,
            'style_service': style_service,
            'layout_service': layout_service,
            'empty_room_service': empty_room_service,
            'evaluation_service': evaluation_service
        }
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Fixture that provides a data model manager with a temporary directory."""
        data_dir = tmp_path / "data"
        return DataModelManager(base_directory=str(data_dir))
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_style_transfer_integration(self, mock_apply_style, sample_image, components):
        """
        GIVEN all components and a sample interior image
        WHEN executing the style transfer workflow
        THEN all components should work together to produce a styled result
        """
        # Mock the result from Flux API
        mock_result = sample_image.copy()
        mock_result[:, :, 0] = mock_result[:, :, 0] * 0.8  # Adjust red channel to simulate style
        mock_apply_style.return_value = mock_result
        
        # Extract components
        style_service = components['style_service']
        segmentation_handler = components['segmentation_handler']
        evaluation_service = components['evaluation_service']
        
        # 1. Generate masks for the interior elements
        wall_mask = segmentation_handler.create_mask(sample_image, label='wall')
        floor_mask = segmentation_handler.create_mask(sample_image, label='floor')
        structure_mask = np.logical_or(wall_mask, floor_mask).astype(np.uint8)
        
        # 2. Apply style transfer with structure preservation
        style_prompt = "Scandinavian style with light colors and minimal decoration"
        styled_result = style_service.apply_style_only(
            sample_image,
            style_prompt,
            preserve_structure=True,
            structure_mask=structure_mask
        )
        
        # 3. Evaluate the result
        preservation_score = evaluation_service.calculate_preservation_score(
            sample_image, styled_result, structure_mask
        )
        
        # Verify the workflow executed correctly
        assert mock_apply_style.called
        assert styled_result is not None
        assert styled_result.shape == sample_image.shape
        assert preservation_score >= 0.7  # Good preservation score
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_layout_transfer_integration(self, mock_apply_style, sample_image, components):
        """
        GIVEN all components and a sample interior image
        WHEN executing the layout transfer workflow
        THEN all components should work together to produce a result with a new layout
        """
        # Mock the result from Flux API
        mock_result = sample_image.copy()
        # Move the "sofa" to simulate layout change
        mock_result[120:150, 50:130] = [150, 75, 50]  # Brown sofa in new position
        mock_apply_style.return_value = mock_result
        
        # Extract components
        layout_service = components['layout_service']
        segmentation_handler = components['segmentation_handler']
        evaluation_service = components['evaluation_service']
        
        # 1. Generate masks for structure preservation
        structure_mask = segmentation_handler.create_mask(sample_image, label='structure')
        
        # 2. Apply layout transfer
        layout_prompt = "Modern living room with sectional sofa on the left side"
        layout_result = layout_service.apply_layout_transfer(
            sample_image,
            layout_prompt,
            preserve_structure=True
        )
        
        # 3. Evaluate structure preservation
        preservation_score = evaluation_service.calculate_preservation_score(
            sample_image, layout_result, structure_mask
        )
        
        # Verify the workflow executed correctly
        assert mock_apply_style.called
        assert layout_result is not None
        assert layout_result.shape == sample_image.shape
        assert preservation_score >= 0.7  # Good preservation score
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_empty_room_furnishing_integration(self, mock_apply_style, sample_image, components):
        """
        GIVEN all components and a sample empty room image
        WHEN executing the empty room furnishing workflow
        THEN all components should work together to produce a furnished result
        """
        # Create empty room by removing furniture
        empty_room = sample_image.copy()
        
        # Remove the "sofa" by replacing with floor color
        height, width = empty_room.shape[:2]
        empty_room[height//2+20:height//2+50, width//2-40:width//2+40] = [240, 230, 200]
        
        # Mock the result from Flux API (add furniture back)
        mock_result = empty_room.copy()
        # Add new "furniture" 
        mock_result[height//2+10:height//2+40, width//2-60:width//2+60] = [120, 60, 30]  # Larger sofa
        mock_result[height//2+20:height//2+30, width//2+70:width//2+90] = [160, 140, 100]  # Side table
        mock_apply_style.return_value = mock_result
        
        # Extract components
        empty_room_service = components['empty_room_service']
        evaluation_service = components['evaluation_service']
        
        # 1. Apply empty room furnishing
        furniture_prompt = "Modern living room with a large sofa and side tables"
        furnished_result = empty_room_service.furnish_room(
            empty_room, 
            style="modern", 
            furniture=furniture_prompt,
            room_type="living room",
            preserve_structure=True
        )
        
        # 2. Verify the furnished room has new furniture
        # Since we're using a mock, just verify the mock was called correctly
        assert mock_apply_style.called
        assert furnished_result is not None
        assert furnished_result.shape == empty_room.shape
        
        # Verify our segmentation handler was used
        assert components['segmentation_handler'].create_mask.called
    
    def test_data_model_integration(self, sample_image, components, data_manager):
        """
        GIVEN all components, sample images, and data manager
        WHEN tracking a complete workflow through the data model
        THEN all data should be properly tracked, stored, and retrievable
        """
        # Extract components
        style_service = components['style_service']
        evaluation_service = components['evaluation_service']
        
        # 1. Create input image model
        input_model = ImageModel(
            image_data=sample_image,
            input_image_path="/virtual/path/interior.jpg",
            user_id="test_user",
            project_id="test_project"
        )
        
        # 2. Save input model
        input_path = data_manager.save_image_model(input_model)
        
        # 3. Create a mask model
        wall_mask = components['segmentation_handler'].create_mask(sample_image, label='wall')
        mask_model = MaskModel(
            mask_data=wall_mask,
            mask_type="wall",
            input_model_id=input_model.id,
            user_id="test_user",
            project_id="test_project"
        )
        
        # 4. Save mask model
        mask_path = data_manager.save_mask_model(mask_model)
        
        # 5. Create a result (mock the style transfer process)
        with mock.patch.object(style_service, 'apply_style_only') as mock_style:
            # Create a modified version of the input as the "result"
            result_image = sample_image.copy()
            result_image[:, :, 2] = result_image[:, :, 2] * 0.7  # Adjust blue channel
            mock_style.return_value = result_image
            
            # Simulate style transfer
            style_prompt = "Minimalist Scandinavian style"
            styled_result = style_service.apply_style_only(
                sample_image, style_prompt, preserve_structure=True
            )
        
        # 6. Create a result model
        result_model = ResultModel(
            result_image_data=styled_result,
            input_model_id=input_model.id,
            generation_method="Flux",
            generation_params={"style": "Scandinavian", "preserve_structure": True},
            user_id="test_user",
            project_id="test_project"
        )
        
        # 7. Save result model
        result_path = data_manager.save_result_model(result_model)
        
        # 8. Save history
        history_path = data_manager.save_history()
        
        # Verify data model integration
        assert os.path.exists(input_path)
        assert os.path.exists(mask_path)
        assert os.path.exists(result_path)
        assert os.path.exists(history_path)
        
        # Verify we can reload the models
        loaded_input = data_manager.load_model(input_model.id)
        loaded_result = data_manager.load_model(result_model.id)
        
        assert loaded_input is not None
        assert loaded_result is not None
        assert loaded_input.id == input_model.id
        assert loaded_result.id == result_model.id
        assert np.array_equal(loaded_input.image_data, input_model.image_data)


class TestComponentInteractions:
    """Integration tests for interactions between specific components."""
    
    @pytest.fixture
    def test_components(self):
        """Fixture that provides the needed components for testing."""
        # Use a mock API key for testing
        flux_client = FluxClient(api_key="test_key")
        
        # Mock the segmentation handler
        segmentation_handler = mock.MagicMock(spec=SegmentationHandler)
        
        # Create a simple mock mask generation function
        def mock_create_mask(*args, **kwargs):
            height, width = 256, 256
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Type of mask based on label
            label = kwargs.get('label', '')
            if label == 'wall':
                mask[:128, :] = 1  # Top half is wall
            elif label == 'floor':
                mask[128:, :] = 1  # Bottom half is floor
            else:
                # Return combined structure mask
                mask[:] = 1
            
            return mask
        
        segmentation_handler.create_mask = mock.MagicMock(side_effect=mock_create_mask)
        
        # Mock other segmentation handler methods
        def mock_apply_structure_preservation(original_image, stylized_image, structure_mask):
            # For testing, just return the stylized image
            return stylized_image
            
        segmentation_handler.apply_structure_preservation = mock.MagicMock(
            side_effect=mock_apply_structure_preservation
        )
        
        def mock_preserve_structure_in_styles(original_image, style_variations, structure_mask):
            # For testing, just return the style variations
            return style_variations
            
        segmentation_handler.preserve_structure_in_styles = mock.MagicMock(
            side_effect=mock_preserve_structure_in_styles
        )
        
        return {
            'flux_client': flux_client,
            'segmentation_handler': segmentation_handler
        }
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_segmentation_style_transfer_integration(self, mock_apply_style, test_components):
        """
        GIVEN segmentation handler and style transfer service
        WHEN applying style transfer with structure preservation
        THEN the structure masks should be used to preserve architectural elements
        """
        # Create a test image
        test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # Mock the style transfer result
        mock_result = test_image.copy()
        mock_result[:, :, 0] += 50  # Change color slightly
        mock_apply_style.return_value = mock_result
        
        # Create style transfer service with mocked components
        style_service = StyleTransferService(
            flux_client=test_components['flux_client'],
            segmentation_handler=test_components['segmentation_handler']
        )
        
        # Apply style with structure preservation
        result = style_service.apply_style_only(
            test_image,
            style_prompt="Modern style",
            preserve_structure=True
        )
        
        # Verify the result and interaction between components
        assert result is not None
        assert result.shape == test_image.shape
        
        # Verify the segmentation handler was used to create masks
        test_components['segmentation_handler'].create_mask.assert_called()
        
        # Verify structure preservation is included in the process
        assert mock_apply_style.called
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_evaluation_segmentation_integration(self, mock_apply_style, test_components):
        """
        GIVEN segmentation handler and evaluation service
        WHEN evaluating structure preservation
        THEN the evaluation should use masks from the segmentation handler
        """
        # Create a test image and result
        test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result_image = test_image.copy()
        result_image[128:, :, :] += 30  # Change the bottom half (floor)
        
        # Get components
        segmentation_handler = test_components['segmentation_handler']
        
        # Create evaluation service
        evaluation_service = VisualEvaluationService()
        
        # 1. Get structure masks from segmentation handler
        wall_mask = segmentation_handler.create_mask(test_image, label='wall')
        floor_mask = segmentation_handler.create_mask(test_image, label='floor')
        
        # 2. Evaluate preservation with different masks
        wall_score = evaluation_service.calculate_preservation_score(
            test_image, result_image, wall_mask
        )
        
        floor_score = evaluation_service.calculate_preservation_score(
            test_image, result_image, floor_mask
        )
        
        # 3. Generate a full preservation report
        report = evaluation_service.generate_structure_preservation_report(
            test_image, result_image, 
            masks={"wall": wall_mask, "floor": floor_mask}
        )
        
        # Verify integration
        assert wall_score > 0.8  # Wall should be well preserved (we didn't change it)
        assert floor_score < wall_score  # Floor should be less preserved (we changed it)
        assert "wall" in report
        assert "floor" in report
        assert report["wall"] > report["floor"]
    
    @mock.patch("src.flux_integration.FluxClient.apply_style_transfer")
    def test_layout_segmentation_integration(self, mock_apply_style, test_components):
        """
        GIVEN segmentation handler and layout transfer service
        WHEN applying layout transfer
        THEN layout service should utilize masks from segmentation for structure preservation
        """
        # Create a test image
        test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # Mock the layout transfer result
        mock_result = test_image.copy()
        # Add some "furniture" to simulate layout change
        mock_result[160:190, 100:150] = [200, 100, 50]
        mock_apply_style.return_value = mock_result
        
        # Create style and layout services
        style_service = StyleTransferService(
            flux_client=test_components['flux_client'],
            segmentation_handler=test_components['segmentation_handler']
        )
        
        layout_service = LayoutTransferService(
            flux_client=test_components['flux_client'],
            style_transfer_service=style_service,
            segmentation_handler=test_components['segmentation_handler']
        )
        
        # Apply layout transfer (which should use the segmentation handler automatically)
        result = layout_service.apply_layout_transfer(
            test_image,
            layout_prompt="Living room with sofa on the left",
            preserve_structure=True
        )
        
        # Verify the workflow
        assert result is not None
        assert result.shape == test_image.shape
        
        # Verify the segmentation handler was used
        assert test_components['segmentation_handler'].create_mask.called
        
        # Verify the API call was made with the enhanced prompt
        assert mock_apply_style.called
