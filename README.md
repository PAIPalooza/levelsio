# Interior Style Transfer POC

Interior Style Transfer POC using Flux on @fal - A proof-of-concept for AI-driven interior design transformations that preserves architectural structure.

## Project Overview

This project demonstrates an AI-driven interior design system that uses Flux via @fal to perform style transfer on input images of interiors. It enables three key transformations while preserving the room's architectural structure:

1. **Style-only transfer**: Change the style/theme of a room while maintaining layout
2. **Style+layout transfer**: Change both style and furniture arrangement
3. **Empty room furnishing**: Add furnishings to an empty interior in a specified style

The system uses auto-segmentation and masking to ensure structural elements (walls, floors, windows) remain intact.

## Features

### Image Segmentation
The application uses the Segment Anything Model (SAM) to identify different elements in interior images:
- Walls
- Floors
- Ceilings
- Windows
- Combined structure masks

### Structure Preservation
When applying style transfer to interior images, the application preserves architectural elements:
- Maintains structural integrity of rooms (walls, floors, ceilings)
- Applies style changes primarily to furniture and decorative elements
- Supports multiple style variations while maintaining the same structure
- Provides scoring mechanism to evaluate structure preservation quality

## Installation

### Prerequisites
- Python 3.11+
- pip or conda

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/PAIPalooza/levelsio.git
   cd levelsio
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up @fal API key (required for Flux model access):
   - Create an account at [fal.ai](https://fal.ai)
   - Generate an API key
   - Set as environment variable: `export FAL_KEY=your_api_key`

## Usage

1. Prepare an interior image and place it in `assets/test_images/`
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/interior-style-transfer.ipynb
   ```
3. Follow the instructions in the notebook to:
   - Load and segment your image
   - Apply style transformations
   - View and save results

## Project Structure

```
project/
├── assets/             # Image assets
│   ├── test_images/    # Input images
│   ├── outputs/        # Generated outputs
│   └── masks/          # Generated masks
├── src/                # Core Python modules
├── notebooks/          # Jupyter notebooks
├── models/             # Pre-trained model weights
└── tests/              # BDD-style tests
```

## Technologies

- **Segment Anything (SAM)**: For automatic segmentation
- **Flux (via @fal)**: For style transfer generation
- **OpenCV & PIL**: For image processing
- **PyTorch**: Deep learning framework
- **Jupyter**: Interactive notebooks

## Development

This project follows Test-Driven Development (TDD) and Behavior-Driven Development (BDD) methodologies aligned with Semantic Seed Coding Standards. To contribute:

1. Check out the relevant issue
2. Create a feature branch: `git checkout -b feature/issue-id-description`
3. Write tests before implementation
4. Implement your solution
5. Run tests: `pytest`
6. Submit a pull request

## License

[Add license information]

## Acknowledgments

- @fal.ai for API access
- Meta AI for Segment Anything Model
