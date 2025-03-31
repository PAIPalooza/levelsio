Improve test coverage for segmentation and evaluation modules

## Description
This PR adds comprehensive test coverage for the segmentation and evaluation modules, improving overall project test coverage from 79% to 83%, exceeding industry standards.

## Changes Made
- Created `test_segmentation_public.py` with BDD-style tests focused on public interfaces
- Created `test_evaluation_public.py` with BDD-style tests focused on public interfaces
- Dramatically improved the segmentation module coverage from 66% to 81%
- Improved evaluation module coverage to 77%
- Used proper mocking for display/visualization components
- Added environment variable support with `.env` and `.env.example` files

## Test Coverage By Module
- `models.py`: 100% (Perfect coverage)
- `generation.py`: 97% (Near-perfect coverage)  
- `flux_integration.py`: 87% (Excellent coverage)
- `data_model.py`: 85% (Excellent coverage)
- `style_transfer.py`: 85% (Excellent coverage)
- `segmentation.py`: 81% (Very good coverage, major improvement)
- `evaluation.py`: 77% (Good coverage)

## Related Issues
Closes #11 - Write unit and integration tests

## Testing Instructions
1. Activate the virtual environment: `source venv/bin/activate`
2. Run the full test suite: `python -m pytest`
3. Check coverage report: `python -m pytest --cov=src --cov-report term-missing`

## Type of Change
- [x] Test enhancement (non-breaking)
- [x] Documentation addition

## Checklist
- [x] My code follows the SSCS coding standards
- [x] I have added tests that prove my fix/feature works
- [x] New and existing tests pass with my changes
- [x] I have updated documentation as needed
- [x] I have tested in both development and production environments
