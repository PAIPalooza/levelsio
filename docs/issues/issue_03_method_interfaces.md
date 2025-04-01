# Issue: Standardize Method Interfaces

## Description
Several tests are failing due to interface changes in methods like `apply_mask`. Following SSCS standards, we need to ensure all method interfaces are consistent and properly documented throughout the codebase.

## User Story (BDD)
**GIVEN** a method interface has changed  
**WHEN** calling the method from different parts of the codebase  
**THEN** all calls should use the updated interface consistently

## Tasks
- [ ] Update all calls to `apply_mask` to match the new method signature
- [ ] Fix test failures related to parameter mismatches
- [ ] Ensure method signatures are consistent across all modules
- [ ] Update method documentation to reflect current parameters
- [ ] Add BDD-style documentation to all updated methods

## Failing Tests
```
FAILED tests/test_segmentation_public.py::TestSegmentationPublicInterfaces::test_apply_mask - TypeError: SegmentationHandler.apply_mask() got an unexpected keyword argument
```

## Affected Methods
- `SegmentationHandler.apply_mask()`
- Any other methods with inconsistent interfaces

## Acceptance Criteria
- All method calls use consistent parameters
- Documentation accurately reflects current method signatures
- Tests pass without parameter errors
- Documentation follows SSCS standards with GIVEN/WHEN/THEN format
- No regression in existing functionality

## Effort Estimate
Story points: 2 (Low-Medium complexity)
