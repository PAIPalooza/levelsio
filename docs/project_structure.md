# Interior Style Transfer API: Project Structure

## Overview
This document outlines the reorganized project structure following Semantic Seed Coding Standards (SSCS) V2.0. The project organization follows the XP-oriented approach with clear separation of concerns and logical grouping of related functionality.

## Directory Structure

```
levelsio/
├── docs/                    # Documentation files
│   ├── api_integration.md   # API integration guidelines
│   ├── github_integration.md# GitHub tools documentation
│   ├── PRD.md               # Product Requirements Document
│   ├── PRD_ANALYSIS.md      # PRD Gap Analysis
│   ├── project_structure.md # This document
│   ├── SSCS_QA_REVIEW.md    # Quality review following SSCS
│   └── issues/              # Issue templates for GitHub
│
├── notebooks/               # Jupyter notebooks for demos
│   ├── basic_demo.py        # Basic demonstration of functionality
│   └── evaluation_reports/  # Generated evaluation reports
│
├── scripts/                 # Utility scripts 
│   ├── github/              # GitHub integration tools
│   │   ├── create_github_issues.py     # Issue creation
│   │   ├── create_github_pr.py         # PR creation
│   │   ├── create_remaining_issues.py  # Batch issue creation
│   │   └── update_github_issue.py      # Issue updates
│   │
│   ├── notebooks/           # Notebook generation tools
│   │   ├── create_notebook.py          # Notebook creator
│   │   └── simple_notebook_creator.py  # Simplified creator
│   │
│   └── server/              # Server tools
│       └── run_api_server.py           # API server runner
│
├── src/                     # Source code
│   ├── api/                 # API implementation
│   │   ├── core/            # Core API functionality
│   │   ├── models/          # Data models
│   │   └── routes/          # API endpoints
│   │
│   ├── data_model.py        # Data model definitions
│   ├── evaluation.py        # Evaluation functionality
│   ├── generation.py        # Image generation
│   ├── segmentation.py      # Image segmentation
│   └── style_transfer.py    # Style transfer logic
│
├── tests/                   # Test suite
│   ├── test_api.py          # API tests
│   ├── test_evaluation.py   # Evaluation tests
│   ├── test_segmentation.py # Segmentation tests
│   └── ...                  # Other test files
│
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
├── PR_DESCRIPTION.md        # PR template for GitHub
├── README.md                # Project overview
└── requirements.txt         # Python dependencies
```

## SSCS Conformance

The project reorganization follows key principles from SSCS V2.0:

### 1. Code Structure
- Clear separation of concerns (src, tests, docs, scripts)
- Logical grouping of related functionality
- Consistent naming conventions

### 2. Development Workflow
- GitHub integration tools follow standard workflow
- Issue templates ensure consistent information
- PR templates follow SSCS guidelines

### 3. Documentation
- Comprehensive documentation in /docs
- API usage examples
- Detailed developer guides
- BDD-style documentation

### 4. Testing Organization
- Tests organized by module
- BDD-style test organization
- Coverage reports

## Usage Guidelines

### Running Scripts
All scripts should now be run from the project root using their paths in the scripts directory:

```bash
# GitHub Issue Creation
python scripts/github/create_github_issues.py

# Running API Server
python scripts/server/run_api_server.py
```

### Running Tests
Tests continue to be run from the project root:

```bash
python -m pytest
```

### Documentation
All documentation is now centralized in the /docs directory, with specific documents for different aspects of the project.

## Future Improvements
- Add script to generate new documentation from code comments
- Create a unified command interface for all scripts
- Implement linting and formatting via pre-commit hooks
