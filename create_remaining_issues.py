#!/usr/bin/env python3
"""
GitHub Issue Creator for Remaining Tasks

This script creates GitHub issues for the remaining coverage gaps and adds a new story 
for Swagger support. It follows the Semantic Seed Coding Standards (SSCS).
"""

import os
import requests
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# GitHub configuration
OWNER = "PAIPalooza"
REPO = "levelsio"
TOKEN = os.getenv("GITHUB_TOKEN")  # Using getenv from dotenv

# API endpoints
API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Define the issues to create
ISSUES = [
    {
        "title": "Improve coverage for segmentation.py private methods",
        "body": """## Description
Currently, the `segmentation.py` module has 81% test coverage, with several private methods remaining untested. 
This technical debt task aims to improve coverage for these methods.

## Missing Coverage Areas
- Lines 29-31: SAM import handling
- Lines 82-83, 123, 138-139, 149-150, 158-163: Checkpoint verification and download
- Lines 201-203, 210-214: Model initialization
- Lines 236-259: Mask generation
- Lines 346-348, 459-462, 487, 490, 535, 573, 576, 587: Visualization and processing

## Acceptance Criteria
1. Create tests that target these specific uncovered areas
2. Improve overall coverage to at least 90%
3. Follow BDD-style testing pattern per SSCS
4. Ensure all tests run in CI pipeline without errors

## Type
- [x] Chore (Technical Debt)

## Points
2
""",
        "labels": ["chore", "test", "technical-debt"]
    },
    {
        "title": "Improve coverage for evaluation.py methods",
        "body": """## Description
Currently, the `evaluation.py` module has 77% test coverage, with several methods remaining untested.
This technical debt task aims to improve coverage for these methods.

## Missing Coverage Areas
- Lines 25-27, 41, 59: SSIM calculation with different environments
- Lines 80-102: MSE calculation with masking
- Lines 121, 130, 137, 142: Metrics calculation
- Lines 241, 262, 267, 269: Preservation scoring
- Lines 303, 311-312, 318-324: Visualization

## Acceptance Criteria
1. Create tests that target these specific uncovered areas
2. Improve overall coverage to at least 90%
3. Follow BDD-style testing pattern per SSCS
4. Ensure all tests run in CI pipeline without errors

## Type
- [x] Chore (Technical Debt)

## Points
2
""",
        "labels": ["chore", "test", "technical-debt"]
    },
    {
        "title": "Improve coverage for style_transfer.py high-risk areas",
        "body": """## Description
Currently, the `style_transfer.py` module has 85% test coverage, but some critical areas remain untested.
This task focuses on testing high-risk areas that could cause issues in production.

## Missing Coverage Areas
- Line 110: Error handling in API responses
- Lines 160-161: Style transfer options parsing
- Line 207: Multi-style processing
- Lines 219-236, 258-259: Batch processing and result handling

## Acceptance Criteria
1. Create tests that target these specific high-risk areas
2. Improve overall coverage to at least 90% 
3. Follow BDD-style testing pattern per SSCS
4. Ensure all tests run in CI pipeline without errors

## Type
- [x] Chore (Technical Debt)

## Points
1
""",
        "labels": ["chore", "test", "technical-debt"]
    },
    {
        "title": "Add Swagger/OpenAPI support for REST API endpoints",
        "body": """## Description
The Interior Style Transfer POC would benefit from Swagger/OpenAPI documentation and interactive API exploration. This feature will make it easier for developers to understand and use our API.

## User Story
As a developer using the Interior Style Transfer API, I want to have interactive API documentation so that I can easily understand the endpoints, parameters, and responses.

## Technical Details
1. Integrate Swagger/OpenAPI with our existing REST API
2. Document all endpoints with:
   - Request parameters and types
   - Response structures and status codes
   - Authentication requirements
   - Example requests and responses
3. Provide an interactive Swagger UI for testing endpoints
4. Generate OpenAPI specification (JSON/YAML) for client code generation

## Acceptance Criteria
1. All endpoints are documented with OpenAPI annotations
2. Swagger UI is accessible at `/api/docs` or similar endpoint
3. Interactive testing works for all endpoints
4. Code examples are provided for Python, JavaScript, and curl
5. Authentication flow is properly documented and integrated with Swagger UI

## Type
- [x] Feature

## Points
3
""",
        "labels": ["feature", "documentation", "api"]
    }
]

def create_issues() -> None:
    """Create GitHub issues for the remaining tasks."""
    if not TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set.")
        print("Set it with: export GITHUB_TOKEN='your-token'")
        return
    
    print(f"Creating {len(ISSUES)} GitHub issues...")
    
    for idx, issue in enumerate(ISSUES, 1):
        print(f"Creating issue {idx}/{len(ISSUES)}: {issue['title']}")
        response = requests.post(API_URL, headers=HEADERS, json=issue)
        
        if response.status_code == 201:
            issue_url = response.json()["html_url"]
            print(f"✅ Success! Issue created at: {issue_url}")
        else:
            print(f"❌ Error creating issue: {response.status_code}")
            print(response.text)
        
        # Be nice to the GitHub API
        if idx < len(ISSUES):
            print("Waiting 2 seconds before creating next issue...")
            import time
            time.sleep(2)
    
    print("Done creating issues!")

if __name__ == "__main__":
    create_issues()
