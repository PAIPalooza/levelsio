#!/usr/bin/env python3
"""
GitHub Pull Request Creator for Interior Style Transfer POC

This script creates pull requests following the Semantic Seed Coding Standards (SSCS).
It helps automate the PR creation process and ensures PR descriptions follow
the defined format in docs/pr_description.md.

Usage:
    1. Ensure you have a GitHub token in .env as GITHUB_TOKEN
    2. Run: python create_github_pr.py --head feature/branch-name --base main
"""

import os
import argparse
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class GitHubPRCreator:
    """Creates GitHub pull requests following SSCS standards."""
    
    def __init__(self, owner: str, repo: str, token: str):
        """
        Initialize the GitHub PR creator.
        
        Args:
            owner: GitHub repository owner/organization
            repo: GitHub repository name
            token: GitHub personal access token
        """
        self.owner = owner
        self.repo = repo
        self.token = token
        self.api_base_url = f"https://api.github.com/repos/{owner}/{repo}"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def create_pull_request(self, head: str, base: str, title: str, body: str, 
                           draft: bool = False) -> Dict[str, Any]:
        """
        Create a new pull request.
        
        Args:
            head: Branch with changes
            base: Branch to merge into
            title: PR title
            body: PR description
            draft: Whether to create as draft PR
            
        Returns:
            API response as dictionary
        """
        url = f"{self.api_base_url}/pulls"
        payload = {
            "head": head,
            "base": base,
            "title": title,
            "body": body,
            "draft": draft
        }
        
        print(f"Creating PR from {head} to {base}")
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            print(f"✅ PR created successfully! URL: {result['html_url']}")
            return result
        except requests.exceptions.HTTPError as e:
            if response.status_code == 422 and "pull request already exists" in response.text.lower():
                print(f"⚠️ A pull request already exists for {head} to {base}")
                # Find the existing PR
                existing_prs = self.list_pull_requests(head=head, base=base)
                if existing_prs:
                    pr_url = existing_prs[0]["html_url"]
                    print(f"Existing PR URL: {pr_url}")
                return {"message": "Pull request already exists", "html_url": pr_url if 'pr_url' in locals() else None}
            print(f"❌ Error creating PR: {e}")
            print(f"Response: {response.text}")
            raise
    
    def list_pull_requests(self, state: str = "open", head: Optional[str] = None, 
                          base: Optional[str] = None) -> list:
        """
        List pull requests with optional filtering.
        
        Args:
            state: PR state (open, closed, all)
            head: Filter by head branch
            base: Filter by base branch
            
        Returns:
            List of PRs
        """
        url = f"{self.api_base_url}/pulls"
        params = {"state": state}
        
        if head:
            # For cross-repository PRs, use format owner:branch
            # For same repository PRs, just use branch name
            params["head"] = f"{self.owner}:{head}" if ":" not in head else head
            
        if base:
            params["base"] = base
            
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_pr_description_template(self) -> str:
        """
        Get PR description template from docs/pr_description.md if available.
        
        Returns:
            PR description template text
        """
        template_path = "docs/pr_description.md"
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()
        else:
            return """# PR: Title

## Story Summary
Brief description of the feature/bug/chore.

## Changes Made
- List significant changes
- Include implementation details
- Highlight key decisions or tradeoffs

## Test Results
- Describe test coverage
- Note any performance impacts
- Mention edge cases tested

## Documentation
- List updated documentation
- Note API changes
- Include screenshots if relevant

## Related Issues
Closes #X (replace X with issue number)

## Checklist
- [ ] Code follows SSCS coding standards
- [ ] Tests for all functionality have been added/updated and pass
- [ ] Documentation has been updated
- [ ] Error handling has been implemented
- [ ] Code has been reviewed for security considerations
- [ ] Branch has been rebased against main/master
"""


def main():
    """Main function to parse arguments and create GitHub PRs."""
    # Load environment variables
    load_dotenv()
    
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ Error: GITHUB_TOKEN not found in environment variables")
        exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create GitHub PRs following SSCS standards")
    parser.add_argument("--head", required=True, help="Branch with changes")
    parser.add_argument("--base", default="main", help="Branch to merge into (default: main)")
    parser.add_argument("--title", help="PR title (default: derived from branch name)")
    parser.add_argument("--body", help="PR description (default: from template)")
    parser.add_argument("--draft", action="store_true", help="Create as draft PR")
    parser.add_argument("--issue", type=int, help="Issue number this PR addresses")
    args = parser.parse_args()
    
    # Initialize PR creator
    pr_creator = GitHubPRCreator(
        owner="PAIPalooza",
        repo="levelsio",
        token=github_token
    )
    
    # Generate title from branch name if not provided
    if not args.title:
        # Extract from branch name like feature/123-description or bug/456-description
        branch_parts = args.head.split("/")
        if len(branch_parts) >= 2:
            branch_type = branch_parts[0].capitalize()  # feature, bug, chore
            # Extract issue number and description if in format X-Y
            if "-" in branch_parts[1]:
                issue_num, description = branch_parts[1].split("-", 1)
                # Convert kebab-case to Title Case
                description = " ".join(word.capitalize() for word in description.split("-"))
                title = f"{branch_type}: {description}"
            else:
                title = f"{branch_type}: {branch_parts[1].capitalize()}"
        else:
            title = args.head.capitalize()
    else:
        title = args.title
    
    # Get PR description
    if not args.body:
        body = pr_creator.get_pr_description_template()
        
        # Add issue reference if provided
        if args.issue:
            body = body.replace("Closes #X", f"Closes #{args.issue}")
    else:
        body = args.body
    
    # Create the PR
    pr = pr_creator.create_pull_request(
        head=args.head,
        base=args.base,
        title=title,
        body=body,
        draft=args.draft
    )
    
    return pr


if __name__ == "__main__":
    main()
