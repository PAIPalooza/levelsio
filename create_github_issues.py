#!/usr/bin/env python3
"""
GitHub Issue Creator for Interior Style Transfer POC

This script reads a backlog.json file and creates GitHub issues based on its content.
It follows the Semantic Seed Coding Standards (SSCS) for code quality and documentation.

Usage:
    1. Set up a GitHub Personal Access Token with 'repo' scope
    2. Run: python create_github_issues.py

Requirements:
    - requests
"""

import json
import os
import requests
import time
from typing import Dict, List, Any


class GitHubIssueCreator:
    """Creates GitHub issues from a JSON backlog file."""
    
    def __init__(self, owner: str, repo: str, token: str):
        """
        Initialize the GitHub issue creator.
        
        Args:
            owner: GitHub repository owner/organization
            repo: GitHub repository name
            token: GitHub personal access token
        """
        self.owner = owner
        self.repo = repo
        self.token = token
        self.api_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def load_backlog(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load issues from a JSON backlog file.
        
        Args:
            file_path: Path to the backlog JSON file
            
        Returns:
            List of issue dictionaries
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def create_issue(self, title: str, body: str, labels: List[str]) -> Dict[str, Any]:
        """
        Create a single GitHub issue.
        
        Args:
            title: Issue title
            body: Issue description
            labels: List of label strings
            
        Returns:
            Response JSON if successful
        """
        payload = {
            "title": title,
            "body": body,
            "labels": labels
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            print(f"Error creating issue '{title}': {response.status_code}")
            print(response.text)
            return None
    
    def create_all_issues(self, issues: List[Dict[str, Any]]) -> None:
        """
        Create all issues from the parsed backlog.
        
        Args:
            issues: List of issue dictionaries
        """
        for i, issue in enumerate(issues):
            print(f"Creating issue {i+1}/{len(issues)}: {issue['title']}")
            result = self.create_issue(
                title=issue["title"],
                body=issue["body"],
                labels=issue["labels"]
            )
            
            if result:
                print(f"✅ Created issue #{result['number']}: {result['title']}")
            else:
                print(f"❌ Failed to create issue: {issue['title']}")
            
            # Rate limiting - avoid hitting GitHub API limits
            time.sleep(1)


def main():
    """Main script execution."""
    # Get GitHub token from environment or user input
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        token = input("Enter your GitHub Personal Access Token: ")
    
    # Repository information
    owner = "PAIPalooza"
    repo = "levelsio"
    backlog_file = "backlog.json"
    
    # Initialize the issue creator
    creator = GitHubIssueCreator(owner, repo, token)
    
    # Load and create issues
    issues = creator.load_backlog(backlog_file)
    print(f"Found {len(issues)} issues in backlog")
    
    # Automatically proceed without confirmation
    print(f"Creating {len(issues)} issues in {owner}/{repo}...")
    
    # Create all issues
    creator.create_all_issues(issues)
    print("Issue creation completed!")


if __name__ == "__main__":
    main()
