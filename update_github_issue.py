#!/usr/bin/env python3
"""
GitHub Issue Updater for Interior Style Transfer POC

This script updates GitHub issues status. It follows the Semantic Seed
Coding Standards (SSCS) for code quality and documentation.

Usage:
    1. Ensure .env has GITHUB_TOKEN set
    2. Run: python update_github_issue.py --issue-number <number> --status <status> [--comment <comment>]
"""

import os
import argparse
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class GitHubIssueUpdater:
    """Updates GitHub issues status and adds comments."""
    
    def __init__(self, owner: str, repo: str, token: str):
        """
        Initialize the GitHub issue updater.
        
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
    
    def update_issue_status(self, issue_number: int, status: str) -> Dict[str, Any]:
        """
        Update the status of a GitHub issue.
        
        Args:
            issue_number: The GitHub issue number to update
            status: New status ('open' or 'closed')
            
        Returns:
            API response as dictionary
        """
        url = f"{self.api_base_url}/issues/{issue_number}"
        payload = {"state": status}
        
        print(f"Updating issue #{issue_number} to status: {status}")
        response = requests.patch(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        print(f"✅ Issue #{issue_number} successfully updated to {status}")
        return response.json()
    
    def add_comment(self, issue_number: int, comment: str) -> Dict[str, Any]:
        """
        Add a comment to a GitHub issue.
        
        Args:
            issue_number: The GitHub issue number to comment on
            comment: Comment text
            
        Returns:
            API response as dictionary
        """
        url = f"{self.api_base_url}/issues/{issue_number}/comments"
        payload = {"body": comment}
        
        print(f"Adding comment to issue #{issue_number}")
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        print(f"✅ Comment successfully added to issue #{issue_number}")
        return response.json()


def main():
    """Main function to parse arguments and update GitHub issues."""
    # Load environment variables
    load_dotenv()
    
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ Error: GITHUB_TOKEN not found in environment variables")
        exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update GitHub issues for Interior Style Transfer POC")
    parser.add_argument("--issue-number", type=int, required=True, help="GitHub issue number to update")
    parser.add_argument("--status", choices=["open", "closed"], help="New status for the issue")
    parser.add_argument("--comment", help="Comment to add to the issue")
    args = parser.parse_args()
    
    # Initialize updater
    updater = GitHubIssueUpdater(
        owner="PAIPalooza",
        repo="levelsio",
        token=github_token
    )
    
    # Update issue status if requested
    if args.status:
        updater.update_issue_status(args.issue_number, args.status)
    
    # Add comment if provided
    if args.comment:
        updater.add_comment(args.issue_number, args.comment)


if __name__ == "__main__":
    main()
