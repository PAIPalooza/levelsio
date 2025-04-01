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
import argparse
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
    
    def list_issues(self, state: str = "open") -> List[Dict[str, Any]]:
        """
        List GitHub issues from the repository.
        
        Args:
            state: Issue state ('open', 'closed', 'all')
            
        Returns:
            List of issue dictionaries
        """
        params = {
            "state": state,
            "per_page": 100
        }
        
        response = requests.get(
            self.api_url,
            headers=self.headers,
            params=params
        )
        
        if response.status_code == 200:
            issues = response.json()
            return issues
        else:
            print(f"Error fetching issues: {response.status_code}")
            print(response.text)
            return []


def main():
    """Main script execution."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="GitHub Issue Creator for Interior Style Transfer POC")
    parser.add_argument("-l", "--list", action="store_true", help="List GitHub issues")
    parser.add_argument("-c", "--create", action="store_true", help="Create GitHub issues from backlog.json")
    parser.add_argument("-s", "--state", type=str, default="open", choices=["open", "closed", "all"], 
                        help="Issue state to list (open, closed, all)")
    args = parser.parse_args()
    
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
    
    if args.list:
        # List issues from GitHub repository
        print(f"Listing {args.state} issues from {owner}/{repo}...")
        issues = creator.list_issues(args.state)
        if issues:
            for i, issue in enumerate(issues):
                issue_number = issue.get("number", "N/A")
                issue_title = issue.get("title", "No title")
                issue_state = issue.get("state", "unknown")
                issue_labels = ", ".join([label.get("name", "") for label in issue.get("labels", [])])
                
                print(f"#{issue_number} [{issue_state}] {issue_title}")
                if issue_labels:
                    print(f"   Labels: {issue_labels}")
            print(f"\nTotal: {len(issues)} {args.state} issues found.")
        else:
            print(f"No {args.state} issues found.")
            
    elif args.create:
        # Load and create issues
        issues = creator.load_backlog(backlog_file)
        print(f"Found {len(issues)} issues in backlog")
        
        # Automatically proceed without confirmation
        print(f"Creating {len(issues)} issues in {owner}/{repo}...")
        
        # Create all issues
        creator.create_all_issues(issues)
        print("Issue creation completed!")
    else:
        # No arguments provided, display help
        parser.print_help()


if __name__ == "__main__":
    main()
