# GitHub Integration Tools

## Overview
This document outlines the GitHub integration tools used in the Interior Style Transfer POC project. These tools follow the Semantic Seed Coding Standards (SSCS) V2.0 for consistent code organization and formatting.

## Available Tools

### 1. GitHub Issue Creator
**File:** `scripts/github/create_github_issues.py`

This script reads a backlog JSON file and creates GitHub issues based on its content.

**Usage:**
```bash
python scripts/github/create_github_issues.py
```

**Key Features:**
- Creates issues from structured backlog data
- Supports labels, assignees, and milestones
- Follows SSCS BDD conventions in descriptions
- Supports issue templates

**Required Environment Variables:**
- `GITHUB_TOKEN`: Personal access token with 'repo' scope

### 2. GitHub PR Creator
**File:** `scripts/github/create_github_pr.py`

Creates pull requests following SSCS standards with proper descriptions and linking to issues.

**Usage:**
```bash
python scripts/github/create_github_pr.py --head feature/branch-name --base main
```

**Key Features:**
- Automatically formats PR description from templates
- Links related issues
- Adds appropriate labels
- Follows SSCS PR structure

**Arguments:**
- `--head`: Source branch name (required)
- `--base`: Target branch for the PR (default: main)
- `--title`: PR title (optional)
- `--template`: PR template file (default: docs/pr_description.md)

### 3. Issue Updater
**File:** `scripts/github/update_github_issue.py`

Updates existing GitHub issues with new information or statuses.

**Usage:**
```bash
python scripts/github/update_github_issue.py --issue 15 --state closed
```

**Key Features:**
- Updates issue status (open/closed)
- Adds comments to issues
- Modifies labels and assignees
- Works with project boards

### 4. Issue Batch Creator
**File:** `scripts/github/create_remaining_issues.py`

Creates multiple GitHub issues in batch mode for sprint planning.

**Usage:**
```bash
python scripts/github/create_remaining_issues.py
```

**Key Features:**
- Creates multiple issues from a predefined list
- Supports batch creation for sprints
- Follows SSCS issue structure

## Integration with Workflow

These GitHub tools are integrated with the development workflow following SSCS principles:

1. **Backlog Management**
   - Issues are created using proper story formats
   - Issue templates ensure consistent information
   - PRs link back to issues for traceability

2. **Story Types & Estimation**
   - Issues include story points and type classification
   - Labels follow SSCS naming conventions

3. **Git/GitHub Workflow**
   - PRs follow the naming convention `type/id-description`
   - Required fields ensure quality documentation

## Best Practices

When using these GitHub integration tools:

1. Always include issue number references in commits
2. Follow SSCS PR description format
3. Use labels consistently for filtering
4. Keep PR descriptions concise but informative
5. Include test instructions when applicable

## Security Considerations

These scripts handle GitHub tokens which should be properly secured:

1. Never commit tokens to the repository
2. Use environment variables or .env files (listed in .gitignore)
3. Use tokens with minimum required permissions
4. Rotate tokens regularly

## Future Improvements

Planned enhancements to GitHub integration tools:

1. Add GitHub Actions integration
2. Implement PR status checking
3. Add automated code review feedbacks
4. Create dashboard for sprint tracking
