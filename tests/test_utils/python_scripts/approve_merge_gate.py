# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3
"""
Approve pending deployments for workflow runs from PRs targeting a specific branch.

Requirements:
    pip install PyGithub

Usage:
    export GH_TOKEN="ghp_..."
    export REPO="NVIDIA/Megatron-LM"
    export TARGET_BRANCH="main"
    export STATUS="approved"
    export COMMENT="Auto-approved by CI"
    
    python approve_pending_deployments.py
"""

import logging
import os
import re
import sys

from github import Github, GithubException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Get environment variables
    github_token = os.environ.get("GH_TOKEN")
    repo_name = os.environ.get("REPO")
    target_branch = os.environ.get("TARGET_BRANCH")
    status = os.environ.get("STATUS")
    comment = os.environ.get("COMMENT", "")

    if not all([github_token, repo_name, target_branch, status]):
        logger.error(
            "Error: GITHUB_TOKEN, REPO, TARGET_BRANCH, and STATUS environment variables must be set"
        )
        sys.exit(1)

    # Initialize GitHub client
    g = Github(github_token)

    try:
        repo = g.get_repo(repo_name)
    except GithubException as e:
        logger.error(f"Error accessing repository: {e}")
        sys.exit(1)

    # Get merge-gate environment ID
    env_id = None
    try:
        # Note: PyGithub doesn't have direct environment support yet,
        # so we use the underlying requester
        response = repo._requester.requestJsonAndCheck("GET", f"{repo.url}/environments")
        for env in response[1].get("environments", []):
            if env.get("name") == "merge-gate":
                env_id = env.get("id")
                break

        if not env_id:
            logger.error("Error: merge-gate environment not found")
            sys.exit(1)
    except GithubException as e:
        logger.error(f"Error fetching environments: {e}")
        sys.exit(1)

    logger.info(f"merge-gate environment ID: {env_id}")

    # Get waiting workflow runs
    try:
        workflow_runs = repo.get_workflow_runs(status="waiting")
    except GithubException as e:
        logger.error(f"Error fetching workflow runs: {e}")
        sys.exit(1)

    logger.info(f"Found {workflow_runs.totalCount} waiting workflow runs")

    # Process each workflow run
    for run in workflow_runs:
        head_branch = run.head_branch

        # Extract PR number from branch pattern pull-request/(\d+)
        match = re.search(r"gh-readonly-queue/([^/]+)/pr-(\d+)-", head_branch)
        if not match:
            logger.info(f"Skipping Run #{run.id} on {head_branch}: not a PR branch")
            continue

        branch_name = match.group(1)
        pr_number = int(match.group(2))
        logger.info(f"Processing PR #{pr_number} from run {run.id}")

        if branch_name != target_branch:
            logger.info(f"Skipping run {run.id}: targets {branch_name}, not {target_branch}")
            continue

        logger.info(f"Processing PR #{pr_number} from run {run.id} (branch: {branch_name})")

        # Approve pending deployment
        try:
            # PyGithub doesn't have direct support for pending deployments API
            # Use the underlying requester
            repo._requester.requestJsonAndCheck(
                "POST",
                f"{repo.url}/actions/runs/{run.id}/pending_deployments",
                input={"environment_ids": [env_id], "state": status, "comment": comment},
            )
            logger.info(f"✓ Successfully updated deployment for run {run.id} (PR #{pr_number})")
        except GithubException as e:
            logger.info(f"✗ Failed to update deployment for run {run.id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
