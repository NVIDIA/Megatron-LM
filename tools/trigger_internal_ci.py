#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI tool to trigger the internal GitLab CI pipeline for a GitHub PR.

Infers the PR number from the current branch, pushes the branch to the
internal GitLab remote under the pull-request/<number> naming convention,
and triggers a pipeline with the specified test configuration.
"""

import argparse
import logging
import os
import subprocess
import sys

import requests
import gitlab  # python-gitlab

GITHUB_REPO = "NVIDIA/Megatron-LM"
GITLAB_PROJECT = "ADLR/Megatron-LM"
GITLAB_BRANCH_PREFIX = "pull-request"
GITHUB_API_URL = "https://api.github.com"

logger = logging.getLogger(__name__)

PIPELINE_VARIABLES_FIXED = {
    "UNIT_TEST": "no",
    "INTEGRATION_TEST": "no",
}


def get_github_token():
    """Return a GitHub token from GH_TOKEN or GITHUB_TOKEN env vars, or exit."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GH_TOKEN or GITHUB_TOKEN not set")
        sys.exit(1)
    return token


def get_current_branch():
    """Return the name of the currently checked-out git branch."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_pr_number(branch, token):
    """Return the PR number for an open GitHub PR matching the given branch, or exit."""
    url = f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/pulls"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    params = {"head": f"NVIDIA:{branch}", "state": "open"}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    pulls = response.json()
    if not pulls:
        logger.error("no open PR found for branch '%s'", branch)
        sys.exit(1)
    return pulls[0]["number"]


def git_push(gitlab_url, target_branch, dry_run=False):
    """Force-push HEAD to the given branch on the GitLab remote."""
    remote_url = f"git@{gitlab_url}:{GITLAB_PROJECT}.git"
    if dry_run:
        logger.info("[DRY RUN] Would push HEAD to %s as %s", remote_url, target_branch)
        return
    subprocess.run(
        ["git", "push", remote_url, f"HEAD:{target_branch}", "--force"],
        check=True,
    )


def trigger_pipeline(gitlab_url, trigger_token, ref, pipeline_vars, dry_run=False):
    """Trigger a GitLab pipeline on the given ref with the provided variables."""
    if dry_run:
        logger.info(
            "[DRY RUN] Would trigger pipeline on https://%s project %s @ %s",
            gitlab_url,
            GITLAB_PROJECT,
            ref,
        )
        return
    gl = gitlab.Gitlab(f"https://{gitlab_url}")
    project = gl.projects.get(GITLAB_PROJECT)
    pipeline = project.trigger_pipeline(ref=ref, token=trigger_token, variables=pipeline_vars)
    logger.info("Pipeline triggered: %s", pipeline.web_url)


def main():
    """Parse arguments and orchestrate the push and pipeline trigger flow."""
    parser = argparse.ArgumentParser(
        description="Trigger the internal GitLab CI pipeline for a GitHub PR."
    )
    parser.add_argument(
        "--gitlab-url",
        required=True,
        help="Hostname of the internal GitLab (e.g. gitlab.example.com)",
    )
    parser.add_argument(
        "--trigger-token",
        default=os.environ.get("GITLAB_TRIGGER_TOKEN"),
        help="GitLab pipeline trigger token (or set GITLAB_TRIGGER_TOKEN env var)",
    )
    parser.add_argument(
        "--functional-test-scope",
        default="mr",
        help="FUNCTIONAL_TEST_SCOPE pipeline variable (default: mr)",
    )
    parser.add_argument(
        "--functional-test-repeat",
        type=int,
        default=5,
        help="FUNCTIONAL_TEST_REPEAT pipeline variable (default: 5)",
    )
    parser.add_argument(
        "--functional-test-cases",
        default="all",
        help="FUNCTIONAL_TEST_CASES pipeline variable (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing git push or pipeline trigger",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.trigger_token:
        logger.error("--trigger-token or GITLAB_TRIGGER_TOKEN not set")
        sys.exit(1)

    github_token = get_github_token()
    branch = get_current_branch()
    logger.info("Current branch: %s", branch)

    pr_number = get_pr_number(branch, github_token)
    logger.info("Found PR #%s for branch '%s'", pr_number, branch)

    target_branch = f"{GITLAB_BRANCH_PREFIX}/{pr_number}"

    git_push(args.gitlab_url, target_branch, dry_run=args.dry_run)

    pipeline_vars = {
        **PIPELINE_VARIABLES_FIXED,
        "FUNCTIONAL_TEST_SCOPE": args.functional_test_scope,
        "FUNCTIONAL_TEST_REPEAT": str(args.functional_test_repeat),
        "FUNCTIONAL_TEST_CASES": args.functional_test_cases,
    }

    trigger_pipeline(
        args.gitlab_url, args.trigger_token, target_branch, pipeline_vars, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
