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

"""CLI tool to trigger the internal GitLab CI pipeline from a local branch.

Pushes the current branch to the internal GitLab remote under the
pull-request/<branch> naming convention and triggers a pipeline with
the specified test configuration.

Requires a GitLab personal access token with at least the 'api' scope.
Set the GITLAB_TOKEN environment variable or pass --access-token.
"""

import argparse
import logging
import os
import subprocess
import sys
from urllib.parse import urlparse

import gitlab  # python-gitlab

GITLAB_PROJECT_ID = 19378
GITLAB_BRANCH_PREFIX = "pull-request"

PIPELINE_VARIABLES_FIXED = {
    "UNIT_TEST": "no",
    "INTEGRATION_TEST": "no",
}

# Scopes whose recipes run full convergence/checkpointing workloads and need a
# long wall-clock budget. The default short-scope time limit is left untouched.
LONG_RUNNING_SCOPES = ("release", "weekly")
LONG_RUNNING_TIME_LIMIT_SECONDS = 4 * 60 * 60

logger = logging.getLogger(__name__)


def resolve_time_limit(scope, override):
    """Resolve the FUNCTIONAL_TEST_TIME_LIMIT value for a functional test scope.

    Args:
        scope: The functional test scope (e.g. ``mr``, ``release``, ``weekly``).
        override: Explicit time limit in seconds, or ``None`` to auto-resolve.

    Returns:
        The time limit in seconds when one applies, otherwise ``None`` so the
        variable is left unset and short-running scopes keep their default.
    """
    if override is not None:
        return override
    if scope in LONG_RUNNING_SCOPES:
        return LONG_RUNNING_TIME_LIMIT_SECONDS
    return None


def get_remote_url(origin):
    """Return the fetch URL configured for the given git remote name."""
    result = subprocess.run(
        ["git", "remote", "get-url", origin],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_gitlab_hostname(remote_url):
    """Extract the hostname (without port) from an SSH or HTTPS remote URL."""
    if remote_url.startswith("git@"):
        hostname = remote_url.split("@", 1)[1].split(":")[0]
    else:
        hostname = urlparse(remote_url).hostname
    return hostname.split(":")[0]


def get_current_branch():
    """Return the name of the currently checked-out git branch."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_push(origin, target_branch, dry_run=False):
    """Force-push HEAD to the given branch on the named git remote."""
    if dry_run:
        logger.info(
            "[DRY RUN] Would push HEAD to remote '%s' as %s", origin, target_branch
        )
        return
    subprocess.run(
        ["git", "push", origin, f"HEAD:{target_branch}", "--force"],
        check=True,
    )


def trigger_pipeline(gitlab_url, access_token, ref, pipeline_vars, dry_run=False):
    """Trigger a GitLab pipeline on the given ref with the provided variables."""
    if dry_run:
        logger.info(
            "[DRY RUN] Would trigger pipeline on https://%s project %s @ %s",
            gitlab_url,
            GITLAB_PROJECT_ID,
            ref,
        )
        return
    logger.info(
        "Triggering pipeline on https://%s project %s @ %s",
        gitlab_url,
        GITLAB_PROJECT_ID,
        ref,
    )
    gl = gitlab.Gitlab(f"https://{gitlab_url}", private_token=access_token)
    project = gl.projects.get(GITLAB_PROJECT_ID, lazy=True)
    variables = [{"key": k, "value": v} for k, v in pipeline_vars.items()]
    pipeline = project.pipelines.create({"ref": ref, "variables": variables})
    logger.info("Pipeline triggered: %s", pipeline.web_url)


def main():
    """Parse arguments and orchestrate the push and pipeline trigger flow."""
    parser = argparse.ArgumentParser(
        description="Trigger the internal GitLab CI pipeline for the current branch."
    )
    parser.add_argument(
        "--gitlab-origin",
        required=True,
        help="Name of the git remote pointing to the internal GitLab (e.g. gitlab)",
    )
    parser.add_argument(
        "--access-token",
        default=os.environ.get("GITLAB_TOKEN"),
        help="GitLab personal access token with 'api' scope (or set GITLAB_TOKEN env var)",
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
        "--functional-test-name",
        default=None,
        help=(
            "FUNCTIONAL_TEST_NAME pipeline variable — names the run for "
            "pre-release/release scopes (used as the run name and W&B experiment). "
            "Defaults to the commit SHA when omitted."
        ),
    )
    parser.add_argument(
        "--functional-test-time-limit",
        type=int,
        default=None,
        help=(
            "FUNCTIONAL_TEST_TIME_LIMIT pipeline variable in seconds. Defaults to "
            "14400 (4h) for the long-running 'release' and 'weekly' scopes and is "
            "left unset for other scopes."
        ),
    )
    parser.add_argument(
        "--cluster-a100",
        default=None,
        help="CLUSTER_A100 pipeline variable (override the default cluster)",
    )
    parser.add_argument(
        "--cluster-h100",
        default=None,
        help="CLUSTER_H100 pipeline variable (override the default cluster)",
    )
    parser.add_argument(
        "--cluster-gb200",
        default=None,
        help="CLUSTER_GB200 pipeline variable (override the default cluster)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing git push or pipeline trigger",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.access_token:
        logger.error("--access-token or GITLAB_TOKEN not set")
        sys.exit(1)

    branch = get_current_branch()
    logger.info("Current branch: %s", branch)

    remote_url = get_remote_url(args.gitlab_origin)
    gitlab_hostname = get_gitlab_hostname(remote_url)

    target_branch = f"{GITLAB_BRANCH_PREFIX}/{branch}"

    git_push(args.gitlab_origin, target_branch, dry_run=args.dry_run)

    pipeline_vars = {
        **PIPELINE_VARIABLES_FIXED,
        "FUNCTIONAL_TEST_SCOPE": args.functional_test_scope,
        "FUNCTIONAL_TEST_REPEAT": str(args.functional_test_repeat),
        "FUNCTIONAL_TEST_CASES": args.functional_test_cases,
    }

    # Only override FUNCTIONAL_TEST_NAME when explicitly provided; otherwise the
    # pipeline default (the commit SHA) applies.
    if args.functional_test_name is not None:
        pipeline_vars["FUNCTIONAL_TEST_NAME"] = args.functional_test_name

    time_limit = resolve_time_limit(
        args.functional_test_scope, args.functional_test_time_limit
    )
    if time_limit is not None:
        pipeline_vars["FUNCTIONAL_TEST_TIME_LIMIT"] = str(time_limit)

    for var, val in [
        ("CLUSTER_A100", args.cluster_a100),
        ("CLUSTER_H100", args.cluster_h100),
        ("CLUSTER_GB200", args.cluster_gb200),
    ]:
        if val is not None:
            pipeline_vars[var] = val

    trigger_pipeline(
        gitlab_hostname,
        args.access_token,
        target_branch,
        pipeline_vars,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
