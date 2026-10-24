# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Adapted from noamelf/codeowner-multi-approval-action/check_code_owners_approval.py.
# Original revision: ab1d2dc274612ab9439b96538f30e88be2570f49 (v0.1).
# https://github.com/noamelf/codeowner-multi-approval-action/blob/ab1d2dc274612ab9439b96538f30e88be2570f49/check_code_owners_approval.py
# Byte-identical upstream source at MIT-licensed revision:
# ff4f388f021bdb1bb3653c2b09e3e84f3885f50c
# https://github.com/noamelf/codeowner-multi-approval-action/blob/ff4f388f021bdb1bb3653c2b09e3e84f3885f50c/check_code_owners_approval.py
# License: MIT. The original v0.1 tree did not include a license file.
# https://github.com/noamelf/codeowner-multi-approval-action/blob/ff4f388f021bdb1bb3653c2b09e3e84f3885f50c/LICENSE
# Local modifications: author-specific approval exemption, shared policy integration,
# and fail-closed handling of GitHub API errors.
#
# The upstream copyright and license notice follows:
#
# MIT License
#
# Copyright (c) 2025 noamelf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Require every CODEOWNERS owner to approve, subject to explicit author exemptions.

Run with ``--pr-number`` and ``--repo-name`` and provide ``GITHUB_TOKEN``.
The standalone CI environment requires codeowners and PyGithub.
"""

import argparse
import logging
import os
import sys
from functools import lru_cache

from codeowner_approval_policy import get_exempt_owners
from codeowners import CodeOwners
from github import Github, GithubException

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """Read the same PR and repository arguments as the previous approval action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--repo-name", required=True)
    return parser.parse_args(argv)


def get_codeowners(repo):
    """Read CODEOWNERS from the default branch, in GitHub's location priority order."""
    for path in (".github/CODEOWNERS", "docs/CODEOWNERS", "CODEOWNERS"):
        try:
            contents = repo.get_contents(path)
        except GithubException as error:
            if error.status == 404:
                continue
            raise
        logger.info("Found CODEOWNERS file at: %s", path)
        return CodeOwners(contents.decoded_content.decode("utf-8"))
    raise FileNotFoundError("CODEOWNERS file not found in the repository.")


def get_modified_files(pr):
    """Collect all changed paths, including paginated GitHub results."""
    return [file.filename for file in pr.get_files()]


def get_approved_reviews(pr):
    """Preserve the previous action's evaluation of reviews in the APPROVED state."""
    return {review.user.login for review in pr.get_reviews() if review.state == "APPROVED"}


@lru_cache(maxsize=None)
def get_team_members(org, team_slug):
    """Fetch team members once per organization and team during a check."""
    return {member.login for member in org.get_team_by_slug(team_slug).get_members()}


def get_missing_approvals_for_file(
    file, codeowners, approved_reviews, org, exempt_owners=frozenset()
):
    """Return unsatisfied owners, exempting only explicitly allowed teams."""
    missing = []
    for owner_type, owner_name in codeowners.of(file):
        if owner_type == "TEAM":
            if owner_name in exempt_owners:
                continue
            members = get_team_members(org, owner_name.rsplit("/", 1)[1])
            approved = bool(members & approved_reviews)
        else:
            approved = owner_name.removeprefix("@") in approved_reviews
        if not approved:
            missing.append(owner_name)
    return missing


def gather_missing_approvals(
    modified_files, codeowners, approved_reviews, org, exempt_owners=frozenset()
):
    """Collect missing owners for each changed file using CODEOWNERS matching."""
    missing = {}
    for file in modified_files:
        owners = get_missing_approvals_for_file(
            file, codeowners, approved_reviews, org, exempt_owners
        )
        if owners:
            missing[file] = owners
    return missing


def check_all_owners_approved(args):
    """Evaluate the PR using its GitHub author, never the workflow actor or reviewer."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError("GITHUB_TOKEN environment variable not found.")

    client = Github(token)
    repo = client.get_repo(args.repo_name)
    pr = repo.get_pull(args.pr_number)
    exempt_owners = get_exempt_owners(pr.user.login)
    if exempt_owners:
        logger.info(
            "Author %s is exempt from approval by: %s", pr.user.login, sorted(exempt_owners)
        )

    codeowners = get_codeowners(repo)
    modified_files = get_modified_files(pr)
    approved_reviews = get_approved_reviews(pr)
    org = client.get_organization(args.repo_name.split("/", 1)[0])
    missing = gather_missing_approvals(
        modified_files, codeowners, approved_reviews, org, exempt_owners
    )
    for file, owners in missing.items():
        print(f"::error::Missing approvals for {file} from: {', '.join(owners)}")
    if missing:
        return False
    logger.info("SUCCESS: All code owners have approved the PR.")
    return True


def main(argv=None):
    """Report failed approvals and API errors as a failing GitHub Actions step."""
    args = parse_args(argv)
    try:
        return 0 if check_all_owners_approved(args) else 1
    except Exception as error:
        print(f"::error::{error}")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
