# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3
"""
GitHub PR Review Reminder Automation
Requirements: pip install PyGithub slack-sdk requests
Usage: GH_TOKEN=ghp_... SLACK_TOKEN=xoxb-... SLACK_WEBHOOK_URL=https://... REPO=NVIDIA/Megatron-LM python github_pr_reminder.py
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import List

import requests
from github import Github

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Reminder:
    id: int
    pr: str
    milestone: str
    author: str
    priority: str
    review_stage: str
    total_review_time: int
    current_stage_time: int
    reviewers: List[str]
    action_message: str


class PRReviewTracker:
    EXPERT_REVIEW = "Expert Review"
    FINAL_REVIEW = "Final Review"
    APPROVED = "Approved"
    EXCLUDED_TEAMS = {"core-adlr", "core-nemo"}

    def __init__(self, token: str, repo_name: str, pr_number: str):
        self.token = token
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)
        self.pr = self.repo.get_pull(pr_number)
        self.stage = self.get_stage(self.pr)
        self.org = self.github.get_organization(self.repo.organization.login)
        self._team_cache = {}
        self._codeowner_rules = self._parse_codeowners()

    def _parse_codeowners(self):
        """Parse CODEOWNERS into ordered list of (pattern, teams) rules."""
        rules = []
        try:
            with open(".github/CODEOWNERS") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    pattern = parts[0]
                    teams = set()
                    for part in parts[1:]:
                        if part.startswith("@NVIDIA/"):
                            teams.add(part.split("/", 1)[1])
                    rules.append((pattern, teams))
        except FileNotFoundError:
            logger.warning("CODEOWNERS file not found")
        logger.info(f"Parsed {len(rules)} CODEOWNERS rules")
        return rules

    @staticmethod
    def _match_file(filepath, pattern):
        """Check if a file path matches a CODEOWNERS pattern.

        Rules:
        - Trailing '/' means directory pattern: matches all files under that directory.
        - Pattern containing '/' is path-relative: exact match or directory prefix.
        - Pattern without '/' matches the filename component anywhere.
        """
        if pattern.endswith('/'):
            return filepath.startswith(pattern)
        if '/' in pattern:
            return filepath == pattern or filepath.startswith(pattern + '/')
        return filepath == pattern or filepath.endswith('/' + pattern)

    def _get_required_teams(self, pr):
        """Determine required review teams from CODEOWNERS rules and PR changed files.

        Uses last-match-wins semantics per file, then unions across all files.
        """
        required_teams = set()
        try:
            changed_files = [f.filename for f in pr.get_files()]
        except Exception as e:
            logger.warning(f"Could not get changed files for PR #{pr.number}: {e}")
            return required_teams

        for filepath in changed_files:
            matched_teams = None
            for pattern, teams in self._codeowner_rules:
                if self._match_file(filepath, pattern):
                    matched_teams = teams
            if matched_teams:
                required_teams.update(matched_teams)
                logger.info(f"  {filepath} → {matched_teams}")

        logger.info(f"Required teams from CODEOWNERS: {required_teams}")
        return required_teams

    def get_stage(self, pr):
        """Get current review stage."""
        labels = {l.name for l in pr.labels}
        if self.APPROVED in labels:
            return self.APPROVED
        if self.FINAL_REVIEW in labels:
            return self.FINAL_REVIEW
        return self.EXPERT_REVIEW

    def _get_team_members(self, slug):
        """Get all members of a team, with caching."""
        if slug not in self._team_cache:
            try:
                self._team_cache[slug] = {
                    m.login for m in self.org.get_team_by_slug(slug).get_members()
                }
            except Exception as e:
                logger.warning(f"Could not get members for team {slug}: {e}")
                self._team_cache[slug] = set()
        return self._team_cache[slug]

    def _get_teams_members(self, slugs):
        """Get all members of multiple teams."""
        members = set()
        for slug in slugs:
            members.update(self._get_team_members(slug))
        return members

    def _get_latest_reviews(self, pr_number):
        """Get the active review state per reviewer using GraphQL latestReviews.

        Unlike the REST API's get_reviews(), this returns only the current
        review per reviewer — matching the green/red checkmarks in the UI.
        Dismissed or removed approvals are not included.
        """
        owner, name = self.repo.full_name.split("/")
        query = """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              latestReviews(first: 100) {
                nodes {
                  author { login }
                  state
                }
              }
            }
          }
        }
        """
        reviews = {}
        try:
            resp = requests.post(
                "https://api.github.com/graphql",
                json={
                    "query": query,
                    "variables": {"owner": owner, "name": name, "number": pr_number},
                },
                headers={"Authorization": f"Bearer {self.token}"},
            )
            resp.raise_for_status()
            nodes = resp.json()["data"]["repository"]["pullRequest"]["latestReviews"]["nodes"]
            for node in nodes:
                if node["author"]:
                    reviews[node["author"]["login"]] = node["state"]
        except Exception as e:
            logger.warning(f"Could not get latest reviews for PR #{pr_number}: {e}")
        logger.info(f"Active reviews: {reviews}")
        return reviews

    def swap_labels(self):
        """Evaluate review state and update labels accordingly."""
        pr = self.pr
        if pr.draft:
            logger.info(f"PR #{pr.number} is a draft. Skipping label swap.")
            return

        # 1. Get the active review state per reviewer via GraphQL latestReviews,
        #    which reflects dismissed/removed approvals the same way the GitHub UI does.
        latest_reviews = self._get_latest_reviews(pr.number)

        approvers = {user for user, state in latest_reviews.items() if state == "APPROVED"}
        non_approvers = {
            user for user, state in latest_reviews.items() if state == "CHANGES_REQUESTED"
        }

        # 2. Determine required teams from CODEOWNERS + changed files
        required_teams = self._get_required_teams(pr)
        if not required_teams:
            logger.info(f"PR #{pr.number}: no CODEOWNERS teams matched changed files. Skipping.")
            return

        expert_required = required_teams - self.EXCLUDED_TEAMS
        final_required = required_teams & self.EXCLUDED_TEAMS
        logger.info(f"Expert teams required: {expert_required}")
        logger.info(f"Final teams required: {final_required}")

        # 3. Check which required teams still need approval (at least one member must approve)
        #    If _get_team_members fails (returns {}), the team stays pending — conservative.
        pending_expert_teams = set()
        for team in expert_required:
            members = self._get_team_members(team)
            if not (members & approvers):
                pending_expert_teams.add(team)

        pending_final_teams = set()
        for team in final_required:
            members = self._get_team_members(team)
            if not (members & approvers):
                pending_final_teams.add(team)

        # 4. Compute pending reviewers: unsatisfied team members + individual blockers
        all_excluded_members = self._get_teams_members(self.EXCLUDED_TEAMS)
        expert_non_approvers = non_approvers - all_excluded_members
        final_non_approvers = non_approvers & all_excluded_members

        pending_expert = (
            self._get_teams_members(pending_expert_teams) | expert_non_approvers
        ) - approvers
        pending_final = (
            self._get_teams_members(pending_final_teams) | final_non_approvers
        ) - approvers

        logger.info(f"Pending expert teams: {pending_expert_teams}, reviewers: {pending_expert}")
        logger.info(f"Pending final teams: {pending_final_teams}, reviewers: {pending_final}")

        needs_final_review = bool(final_required)

        # 5. State machine: update labels based on current stage and pending reviewers
        if self.stage == self.APPROVED:
            self._handle_approved_stage(pr, pending_expert, pending_final)
        elif self.stage == self.FINAL_REVIEW:
            self._handle_final_review_stage(pr, pending_expert, pending_final)
        else:
            self._handle_expert_review_stage(pr, pending_expert, pending_final, needs_final_review)

    def _handle_approved_stage(self, pr, pending_expert, pending_final):
        """Handle PRs that already have the Approved label."""
        if len(pending_expert) > 0 or len(pending_final) > 0:
            # New reviewers appeared — revert
            try:
                pr.remove_from_labels(self.APPROVED)
                logger.info(f'Removed "{self.APPROVED}" from PR #{pr.number}')
            except Exception as e:
                logger.warning(f'Failed to remove "{self.APPROVED}" from PR #{pr.number}: {e}')

            if len(pending_expert) > 0:
                # Back to expert review — also remove Final Review if present
                try:
                    pr.remove_from_labels(self.FINAL_REVIEW)
                except Exception:
                    pass
                logger.info(
                    f'Reverted PR #{pr.number} to expert review — pending: {pending_expert}'
                )
            else:
                # Expert review done but final review needed again
                try:
                    pr.add_to_labels(self.FINAL_REVIEW)
                except Exception:
                    pass
                logger.info(f'Reverted PR #{pr.number} to final review — pending: {pending_final}')
        else:
            logger.info(f"PR #{pr.number} is approved. No changes needed.")

    def _handle_final_review_stage(self, pr, pending_expert, pending_final):
        """Handle PRs in the Final Review stage."""
        if len(pending_expert) > 0:
            # New expert reviewers appeared — revert to expert review
            try:
                pr.remove_from_labels(self.FINAL_REVIEW)
                logger.info(
                    f'Removed "{self.FINAL_REVIEW}" from PR #{pr.number} — '
                    f'new expert reviewers pending: {pending_expert}'
                )
            except Exception as e:
                logger.warning(f'Failed to remove "{self.FINAL_REVIEW}" from PR #{pr.number}: {e}')
        elif len(pending_final) == 0:
            # All final reviewers approved — move to Approved, remove Final Review
            try:
                pr.remove_from_labels(self.FINAL_REVIEW)
                logger.info(f'Removed "{self.FINAL_REVIEW}" from PR #{pr.number}')
            except Exception as e:
                logger.warning(f'Failed to remove "{self.FINAL_REVIEW}" from PR #{pr.number}: {e}')
            try:
                pr.add_to_labels(self.APPROVED)
                logger.info(f'Added "{self.APPROVED}" to PR #{pr.number}')
            except Exception as e:
                logger.warning(f'Failed to add "{self.APPROVED}" to PR #{pr.number}: {e}')
        else:
            logger.info(f"PR #{pr.number} is in final review. Pending: {pending_final}")

    def _handle_expert_review_stage(self, pr, pending_expert, pending_final, needs_final_review):
        """Handle PRs in the Expert Review stage (no review labels yet)."""
        if len(pending_expert) > 0:
            logger.info(f"PR #{pr.number} is in expert review. Pending: {pending_expert}")
            return

        # All expert reviewers approved
        if needs_final_review and len(pending_final) > 0:
            # Final review teams are assigned and still pending
            try:
                pr.add_to_labels(self.FINAL_REVIEW)
                logger.info(f'Added "{self.FINAL_REVIEW}" to PR #{pr.number}')
            except Exception as e:
                logger.warning(f'Failed to add "{self.FINAL_REVIEW}" to PR #{pr.number}: {e}')
        else:
            # No final review needed, or final reviewers already approved
            try:
                pr.add_to_labels(self.APPROVED)
                logger.info(f'Added "{self.APPROVED}" to PR #{pr.number}')
            except Exception as e:
                logger.warning(f'Failed to add "{self.APPROVED}" to PR #{pr.number}: {e}')


def main():
    token = os.environ.get("GH_TOKEN")
    repo = os.environ.get("REPO", "NVIDIA/Megatron-LM")
    pr_number = int(os.environ.get("PR_NUMBER"))

    if not token:
        logger.error("GH_TOKEN environment variable is required")
        sys.exit(1)

    logger.info(f"Starting PR review reminder for {repo}")
    tracker = PRReviewTracker(token, repo, pr_number)
    tracker.swap_labels()


if __name__ == "__main__":
    main()
