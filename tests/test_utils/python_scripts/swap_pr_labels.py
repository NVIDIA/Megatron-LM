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
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)
        self.pr = self.repo.get_pull(pr_number)
        self.stage = self.get_stage(self.pr)
        self.org = self.github.get_organization(self.repo.organization.login)
        self._team_cache = {}
        self._codeowner_teams = self._parse_codeowner_teams()

    def _parse_codeowner_teams(self):
        """Parse CODEOWNERS to get the set of teams that are blocking reviewers."""
        teams = set()
        try:
            with open(".github/CODEOWNERS") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    for token in line.split():
                        if token.startswith("@NVIDIA/"):
                            teams.add(token.split("/", 1)[1])
        except FileNotFoundError:
            logger.warning("CODEOWNERS file not found")
        logger.info(f"CODEOWNERS teams: {teams}")
        return teams

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

    def swap_labels(self):
        """Evaluate review state and update labels accordingly."""
        pr = self.pr
        if pr.draft:
            logger.info(f"PR #{pr.number} is a draft. Skipping label swap.")
            return

        # 1. Get the latest review state for everyone who has submitted a review
        latest_reviews = {}
        try:
            for review in pr.get_reviews():
                if not review.user:
                    continue
                if review.state in ("APPROVED", "CHANGES_REQUESTED"):
                    if (
                        review.user.login not in latest_reviews
                        or review.submitted_at > latest_reviews[review.user.login].submitted_at
                    ):
                        latest_reviews[review.user.login] = review
        except Exception as e:
            logger.warning(f"Could not get reviews for PR #{pr.number}: {e}")

        approvers = {user for user, review in latest_reviews.items() if review.state == "APPROVED"}
        non_approvers = {
            user for user, review in latest_reviews.items() if review.state == "CHANGES_REQUESTED"
        }

        # 2. Get all currently pending review requests
        try:
            pending_users_req, pending_teams_req = pr.get_review_requests()
            pending_individuals = {r.login for r in pending_users_req}
            pending_team_slugs = {t.slug for t in pending_teams_req}
        except Exception as e:
            logger.warning(f"Could not get review requests for PR #{pr.number}: {e}")
            pending_individuals = set()
            pending_team_slugs = set()

        # 3. Filter to only CODEOWNERS teams (ignore optional teams like mcore-oncall)
        pending_team_slugs = pending_team_slugs & self._codeowner_teams
        logger.info(f"Pending CODEOWNERS teams: {pending_team_slugs}")

        # 4. Classify teams into expert vs final (excluded)
        expert_team_slugs = pending_team_slugs - self.EXCLUDED_TEAMS
        final_team_slugs = pending_team_slugs & self.EXCLUDED_TEAMS

        # 5. Get team members
        expert_team_members = self._get_teams_members(expert_team_slugs)
        all_excluded_members = self._get_teams_members(self.EXCLUDED_TEAMS)

        # 6. Compute pending expert reviewers
        expert_non_approvers = non_approvers - all_excluded_members
        pending_expert = (
            pending_individuals | expert_team_members | expert_non_approvers
        ) - approvers
        logger.info(f"Pending expert reviewers: {pending_expert}")

        # 7. Compute pending final reviewers
        final_pending_members = self._get_teams_members(final_team_slugs)
        final_non_approvers = non_approvers & all_excluded_members
        pending_final = (final_pending_members | final_non_approvers) - approvers
        logger.info(f"Pending final reviewers: {pending_final}")

        # 8. Determine if final review is needed at all (excluded teams are assigned)
        excluded_who_reviewed = (approvers | non_approvers) & all_excluded_members
        needs_final_review = bool(final_team_slugs) or bool(excluded_who_reviewed)

        # 9. Guard: if no codeowner reviewers exist at all, the review process hasn't started yet.
        has_any_reviewers = pending_individuals or pending_team_slugs or approvers or non_approvers
        if not has_any_reviewers and self.stage == self.EXPERT_REVIEW:
            logger.info(f"PR #{pr.number} has no reviewers assigned yet. Skipping.")
            return

        # 10. State machine: update labels based on current stage and pending reviewers
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
