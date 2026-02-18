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
    EXCLUDED_TEAMS = {"core-adlr", "core-nemo"}

    def __init__(self, token: str, repo_name: str, pr_number: str):
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)
        self.pr = self.repo.get_pull(pr_number)
        self.stage = self.get_stage(self.pr)
        self.org = self.github.get_organization(self.repo.organization.login)

    def get_stage(self, pr):
        """Get current review stage."""
        labels = {l.name for l in pr.labels}
        return self.FINAL_REVIEW if self.FINAL_REVIEW in labels else self.EXPERT_REVIEW

    def swap_labels(self):
        """Get filtered reviewer emails who haven't approved yet."""
        pr = self.pr
        if self.stage == self.FINAL_REVIEW:
            logger.info(f"PR #{self.pr.number} is in the {self.stage} stage. No reviewers needed.")
            return

        # 1. Get the latest review state for everyone who has submitted a review
        latest_reviews = {}
        try:
            for review in pr.get_reviews():
                if not review.user:  # Handle rare cases of deleted users
                    continue
                # Only track 'APPROVED' or 'CHANGES_REQUESTED' as definitive states
                if review.state in ("APPROVED", "CHANGES_REQUESTED"):
                    if (
                        review.user.login not in latest_reviews
                        or review.submitted_at > latest_reviews[review.user.login].submitted_at
                    ):
                        latest_reviews[review.user.login] = review
        except Exception as e:
            logger.warning(f"Could not get reviews for PR #{pr.number}: {e}")

        # 2. Separate reviewers into approvers (List B) and non-approvers
        approvers = {user for user, review in latest_reviews.items() if review.state == "APPROVED"}
        non_approving_reviewers = {
            user for user, review in latest_reviews.items() if review.state == "CHANGES_REQUESTED"
        }

        # 3. Get all *currently pending* review requests
        try:
            pending_users_req, pending_teams_req = pr.get_review_requests()
            pending_individuals = {r.login for r in pending_users_req}
            pending_teams_slugs = {t.slug for t in pending_teams_req}
        except Exception as e:
            logger.warning(f"Could not get review requests for PR #{pr.number}: {e}")
            pending_individuals = set()
            pending_teams_slugs = set()

        # 4. Filter pending teams based on the current stage
        teams_to_query = (
            pending_teams_slugs - self.EXCLUDED_TEAMS
            if self.stage == self.EXPERT_REVIEW
            else pending_teams_slugs & self.EXCLUDED_TEAMS
        )

        # 5. Get members from the required pending teams
        pending_team_members = set()
        for slug in teams_to_query:
            try:
                pending_team_members.update(
                    m.login for m in self.org.get_team_by_slug(slug).get_members()
                )
            except Exception as e:
                logger.warning(f"Could not get members for team {slug} on PR #{pr.number}: {e}")

        # 6. "List A": Combine all users who *still need to review*
        all_required_reviewers = (
            pending_individuals | pending_team_members | non_approving_reviewers
        )

        # 7. Final list (List A - List B):
        pending_reviewers = all_required_reviewers - approvers
        logger.info(f"Pending reviewers: {pending_reviewers}")
        if len(pending_reviewers) == 0:
            try:
                pr.remove_from_labels(self.EXPERT_REVIEW)
                logger.info(f'Removed "{self.EXPERT_REVIEW}" label from PR #{pr.number}')
            except Exception as e:
                logger.warning(
                    f'Failed to remove "{self.EXPERT_REVIEW}" label from PR #{pr.number}: {e}'
                )

            try:
                pr.add_to_labels(self.FINAL_REVIEW)
                logger.info(f'Added "{self.FINAL_REVIEW}" label to PR #{pr.number}')
            except Exception as e:
                logger.warning(f'Failed to add "{self.FINAL_REVIEW}" label to PR #{pr.number}: {e}')


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
