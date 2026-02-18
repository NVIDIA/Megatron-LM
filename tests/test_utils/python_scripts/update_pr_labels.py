# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3
"""
GitHub PR Label Automation

Supports two actions:
  - swap:   On approval, check if all expert reviewers have approved and
            swap "Expert Review" -> "Final Review" when appropriate.
  - assign: On PR opened / ready-for-review, assign the correct initial
            label ("Expert Review", "Final Review", or none) based on the
            requested review teams.

Requirements: pip install PyGithub
Usage:
  GH_TOKEN=ghp_... python update_pr_labels.py --pr 42 swap
  GH_TOKEN=ghp_... python update_pr_labels.py --pr 42 assign
"""

import argparse
import logging
import os
import sys

from github import Github

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PRReviewTracker:
    EXPERT_REVIEW = "Expert Review"
    FINAL_REVIEW = "Final Review"
    REVIEW_LABELS = {EXPERT_REVIEW, FINAL_REVIEW}
    EXCLUDED_TEAMS = {"core-adlr", "core-nemo"}
    NON_REQUIRED_TEAMS = {"mcore-oncall"}

    def __init__(self, token: str, repo_name: str, pr_number: int):
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)
        self.pr = self.repo.get_pull(pr_number)
        self.org = self.github.get_organization(self.repo.organization.login)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_label(self):
        labels = {l.name for l in self.pr.labels}
        if self.FINAL_REVIEW in labels:
            return self.FINAL_REVIEW
        if self.EXPERT_REVIEW in labels:
            return self.EXPERT_REVIEW
        return None

    def _get_pending_review_teams(self):
        try:
            _, pending_teams_req = self.pr.get_review_requests()
            return {t.slug for t in pending_teams_req}
        except Exception as e:
            logger.warning(f"Could not get review requests for PR #{self.pr.number}: {e}")
            return set()

    def _set_label(self, desired_label):
        """Ensure only *desired_label* (or none) from the review-label set is present."""
        pr = self.pr
        current_labels = {l.name for l in pr.labels}

        for label in self.REVIEW_LABELS:
            if label != desired_label and label in current_labels:
                try:
                    pr.remove_from_labels(label)
                    logger.info(f'Removed "{label}" label from PR #{pr.number}')
                except Exception as e:
                    logger.warning(f'Failed to remove "{label}" label from PR #{pr.number}: {e}')

        if desired_label and desired_label not in current_labels:
            try:
                pr.add_to_labels(desired_label)
                logger.info(f'Added "{desired_label}" label to PR #{pr.number}')
            except Exception as e:
                logger.warning(f'Failed to add "{desired_label}" label to PR #{pr.number}: {e}')

    # ------------------------------------------------------------------
    # Action: assign  (PR opened / ready-for-review)
    # ------------------------------------------------------------------

    def assign_initial_label(self):
        """Determine the correct label for a newly opened / ready-for-review PR.

        Logic (mirrors the PR template):
          - No review teams requested  → no review label
          - Only core-adlr / core-nemo / mcore-oncall → "Final Review"
          - Any *other* expert teams requested          → "Expert Review"
        """
        pr = self.pr
        if pr.draft:
            logger.info(f"PR #{pr.number} is still a draft — skipping label assignment.")
            return

        current_label = self._get_current_label()
        if current_label is not None:
            logger.info(
                f'PR #{pr.number} already has "{current_label}" — skipping initial assignment.'
            )
            return

        pending_teams = self._get_pending_review_teams()
        meaningful_teams = pending_teams - self.NON_REQUIRED_TEAMS

        if not meaningful_teams:
            logger.info(f"PR #{pr.number} has no meaningful review teams — no label assigned.")
            return

        expert_teams = meaningful_teams - self.EXCLUDED_TEAMS
        if expert_teams:
            logger.info(
                f"PR #{pr.number} has expert review teams {expert_teams} — "
                f'assigning "{self.EXPERT_REVIEW}".'
            )
            self._set_label(self.EXPERT_REVIEW)
        else:
            logger.info(
                f"PR #{pr.number} only has core teams {meaningful_teams} — "
                f'assigning "{self.FINAL_REVIEW}".'
            )
            self._set_label(self.FINAL_REVIEW)

    # ------------------------------------------------------------------
    # Action: swap  (review submitted)
    # ------------------------------------------------------------------

    def swap_labels(self):
        """After an approval, check if all expert reviewers are satisfied.

        If every expert reviewer has approved, swap "Expert Review" → "Final Review".
        """
        pr = self.pr
        stage = self._get_current_label()

        if stage == self.FINAL_REVIEW:
            logger.info(f"PR #{pr.number} is already in Final Review — nothing to swap.")
            return

        if stage is None:
            logger.info(f"PR #{pr.number} has no review label — nothing to swap.")
            return

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
        non_approving_reviewers = {
            user for user, review in latest_reviews.items() if review.state == "CHANGES_REQUESTED"
        }

        try:
            pending_users_req, pending_teams_req = pr.get_review_requests()
            pending_individuals = {r.login for r in pending_users_req}
            pending_teams_slugs = {t.slug for t in pending_teams_req}
        except Exception as e:
            logger.warning(f"Could not get review requests for PR #{pr.number}: {e}")
            pending_individuals = set()
            pending_teams_slugs = set()

        expert_teams = pending_teams_slugs - self.EXCLUDED_TEAMS

        pending_team_members = set()
        for slug in expert_teams:
            try:
                pending_team_members.update(
                    m.login for m in self.org.get_team_by_slug(slug).get_members()
                )
            except Exception as e:
                logger.warning(f"Could not get members for team {slug} on PR #{pr.number}: {e}")

        all_required_reviewers = (
            pending_individuals | pending_team_members | non_approving_reviewers
        )
        pending_reviewers = all_required_reviewers - approvers
        logger.info(f"Pending expert reviewers: {pending_reviewers}")

        if len(pending_reviewers) == 0:
            logger.info(
                f'All expert reviewers approved — swapping to "{self.FINAL_REVIEW}" '
                f"on PR #{pr.number}."
            )
            self._set_label(self.FINAL_REVIEW)


def main():
    parser = argparse.ArgumentParser(description="GitHub PR label automation")
    parser.add_argument("action", choices=["swap", "assign"])
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    parser.add_argument("--repo", default="NVIDIA/Megatron-LM", help="GitHub repo (owner/name)")
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN")
    if not token:
        logger.error("GH_TOKEN environment variable is required")
        sys.exit(1)

    logger.info(f"Running action={args.action} for PR #{args.pr} in {args.repo}")
    tracker = PRReviewTracker(token, args.repo, args.pr)

    if args.action == "assign":
        tracker.assign_initial_label()
    else:
        tracker.swap_labels()


if __name__ == "__main__":
    main()
