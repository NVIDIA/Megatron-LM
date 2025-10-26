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
from datetime import datetime, timezone
from typing import List

import requests
from github import Github
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

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

    def __init__(
        self, token: str, repo_name: str, slack_token: str = None, webhook_url: str = None
    ):
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)
        self.email_cache = {}
        self.slack_id_cache = {}
        self.slack_client = WebClient(token=slack_token) if slack_token else None
        self.webhook_url = webhook_url

    def get_user_email(self, username: str):
        """Get user's email, prioritizing public profile, then recent commits."""
        if username in self.email_cache:
            return self.email_cache[username]

        try:
            user = self.github.get_user(username)

            # 1. Try public profile email first
            if user.email and not user.email.endswith("@users.noreply.github.com"):
                self.email_cache[username] = user.email
                return user.email

            # 2. If no public email, check recent commits on the main repo
            try:
                # Use get_commits(author=...) which is more direct than search_commits
                for commit in self.repo.get_commits(author=user)[:10]:
                    email = commit.commit.author.email
                    if email and not email.endswith("@users.noreply.github.com"):
                        self.email_cache[username] = email
                        return email
            except Exception as e:
                logger.debug(f"Could not check commits for {username}: {e}")

            # 3. Fallback to public email (even if noreply) or a constructed noreply
            email = user.email or f"{username}@users.noreply.github.com"
            self.email_cache[username] = email
            return email

        except Exception as e:
            logger.warning(f"Could not get user object for {username}: {e}")
            email = f"{username}@users.noreply.github.com"
            self.email_cache[username] = email
            return email

    def get_slack_user_id(self, email: str):
        """Get Slack user ID from email."""
        if not self.slack_client:
            return email
        if email in self.slack_id_cache:
            return self.slack_id_cache[email]
        try:
            response = self.slack_client.users_lookupByEmail(email=email)
            user_id = response["user"]["id"]
            self.slack_id_cache[email] = f"<@{user_id}>"
            return self.slack_id_cache[email]
        except SlackApiError as e:
            logger.warning(f"Could not find Slack user for {email}: {e.response['error']}")
            self.slack_id_cache[email] = email
            return email

    def get_label_date(self, pr, label: str):
        """Get most recent date when label was attached."""
        dates = [
            e.created_at
            for e in pr.as_issue().get_events()
            if e.event == "labeled" and e.label and e.label.name == label
        ]
        return max(dates) if dates else None

    def days_since(self, date):
        """Calculate days since given date."""
        if not date:
            return 0
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - date).days

    def get_stage(self, pr):
        """Get current review stage."""
        labels = {l.name for l in pr.labels}
        return self.FINAL_REVIEW if self.FINAL_REVIEW in labels else self.EXPERT_REVIEW

    def get_reviewers(self, pr):
        """Get filtered reviewer emails who haven't approved yet."""
        stage = self.get_stage(pr)
        org = self.github.get_organization(self.repo.organization.login)

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
            if stage == self.EXPERT_REVIEW
            else pending_teams_slugs & self.EXCLUDED_TEAMS
        )

        # 5. Get members from the required pending teams
        pending_team_members = set()
        for slug in teams_to_query:
            try:
                pending_team_members.update(
                    m.login for m in org.get_team_by_slug(slug).get_members()
                )
            except Exception as e:
                logger.warning(f"Could not get members for team {slug} on PR #{pr.number}: {e}")

        # 6. "List A": Combine all users who *still need to review*
        all_required_reviewers = (
            pending_individuals | pending_team_members | non_approving_reviewers
        )

        # 7. Final list (List A - List B):
        pending_reviewers = all_required_reviewers - approvers
        reviewer_emails = sorted([self.get_user_email(u) for u in pending_reviewers])
        action_message = "Please review the PR."

        # 8. Handle the original edge cases
        if len(reviewer_emails) == 0:
            if stage == self.EXPERT_REVIEW:
                # Assign to PR author
                reviewer_emails = [self.get_user_email(pr.user.login)]
                action_message = "All Expert Reviewers approved the PR. Please attach the Final Review label to proceed with the review."
            elif stage == self.FINAL_REVIEW:
                # Assign to mcore-reviewers who approved
                try:
                    mcore_team = org.get_team_by_slug("mcore-reviewers")
                    mcore_members = {m.login for m in mcore_team.get_members()}
                    valid_approvers = approvers & mcore_members
                    reviewer_emails = sorted([self.get_user_email(u) for u in valid_approvers])
                    action_message = "All Final Reviewers approved the PR. Please ping an Expert or Final Reviewer to merge the PR."

                except Exception as e:
                    logger.warning(
                        f"Could not get mcore-reviewers approvers for PR #{pr.number}: {e}"
                    )

        return reviewer_emails, action_message

    def create_reminder(self, pr):
        """Create reminder for PR."""
        stage = self.get_stage(pr)
        stage_days = self.days_since(self.get_label_date(pr, stage))
        author_email = self.get_user_email(pr.user.login)
        reviewer_emails, action_message = self.get_reviewers(pr)

        return Reminder(
            id=pr.number,
            pr=f"<{pr.html_url}|#{pr.number} - {pr.title}>",
            milestone=pr.milestone.title if pr.milestone else "No Milestone",
            author=self.get_slack_user_id(author_email),
            priority="P0" if stage_days > 3 else "P1" if stage_days >= 1 else "P2",
            review_stage=stage,
            total_review_time=self.days_since(self.get_label_date(pr, self.EXPERT_REVIEW)),
            current_stage_time=stage_days,
            reviewers=[self.get_slack_user_id(email) for email in reviewer_emails],
            action_message=action_message,
        )

    def generate_reminders(self):
        """Generate all reminders."""
        milestones = list(self.repo.get_milestones(state="open", sort="due_on", direction="desc"))[
            :2
        ]
        logger.info(f"Found milestones: {', '.join(m.title for m in milestones)}")

        reminders = []
        for milestone in milestones:
            # Find issues with the 'Expert Review' or 'Final Review' label
            query = (
                f'repo:"{self.repo.full_name}" '
                f'milestone:"{milestone.title}" '
                f'is:open is:pr '
                f'label:"{self.EXPERT_REVIEW}","{self.FINAL_REVIEW}"'
            )
            try:
                # Use search_issues for a more direct query instead of get_issues + filtering
                issues = self.github.search_issues(query)
                for issue in issues:
                    try:
                        reminders.append(self.create_reminder(issue.as_pull_request()))
                        logger.info(f"Processed PR #{issue.number}")
                    except Exception as e:
                        logger.error(f"Failed to process PR #{issue.number}: {e}")
            except Exception as e:
                logger.error(f"Failed to search issues for milestone {milestone.title}: {e}")

        return sorted(reminders, key=lambda r: (r.priority, -r.current_stage_time))

    def send_slack_notification(self, reminder: Reminder):
        """Send Slack notification via webhook."""
        if not self.webhook_url:
            return

        reviewers_str = ', '.join(reminder.reviewers) if reminder.reviewers else 'None'
        message = [
            f"*PR*: {reminder.pr}",
            f"*Milestone*: {reminder.milestone}",
            f"*Author*: {reminder.author}",
            f"*Priority*: {reminder.priority}",
            f"*Review stage*: {reminder.review_stage}",
            f"*Days in review*: {reminder.total_review_time}",
            f"*Days in {reminder.review_stage}*: {reminder.current_stage_time}",
            f"*Reviewers*: {reviewers_str}",
        ]

        payload = {
            "text": f"PR Review Reminder: {reminder.priority} - PR #{reminder.id}",
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(message)}}],
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Sent Slack notification for PR #{reminder.id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification for PR #{reminder.id}: {e}")


def main():
    token = os.environ.get("GH_TOKEN")
    slack_token = os.environ.get("SLACK_TOKEN")
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    repo = os.environ.get("REPO", "NVIDIA/Megatron-LM")

    if not token:
        logger.error("GH_TOKEN environment variable is required")
        sys.exit(1)

    logger.info(f"Starting PR review reminder for {repo}")
    tracker = PRReviewTracker(token, repo, slack_token, webhook_url)
    reminders = tracker.generate_reminders()
    logger.info(f"Generated {len(reminders)} reminders\n{'=' * 80}")

    if not reminders:
        logger.info("No reminders to send.")
        return

    for r in reminders:
        logger.info(f"{r.priority} | PR #{r.id} | {r.milestone}")
        logger.info(f"   Author: {r.author} | Stage: {r.review_stage}")
        logger.info(f"   Stage time: {r.current_stage_time}d | Total: {r.total_review_time}")
        logger.info(f"   Reviewers: {', '.join(r.reviewers) if r.reviewers else 'None'}")
        logger.info(f"   Action message: {r.action_message}")
        logger.info("-" * 80)
        if webhook_url:
            tracker.send_slack_notification(r)

    logger.info("All reminders processed.")


if __name__ == "__main__":
    main()
