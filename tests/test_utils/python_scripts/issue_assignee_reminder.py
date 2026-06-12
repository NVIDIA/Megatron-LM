# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3
"""
GitHub Issue Assignee Reminder Automation
Requirements: pip install PyGithub slack-sdk requests
Usage: GH_TOKEN=ghp_... SLACK_TOKEN=xoxb-... SLACK_WEBHOOK_URL=https://... REPO=NVIDIA/Megatron-LM python issue_assignee_reminder.py
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
    issue: str
    milestone: str
    author: str
    priority: str
    days_open: int
    days_since_update: int
    assignees: List[str]
    action_message: str


class IssueTracker:
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
            public_email = None

            # 1. Try public profile email first
            if user.email and not user.email.endswith("@users.noreply.github.com"):
                if user.email.endswith("@nvidia.com"):
                    self.email_cache[username] = user.email
                    return user.email
                else:
                    public_email = user.email

            # 2. If no public email, check recent commits on the main repo
            try:
                # Use get_commits(author=...) which is more direct than search_commits
                for commit in self.repo.get_commits(author=user)[:10]:
                    email = commit.commit.author.email
                    if (
                        email
                        and not email.endswith("@users.noreply.github.com")
                        and email.endswith("@nvidia.com")
                    ):
                        self.email_cache[username] = email
                        return email
                    elif (
                        email
                        and not email.endswith("@users.noreply.github.com")
                        and public_email is None
                    ):
                        public_email = email
            except Exception as e:
                logger.debug(f"Could not check commits for {username}: {e}")

            if public_email is None:
                public_email = f"{username}@users.noreply.github.com"

            self.email_cache[username] = public_email
            return public_email

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

    def days_since(self, date):
        """Calculate days since given date."""
        if not date:
            return 0
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - date).days

    def get_assignees(self, issue):
        """Get list of assignee emails."""
        assignee_emails = []
        if issue.assignees:
            for user in issue.assignees:
                assignee_emails.append(self.get_user_email(user.login))
        return sorted(assignee_emails)

    def should_remind(self, issue):
        """Check if reminder should be sent."""
        # 1. Check if last update was > 24 hours ago
        if self.days_since(issue.updated_at) < 1:
            logger.info(f"Skipping Issue #{issue.number}: updated less than 24h ago")
            return False

        # 2. Check if the most recent comment was made by an assignee
        try:
            comments = issue.get_comments(sort="created", direction="desc")
            if comments.totalCount > 0:
                last_comment = comments[0]
                assignee_logins = {a.login for a in issue.assignees}
                if last_comment.user.login in assignee_logins:
                    logger.info(
                        f"Skipping Issue #{issue.number}: last comment by assignee {last_comment.user.login}"
                    )
                    return False
        except Exception as e:
            logger.warning(f"Could not check comments for Issue #{issue.number}: {e}")

        return True

    def create_reminder(self, issue):
        """Create reminder for Issue."""
        days_open = self.days_since(issue.created_at)
        days_update = self.days_since(issue.updated_at)
        author_email = self.get_user_email(issue.user.login)
        assignee_emails = self.get_assignees(issue)

        # Determine priority based on days open (example logic)
        if days_open > 30:
            priority = "P0"
        elif days_open > 14:
            priority = "P1"
        else:
            priority = "P2"

        return Reminder(
            id=issue.number,
            issue=f"<{issue.html_url}|#{issue.number} - {issue.title}>",
            milestone=issue.milestone.title if issue.milestone else "No Milestone",
            author=self.get_slack_user_id(author_email),
            priority=priority,
            days_open=days_open,
            days_since_update=days_update,
            assignees=[self.get_slack_user_id(email) for email in assignee_emails],
            action_message="This issue is assigned to you.",
        )

    def generate_reminders(self):
        """Generate all reminders."""
        milestones = list(self.repo.get_milestones(state="open", sort="due_on", direction="desc"))[
            :2
        ]
        logger.info(f"Found milestones: {', '.join(m.title for m in milestones)}")

        reminders = []
        for milestone in milestones:
            # Find open issues in the milestone that are NOT PRs
            query = (
                f'repo:"{self.repo.full_name}" '
                f'milestone:"{milestone.title}" '
                f'is:open is:issue'
            )
            try:
                issues = self.github.search_issues(query)
                for issue in issues:
                    if not issue.assignees:
                        continue  # Skip unassigned issues

                    if not self.should_remind(issue):
                        continue

                    try:
                        reminders.append(self.create_reminder(issue))
                        logger.info(f"Processed Issue #{issue.number}")
                    except Exception as e:
                        logger.error(f"Failed to process Issue #{issue.number}: {e}")
            except Exception as e:
                logger.error(f"Failed to search issues for milestone {milestone.title}: {e}")

        return sorted(reminders, key=lambda r: (r.priority, -r.days_open))

    def send_slack_notification(self, reminder: Reminder):
        """Send Slack notification via webhook."""
        if not self.webhook_url:
            return

        assignees_str = ', '.join(reminder.assignees) if reminder.assignees else 'None'
        message = [
            f"*Issue*: {reminder.issue}",
            f"*Milestone*: {reminder.milestone}",
            f"*Author*: {reminder.author}",
            f"*Priority*: {reminder.priority}",
            f"*Days open*: {reminder.days_open}",
            f"*Days since update*: {reminder.days_since_update}",
            f"*Assignees*: {assignees_str}",
        ]

        payload = {
            "text": f"Issue Assignee Reminder: {reminder.priority} - Issue #{reminder.id}",
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(message)}}],
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Sent Slack notification for Issue #{reminder.id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification for Issue #{reminder.id}: {e}")


def main():
    token = os.environ.get("GH_TOKEN")
    slack_token = os.environ.get("SLACK_TOKEN")
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    repo = os.environ.get("REPO", "NVIDIA/Megatron-LM")

    if not token:
        logger.error("GH_TOKEN environment variable is required")
        sys.exit(1)

    logger.info(f"Starting Issue assignee reminder for {repo}")
    tracker = IssueTracker(token, repo, slack_token, webhook_url)
    reminders = tracker.generate_reminders()
    logger.info(f"Generated {len(reminders)} reminders\n{'=' * 80}")

    if not reminders:
        logger.info("No reminders to send.")
        return

    for r in reminders:
        logger.info(f"{r.priority} | Issue #{r.id} | {r.milestone}")
        logger.info(f"   Author: {r.author} | Days open: {r.days_open}")
        logger.info(f"   Assignees: {', '.join(r.assignees) if r.assignees else 'None'}")
        logger.info("-" * 80)
        if webhook_url:
            tracker.send_slack_notification(r)

    logger.info("All reminders processed.")


if __name__ == "__main__":
    main()


