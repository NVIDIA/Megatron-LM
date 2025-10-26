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
        """Get user's email from their recent commits in fork or main repo."""
        if username in self.email_cache:
            return self.email_cache[username]

        try:
            user = self.github.get_user(username)
            repos = []

            # Try user's fork first
            try:
                repos.append(user.get_repo(self.repo.name))
            except:
                pass
            repos.append(self.repo)

            # Search commits in fork then main repo
            for repo in repos:
                try:
                    commits = self.github.search_commits(
                        f"author:{username} repo:{repo.full_name}", sort="author-date", order="desc"
                    )
                    for commit in commits[:5]:
                        try:
                            email = repo.get_commit(commit.sha).commit.author.email
                            if email and not email.endswith("@users.noreply.github.com"):
                                self.email_cache[username] = email
                                return email
                        except:
                            continue
                except:
                    continue

            # Fallback to public email or noreply
            email = user.email or f"{username}@users.noreply.github.com"
            self.email_cache[username] = email
            return email

        except Exception as e:
            logger.warning(f"Could not get email for {username}: {e}")
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
        """Get filtered reviewer emails."""
        stage = self.get_stage(pr)
        teams = {t.slug for t in pr.get_review_requests()[1]}

        teams = (
            teams - self.EXCLUDED_TEAMS
            if stage == self.EXPERT_REVIEW
            else teams & self.EXCLUDED_TEAMS
        )

        reviewers = set()
        org = self.github.get_organization(self.repo.organization.login)
        for slug in teams:
            try:
                reviewers.update(m.login for m in org.get_team_by_slug(slug).get_members())
            except:
                pass

        reviewers.update(r.login for r in pr.get_review_requests()[0])
        reviewer_emails = sorted([self.get_user_email(u) for u in reviewers])

        # Edge case: Expert Review with no reviewers - assign to PR author
        if len(reviewer_emails) == 0 and stage == self.EXPERT_REVIEW:
            pr_author_email = self.get_user_email(pr.user.login)
            reviewer_emails = [pr_author_email]

        # Edge case: Final Review with no reviewers - get approvers from mcore-reviewers team
        if len(reviewer_emails) == 0 and stage == self.FINAL_REVIEW:
            try:
                # Get all approvers (users who approved the PR)
                approvers = {
                    review.user.login for review in pr.get_reviews() if review.state == "APPROVED"
                }

                # Get mcore-reviewers team members
                mcore_team = org.get_team_by_slug("mcore-reviewers")
                mcore_members = {m.login for m in mcore_team.get_members()}

                # Intersection: approvers who are in mcore-reviewers
                valid_approvers = approvers & mcore_members
                reviewer_emails = sorted([self.get_user_email(u) for u in valid_approvers])
            except Exception as e:
                logger.warning(f"Could not get mcore-reviewers approvers for PR #{pr.number}: {e}")

        return reviewer_emails

    def create_reminder(self, pr):
        """Create reminder for PR."""
        stage = self.get_stage(pr)
        stage_days = self.days_since(self.get_label_date(pr, stage))
        author_email = self.get_user_email(pr.user.login)
        reviewer_emails = self.get_reviewers(pr)

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
        )

    def generate_reminders(self):
        """Generate all reminders."""
        milestones = list(self.repo.get_milestones(state="open", sort="due_on", direction="desc"))[
            :2
        ]
        logger.info(f"Found milestones: {', '.join(m.title for m in milestones)}")

        reminders = []
        for milestone in milestones:
            for issue in self.repo.get_issues(state="open", milestone=milestone):
                if not issue.pull_request:
                    continue
                labels = {l.name for l in issue.labels}
                if self.EXPERT_REVIEW in labels or self.FINAL_REVIEW in labels:
                    try:
                        reminders.append(self.create_reminder(self.repo.get_pull(issue.number)))
                        logger.info(f"Processed PR #{issue.number}")
                    except Exception as e:
                        logger.error(f"Failed to process PR #{issue.number}: {e}")

        return sorted(reminders, key=lambda r: (r.priority, -r.current_stage_time))

    def send_slack_notification(self, reminder: Reminder):
        """Send Slack notification via webhook."""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured, skipping notification")
            return

        message = []
        message.append(f"*PR*: {reminder.pr}")
        message.append(f"*Milestone*: {reminder.milestone}")
        message.append(f"*Author*: {reminder.author}")
        message.append(f"*Priority*: {reminder.priority}")
        message.append(f"*Review stage*: {reminder.review_stage}")
        message.append(f"*Days in review*: {reminder.total_review_time}")
        message.append(f"*Days in {reminder.review_stage}*: {reminder.current_stage_time}")
        message.append(
            f"*Reviewers*: {', '.join(reminder.reviewers) if reminder.reviewers else 'None'}"
        )

        payload = {
            "text": f"PR Review Reminder: {reminder.priority} - PR #{reminder.id}",
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(message)}}],
        }

        try:
            response = requests.post(self.webhook_url, json=payload)
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

    for r in reminders:
        logger.info(f"{r.priority} | PR #{r.id} | {r.milestone}")
        logger.info(f"   Author: {r.author} | Stage: {r.review_stage}")
        logger.info(f"   Stage time: {r.current_stage_time}d | Total: {r.total_review_time}d")
        logger.info(f"   Reviewers: {', '.join(r.reviewers) if r.reviewers else 'None'}")

        # Send Slack notification via webhook
        if webhook_url:
            tracker.send_slack_notification(r)

    return reminders


if __name__ == "__main__":
    main()
