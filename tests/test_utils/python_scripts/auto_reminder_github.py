#!/usr/bin/env python3
"""
GitHub PR Review Reminder Automation
Requirements: pip install PyGithub
Usage: GH_TOKEN=ghp_... REPO=NVIDIA/Megatron-LM python github_pr_reminder.py
"""

import os
import sys
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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


class PRReviewTracker:
    EXPERT_REVIEW = "Expert Review"
    FINAL_REVIEW = "Final Review"
    EXCLUDED_TEAMS = {"core-adlr", "core-nemo"}
    
    def __init__(self, token: str, repo_name: str):
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)
        self.email_cache = {}
    
    def get_user_email(self, username: str):
        """Get user's email address from GitHub API."""
        if username in self.email_cache:
            return self.email_cache[username]
        
        try:
            user = self.github.get_user(username)
            email = user.email
            
            if not email:
                events = list(user.get_events()[:10])
                for event in events:
                    if event.type == "PushEvent" and event.payload.get("commits"):
                        for commit in event.payload["commits"]:
                            if commit.get("author", {}).get("email"):
                                email = commit["author"]["email"]
                                break
                        if email:
                            break
            
            if not email:
                email = f"{username}@users.noreply.github.com"
            
            self.email_cache[username] = email
            return email
        except Exception as e:
            logger.warning(f"Could not get email for {username}: {e}")
            email = f"{username}@users.noreply.github.com"
            self.email_cache[username] = email
            return email
    
    def get_label_date(self, pr, label: str):
        """Get most recent date when label was attached."""
        dates = [e.created_at for e in pr.as_issue().get_events() 
                 if e.event == "labeled" and e.label and e.label.name == label]
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
        
        if stage == self.EXPERT_REVIEW:
            teams -= self.EXCLUDED_TEAMS
        else:
            teams &= self.EXCLUDED_TEAMS
        
        reviewers = set()
        org = self.github.get_organization(self.repo.organization.login)
        for slug in teams:
            try:
                reviewers.update(m.login for m in org.get_team_by_slug(slug).get_members())
            except:
                pass
        
        reviewers.update(r.login for r in pr.get_review_requests()[0])
        return sorted([self.get_user_email(u) for u in reviewers])
    
    def create_reminder(self, pr):
        """Create reminder for PR."""
        stage = self.get_stage(pr)
        stage_days = self.days_since(self.get_label_date(pr, stage))
        
        return Reminder(
            id=pr.number,
            pr=f"<{pr.html_url}|#{pr.number} - {pr.title}>",
            milestone=pr.milestone.title if pr.milestone else "No Milestone",
            author=self.get_user_email(pr.user.login),
            priority="P0" if stage_days > 3 else "P1" if stage_days >= 1 else "P2",
            review_stage=stage,
            total_review_time=self.days_since(self.get_label_date(pr, self.EXPERT_REVIEW)),
            current_stage_time=stage_days,
            reviewers=self.get_reviewers(pr)
        )
    
    def generate_reminders(self):
        """Generate all reminders."""
        milestones = list(self.repo.get_milestones(state="open", sort="due_on", direction="desc"))[:2]
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


def main():
    token = os.environ.get("GH_TOKEN")
    repo = os.environ.get("REPO", "NVIDIA/Megatron-LM")
    
    if not token:
        logger.error("GH_TOKEN environment variable is required")
        sys.exit(1)
    
    logger.info(f"Starting PR review reminder for {repo}")
    reminders = PRReviewTracker(token, repo).generate_reminders()
    logger.info(f"Generated {len(reminders)} reminders")
    
    print("\n" + "=" * 80)
    for r in reminders:
        print(f"\n{r.priority} | PR #{r.id} | {r.milestone}")
        print(f"   Author: {r.author} | Stage: {r.review_stage}")
        print(f"   Stage time: {r.current_stage_time}d | Total: {r.total_review_time}d")
        print(f"   Reviewers: {', '.join(r.reviewers) if r.reviewers else 'None'}")
    
    return reminders


if __name__ == "__main__":
    main()