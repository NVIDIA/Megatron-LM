#!/usr/bin/env python3
"""
GitHub PR Review Reminder Automation
Requirements: pip install PyGithub
Usage: GITHUB_TOKEN=ghp_... REPO=NVIDIA/Megatron-LM python github_pr_reminder.py
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Set
from github import Github


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
        self.repo = Github(token).get_repo(repo_name)
        self.github = Github(token)
    
    def get_label_date(self, pr, label: str):
        """Get most recent date when label was attached."""
        dates = [e.created_at for e in pr.as_issue().get_events() 
                 if e.event == "labeled" and e.label and e.label.name == label]
        return max(dates) if dates else None
    
    def days_since(self, date):
        """Calculate days since given date."""
        if not date:
            return 0
        now = datetime.now(timezone.utc)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return (now - date).days
    
    def get_stage(self, pr):
        """Get current review stage."""
        labels = {l.name for l in pr.labels}
        return self.FINAL_REVIEW if self.FINAL_REVIEW in labels else self.EXPERT_REVIEW
    
    def get_reviewers(self, pr):
        """Get filtered reviewer usernames."""
        stage = self.get_stage(pr)
        teams = {t.slug for t in pr.get_review_requests()[1]}
        
        # Filter teams based on stage
        if stage == self.EXPERT_REVIEW:
            teams -= self.EXCLUDED_TEAMS
        else:  # FINAL_REVIEW
            teams &= self.EXCLUDED_TEAMS
        
        # Get team members
        reviewers = set()
        org = self.github.get_organization(self.repo.organization.login)
        for slug in teams:
            try:
                team = org.get_team_by_slug(slug)
                reviewers.update(m.login for m in team.get_members())
            except:
                pass
        
        # Add individual reviewers
        reviewers.update(r.login for r in pr.get_review_requests()[0])
        return sorted(reviewers)
    
    def create_reminder(self, pr):
        """Create reminder for PR."""
        stage = self.get_stage(pr)
        expert_days = self.days_since(self.get_label_date(pr, self.EXPERT_REVIEW))
        stage_days = self.days_since(self.get_label_date(pr, stage))
        priority = "P0" if stage_days > 3 else "P1" if stage_days >= 1 else "P2"
        
        return Reminder(
            id=pr.number,
            pr=f"<{pr.html_url}|#{pr.number} - {pr.title}>",
            milestone=pr.milestone.title if pr.milestone else "No Milestone",
            author=f"@{pr.user.login}",
            priority=priority,
            review_stage=stage,
            total_review_time=expert_days,
            current_stage_time=stage_days,
            reviewers=self.get_reviewers(pr)
        )
    
    def generate_reminders(self):
        """Generate all reminders."""
        # Get top 2 milestones
        milestones = list(self.repo.get_milestones(state="open", sort="due_on", direction="desc"))[:2]
        print(f"📋 Milestones: {', '.join(m.title for m in milestones)}")
        
        # Get PRs with required labels
        reminders = []
        for milestone in milestones:
            for issue in self.repo.get_issues(state="open", milestone=milestone):
                if not issue.pull_request:
                    continue
                labels = {l.name for l in issue.labels}
                if self.EXPERT_REVIEW in labels or self.FINAL_REVIEW in labels:
                    try:
                        pr = self.repo.get_pull(issue.number)
                        reminders.append(self.create_reminder(pr))
                        print(f"✅ PR #{pr.number}")
                    except Exception as e:
                        print(f"❌ PR #{issue.number}: {e}")
        
        return sorted(reminders, key=lambda r: (r.priority, -r.current_stage_time))


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("REPO", "NVIDIA/Megatron-LM")
    
    if not token:
        print("❌ GITHUB_TOKEN required")
        sys.exit(1)
    
    tracker = PRReviewTracker(token, repo)
    reminders = tracker.generate_reminders()
    
    print(f"\n📊 {len(reminders)} reminders\n" + "=" * 80)
    for r in reminders:
        print(f"\n{r.priority} | PR #{r.id} | {r.milestone}")
        print(f"   Author: {r.author} | Stage: {r.review_stage}")
        print(f"   Stage time: {r.current_stage_time}d | Total: {r.total_review_time}d")
        print(f"   Reviewers: {', '.join(r.reviewers) if r.reviewers else 'None'}")
    
    return reminders


if __name__ == "__main__":
    main()