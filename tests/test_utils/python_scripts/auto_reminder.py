import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import click
import gitlab
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Get environment variables
PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
GITLAB_ENDPOINT = os.getenv('GITLAB_ENDPOINT')
RO_API_TOKEN = os.getenv("RO_API_TOKEN")
SLACK_WEBHOOK_URL = os.getenv("SLACK_REMINDER_HOOK")  # Webhook URL for the channel
SLACK_API_TOKEN = os.getenv("SLACK_API_TOKEN")  # For user lookups only

# Validate required environment variables
if not GITLAB_ENDPOINT or GITLAB_ENDPOINT == 'none':
    raise ValueError("GITLAB_ENDPOINT environment variable is not set or is invalid")
if not RO_API_TOKEN:
    raise ValueError("RO_API_TOKEN environment variable is not set")

# Required reviewers
REQUIRED_REVIEWERS = {
    "final_reviewers": [
        "jcasper@nvidia.com",
        "dnarayanan@nvidia.com",
        "eharper@nvidia.com",
        "shanmugamr@nvidia.com",
        "yuya@nvidia.com",
        "ansubramania@nvidia.com",
    ],
    "expert_reviewers": [],
}
CI_MAINTAINER = ["okoenig@nvidia.com"]  # Using email address

# Initialize Slack client for user lookups only
slack_client = WebClient(token=SLACK_API_TOKEN) if SLACK_API_TOKEN else None

# Cache for Slack user IDs
slack_user_cache = {}


@dataclass
class Reminder:
    iid: int
    mr: str
    milestone: str
    author: str
    priority: str
    review_stage: str
    total_review_time: str
    current_stage_time: str
    reviewers: list[str]


def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """Retry a function with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed after {max_retries} attempts. Last error: {e}")
                raise
            print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff


def get_gitlab_handle():
    """Get GitLab handle with retry logic."""

    def _get_handle():
        try:
            return gitlab.Gitlab(
                f"https://{GITLAB_ENDPOINT}", private_token=RO_API_TOKEN, timeout=30  # Add timeout
            )
        except Exception as e:
            print(f"Error creating GitLab handle: {e}")
            print(f"Using endpoint: https://{GITLAB_ENDPOINT}")
            raise

    return retry_with_backoff(_get_handle)


def get_recent_milestones(project):
    """Get the two most recent milestones from the project."""
    milestones = project.milestones.list(state='active', sort='due_date_desc')
    if not milestones:
        return None, None
    return milestones[0], milestones[1] if len(milestones) > 1 else None


def get_current_review_stage(mr):
    """Get the current review stage of the MR."""
    if 'Final Review' in mr.labels:
        return 'Final Review'
    elif 'Expert Review' in mr.labels:
        return 'Expert Review'
    return None


def get_mcore_reviewers():
    """Get all members of mcore-reviewers group and its subgroups recursively."""
    mcore_group = get_gitlab_handle().groups.get('mcore-reviewers')
    reviewers = set()

    def get_group_members(group):
        # Get direct members of the group
        for member in group.members.list(get_all=True):
            reviewers.add(f"{member.username}@nvidia.com")

        # Recursively get members of subgroups
        for subgroup in group.subgroups.list(get_all=True):
            subgroup_obj = get_gitlab_handle().groups.get(subgroup.id)
            get_group_members(subgroup_obj)

    get_group_members(mcore_group)
    return list(reviewers)


def get_days_in_stage(mr, stage):
    """Get the latest time when each review label was added."""
    for event in sorted(
        mr.resourcelabelevents.list(get_all=True), key=lambda x: x.created_at, reverse=True
    ):
        if (
            event.label.get('name') == stage
            and event.action == 'add'
            and '_bot_' not in event.user.get('username')
        ):
            return get_days_since(event.created_at)


def get_days_since(dt_str):
    """Calculate number of days since the given datetime string."""
    if not dt_str:
        return 0
    now = datetime.now(timezone.utc)
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    delta = now - dt
    return max(1, delta.days)  # Round up to at least 1 day


def get_required_reviewers(mr):
    """Get list of required reviewers who haven't approved yet."""
    print(f"MR #{mr.iid} - {mr.title}")

    # Extract user information from approvals
    approved_users = []
    for approval in mr.approvals.get().approved_by:
        if username := approval['user'].get('username'):
            approved_users.append(f"{username}@nvidia.com")

    # Get assigned reviewers from GitLab API
    assigned_reviewers = []
    for reviewer in mr.reviewers:
        if username := reviewer.get('username'):
            assigned_reviewers.append(f"{username}@nvidia.com")

    print(f"Assigned reviewers: {assigned_reviewers}")
    print(f"Approved users: {approved_users}")

    # Get reviewers based on current stage
    if get_current_review_stage(mr) == 'Expert Review':
        review_group = REQUIRED_REVIEWERS["expert_reviewers"]

    elif get_current_review_stage(mr) == 'Final Review':
        review_group = REQUIRED_REVIEWERS["final_reviewers"]
        # Get pipeline status
        mr_pipelines = mr.pipelines.list(sort='desc', order_by='created_at')
        pipeline = mr_pipelines[0] if mr_pipelines else None
        if pipeline and pipeline.status != 'success':
            review_group = []

    else:
        review_group = []

    review_group = [
        reviewer
        for reviewer in review_group
        if (reviewer in assigned_reviewers) and (reviewer not in approved_users)
    ]

    if review_group is None or len(review_group) == 0:
        review_group = ["okoenig@nvidia.com"]

    print(f"Reviewer: {review_group}")

    return ", ".join(get_slack_user_id(reviewer) for reviewer in review_group)


def get_priority(days_in_current_stage):
    """Get priority based on age category with custom Slack emojis."""

    if days_in_current_stage <= 1:
        return "P2 :sparkles:"  # Custom sparkles for normal
    elif days_in_current_stage <= 3:
        return "P1 :yellow_alert:"  # Custom yellow alert for important
    else:
        return "P0 :alert:"  # Custom alert emoji for critical


def get_slack_user_id(email):
    """Look up Slack user ID by email with caching and retries."""
    if not slack_client:
        return None

    def lookup_user():
        if email in slack_user_cache:
            return slack_user_cache[email]

        try:
            response = slack_client.users_lookupByEmail(email=email)
            if response["ok"]:
                user_id = response["user"]["id"]
                # Cache the result
                slack_user_cache[email] = f"<@{user_id}>"
                return slack_user_cache[email]
            return None
        except SlackApiError as e:
            if e.response["error"] == "users_not_found":
                # Cache None if user not found
                return ""
            raise

    try:
        return retry_with_backoff(lookup_user)
    except SlackApiError as e:
        print(f"Error looking up Slack user after retries: {e}")
        return ""


def send_to_slack(message, dry_run=False):
    """Send message to Slack using webhook."""
    if not SLACK_WEBHOOK_URL:
        print("Warning: SLACK_REMINDER_HOOK not set, skipping Slack notification")
        return

    if dry_run:
        print("\n=== DRY RUN - Would send to Slack ===")
        print(message)
        print("====================================\n")
        return

    payload = {"text": message, "mrkdwn": True}

    def _send():
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10,  # Add timeout
        )
        response.raise_for_status()
        return response

    try:
        retry_with_backoff(_send)
    except Exception as e:
        print(f"Error sending to Slack webhook after retries: {e}")


def process_mrs(project, milestones, labels, dry_run=False):
    """Process all MRs from given milestones."""
    if not any(milestones):
        print("No milestones found")
        return ""

    reminders = [
        Reminder(
            iid=mr.iid,
            mr=f"<{mr.web_url}|#{mr.iid} - {mr.title}>",
            milestone=mr.milestone['title'],
            author=get_slack_user_id(f"{mr.author['username']}@nvidia.com"),
            priority=get_priority(get_days_in_stage(mr, stage=get_current_review_stage(mr))),
            review_stage=get_current_review_stage(mr),
            total_review_time=get_days_in_stage(mr, stage="Expert Review"),
            current_stage_time=get_days_in_stage(mr, stage=get_current_review_stage(mr)),
            reviewers=get_required_reviewers(mr),
        )
        for m in milestones
        for label in labels
        for mr in project.mergerequests.list(
            state='opened',  # Only get open MRs
            milestone=m.title,  # Filter by milestone
            labels=[label],  # Filter by label
            order_by='updated_at',  # Order by update date
            sort='desc',  # Most recent first
        )
    ]

    reminders.sort(key=lambda x: x.current_stage_time)

    # Build and send individual messages for each MR
    for reminder in reminders:

        # Build message for this MR
        message = []
        message.append(f"*MR*: {reminder.mr}")
        message.append(f"*Milestone*: {reminder.milestone}")
        message.append(f"*Author*: {reminder.author}")
        message.append(f"*Priority*: {reminder.priority}")
        message.append(f"*Review stage*: {reminder.review_stage}")
        message.append(f"*Days in review*: {reminder.total_review_time}")
        message.append(f"*Days in {reminder.review_stage}*: {reminder.current_stage_time}")
        message.append(f"*Reviewers*: {reminder.reviewers}")
        # Send individual message for this MR
        print(f"Sending message for MR #{reminder.iid}")
        send_to_slack("\n".join(message), dry_run)


@click.command()
@click.option('--dry-run', is_flag=True, help='Run in dry-run mode without sending to Slack')
def main(dry_run):
    """Auto reminder script for MR reviews."""
    if dry_run:
        print("Running in DRY-RUN mode - no messages will be sent to Slack")

    REQUIRED_REVIEWERS["expert_reviewers"] = list(
        set(get_mcore_reviewers()) - set(REQUIRED_REVIEWERS["final_reviewers"])
    )
    print(REQUIRED_REVIEWERS)

    try:
        gl = get_gitlab_handle()
        project = gl.projects.get(PROJECT_ID)

        process_mrs(
            project,
            milestones=get_recent_milestones(project),
            labels=["Expert Review", "Final Review"],
            dry_run=dry_run,
        )
    except Exception as e:
        error_message = f"Error in main execution: {str(e)}"
        print(error_message)
        send_to_slack(error_message, dry_run)
        raise


if __name__ == "__main__":
    main()
