import os
import re
import time
from datetime import datetime, timedelta, timezone

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


def get_label_times(mr):
    """Get the earliest time when each review label was first added."""
    # Get resource label events with all pages
    events = mr.resourcelabelevents.list(get_all=True)
    print(f"Processing MR #{mr.iid}: Found {len(events)} label events")

    # Initialize times
    expert_review_requested_time = None
    final_review_requested_time = None
    current_stage = None

    # Process all events in chronological order
    for event in sorted(events, key=lambda x: x.created_at):
        if not event.label or event.action != 'add':
            continue

        label_name = event.label.get('name')
        if label_name == 'Expert Review' and expert_review_requested_time is None:
            expert_review_requested_time = event.created_at
            print(f"First Expert Review added at: {expert_review_requested_time}")
        elif label_name == 'Final Review' and final_review_requested_time is None:
            final_review_requested_time = event.created_at
            print(f"First Final Review added at: {final_review_requested_time}")

    # If no review labels found, use MR creation time as fallback
    if not expert_review_requested_time:
        expert_review_requested_time = mr.created_at
        print(
            f"No Expert Review label found, using MR creation time: {expert_review_requested_time}"
        )

    # Determine current stage based on labels
    if 'Final Review' in mr.labels:
        current_stage = 'Final Review'
        current_stage_time = final_review_requested_time
    elif 'Expert Review' in mr.labels:
        current_stage = 'Expert Review'
        current_stage_time = expert_review_requested_time
    else:
        current_stage = None
        current_stage_time = expert_review_requested_time

    return expert_review_requested_time, current_stage_time, current_stage


def get_days_since(dt_str):
    """Calculate number of days since the given datetime string."""
    if not dt_str:
        return 0
    now = datetime.now(timezone.utc)
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    delta = now - dt
    return max(1, delta.days)  # Round up to at least 1 day


def get_age_category(label_added_time):
    """Categorize MRs by their age based on when 'Final Review' label was added."""
    now = datetime.now(timezone.utc)  # Make now timezone-aware
    added = datetime.fromisoformat(label_added_time.replace('Z', '+00:00'))
    age = now - added

    if age <= timedelta(days=1):
        return "Last 24 hours"
    elif age <= timedelta(days=3):
        return "Last 3 days"
    else:
        return "Older than 3 days"


def extract_expert_reviewers(mr_description):
    """Extract expert reviewers from MR description.
    Looks for usernames under '(Step 2): Assign expert reviewers'.
    Format is:
    Category: username1, username2
    """
    if not mr_description:
        return []

    # Find the Step 2 section - capture everything until next section or end
    step2_pattern = r"\(Step 2\): Assign expert reviewers[\s\S]*?(?=\(Step \d+\):|$)"
    step2_match = re.search(step2_pattern, mr_description, re.IGNORECASE)

    if not step2_match:
        return []

    step2_section = step2_match.group(0)

    # Split into lines and process each line
    lines = step2_section.split('\n')
    reviewers = set()

    # Skip the header line and process each line
    for line in lines[1:]:  # Skip the header line
        line = line.strip()
        if not line or ':' not in line:
            continue

        # Split on colon and take the right side
        category, usernames = line.split(':', 1)
        # Skip if category is empty or looks like a header
        if not category.strip() or category.strip().lower() in ['assign expert reviewers']:
            continue

        # Split usernames by commas and spaces
        for username in re.split(r'[,\s]+', usernames.strip()):
            if username:
                email = f"{username.lower()}@nvidia.com"
                reviewers.add(email)

    return sorted(reviewers)  # Return sorted list for consistency


def was_label_removed(mr, label_name):
    """Check if a label was ever removed from the MR."""
    try:
        # Get resource label events with all pages
        events = mr.resourcelabelevents.list(get_all=True)
        print(f"Checking label events for MR #{mr.iid} - {label_name}")
        print(f"Total events found: {len(events)}")

        # Find all events for this label
        label_events = [
            event for event in events if event.label and event.label.get('name') == label_name
        ]
        print(f"Events for {label_name}: {len(label_events)}")

        # Check if there was any remove event
        for event in label_events:
            print(f"Label event: action={event.action}, created_at={event.created_at}")
            if event.action == 'remove':
                print(f"Found remove event for {label_name} at {event.created_at}")
                return True

        print(f"No remove events found for {label_name}")
        return False
    except Exception as e:
        print(f"Error checking label events: {e}")
        # If we can't check the events, assume the label wasn't removed
        return False


def get_required_reviewers(mr):
    """Get list of required reviewers who haven't approved yet."""

    def _get_approvals():
        return mr.approvals.get()

    # Get all approvals with retry
    approvals = retry_with_backoff(_get_approvals)

    approved_users = []

    # Extract user information from approvals
    for approval in approvals.approved_by:
        user = approval['user']
        print(f"User data: {user}")  # Debug print
        # Get username instead of email
        username = user.get('username')
        if username:
            approved_users.append(username)

    # Get assigned reviewers from GitLab API
    assigned_reviewers = []
    for reviewer in mr.reviewers:
        if reviewer.get('username'):
            assigned_reviewers.append(f"{reviewer['username']}@nvidia.com")
    print(f"Assigned reviewers from GitLab: {assigned_reviewers}")

    # Initialize empty list for required reviewers
    required_reviewers = []

    # Get reviewers based on current stage
    if 'Expert Review' in mr.labels:
        # Extract expert reviewers from MR description
        expert_reviewers = extract_expert_reviewers(mr.description)
        print(f"Expert reviewers from description: {expert_reviewers}")

        # Only keep reviewers who are both assigned and in expert reviewers list
        required_reviewers = [r for r in assigned_reviewers if r in expert_reviewers]
        print(f"Expert reviewers after filtering: {required_reviewers}")

    elif 'Final Review' in mr.labels:
        # Get code-owners groups and their members
        code_owners_groups = {}
        for reviewer in mr.reviewers:
            if reviewer.get('username'):
                email = f"{reviewer['username']}@nvidia.com"
                # Get groups this reviewer is part of
                groups = reviewer.get('groups', [])
                for group in groups:
                    if group not in code_owners_groups:
                        code_owners_groups[group] = set()
                    code_owners_groups[group].add(email)

        print(f"Code-owners groups: {code_owners_groups}")

        # Track which groups are satisfied
        satisfied_groups = set()
        for group, members in code_owners_groups.items():
            # A group is satisfied if at least one member has approved
            if any(member.split('@')[0] in approved_users for member in members):
                satisfied_groups.add(group)
                print(f"Group {group} is satisfied")

        # Get final reviewers who are not in satisfied groups
        remaining_final_reviewers = []
        for reviewer in REQUIRED_REVIEWERS["final_reviewers"]:
            # Check if reviewer is in any satisfied group
            reviewer_groups = set()
            for group, members in code_owners_groups.items():
                if reviewer in members:
                    reviewer_groups.add(group)

            # Only keep reviewer if they are not in any satisfied group
            if not any(group in satisfied_groups for group in reviewer_groups):
                remaining_final_reviewers.append(reviewer)
                print(f"Keeping final reviewer {reviewer} (groups: {reviewer_groups})")

        required_reviewers = remaining_final_reviewers
        print(f"Remaining final reviewers after code-owners check: {required_reviewers}")

    # Filter out reviewers who have already approved
    remaining_reviewers = [
        reviewer
        for reviewer in required_reviewers
        if reviewer.split('@')[0] not in approved_users  # Compare usernames
    ]
    print(f"Remaining reviewers after approval check: {remaining_reviewers}")  # Debug print
    return remaining_reviewers


def format_datetime(dt_str):
    """Format datetime string to YYYY-MM-DD HH:MM PT."""
    # Parse the input datetime
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

    # Convert to PT timezone
    pt_tz = timezone(timedelta(hours=-7))  # PT is UTC-7
    dt_pt = dt.astimezone(pt_tz)

    # Format as YYYY-MM-DD HH:MM PT
    return dt_pt.strftime("%Y-%m-%d %H:%M PT")


def get_priority(age_category):
    """Get priority based on age category with custom Slack emojis."""
    if age_category == "Older than 3 days":
        return "P0 :alert:"  # Custom alert emoji for critical
    elif age_category == "Last 3 days":
        return "P1 :yellow_alert:"  # Custom yellow alert for important
    else:  # Last 24 hours
        return "P2 :sparkles:"  # Custom sparkles for normal


def process_mrs(project, milestones, dry_run=False):
    """Process all MRs from given milestones."""
    if not any(milestones):
        print("No milestones found")
        return ""

    # Initialize age groups
    age_groups = {"Last 24 hours": [], "Last 3 days": [], "Older than 3 days": []}

    # Process MRs from all milestones
    for milestone in milestones:
        if not milestone:
            continue

        print(f"\nProcessing milestone: {milestone.title}")

        # Fetch merge requests for this milestone
        mrs = project.mergerequests.list(
            state='opened',  # Only get open MRs
            milestone=milestone.title,  # Filter by milestone
            order_by='updated_at',  # Order by update date
            sort='desc',  # Most recent first
        )
        print(f"Found {len(mrs)} open MRs in milestone")

        # Filter MRs with either review label
        review_mrs = [
            mr
            for mr in mrs
            if any(label in ['Final Review', 'Expert Review'] for label in mr.labels)
        ]
        print(f"Found {len(review_mrs)} MRs with review labels")

        # Add MRs to appropriate age groups
        for mr in review_mrs:
            expert_review_time, current_stage_time, review_stage = get_label_times(mr)
            age_category = get_age_category(current_stage_time)
            age_groups[age_category].append(
                (mr, milestone, expert_review_time, current_stage_time, review_stage)
            )
            print(f"Added MR #{mr.iid} to {age_category} group")

    # Build and send individual messages for each MR
    for age_group, mrs_in_group in age_groups.items():
        if mrs_in_group:
            print(f"\nProcessing {len(mrs_in_group)} MRs in {age_group} group")

            # Sort MRs by total review time (days since Expert Review) in ascending order
            mrs_in_group.sort(key=lambda x: get_days_since(x[2]))  # x[2] is expert_review_time

            for mr, milestone, expert_review_time, current_stage_time, review_stage in mrs_in_group:
                # Build message for this MR
                message = []
                message.append(f"*MR*: <{mr.web_url}|#{mr.iid} - {mr.title}>")
                message.append(f"*Milestone*: {milestone.title}")

                # Get author's Slack ID and create mention
                author_email = f"{mr.author['username']}@nvidia.com"
                author_id = get_slack_user_id(author_email)
                author_mention = f"<@{author_id}>" if author_id else mr.author['name']
                message.append(f"*Author*: {author_mention}")

                message.append(f"*Priority*: {get_priority(age_group)}")
                message.append(f"*Review stage*: {review_stage or 'No review label'}")
                message.append(f"*Days in review*: {get_days_since(expert_review_time)}")
                message.append(f"*Days in {review_stage}*: {get_days_since(current_stage_time)}")

                # Get and display required reviewers and missing approvals
                required_reviewers = get_required_reviewers(mr)
                if required_reviewers:
                    # Convert email addresses to Slack mentions
                    reviewer_mentions = []
                    for reviewer in required_reviewers:
                        user_id = get_slack_user_id(reviewer)
                        if user_id:
                            reviewer_mentions.append(f"<@{user_id}>")
                        else:
                            reviewer_mentions.append(reviewer)
                    message.append("*Required reviewers*: " + ", ".join(reviewer_mentions))
                else:
                    # Convert maintainer emails to Slack mentions
                    maintainer_mentions = []
                    for maintainer in CI_MAINTAINER:
                        user_id = get_slack_user_id(maintainer)
                        if user_id:
                            maintainer_mentions.append(f"<@{user_id}>")
                        else:
                            maintainer_mentions.append(maintainer)
                    message.append(
                        f"*All required reviewers have approved (cc: {', '.join(maintainer_mentions)})*"
                    )

                # Send individual message for this MR
                print(f"Sending message for MR #{mr.iid}")
                send_to_slack("\n".join(message), dry_run)

    # Return empty string since we're sending messages directly
    return ""


def get_slack_user_id(email):
    """Look up Slack user ID by email with caching and retries."""
    if not slack_client:
        return None

    # Check cache first
    if email in slack_user_cache:
        return slack_user_cache[email]

    def lookup_user():
        try:
            response = slack_client.users_lookupByEmail(email=email)
            if response["ok"]:
                user_id = response["user"]["id"]
                # Cache the result
                slack_user_cache[email] = user_id
                return user_id
            return None
        except SlackApiError as e:
            if e.response["error"] == "users_not_found":
                # Cache None if user not found
                slack_user_cache[email] = None
                return None
            raise

    try:
        return retry_with_backoff(lookup_user)
    except SlackApiError as e:
        print(f"Error looking up Slack user after retries: {e}")
        return None


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


@click.command()
@click.option('--dry-run', is_flag=True, help='Run in dry-run mode without sending to Slack')
def main(dry_run):
    """Auto reminder script for MR reviews."""
    if dry_run:
        print("Running in DRY-RUN mode - no messages will be sent to Slack")

    try:
        gl = get_gitlab_handle()
        project = gl.projects.get(PROJECT_ID)

        # Get the most recent milestones
        current_milestone, previous_milestone = get_recent_milestones(project)
        if not current_milestone and not previous_milestone:
            message = "No active milestones found"
            print(message)
            send_to_slack(message, dry_run)
            return

        # Process all MRs from both milestones
        process_mrs(project, [current_milestone, previous_milestone], dry_run)
    except Exception as e:
        error_message = f"Error in main execution: {str(e)}"
        print(error_message)
        send_to_slack(error_message, dry_run)
        raise


if __name__ == "__main__":
    main()
