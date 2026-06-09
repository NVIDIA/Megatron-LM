# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import requests
import argparse
from datetime import datetime, timedelta, timezone

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Constants
GITHUB_API_URL = "https://api.github.com"
SCHEDULE_FILE = ".github/oncall_schedule.json"
ROTATION_TEAM_SLUG = "mcore-oncall-rotation"
ACTIVE_ONCALL_TEAM_SLUG = "mcore-oncall"
SLACK_USERGROUP_HANDLE = "mcore-oncall"
COMMUNITY_REQUEST_LABEL = "community-request"
TARGET_WEEKS = 12
SLACK_ROTATION_FALLBACK_EMAILS = ["ppetrakian@nvidia.com"]

# Caches for email and Slack lookups
_email_cache = {}
_slack_id_cache = {}


def get_slack_error(error):
    response = getattr(error, "response", None)
    if response is not None:
        try:
            return response["error"]
        except (KeyError, TypeError):
            pass
    return str(error)


def get_headers():
    token = os.environ.get("GH_TOKEN")
    if not token:
        # Fallback to GITHUB_TOKEN if GH_TOKEN not set
        token = os.environ.get("GITHUB_TOKEN")

    if not token:
        print("Error: GH_TOKEN or GITHUB_TOKEN not set")
        sys.exit(1)

    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}


def get_repo_info():
    """Returns (owner, repo) from GITHUB_REPOSITORY env var."""
    repo_env = os.environ.get("GITHUB_REPOSITORY")
    if not repo_env:
        print("Error: GITHUB_REPOSITORY environment variable not set")
        sys.exit(1)
    parts = repo_env.split("/")
    return parts[0], parts[1]


def get_team_members(org, team_slug):
    """Fetches members of the GitHub team."""
    url = f"{GITHUB_API_URL}/orgs/{org}/teams/{team_slug}/members"
    headers = get_headers()

    members = set()
    page = 1
    while True:
        resp = requests.get(f"{url}?per_page=100&page={page}", headers=headers)
        if resp.status_code != 200:
            print(f"Error fetching team members: {resp.status_code} {resp.text}")
            sys.exit(1)

        data = resp.json()
        if not data:
            break

        members.update([m['login'] for m in data])
        if len(data) < 100:
            break
        page += 1

    return members


def get_user_email(username):
    """Get user's email from GitHub, prioritizing @nvidia.com emails.

    Checks in order:
    1. Public profile email
    2. Recent commits in the repository
    """
    if username in _email_cache:
        return _email_cache[username]

    headers = get_headers()
    public_email = None

    try:
        # 1. Try to get user's public profile email first
        resp = requests.get(f"{GITHUB_API_URL}/users/{username}", headers=headers)
        if resp.status_code == 200:
            user_data = resp.json()
            email = user_data.get('email')
            if email and not email.endswith("@users.noreply.github.com"):
                if email.endswith("@nvidia.com"):
                    _email_cache[username] = email
                    return email
                # Store non-nvidia email as fallback
                public_email = email

        # 2. Check recent commits in the repository for @nvidia.com email
        repo_env = os.environ.get("GITHUB_REPOSITORY", "NVIDIA/Megatron-LM")
        commits_url = f"{GITHUB_API_URL}/repos/{repo_env}/commits?author={username}&per_page=10"
        resp = requests.get(commits_url, headers=headers)

        if resp.status_code == 200:
            commits = resp.json()
            for commit in commits:
                # Get email from commit author
                commit_data = commit.get('commit', {})
                author_data = commit_data.get('author', {})
                email = author_data.get('email')

                if email and not email.endswith("@users.noreply.github.com"):
                    if email.endswith("@nvidia.com"):
                        _email_cache[username] = email
                        print(f"Found @nvidia.com email for {username} from commits: {email}")
                        return email
                    elif public_email is None:
                        public_email = email

        # 3. Use public email if found, otherwise fallback
        if public_email:
            _email_cache[username] = public_email
            print(f"Using public email for {username}: {public_email}")
            return public_email

        # Fallback to noreply email
        fallback = f"{username}@users.noreply.github.com"
        _email_cache[username] = fallback
        print(f"Warning: No email found for {username}, using fallback: {fallback}")
        return fallback

    except Exception as e:
        print(f"Warning: Could not get email for {username}: {e}")
        fallback = f"{username}@users.noreply.github.com"
        _email_cache[username] = fallback
        return fallback


def get_slack_client():
    """Get Slack WebClient if token is available."""
    slack_token = os.environ.get("SLACK_TOKEN")
    if not slack_token:
        return None

    return WebClient(token=slack_token)


def get_slack_user_id(slack_client, email):
    """Get Slack user ID from email."""
    if not slack_client:
        return None

    if email in _slack_id_cache:
        return _slack_id_cache[email]

    try:
        response = slack_client.users_lookupByEmail(email=email)
        user_id = response["user"]["id"]
        _slack_id_cache[email] = user_id
        return user_id
    except SlackApiError as e:
        print(f"Warning: Could not find Slack user for {email}: {get_slack_error(e)}")
        _slack_id_cache[email] = None
        return None


def get_slack_usergroup_id(slack_client, handle):
    """Get Slack usergroup ID from handle."""
    if not slack_client:
        return None

    try:
        response = slack_client.usergroups_list(include_users=True)
        for usergroup in response.get("usergroups", []):
            if usergroup.get("handle") == handle:
                return usergroup.get("id"), usergroup.get("users", [])
        print(f"Warning: Slack usergroup '{handle}' not found")
        return None, []
    except SlackApiError as e:
        print(f"Warning: Could not list Slack usergroups: {get_slack_error(e)}")
        return None, []


def get_slack_rotation_fallback_emails():
    emails = os.environ.get("ONCALL_SLACK_NOTIFY_EMAILS", "")
    if emails:
        parsed_emails = [email.strip() for email in emails.split(",") if email.strip()]
        if parsed_emails:
            return parsed_emails
    return SLACK_ROTATION_FALLBACK_EMAILS


def get_github_actions_run_url():
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not repo or not run_id:
        return None

    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server_url}/{repo}/actions/runs/{run_id}"


def get_slack_user_ids_for_emails(slack_client, emails):
    slack_user_ids = []
    seen = set()
    for email in emails:
        slack_id = get_slack_user_id(slack_client, email)
        if slack_id and slack_id not in seen:
            slack_user_ids.append(slack_id)
            seen.add(slack_id)
    return slack_user_ids


def get_slack_user_ids_for_usernames(slack_client, usernames):
    emails = [get_user_email(username) for username in usernames]
    return get_slack_user_ids_for_emails(slack_client, emails)


def get_slack_rotation_notification_recipients(slack_client, old_members_usernames):
    previous_oncall_recipients = get_slack_user_ids_for_usernames(
        slack_client, old_members_usernames
    )
    if previous_oncall_recipients:
        return previous_oncall_recipients, "previous oncall"

    fallback_emails = get_slack_rotation_fallback_emails()
    fallback_recipients = get_slack_user_ids_for_emails(slack_client, fallback_emails)
    return fallback_recipients, "fallback contacts"


def send_slack_rotation_failure_notification(
    slack_client, reason, new_oncall_username, new_oncall_email, old_members_usernames
):
    recipients, recipient_source = get_slack_rotation_notification_recipients(
        slack_client, old_members_usernames
    )

    run_url = get_github_actions_run_url()
    message_lines = [
        "MCore on-call Slack rotation needs manual attention.",
        f"Reason: {reason}",
        f"Target GitHub oncall: {new_oncall_username}",
        f"Email tried for Slack lookup: {new_oncall_email}",
        f"Slack usergroup not updated: @{SLACK_USERGROUP_HANDLE}",
        "Please update the Slack usergroup manually or fix the GitHub-to-Slack email mapping.",
    ]
    if run_url:
        message_lines.append(f"Workflow run: {run_url}")
    message = "\n".join(message_lines)

    if not recipients:
        print("Warning: Could not resolve Slack notification recipients for rotation failure")
        print(message)
        return False

    sent = False
    for recipient in recipients:
        try:
            slack_client.chat_postMessage(channel=recipient, text=message)
            print(f"Sent Slack rotation failure notification to {recipient} ({recipient_source})")
            sent = True
        except SlackApiError as e:
            print(
                f"Warning: Could not send Slack rotation failure notification to {recipient}: {get_slack_error(e)}"
            )

    if not sent:
        print("Warning: Failed to send Slack rotation failure notification")
    return sent


def update_slack_usergroup(new_oncall_username, old_members_usernames):
    """
    Updates the Slack usergroup to contain only the new oncall user.
    Adds new oncall first, then removes old members (usergroups need at least one member).
    """
    slack_client = get_slack_client()
    if not slack_client:
        print("Slack token not configured, skipping Slack usergroup update")
        return

    # Get the new oncall's email and Slack user ID
    new_email = get_user_email(new_oncall_username)
    new_slack_id = get_slack_user_id(slack_client, new_email)

    if not new_slack_id:
        reason = f"Could not find Slack user ID for {new_oncall_username} ({new_email})"
        print(f"{reason}, skipping Slack update")
        send_slack_rotation_failure_notification(
            slack_client, reason, new_oncall_username, new_email, old_members_usernames
        )
        return

    # Get the usergroup ID and current members
    usergroup_id, current_slack_members = get_slack_usergroup_id(
        slack_client, SLACK_USERGROUP_HANDLE
    )

    if not usergroup_id:
        reason = f"Could not find Slack usergroup '{SLACK_USERGROUP_HANDLE}'"
        print(f"{reason}, skipping Slack update")
        send_slack_rotation_failure_notification(
            slack_client, reason, new_oncall_username, new_email, old_members_usernames
        )
        return

    try:
        # Step 1: Add new oncall first (include current members to avoid removing anyone yet)
        # This ensures usergroup always has at least one member
        if new_slack_id not in current_slack_members:
            updated_members = current_slack_members + [new_slack_id]
            slack_client.usergroups_users_update(usergroup=usergroup_id, users=updated_members)
            print(f"Added {new_oncall_username} to Slack usergroup '{SLACK_USERGROUP_HANDLE}'")

        # Step 2: Now set the usergroup to contain only the new oncall
        slack_client.usergroups_users_update(usergroup=usergroup_id, users=[new_slack_id])
        print(
            f"Updated Slack usergroup '{SLACK_USERGROUP_HANDLE}' to contain only {new_oncall_username}"
        )

    except SlackApiError as e:
        error = get_slack_error(e)
        print(f"Failed to update Slack usergroup: {error}")
        send_slack_rotation_failure_notification(
            slack_client,
            f"Failed to update Slack usergroup: {error}",
            new_oncall_username,
            new_email,
            old_members_usernames,
        )


def load_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        return []
    try:
        with open(SCHEDULE_FILE, 'r') as f:
            data = json.load(f)
            # Normalize to list of dicts if it's a list of strings
            schedule = []
            for item in data:
                if isinstance(item, str):
                    schedule.append({"user": item, "date": "YYYY-MM-DD"})
                else:
                    schedule.append(item)
            return schedule
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_schedule(schedule):
    with open(SCHEDULE_FILE, 'w') as f:
        json.dump(schedule, f, indent=4)
        f.write('\n')  # trailing newline


def update_active_oncall_team(org, new_oncall):
    """Updates the active oncall team to contain only the new oncall user."""
    # 1. Get current members of the active team
    current_members = get_team_members(org, ACTIVE_ONCALL_TEAM_SLUG)

    # 2. Add the new oncall if not present
    if new_oncall not in current_members:
        url = (
            f"{GITHUB_API_URL}/orgs/{org}/teams/{ACTIVE_ONCALL_TEAM_SLUG}/memberships/{new_oncall}"
        )
        resp = requests.put(url, headers=get_headers())
        if resp.status_code == 200:
            print(f"Added {new_oncall} to {ACTIVE_ONCALL_TEAM_SLUG}")
        else:
            print(
                f"Failed to add {new_oncall} to {ACTIVE_ONCALL_TEAM_SLUG}: {resp.status_code} {resp.text}"
            )

    # 3. Remove everyone else
    old_members = []
    for member in current_members:
        if member not in [new_oncall, 'svcnvidia-nemo-ci']:
            old_members.append(member)
            url = (
                f"{GITHUB_API_URL}/orgs/{org}/teams/{ACTIVE_ONCALL_TEAM_SLUG}/memberships/{member}"
            )
            resp = requests.delete(url, headers=get_headers())
            if resp.status_code == 204:
                print(f"Removed {member} from {ACTIVE_ONCALL_TEAM_SLUG}")
            else:
                print(
                    f"Failed to remove {member} from {ACTIVE_ONCALL_TEAM_SLUG}: {resp.status_code} {resp.text}"
                )

    # 4. Update Slack usergroup (add new oncall first, then remove old members)
    update_slack_usergroup(new_oncall, old_members)


def rotate_schedule(repo_owner, dry_run=False):
    schedule = load_schedule()
    print(f"Current schedule length: {len(schedule)}")

    # 1. Rotate (Remove past week)
    # Only if schedule is not empty.
    if schedule:
        # Check date of first entry
        first_entry = schedule[0]
        try:
            # We assume the date is the *start* of the oncall shift (Wednesday).
            # The shift ends 7 days later.
            start_date = datetime.strptime(first_entry['date'], "%Y-%m-%d").date()
            end_date = start_date + timedelta(days=7)

            today = datetime.now(timezone.utc).date()

            # If today is >= end_date, the shift is over.
            # (e.g. Started last Wed, ends today Wed. If today is Wed, we rotate)
            if today >= end_date:
                removed = schedule.pop(0)
                print(f"Rotated out: {removed} (Ended {end_date})")
            else:
                print(
                    f"First entry {first_entry} has not ended yet (Ends {end_date}). Not removing."
                )
        except ValueError:
            # Fallback if date is invalid, rotate anyway
            removed = schedule.pop(0)
            print(f"Rotated out (invalid date): {removed}")
    else:
        print("Schedule empty, nothing to rotate.")

    # 2. Replenish
    ensure_schedule_filled(schedule, repo_owner)

    # 3. Update active oncall team
    if schedule:
        current_oncall = schedule[0]['user']
        print(f"New active oncall: {current_oncall}")
        if not dry_run:
            update_active_oncall_team(repo_owner, current_oncall)
        else:
            print(
                f"Dry run: Would update {ACTIVE_ONCALL_TEAM_SLUG} to contain only {current_oncall}"
            )

    if not dry_run:
        save_schedule(schedule)
        print("Schedule updated and saved.")
    else:
        print("Dry run: Schedule not saved.")
        print(json.dumps(schedule, indent=4))


def get_last_wednesday():
    today = datetime.now(timezone.utc).date()
    # Monday=0, Wednesday=2
    offset = (today.weekday() - 2) % 7
    return today - timedelta(days=offset)


def ensure_schedule_filled(schedule, repo_owner):
    """Appends users to schedule until it reaches TARGET_WEEKS."""
    members = get_team_members(repo_owner, ROTATION_TEAM_SLUG)
    if not members:
        print(f"Warning: No team members found in {ROTATION_TEAM_SLUG}.")
        return
    if 'svcnvidia-nemo-ci' in members:
        members.remove('svcnvidia-nemo-ci')
    members = list(members)

    members.sort()  # Deterministic order

    while len(schedule) < TARGET_WEEKS:
        # Determine start date for the new entry
        if not schedule:
            # Start with the most recent Wednesday if list is empty
            next_date = get_last_wednesday()

            # Start with the first member alphabetically if list is empty
            next_user = members[0]
        else:
            last_entry = schedule[-1]
            last_user = last_entry['user']

            # Parse last date and add 7 days
            try:
                last_date = datetime.strptime(last_entry['date'], "%Y-%m-%d").date()
                next_date = last_date + timedelta(days=7)
            except ValueError:
                # Fallback if date is invalid/placeholder
                next_date = get_last_wednesday() + timedelta(days=7 * len(schedule))

            try:
                # Find index of last scheduled user in the team list
                if last_user in members:
                    last_idx = members.index(last_user)
                    next_idx = (last_idx + 1) % len(members)
                    next_user = members[next_idx]
                else:
                    # Last user not in team, just pick first member
                    next_user = members[0]
            except ValueError:
                next_user = members[0]

        new_entry = {"user": next_user, "date": next_date.strftime("%Y-%m-%d")}
        schedule.append(new_entry)
        print(f"Appended: {new_entry}")


def assign_reviewer(pr_number):
    """Assigns mcore-oncall if no reviewers are set or community-request is applied."""
    owner, repo = get_repo_info()

    pr_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pr_number}"
    pr_resp = requests.get(pr_url, headers=get_headers())

    if pr_resp.status_code != 200:
        print(f"Failed to fetch PR: {pr_resp.status_code} {pr_resp.text}")
        sys.exit(1)

    pr_data = pr_resp.json()
    requested_reviewers = pr_data.get("requested_reviewers", [])
    requested_teams = pr_data.get("requested_teams", [])
    labels = {label.get("name") for label in pr_data.get("labels", [])}
    requested_team_slugs = {team.get("slug") for team in requested_teams}
    has_community_request_label = COMMUNITY_REQUEST_LABEL in labels

    if ACTIVE_ONCALL_TEAM_SLUG in requested_team_slugs:
        print(
            f"Skipping reviewer request: team NVIDIA/{ACTIVE_ONCALL_TEAM_SLUG} "
            "is already requested"
        )
        return

    if not has_community_request_label and (requested_reviewers or requested_teams):
        print(
            "Skipping reviewer request: PR already has "
            f"{len(requested_reviewers)} user reviewer(s) and "
            f"{len(requested_teams)} team reviewer(s)"
        )
        return

    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers"

    # Assign the oncall team as reviewer
    data = {"team_reviewers": [ACTIVE_ONCALL_TEAM_SLUG]}
    resp = requests.post(url, headers=get_headers(), json=data)

    if resp.status_code in [201, 200]:
        print(f"Successfully requested review from team NVIDIA/{ACTIVE_ONCALL_TEAM_SLUG}")
    else:
        print(f"Failed to request review: {resp.status_code} {resp.text}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Manage Oncall Schedule")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Rotate command
    parser_rotate = subparsers.add_parser(
        "rotate", help="Rotate the schedule (remove first, append new)"
    )
    parser_rotate.add_argument("--dry-run", action="store_true", help="Do not save changes")

    # Fill command (just fill up to 12 without rotating - useful for init)
    parser_fill = subparsers.add_parser(
        "fill", help="Fill the schedule to 12 weeks without rotating"
    )

    # Assign command
    parser_assign = subparsers.add_parser("assign", help="Assign current oncall to PR")
    parser_assign.add_argument("--pr", type=int, required=True, help="PR number")

    args = parser.parse_args()

    owner, _ = get_repo_info()

    if args.command == "rotate":
        rotate_schedule(owner, dry_run=args.dry_run)
    elif args.command == "fill":
        schedule = load_schedule()
        ensure_schedule_filled(schedule, owner)
        save_schedule(schedule)
        print("Schedule filled and saved.")
    elif args.command == "assign":
        assign_reviewer(args.pr)


if __name__ == "__main__":
    main()
