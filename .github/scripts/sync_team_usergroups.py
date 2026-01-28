# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Syncs GitHub team membership to Slack user groups.

This script reads members from GitHub teams and updates the corresponding
Slack user groups to match.
"""

import os
import sys
import argparse
import requests

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Constants
GITHUB_API_URL = "https://api.github.com"
PARENT_TEAM_SLUG = "mcore-reviewers"

# Caches for email and Slack lookups
_email_cache = {}
_slack_id_cache = {}
_usergroups_cache = None


def get_headers():
    """Get GitHub API headers with authentication."""
    token = os.environ.get("GH_TOKEN")
    if not token:
        token = os.environ.get("GITHUB_TOKEN")

    if not token:
        print("Error: GH_TOKEN or GITHUB_TOKEN not set")
        sys.exit(1)

    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def get_org():
    """Returns the organization from GITHUB_REPOSITORY env var or default."""
    repo_env = os.environ.get("GITHUB_REPOSITORY", "NVIDIA/Megatron-LM")
    return repo_env.split("/")[0]


def github_team_to_slack_usergroup(team_slug):
    """Convert a GitHub team slug to a Slack usergroup handle.

    Rules:
    - Base pattern: "test" -> "mcore-test"
    - Remove "core-" prefix: "core-test" -> "mcore-test"
    - Remove "megatron-" prefix: "megatron-test" -> "mcore-test"
    - Remove "-and-": "test1-and-test2" -> "mcore-test1-test2"
    - Shorten "mixture-of-experts" to "moe"
    - Shorten "pipeline-parallelism" to "pp"
    - Shorten "reinforcement-learning" to "rl"
    """
    name = team_slug

    # Apply shortenings first (before removing prefixes)
    name = name.replace("mixture-of-experts", "moe")
    name = name.replace("pipeline-parallelism", "pp")
    name = name.replace("reinforcement-learning", "rl")

    # Remove prefixes
    if name.startswith("core-"):
        name = name[5:]  # Remove "core-"
    elif name.startswith("megatron-"):
        name = name[9:]  # Remove "megatron-"

    # Remove "-and-"
    name = name.replace("-and-", "-")

    return f"mcore-{name}"


def get_child_teams(org, parent_team_slug):
    """Fetches child teams of a parent GitHub team."""
    # First get the team ID
    url = f"{GITHUB_API_URL}/orgs/{org}/teams/{parent_team_slug}"
    headers = get_headers()

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Error fetching parent team '{parent_team_slug}': {resp.status_code} {resp.text}")
        return []

    parent_team_id = resp.json().get("id")
    if not parent_team_id:
        print(f"Error: Could not get ID for team '{parent_team_slug}'")
        return []

    # Now fetch child teams
    url = f"{GITHUB_API_URL}/orgs/{org}/teams/{parent_team_slug}/teams"
    child_teams = []
    page = 1

    while True:
        resp = requests.get(f"{url}?per_page=100&page={page}", headers=headers)
        if resp.status_code != 200:
            print(f"Error fetching child teams: {resp.status_code} {resp.text}")
            return child_teams

        data = resp.json()
        if not data:
            break

        child_teams.extend([team["slug"] for team in data])
        if len(data) < 100:
            break
        page += 1

    return child_teams


def get_team_members(org, team_slug):
    """Fetches members of the GitHub team."""
    url = f"{GITHUB_API_URL}/orgs/{org}/teams/{team_slug}/members"
    headers = get_headers()

    members = set()
    page = 1
    while True:
        resp = requests.get(f"{url}?per_page=100&page={page}", headers=headers)
        if resp.status_code == 404:
            print(f"Warning: Team '{team_slug}' not found in org '{org}'")
            return set()
        if resp.status_code != 200:
            print(f"Error fetching team members: {resp.status_code} {resp.text}")
            return set()

        data = resp.json()
        if not data:
            break

        members.update([m["login"] for m in data])
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
                        print(f"Found @nvidia.com email for {username} from commits")
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
        print(f"Warning: Could not find Slack user for {email}: {e.response['error']}")
        _slack_id_cache[email] = None
        return None


def fetch_all_usergroups(slack_client):
    """Fetch all Slack usergroups once and cache them."""
    global _usergroups_cache

    if _usergroups_cache is not None:
        return _usergroups_cache

    if not slack_client:
        _usergroups_cache = {}
        return _usergroups_cache

    try:
        print("Fetching Slack usergroups...")
        response = slack_client.usergroups_list(include_users=True)
        _usergroups_cache = {}
        for usergroup in response.get("usergroups", []):
            handle = usergroup.get("handle")
            if handle:
                _usergroups_cache[handle] = {
                    "id": usergroup.get("id"),
                    "users": usergroup.get("users", []),
                }
        print(f"Fetched {len(_usergroups_cache)} usergroups")
        return _usergroups_cache
    except SlackApiError as e:
        print(f"Warning: Could not list Slack usergroups: {e.response['error']}")
        _usergroups_cache = {}
        return _usergroups_cache


def get_slack_usergroup_id(slack_client, handle):
    """Get Slack usergroup ID from handle."""
    usergroups = fetch_all_usergroups(slack_client)

    if handle in usergroups:
        return usergroups[handle]["id"], usergroups[handle]["users"]

    return None, []


def github_team_to_usergroup_name(team_slug):
    """Convert a GitHub team slug to a Slack usergroup display name.

    Example: "test3" -> "Megatron Core Experts: Test3"
    """
    # Title case each word separated by hyphens, then join with spaces
    words = team_slug.split("-")
    title_cased = " ".join(word.capitalize() for word in words)
    return f"Megatron Core Experts: {title_cased}"


def create_slack_usergroup(slack_client, handle, team_slug):
    """Create a new Slack usergroup.

    Args:
        slack_client: Slack WebClient instance
        handle: The usergroup handle (e.g., "mcore-test")
        team_slug: The GitHub team slug (used for name and description)

    Returns:
        The usergroup ID if created successfully, None otherwise
    """
    global _usergroups_cache

    name = github_team_to_usergroup_name(team_slug)
    description = f'Expert review group "{team_slug}"'

    try:
        print(f"Creating Slack usergroup '@{handle}' with name '{name}'...")
        response = slack_client.usergroups_create(
            name=name,
            handle=handle,
            description=description,
        )
        usergroup = response.get("usergroup", {})
        usergroup_id = usergroup.get("id")

        if usergroup_id:
            # Update cache with new usergroup
            if _usergroups_cache is not None:
                _usergroups_cache[handle] = {
                    "id": usergroup_id,
                    "users": [],
                }
            print(f"Successfully created Slack usergroup '@{handle}'")
            return usergroup_id
        else:
            print(f"Error: Usergroup created but no ID returned")
            return None

    except SlackApiError as e:
        print(f"Error creating Slack usergroup '@{handle}': {e.response['error']}")
        return None


def sync_team_to_usergroup(team_slug, usergroup_handle, dry_run=False):
    """Sync a GitHub team to a Slack usergroup."""
    print(f"\n{'='*60}")
    print(f"Syncing GitHub team '{team_slug}' -> Slack usergroup '@{usergroup_handle}'")
    print(f"{'='*60}")

    org = get_org()
    slack_client = get_slack_client()

    if not slack_client:
        print("Error: Slack token not configured")
        return False

    # 1. Get GitHub team members
    members = get_team_members(org, team_slug)
    if not members:
        print(f"No members found in GitHub team '{team_slug}'")
        return False

    # Filter out service accounts
    members = {m for m in members if not m.startswith("svc")}
    print(f"GitHub team members ({len(members)}): {sorted(members)}")

    # 2. Get Slack user IDs for each member
    slack_user_ids = []
    missing_users = []

    for username in sorted(members):
        email = get_user_email(username)
        slack_id = get_slack_user_id(slack_client, email)
        if slack_id:
            slack_user_ids.append(slack_id)
        else:
            missing_users.append((username, email, "not found in Slack"))

    if missing_users:
        print(f"\nWarning: Could not resolve {len(missing_users)} users:")
        for username, email, reason in missing_users:
            print(f"  - {username}: {reason}" + (f" (tried {email})" if email else ""))

    if not slack_user_ids:
        print(f"Error: No Slack users found for team '{team_slug}'")
        return False

    # 3. Get current Slack usergroup membership (or create if it doesn't exist)
    usergroup_id, current_members = get_slack_usergroup_id(slack_client, usergroup_handle)

    if not usergroup_id:
        print(f"Slack usergroup '@{usergroup_handle}' not found, creating it...")
        if dry_run:
            print(f"Dry run: Would create usergroup '@{usergroup_handle}'")
            current_members = []
        else:
            usergroup_id = create_slack_usergroup(slack_client, usergroup_handle, team_slug)
            if not usergroup_id:
                print(f"Error: Failed to create Slack usergroup '@{usergroup_handle}'")
                return False
            current_members = []

    # 4. Compare and update
    current_set = set(current_members)
    new_set = set(slack_user_ids)

    to_add = new_set - current_set
    to_remove = current_set - new_set

    print(f"\nCurrent usergroup members: {len(current_members)}")
    print(f"New members to set: {len(slack_user_ids)}")
    print(f"  Adding: {len(to_add)} users")
    print(f"  Removing: {len(to_remove)} users")

    if current_set == new_set:
        print("No changes needed - usergroup is already in sync")
        return True

    if dry_run:
        print(f"\nDry run: Would update '@{usergroup_handle}' with {len(slack_user_ids)} members")
        return True

    # 5. Update the usergroup
    try:
        slack_client.usergroups_users_update(
            usergroup=usergroup_id, users=slack_user_ids
        )
        print(f"\nSuccessfully updated '@{usergroup_handle}' with {len(slack_user_ids)} members")
        return True
    except SlackApiError as e:
        print(f"Error updating usergroup: {e.response['error']}")
        return False


def get_team_to_usergroup_mapping():
    """Fetch child teams of mcore-reviewers and generate the mapping."""
    org = get_org()
    child_teams = get_child_teams(org, PARENT_TEAM_SLUG)

    if not child_teams:
        print(f"Error: No child teams found under '{PARENT_TEAM_SLUG}'")
        return {}

    mapping = {}
    for team_slug in child_teams:
        usergroup_handle = github_team_to_slack_usergroup(team_slug)
        mapping[team_slug] = usergroup_handle

    return mapping


def sync_all_teams(dry_run=False):
    """Sync all GitHub teams under mcore-reviewers to their Slack usergroups."""
    print(f"Fetching child teams of '{PARENT_TEAM_SLUG}'...")
    team_to_usergroup = get_team_to_usergroup_mapping()

    if not team_to_usergroup:
        return False

    print(f"Found {len(team_to_usergroup)} teams to sync")
    print("\nTeam to usergroup mapping:")
    for team, usergroup in sorted(team_to_usergroup.items()):
        print(f"  {team} -> @{usergroup}")

    results = {"success": [], "failed": []}

    for team_slug, usergroup_handle in team_to_usergroup.items():
        success = sync_team_to_usergroup(team_slug, usergroup_handle, dry_run=dry_run)
        if success:
            results["success"].append(team_slug)
        else:
            results["failed"].append(team_slug)

    # Summary
    print(f"\n{'='*60}")
    print("SYNC SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["failed"]:
        print(f"\nFailed teams: {', '.join(results['failed'])}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync GitHub team membership to Slack user groups"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all configured team-to-usergroup mappings",
    )

    args = parser.parse_args()

    if args.list:
        print(f"Fetching child teams of '{PARENT_TEAM_SLUG}'...")
        team_to_usergroup = get_team_to_usergroup_mapping()
        if not team_to_usergroup:
            sys.exit(1)
        print("\nTeam-to-usergroup mappings:")
        print(f"{'GitHub Team':<35} {'Slack Usergroup':<30}")
        print("-" * 65)
        for team, usergroup in sorted(team_to_usergroup.items()):
            print(f"{team:<35} @{usergroup:<29}")
        return

    success = sync_all_teams(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
