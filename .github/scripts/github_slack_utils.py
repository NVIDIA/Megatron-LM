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

"""Shared GitHub-to-Slack user lookup helpers for repository automation."""

import os
import re
import sys

try:
    import requests
except ImportError:  # pragma: no cover - workflow environments install requests.
    requests = None

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:  # pragma: no cover - workflow environments install slack-sdk.
    WebClient = None
    SlackApiError = Exception


GITHUB_API_URL = "https://api.github.com"

_email_cache = {}
_slack_id_cache = {}


def get_headers() -> dict[str, str]:
    """Return GitHub API headers from the configured workflow token."""

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GH_TOKEN or GITHUB_TOKEN not set")
        sys.exit(1)

    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}


def get_user_email(username: str) -> str:
    """Resolve a GitHub username to an email, preferring @nvidia.com addresses."""

    if username in _email_cache:
        return _email_cache[username]

    if requests is None:
        print("Error: requests is not installed")
        sys.exit(1)

    headers = get_headers()
    public_email = None

    try:
        response = requests.get(f"{GITHUB_API_URL}/users/{username}", headers=headers, timeout=30)
        if response.status_code == 200:
            user_data = response.json()
            email = user_data.get("email")
            if email and not email.endswith("@users.noreply.github.com"):
                if email.endswith("@nvidia.com"):
                    _email_cache[username] = email
                    return email
                public_email = email

        repo_env = os.environ.get("GITHUB_REPOSITORY", "NVIDIA/Megatron-LM")
        commits_url = f"{GITHUB_API_URL}/repos/{repo_env}/commits?author={username}&per_page=10"
        response = requests.get(commits_url, headers=headers, timeout=30)
        if response.status_code == 200:
            for commit in response.json():
                commit_data = commit.get("commit", {})
                author_data = commit_data.get("author", {})
                email = author_data.get("email")

                if email and not email.endswith("@users.noreply.github.com"):
                    if email.endswith("@nvidia.com"):
                        _email_cache[username] = email
                        print(f"Found @nvidia.com email for {username} from commits")
                        return email
                    if public_email is None:
                        public_email = email

                signoff_matches = re.findall(
                    r"Signed-off-by:.*<([^>]+@nvidia\.com)>", commit_data.get("message", "")
                )
                if signoff_matches:
                    _email_cache[username] = signoff_matches[0]
                    print(f"Found @nvidia.com email for {username} from Signed-off-by")
                    return signoff_matches[0]

        if public_email:
            _email_cache[username] = public_email
            print(f"Using public email for {username}: {public_email}")
            return public_email

    except Exception as exc:
        print(f"Warning: Could not get email for {username}: {exc}")

    fallback = f"{username}@users.noreply.github.com"
    _email_cache[username] = fallback
    print(f"Warning: No email found for {username}, using fallback: {fallback}")
    return fallback


def get_slack_client(require_slack: bool = False):
    """Return a Slack WebClient, or None when Slack is optional and not configured."""

    slack_token = os.environ.get("SLACK_TOKEN")
    if not slack_token:
        if require_slack:
            print("Error: SLACK_TOKEN is required")
            sys.exit(1)
        return None

    if WebClient is None:
        print("Error: slack-sdk is not installed")
        sys.exit(1)

    return WebClient(token=slack_token)


def get_slack_user_id(slack_client, email: str) -> str | None:
    """Resolve an email address to a Slack user ID."""

    if not slack_client:
        return None

    if email in _slack_id_cache:
        return _slack_id_cache[email]

    try:
        response = slack_client.users_lookupByEmail(email=email)
        user_id = response["user"]["id"]
        _slack_id_cache[email] = user_id
        return user_id
    except SlackApiError as exc:
        print(f"Warning: Could not find Slack user for {email}: {exc.response['error']}")
        _slack_id_cache[email] = None
        return None
