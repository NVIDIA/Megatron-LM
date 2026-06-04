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

"""Assign community-request issues from Claude analysis and notify owners in Slack."""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

try:
    import requests
except ImportError:  # pragma: no cover - workflow installs requests.
    requests = None

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:  # pragma: no cover - workflow installs slack-sdk.
    WebClient = None
    SlackApiError = Exception


GITHUB_API_URL = "https://api.github.com"
ACTIVE_ONCALL_TEAM_SLUG = "mcore-oncall"
MCORE_ONCALL_SLACK_USERGROUP_ID = "S0A7B4U1T3P"
CONFIDENCE_THRESHOLD = 0.75
SERVICE_ACCOUNT_LOGINS = {"svcnvidia-nemo-ci"}

_email_cache = {}
_slack_id_cache = {}


@dataclass(frozen=True)
class IssueContext:
    """Minimal issue metadata needed for assignment and notification."""

    owner: str
    repo: str
    number: int
    title: str
    url: str
    author: str


@dataclass(frozen=True)
class AssignmentPlan:
    """Validated assignment decision."""

    mode: str
    assignees: list[str]
    notify_users: list[str]
    confidence: float
    rationale: str
    relevant_paths: list[str]


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        print(f"Error: {name} is required")
        sys.exit(1)
    return value


def get_headers() -> dict[str, str]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GH_TOKEN or GITHUB_TOKEN not set")
        sys.exit(1)

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_repo_info() -> tuple[str, str]:
    repo_env = get_required_env("GITHUB_REPOSITORY")
    owner, repo = repo_env.split("/", maxsplit=1)
    return owner, repo


def get_issue_context() -> IssueContext:
    owner, repo = get_repo_info()
    return IssueContext(
        owner=owner,
        repo=repo,
        number=int(get_required_env("ISSUE_NUMBER")),
        title=get_required_env("ISSUE_TITLE"),
        url=get_required_env("ISSUE_URL"),
        author=get_required_env("ISSUE_AUTHOR"),
    )


def request_json(method: str, url: str, **kwargs):
    if requests is None:
        print("Error: requests is not installed")
        sys.exit(1)

    response = requests.request(method, url, headers=get_headers(), timeout=30, **kwargs)
    if response.status_code >= 400:
        print(f"GitHub API request failed: {method} {url}: {response.status_code} {response.text}")
        sys.exit(1)

    if response.status_code == 204 or not response.text:
        return None

    return response.json()


def parse_analysis(raw_analysis: str) -> dict:
    try:
        analysis = json.loads(raw_analysis)
    except json.JSONDecodeError as exc:
        print(f"Error: Claude analysis was not valid JSON: {exc}")
        sys.exit(1)

    if not isinstance(analysis, dict):
        print("Error: Claude analysis must be a JSON object")
        sys.exit(1)

    return analysis


def normalize_login(login: str | None) -> str | None:
    if not login:
        return None

    normalized = login.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:]
    if "/" in normalized:
        return None
    return normalized or None


def is_service_account(login: str) -> bool:
    return login in SERVICE_ACCOUNT_LOGINS or login.startswith("svc")


def human_members(members: set[str] | list[str]) -> list[str]:
    return sorted(member for member in members if not is_service_account(member))


def analysis_confidence(analysis: dict) -> float:
    try:
        confidence = float(analysis.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return max(0.0, min(confidence, 1.0))


def analysis_relevant_paths(analysis: dict) -> list[str]:
    paths = analysis.get("relevant_paths", [])
    if not isinstance(paths, list):
        return []
    return [path for path in paths if isinstance(path, str)][:5]


def analysis_rationale(analysis: dict) -> str:
    rationale = analysis.get("rationale", "")
    if not isinstance(rationale, str) or not rationale.strip():
        return "Claude did not provide a rationale."
    return rationale.strip()


def should_use_candidate(analysis: dict, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    candidate = normalize_login(analysis.get("assignee"))
    if not candidate or is_service_account(candidate):
        return False
    if bool(analysis.get("fallback_to_oncall", False)):
        return False
    return analysis_confidence(analysis) >= confidence_threshold


def check_assignable(issue: IssueContext, login: str) -> bool:
    url = f"{GITHUB_API_URL}/repos/{issue.owner}/{issue.repo}/assignees/{login}"
    if requests is None:
        print("Error: requests is not installed")
        sys.exit(1)

    response = requests.get(url, headers=get_headers(), timeout=30)
    if response.status_code == 204:
        return True
    if response.status_code == 404:
        return False

    print(f"GitHub API request failed: GET {url}: {response.status_code} {response.text}")
    sys.exit(1)


def get_team_members(org: str, team_slug: str) -> set[str]:
    members = set()
    page = 1

    while True:
        url = f"{GITHUB_API_URL}/orgs/{org}/teams/{team_slug}/members?per_page=100&page={page}"
        data = request_json("GET", url)
        if not data:
            break

        members.update(member["login"] for member in data)
        if len(data) < 100:
            break
        page += 1

    return members


def assign_issue(issue: IssueContext, assignees: list[str], dry_run: bool = False) -> None:
    if not assignees:
        print("No assignable users found; skipping issue assignment")
        return

    print(f"Assigning issue #{issue.number} to: {', '.join(assignees)}")
    if dry_run:
        return

    url = f"{GITHUB_API_URL}/repos/{issue.owner}/{issue.repo}/issues/{issue.number}/assignees"
    request_json("POST", url, json={"assignees": assignees[:10]})


def create_assignment_plan(analysis: dict, issue: IssueContext) -> AssignmentPlan:
    confidence = analysis_confidence(analysis)
    candidate = normalize_login(analysis.get("assignee"))
    rationale = analysis_rationale(analysis)
    relevant_paths = analysis_relevant_paths(analysis)

    if should_use_candidate(analysis) and candidate and check_assignable(issue, candidate):
        return AssignmentPlan(
            mode="candidate",
            assignees=[candidate],
            notify_users=[candidate],
            confidence=confidence,
            rationale=rationale,
            relevant_paths=relevant_paths,
        )

    if candidate:
        print(f"Falling back to {ACTIVE_ONCALL_TEAM_SLUG}; candidate was {candidate} with confidence {confidence:.2f}")
    else:
        print(f"Falling back to {ACTIVE_ONCALL_TEAM_SLUG}; Claude did not provide a usable candidate")

    oncall_members = human_members(get_team_members(issue.owner, ACTIVE_ONCALL_TEAM_SLUG))
    assignable_oncall = [member for member in oncall_members if check_assignable(issue, member)]

    return AssignmentPlan(
        mode="oncall",
        assignees=assignable_oncall,
        notify_users=oncall_members,
        confidence=confidence,
        rationale=rationale,
        relevant_paths=relevant_paths,
    )


def get_user_email(username: str) -> str:
    """Get user's email from GitHub, preferring @nvidia.com addresses."""

    if username in _email_cache:
        return _email_cache[username]

    public_email = None

    try:
        user_data = request_json("GET", f"{GITHUB_API_URL}/users/{username}")
        email = user_data.get("email") if user_data else None
        if email and not email.endswith("@users.noreply.github.com"):
            if email.endswith("@nvidia.com"):
                _email_cache[username] = email
                return email
            public_email = email

        repo_env = os.environ.get("GITHUB_REPOSITORY", "NVIDIA/Megatron-LM")
        commits = request_json(
            "GET", f"{GITHUB_API_URL}/repos/{repo_env}/commits?author={username}&per_page=10"
        )
        for commit in commits or []:
            commit_data = commit.get("commit", {})
            author_data = commit_data.get("author", {})
            email = author_data.get("email")
            if email and not email.endswith("@users.noreply.github.com"):
                if email.endswith("@nvidia.com"):
                    _email_cache[username] = email
                    return email
                if public_email is None:
                    public_email = email

            signoff_matches = re.findall(
                r"Signed-off-by:.*<([^>]+@nvidia\.com)>", commit_data.get("message", "")
            )
            if signoff_matches:
                _email_cache[username] = signoff_matches[0]
                return signoff_matches[0]

    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive API fallback.
        print(f"Warning: Could not get email for {username}: {exc}")

    fallback = public_email or f"{username}@users.noreply.github.com"
    _email_cache[username] = fallback
    return fallback


def get_slack_client(require_slack: bool):
    slack_token = os.environ.get("SLACK_TOKEN")
    if not slack_token:
        if require_slack:
            print("Error: SLACK_TOKEN is required to notify the assignee")
            sys.exit(1)
        print("Slack token not configured, skipping Slack notification")
        return None

    if WebClient is None:
        print("Error: slack-sdk is not installed")
        sys.exit(1)

    return WebClient(token=slack_token)


def get_slack_user_id(slack_client, email: str) -> str | None:
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


def build_slack_message(issue: IssueContext, plan: AssignmentPlan) -> str:
    paths = ", ".join(plan.relevant_paths) if plan.relevant_paths else "none identified"
    oncall_mention = f"<!subteam^{MCORE_ONCALL_SLACK_USERGROUP_ID}|mcore-oncall>"
    if plan.mode == "candidate":
        return (
            f"I (Megatron Issue Bot) have assigned you to the newly created community issue: <{issue.url}|{issue.url}>.\n\n"
            "I determined that you are the best individual to answer this community issue. "
            "Please take action at your earliest convenience, at latest within 1 business day. "
            "If I made a mistake or if you are unsure how to proceed, please reach out to "
            f"{oncall_mention} directly."
        )

    return (
        f"Community request <{issue.url}|#{issue.number}: {issue.title}> needs on-call triage.\n"
        f"I could not confidently identify a direct assignee; confidence: {plan.confidence:.2f}\n"
        f"Relevant paths: {paths}\n"
        f"Rationale: {plan.rationale}"
    )


def send_slack_notifications(issue: IssueContext, plan: AssignmentPlan, dry_run: bool, require_slack: bool) -> None:
    if not plan.notify_users:
        print("No users to notify in Slack")
        if require_slack:
            sys.exit(1)
        return

    slack_client = get_slack_client(require_slack=require_slack)
    if not slack_client:
        return

    message = build_slack_message(issue, plan)
    missing_users = []

    for username in plan.notify_users:
        email = get_user_email(username)
        slack_user_id = get_slack_user_id(slack_client, email)
        if not slack_user_id:
            missing_users.append(f"{username} ({email})")
            continue

        print(f"Sending Slack notification to {username}")
        if dry_run:
            continue

        conversation = slack_client.conversations_open(users=slack_user_id)
        channel_id = conversation["channel"]["id"]
        slack_client.chat_postMessage(channel=channel_id, text=message, unfurl_links=False, unfurl_media=False)

    if missing_users:
        print("Could not send Slack notifications to: " + ", ".join(missing_users))
        if require_slack:
            sys.exit(1)


def run(dry_run: bool = False, require_slack: bool = True) -> AssignmentPlan:
    issue = get_issue_context()
    analysis = parse_analysis(get_required_env("ANALYSIS_JSON"))
    plan = create_assignment_plan(analysis, issue)

    assign_issue(issue, plan.assignees, dry_run=dry_run)
    send_slack_notifications(issue, plan, dry_run=dry_run, require_slack=require_slack)

    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign and notify owners for community-request issues")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing to GitHub or Slack")
    parser.add_argument(
        "--allow-missing-slack",
        action="store_true",
        help="Do not fail when Slack cannot be notified",
    )
    args = parser.parse_args()

    run(dry_run=args.dry_run, require_slack=not args.allow_missing_slack)


if __name__ == "__main__":
    main()
