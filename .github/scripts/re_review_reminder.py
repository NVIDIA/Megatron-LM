#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Send a Slack DM when an MCore engineer re-requests a PR review."""

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from github_slack_utils import get_headers, get_slack_client, get_slack_user_id, get_user_email

try:
    import requests
except ImportError:  # pragma: no cover - the workflow installs requests.
    requests = None


GITHUB_API_URL = "https://api.github.com"
EXPECTED_REPOSITORY = "NVIDIA/Megatron-LM"
EXPECTED_BASE_BRANCH = "main"
MCORE_ENGINEERS_TEAM = "mcore-engineers"
SUBMITTED_REVIEW_STATES = {"COMMENTED", "APPROVED", "CHANGES_REQUESTED", "DISMISSED"}
PER_PAGE = 100
REQUEST_TIMEOUT_SECONDS = 30

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class NotifierError(RuntimeError):
    """Raised when an eligible notification cannot be evaluated or delivered."""


@dataclass(frozen=True)
class ReviewRequest:
    """Validated fields from an individual review-request event."""

    number: int
    requester: str
    reviewer: str
    event_timestamp: datetime

    @property
    def url(self) -> str:
        """Return the canonical pull-request URL."""

        return f"https://github.com/{EXPECTED_REPOSITORY}/pull/{self.number}"


def _parse_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise NotifierError(f"Review-request payload is missing {field_name}")

    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NotifierError(f"Review-request payload has invalid {field_name}") from exc

    if timestamp.tzinfo is None:
        raise NotifierError(f"Review-request payload has timezone-less {field_name}")
    return timestamp


def _extract_review_request(event: dict) -> ReviewRequest | None:
    if event.get("action") != "review_requested":
        logger.info("Skipping event because it is not a review request")
        return None

    repository = event.get("repository") or {}
    if str(repository.get("full_name", "")).casefold() != EXPECTED_REPOSITORY.casefold():
        logger.info("Skipping review request outside NVIDIA/Megatron-LM")
        return None

    pull_request = event.get("pull_request") or {}
    if (pull_request.get("base") or {}).get("ref") != EXPECTED_BASE_BRANCH:
        logger.info("Skipping review request for a pull request not targeting main")
        return None

    requested_reviewer = event.get("requested_reviewer")
    if requested_reviewer is None:
        logger.info("Skipping review request because it targets a team, not an individual")
        return None
    if not isinstance(requested_reviewer, dict):
        raise NotifierError("Review-request payload has an invalid requested_reviewer")

    requester = str((event.get("sender") or {}).get("login", "")).strip()
    reviewer = str(requested_reviewer.get("login", "")).strip()
    if not requester or not reviewer:
        raise NotifierError("Review-request payload is missing the requester or reviewer")

    if requester.casefold() == reviewer.casefold():
        logger.info("Skipping review request because the requester and reviewer are the same user")
        return None

    number = event.get("number", pull_request.get("number"))
    if not isinstance(number, int) or number <= 0:
        raise NotifierError("Review-request payload has an invalid pull-request number")

    return ReviewRequest(
        number=number,
        requester=requester,
        reviewer=reviewer,
        event_timestamp=_parse_timestamp(pull_request.get("updated_at"), "pull_request.updated_at"),
    )


def _request_json(path: str):
    if requests is None:
        raise NotifierError("requests is not installed")

    url = f"{GITHUB_API_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(url, headers=get_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise NotifierError(f"GitHub API request failed: GET {url}") from exc

    if response.status_code != 200:
        raise NotifierError(f"GitHub API request failed: GET {url}: HTTP {response.status_code}")

    try:
        return response.json()
    except ValueError as exc:
        raise NotifierError(f"GitHub API returned invalid JSON: GET {url}") from exc


def _get_paginated_list(path: str) -> list[dict]:
    items = []
    page = 1

    while True:
        separator = "&" if "?" in path else "?"
        page_items = _request_json(f"{path}{separator}per_page={PER_PAGE}&page={page}")
        if not isinstance(page_items, list):
            raise NotifierError(f"GitHub API returned an unexpected response for {path}")

        items.extend(page_items)
        if len(page_items) < PER_PAGE:
            return items
        page += 1


def _get_mcore_engineers() -> set[str]:
    members = _get_paginated_list(f"orgs/NVIDIA/teams/{MCORE_ENGINEERS_TEAM}/members")
    logins = set()
    for member in members:
        login = member.get("login") if isinstance(member, dict) else None
        if not isinstance(login, str) or not login:
            raise NotifierError("GitHub team response is missing a member login")
        logins.add(login.casefold())
    return logins


def _reviewer_is_still_requested(review_request: ReviewRequest) -> bool:
    response = _request_json(f"repos/{EXPECTED_REPOSITORY}/pulls/{review_request.number}")
    if (
        not isinstance(response, dict)
        or response.get("state") not in {"open", "closed"}
        or not isinstance(response.get("requested_reviewers"), list)
    ):
        raise NotifierError("GitHub pull-request response has an unexpected format")

    if response.get("state") != "open":
        return False

    reviewer = review_request.reviewer.casefold()
    return any(
        str(user.get("login", "")).casefold() == reviewer
        for user in response["requested_reviewers"]
    )


def _has_prior_review(review_request: ReviewRequest) -> bool:
    reviews = _get_paginated_list(
        f"repos/{EXPECTED_REPOSITORY}/pulls/{review_request.number}/reviews"
    )
    reviewer = review_request.reviewer.casefold()

    for review in reviews:
        if str((review.get("user") or {}).get("login", "")).casefold() != reviewer:
            continue

        if review.get("state") not in SUBMITTED_REVIEW_STATES:
            continue

        submitted_at = review.get("submitted_at")
        if submitted_at is None:
            continue

        if _parse_timestamp(submitted_at, "review.submitted_at") < review_request.event_timestamp:
            return True

    return False


def build_slack_message(review_request: ReviewRequest) -> str:
    """Build the direct-message text for a qualifying review re-request."""

    return (
        "You have previously reviewed Megatron-LM PR "
        f"<{review_request.url}|#{review_request.number}> "
        "and your review has been re-requested. Please take another look."
    )


def _send_slack_dm(review_request: ReviewRequest) -> None:
    email = get_user_email(review_request.reviewer)
    if not isinstance(email, str) or not email.lower().endswith("@nvidia.com"):
        raise NotifierError(
            f"Could not resolve GitHub user {review_request.reviewer} to an NVIDIA Slack identity"
        )

    slack_client = get_slack_client(require_slack=True)
    if slack_client is None:
        raise NotifierError("Slack client is not configured")
    slack_user_id = get_slack_user_id(slack_client, email)
    if not slack_user_id:
        raise NotifierError(
            f"Could not resolve GitHub user {review_request.reviewer} to a Slack user"
        )

    conversation = slack_client.conversations_open(users=slack_user_id)
    channel_id = (conversation.get("channel") or {}).get("id")
    if not channel_id:
        raise NotifierError(f"Slack did not return a DM channel for {review_request.reviewer}")

    slack_client.chat_postMessage(
        channel=channel_id,
        text=build_slack_message(review_request),
        unfurl_links=False,
        unfurl_media=False,
    )


def process_event(event: dict) -> bool:
    """Evaluate one webhook payload and send a DM when it is a review re-request."""

    review_request = _extract_review_request(event)
    if review_request is None:
        return False

    members = _get_mcore_engineers()
    if review_request.requester.casefold() not in members:
        logger.info(
            f"Skipping review request because {review_request.requester} is not in mcore-engineers"
        )
        return False
    if review_request.reviewer.casefold() not in members:
        logger.info(
            f"Skipping review request because {review_request.reviewer} is not in mcore-engineers"
        )
        return False

    if not _reviewer_is_still_requested(review_request):
        logger.info(
            f"Skipping review request because {review_request.reviewer} is no longer pending"
        )
        return False

    if not _has_prior_review(review_request):
        logger.info(f"Skipping initial review request for {review_request.reviewer}")
        return False

    _send_slack_dm(review_request)
    logger.info(
        f"Sent review re-request reminder to {review_request.reviewer} "
        f"for PR #{review_request.number}"
    )
    return True


def run(event_path: str | Path | None = None) -> bool:
    """Load a GitHub event payload and process its review request."""

    raw_event_path = event_path or os.environ.get("GITHUB_EVENT_PATH")
    if not raw_event_path:
        raise NotifierError("GITHUB_EVENT_PATH is required")
    path = Path(raw_event_path)

    try:
        with path.open(encoding="utf-8") as event_file:
            event = json.load(event_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise NotifierError(f"Could not read GitHub event payload from {path}") from exc

    if not isinstance(event, dict):
        raise NotifierError("GitHub event payload must be a JSON object")
    return process_event(event)


def main() -> None:
    """Run the notifier from GitHub Actions."""

    try:
        run()
    except NotifierError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
