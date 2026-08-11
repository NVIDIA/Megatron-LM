# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve the scheduled Megatron-LM on-call user to an NVIDIA email address."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

GITHUB_API_URL = "https://api.github.com"
GITHUB_AUDITS_REPOSITORY = "NVIDIA-GitHub-Management/github-audits"
GITHUB_AUDITS_RELEASE = "v0.1.0"
SSO_USERS_ASSET = "users_sso.json"
GITHUB_TOKEN_ENV = "NVIDIA_MANAGEMENT_ORG_PAT"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_ATTEMPTS = 3
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


class AssigneeResolutionError(RuntimeError):
    """Raised when the current on-call Linear assignee cannot be resolved safely."""


def load_schedule(path: Path) -> list[dict[str, object]]:
    """Load and validate the top-level on-call schedule structure."""

    try:
        schedule = json.loads(path.read_text())
    except OSError as exc:
        raise AssigneeResolutionError(f"could not read on-call schedule {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AssigneeResolutionError(f"on-call schedule {path} is not valid JSON") from exc

    if not isinstance(schedule, list) or not schedule:
        raise AssigneeResolutionError(f"on-call schedule {path} must be a non-empty list")
    return schedule


def current_oncall(schedule: list[dict[str, object]]) -> str:
    """Return the current on-call user, matching the existing rotation manager."""

    if not schedule:
        raise AssigneeResolutionError("on-call schedule must be a non-empty list")
    entry = schedule[0]
    if not isinstance(entry, dict):
        raise AssigneeResolutionError("the current on-call schedule entry must be an object")

    username = entry.get("user")
    if not isinstance(username, str) or not username.strip():
        raise AssigneeResolutionError("the current on-call schedule entry must have a user")
    return username.strip()


def _get_json(
    http_get: Callable[..., requests.Response], url: str, description: str, **kwargs: object
) -> object:
    """Fetch JSON with bounded retries and convert failures to a safe diagnostic."""

    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = http_get(url, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            retryable = status is None or status in RETRYABLE_HTTP_STATUSES
            if retryable and attempt + 1 < REQUEST_ATTEMPTS:
                time.sleep(2**attempt)
                continue
            raise AssigneeResolutionError(f"failed to download {description}: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise AssigneeResolutionError(f"{description} is not valid JSON") from exc

    raise AssertionError("request retry loop exited unexpectedly")


def download_sso_users(
    token: str, http_get: Callable[..., requests.Response] = requests.get
) -> dict[str, object]:
    """Download the NVIDIA SSO map from the private github-audits release."""

    token = token.strip()
    if not token or any(character.isspace() for character in token):
        raise AssigneeResolutionError(f"{GITHUB_TOKEN_ENV} is missing or invalid")

    release_url = (
        f"{GITHUB_API_URL}/repos/{GITHUB_AUDITS_REPOSITORY}/releases/tags/"
        f"{GITHUB_AUDITS_RELEASE}"
    )
    api_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    release = _get_json(
        http_get,
        release_url,
        f"{GITHUB_AUDITS_REPOSITORY} release {GITHUB_AUDITS_RELEASE}",
        headers=api_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not isinstance(release, dict) or not isinstance(release.get("assets"), list):
        raise AssigneeResolutionError("github-audits release metadata has no assets list")

    matching_assets = [
        asset
        for asset in release["assets"]
        if isinstance(asset, dict) and asset.get("name") == SSO_USERS_ASSET
    ]
    if len(matching_assets) != 1:
        raise AssigneeResolutionError(
            f"expected one {SSO_USERS_ASSET} asset, found {len(matching_assets)}"
        )

    asset_url = matching_assets[0].get("url")
    if not isinstance(asset_url, str):
        raise AssigneeResolutionError("github-audits returned an invalid release asset URL")
    parsed_asset_url = urlparse(asset_url)
    if parsed_asset_url.scheme != "https" or parsed_asset_url.netloc != "api.github.com":
        raise AssigneeResolutionError("github-audits returned an invalid release asset URL")

    asset_headers = {**api_headers, "Accept": "application/octet-stream"}
    # requests removes Authorization when GitHub redirects the asset download to
    # a different host, so the private-repo token is not forwarded to object storage.
    users = _get_json(
        http_get,
        asset_url,
        SSO_USERS_ASSET,
        headers=asset_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    if not isinstance(users, dict):
        raise AssigneeResolutionError(f"{SSO_USERS_ASSET} must contain a JSON object")
    return users


def resolve_nvidia_email(users: dict[str, object], github_login: str) -> str:
    """Resolve a GitHub login to its verified ``nvidia_email`` SSO field."""

    matching_logins = [login for login in users if login.casefold() == github_login.casefold()]
    if len(matching_logins) != 1:
        raise AssigneeResolutionError(
            f"expected one SSO record for GitHub user {github_login!r}, found {len(matching_logins)}"
        )

    record = users[matching_logins[0]]
    if not isinstance(record, dict):
        raise AssigneeResolutionError(f"SSO record for GitHub user {github_login!r} is invalid")

    email = record.get("nvidia_email")
    if not isinstance(email, str):
        raise AssigneeResolutionError(
            f"SSO record for GitHub user {github_login!r} has no nvidia_email"
        )
    email = email.strip()
    local_part, separator, domain = email.rpartition("@")
    if (
        separator != "@"
        or not local_part
        or "@" in local_part
        or domain.casefold() != "nvidia.com"
        or any(character.isspace() for character in email)
    ):
        raise AssigneeResolutionError(
            f"SSO record for GitHub user {github_login!r} has an invalid NVIDIA email"
        )
    return email


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Resolve the scheduled Megatron-LM on-call user's NVIDIA email"
    )
    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=Path(".github/oncall_schedule.json"),
        help="path to the dated GitHub on-call schedule",
    )
    parser.add_argument(
        "--token-env",
        default=GITHUB_TOKEN_ENV,
        help=f"environment variable containing the github-audits token (default: {GITHUB_TOKEN_ENV})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print only the current on-call user's NVIDIA email to stdout."""

    args = parse_args(argv)
    try:
        username = current_oncall(load_schedule(args.schedule_file))
        token = os.environ.get(args.token_env, "")
        email = resolve_nvidia_email(download_sso_users(token), username)
    except AssigneeResolutionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Resolved scheduled on-call GitHub user {username!r}.", file=sys.stderr)
    print(email)
    return 0


if __name__ == "__main__":
    sys.exit(main())
