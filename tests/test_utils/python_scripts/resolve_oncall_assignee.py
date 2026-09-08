# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve the scheduled Megatron-LM on-call user to an NVIDIA email address."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

_GITHUB_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / ".github" / "scripts"
sys.path.insert(0, str(_GITHUB_SCRIPTS_DIR))

from github_slack_utils import get_user_email  # noqa: E402


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


def resolve_nvidia_email(github_login: str) -> str:
    """Resolve a GitHub login with the repository's shared email lookup helper."""

    # The shared helper reports lookup details on stdout. Suppress those messages
    # so command substitution receives only the email printed by ``main`` and CI
    # logs do not repeat email addresses from fallback diagnostics.
    try:
        with redirect_stdout(StringIO()):
            email = get_user_email(github_login)
    except SystemExit as exc:
        raise AssigneeResolutionError(
            f"could not look up an email for GitHub user {github_login!r}"
        ) from exc

    if not isinstance(email, str):
        raise AssigneeResolutionError(
            f"GitHub user {github_login!r} did not resolve to an NVIDIA email"
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
            f"GitHub user {github_login!r} did not resolve to a valid NVIDIA email"
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print only the current on-call user's NVIDIA email to stdout."""

    args = parse_args(argv)
    try:
        username = current_oncall(load_schedule(args.schedule_file))
        email = resolve_nvidia_email(username)
    except AssigneeResolutionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Resolved scheduled on-call GitHub user {username!r}.", file=sys.stderr)
    print(email)
    return 0


if __name__ == "__main__":
    sys.exit(main())
