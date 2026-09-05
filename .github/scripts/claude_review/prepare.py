# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Authenticate commands and capture immutable pull-request revisions."""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

from .common import (  # isort: skip
    GitHubAPI,
    ReviewError,
    _full_sha,
    _normalise_text,
    actor_authorized,
    parse_trigger,
)


def _write_outputs(values: dict[str, Any]) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a", encoding="utf-8") as stream:
        for key, value in values.items():
            text = str(value).lower() if isinstance(value, bool) else str(value)
            if "\n" in text:
                raise ReviewError(f"multiline GitHub output is not allowed: {key}")
            stream.write(f"{key}={text}\n")


def prepare_event(event_path: Path, output_path: Path, acknowledge: bool) -> dict[str, Any] | None:
    event = json.loads(event_path.read_text(encoding="utf-8"))
    issue = event.get("issue") or {}
    parsed = parse_trigger((event.get("comment") or {}).get("body", ""))
    if not issue.get("pull_request") or parsed is None:
        _write_outputs({"triggered": False})
        return None

    mode, requested_sha = parsed
    repository = (event.get("repository") or {}).get("full_name")
    pr_number = issue.get("number")
    actor = (event.get("comment") or {}).get("user") or {}
    if not isinstance(repository, str) or not isinstance(pr_number, int):
        raise ReviewError("invalid issue_comment event")

    api = GitHubAPI(
        os.environ.get("GITHUB_TOKEN", ""),
        os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    bot_allowlist = os.environ.get("CLAUDE_REVIEW_BOT_ALLOWLIST", "").split(",")
    login = str(actor.get("login", ""))
    is_bot = actor.get("type") == "Bot" or login.endswith("[bot]")
    if is_bot:
        permission = (
            "bot-allowlist" if login in {name.strip() for name in bot_allowlist} else "none"
        )
    else:
        quoted_login = urllib.parse.quote(login, safe="")
        permission_data = api.request(
            "GET", f"/repos/{repository}/collaborators/{quoted_login}/permission"
        )
        permission = str(permission_data.get("permission", "none"))
    if not actor_authorized(actor, permission, bot_allowlist):
        _write_outputs({"triggered": True, "authorized": False, "mode": mode})
        return None

    pr = api.request("GET", f"/repos/{repository}/pulls/{pr_number}")
    base = pr.get("base") or {}
    head = pr.get("head") or {}
    base_repo = (base.get("repo") or {}).get("full_name")
    head_repo = (head.get("repo") or {}).get("full_name")
    base_ref = base.get("ref")
    head_ref = head.get("ref")
    head_sha = _full_sha(head.get("sha"), "HEAD_SHA")
    if not all(
        isinstance(item, str) and item for item in (base_repo, head_repo, base_ref, head_ref)
    ):
        raise ReviewError("pull request repository references are incomplete")
    ref_path = urllib.parse.quote(base_ref, safe="")
    base_data = api.request("GET", f"/repos/{base_repo}/git/ref/heads/{ref_path}")
    base_sha = _full_sha(((base_data.get("object") or {}).get("sha")), "BASE_SHA")

    if requested_sha is not None and requested_sha != head_sha:
        if acknowledge:
            api.request(
                "POST",
                f"/repos/{repository}/issues/{pr_number}/comments",
                {
                    "body": "Claude review was not started: the requested revision does not match the current PR head. Re-run the command with the current full SHA."
                },
            )
        _write_outputs(
            {"triggered": True, "authorized": True, "revision_match": False, "mode": mode}
        )
        return None

    metadata = {
        "version": 1,
        "repository": repository,
        "pr_number": pr_number,
        "mode": mode,
        "trigger": {
            "comment_id": (event.get("comment") or {}).get("id"),
            "actor": _normalise_text(actor.get("login"), 128),
            "permission": permission,
            "requested_sha": requested_sha,
        },
        "base_sha": base_sha,
        "head_sha": head_sha,
        "base_repository": base_repo,
        "head_repository": head_repo,
        "base_ref": _normalise_text(base_ref, 255),
        "head_ref": _normalise_text(head_ref, 255),
        "title": _normalise_text(pr.get("title"), 2_000),
        "body": _normalise_text(pr.get("body"), 20_000),
        "author": _normalise_text(((pr.get("user") or {}).get("login")), 128),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if acknowledge:
        api.request(
            "POST",
            f"/repos/{repository}/issues/comments/{metadata['trigger']['comment_id']}/reactions",
            {"content": "eyes"},
        )
        api.request(
            "POST",
            f"/repos/{repository}/issues/{pr_number}/comments",
            {
                "body": f"Started isolated Claude {mode} review for `{head_sha}`. A later push makes this result stale and requires a new review."
            },
        )
    _write_outputs(
        {
            "triggered": True,
            "authorized": True,
            "revision_match": True,
            "mode": mode,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "head_repository": head_repo,
            "pr_number": pr_number,
        }
    )
    return metadata
