# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared policy, limits, and GitHub transport for isolated reviews."""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import PurePosixPath
from typing import Any, Iterable

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
COMMAND_RE = re.compile(r"^/claude (review|strict-review)(?: ([0-9a-fA-F]{40}))?$")
SAFE_PERMISSIONS = frozenset({"admin", "maintain", "write", "triage"})
REPORT_VERSION = 1
MAX_CHANGED_FILES = 2_000
MAX_DIFF_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_FILE_READ = 256 * 1024
MAX_SEARCH_BYTES = 32 * 1024 * 1024
MAX_INLINE_FINDINGS = 30
MAX_GENERAL_FINDINGS = 10
MAX_VALID_LINES = 20_000
FAILURE_REASONS = frozenset(
    {"none", "context_incomplete", "budget_exhausted", "timeout", "analysis_failed", "invalid_output"}
)
SEVERITIES = frozenset({"critical", "important", "suggestion"})
SIDES = frozenset({"LEFT", "RIGHT"})
TRUSTED_CODE_PATHS = (
    ".github/scripts/claude_review.py",
    ".github/scripts/claude_review/__init__.py",
    ".github/scripts/claude_review/common.py",
    ".github/scripts/claude_review/prepare.py",
    ".github/scripts/claude_review/context.py",
    ".github/scripts/claude_review/publisher.py",
    ".github/scripts/claude_review/cli.py",
)


class ReviewError(RuntimeError):
    """A deterministic, fail-closed review error."""


def _full_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ReviewError(f"{name} must be a lowercase 40-character SHA")
    return value


def _safe_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > 4096:
        raise ReviewError("path is missing or too long")
    if value.startswith("./") or "/./" in value:
        raise ReviewError("path must be normalized")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReviewError("path must be a normalized repository-relative path")
    if any(ord(char) < 32 for char in value):
        raise ReviewError("path contains a control character")
    return path.as_posix()


def _normalise_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(char for char in value if char in "\n\t" or ord(char) >= 32)
    return value[:limit]


def parse_trigger(body: str) -> tuple[str, str | None] | None:
    """Parse exactly one normalized first-line review command.

    NFKC normalization and surrounding whitespace removal are the only
    normalization.  Matching remains case-sensitive; quoting, command suffixes,
    multiple spaces, abbreviated SHAs, and near matches are rejected.
    """

    if not isinstance(body, str):
        return None
    first_line = body.splitlines()[0] if body.splitlines() else body
    first_line = unicodedata.normalize("NFKC", first_line).strip(" \t\r")
    match = COMMAND_RE.fullmatch(first_line)
    if not match:
        return None
    mode = "light" if match.group(1) == "review" else "strict"
    return mode, match.group(2).lower() if match.group(2) else None


def actor_authorized(actor: dict[str, Any], permission: str, bot_allowlist: Iterable[str] = ()) -> bool:
    """Apply the explicit repository reviewer policy."""

    login = actor.get("login")
    actor_type = actor.get("type")
    if not isinstance(login, str) or not login:
        return False
    allowed_bots = {name.strip() for name in bot_allowlist if name.strip()}
    if actor_type == "Bot" or login.endswith("[bot]"):
        return login in allowed_bots
    return actor_type == "User" and permission in SAFE_PERMISSIONS


class GitHubAPI:
    """Small JSON-only GitHub client; callers choose every allowed endpoint."""

    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        if not token:
            raise ReviewError("GitHub token is required")
        self.token = token
        self.api_url = api_url.rstrip("/")

    def request(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "megatron-claude-review/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.api_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode(errors="replace")
            raise ReviewError(f"GitHub {method} {path} failed: {error.code}: {detail}") from error
        if len(raw) > 4 * 1024 * 1024:
            raise ReviewError("GitHub response exceeded 4 MiB")
        return json.loads(raw) if raw else {}
