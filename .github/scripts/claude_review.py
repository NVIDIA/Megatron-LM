#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Isolated, read-only review context and deterministic publication.

This module is intentionally self-contained and uses only the Python standard
library.  The trusted target revision runs it in three separate security
boundaries:

* ``prepare-event`` authenticates a human command and captures immutable PR
  revisions;
* ``build-context`` turns Git objects into inert, bounded review data, while
  ``serve`` is the only repository interface exposed to the model;
* ``publish`` validates the model's JSON as data and uses an OIDC-exchanged
  claude[bot] token to publish COMMENT feedback.

No subcommand executes content from the pull request.  Git is invoked only with
fixed read-only commands and revisions validated as full commit object IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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

    api = GitHubAPI(os.environ.get("GITHUB_TOKEN", ""), os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    bot_allowlist = os.environ.get("CLAUDE_REVIEW_BOT_ALLOWLIST", "").split(",")
    login = str(actor.get("login", ""))
    is_bot = actor.get("type") == "Bot" or login.endswith("[bot]")
    if is_bot:
        permission = "bot-allowlist" if login in {name.strip() for name in bot_allowlist} else "none"
    else:
        quoted_login = urllib.parse.quote(login, safe="")
        permission_data = api.request("GET", f"/repos/{repository}/collaborators/{quoted_login}/permission")
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
    if not all(isinstance(item, str) and item for item in (base_repo, head_repo, base_ref, head_ref)):
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
        _write_outputs({"triggered": True, "authorized": True, "revision_match": False, "mode": mode})
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


def _run(repo: Path, args: list[str], *, text: bool = True) -> str | bytes:
    process = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        timeout=120,
    )
    if process.returncode != 0:
        stderr = process.stderr if text else process.stderr.decode(errors="replace")
        raise ReviewError(f"read-only git command failed: {args[0]}: {stderr[:1000]}")
    return process.stdout


def _run_limited(repo: Path, args: list[str], limit: int) -> tuple[bytes, bool]:
    process = subprocess.Popen(
        ["git", "-c", "core.hooksPath=/dev/null", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None
    data = process.stdout.read(limit + 1)
    truncated = len(data) > limit
    if truncated:
        process.kill()
    _, stderr = process.communicate(timeout=30)
    if not truncated and process.returncode != 0:
        raise ReviewError(f"read-only git command failed: {args[0]}: {stderr[:1000].decode(errors='replace')}")
    return data[:limit], truncated


def _tree(repo: Path, revision: str) -> dict[str, dict[str, Any]]:
    raw = _run(repo, ["ls-tree", "-rlz", "-r", revision], text=False)
    assert isinstance(raw, bytes)
    result: dict[str, dict[str, Any]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        parts = meta.decode().split()
        if len(parts) != 4:
            raise ReviewError("unexpected ls-tree record")
        mode, kind, oid, size = parts
        path = _safe_path(raw_path.decode("utf-8", errors="strict"))
        result[path] = {"mode": mode, "kind": kind, "oid": oid, "size": None if size == "-" else int(size)}
    return result


def _name_status(repo: Path, merge_base: str, head: str) -> list[tuple[str, str | None, str]]:
    raw = _run(repo, ["diff", "--name-status", "-z", "--find-renames", merge_base, head], text=False)
    assert isinstance(raw, bytes)
    fields = raw.split(b"\0")
    changes: list[tuple[str, str | None, str]] = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index].decode("ascii")
        index += 1
        if status.startswith(("R", "C")):
            old_path = _safe_path(fields[index].decode())
            path = _safe_path(fields[index + 1].decode())
            index += 2
        else:
            old_path = None
            path = _safe_path(fields[index].decode())
            index += 1
        changes.append((status, old_path, path))
    return changes


def _diff_stats(repo: Path, merge_base: str, head: str, paths: list[str]) -> tuple[int | None, int | None]:
    raw = _run(repo, ["diff", "--numstat", merge_base, head, "--", *paths])
    assert isinstance(raw, str)
    additions = deletions = 0
    for line in raw.splitlines():
        fields = line.split("\t", 2)
        if len(fields) < 2 or fields[0] == "-" or fields[1] == "-":
            return None, None
        additions += int(fields[0])
        deletions += int(fields[1])
    return additions, deletions


def _valid_lines(patch: str) -> tuple[list[int], list[int], bool]:
    left: set[int] = set()
    right: set[int] = set()
    for match in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", patch, re.MULTILINE):
        old_start, old_count, new_start, new_count = (
            int(match.group(1)),
            int(match.group(2) or "1"),
            int(match.group(3)),
            int(match.group(4) or "1"),
        )
        left.update(range(old_start, old_start + old_count))
        right.update(range(new_start, new_start + new_count))
    truncated = len(left) > MAX_VALID_LINES or len(right) > MAX_VALID_LINES
    return sorted(left)[:MAX_VALID_LINES], sorted(right)[:MAX_VALID_LINES], truncated


def _archive(repo: Path, revision: str, destination: Path) -> None:
    data, truncated = _run_limited(repo, ["archive", "--format=tar", revision], MAX_ARCHIVE_BYTES)
    if truncated:
        raise ReviewError("repository archive exceeds the 256 MiB safety limit")
    destination.write_bytes(data)


def build_context(repo: Path, metadata_path: Path, output_dir: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base_sha = _full_sha(metadata.get("base_sha"), "BASE_SHA")
    head_sha = _full_sha(metadata.get("head_sha"), "HEAD_SHA")
    for name, revision in (("BASE_SHA", base_sha), ("HEAD_SHA", head_sha)):
        resolved = str(_run(repo, ["rev-parse", f"{revision}^{{commit}}"])).strip()
        if resolved != revision:
            raise ReviewError(f"{name} does not resolve to the captured commit")
    merge_base = str(_run(repo, ["merge-base", base_sha, head_sha])).strip()
    _full_sha(merge_base, "MERGE_BASE_SHA")
    if merge_base == head_sha and head_sha != base_sha:
        raise ReviewError("HEAD_SHA cannot be the trusted base ancestor")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_tree = _tree(repo, base_sha)
    head_tree = _tree(repo, head_sha)
    statuses = _name_status(repo, merge_base, head_sha)
    context_complete = len(statuses) <= MAX_CHANGED_FILES
    changes: list[dict[str, Any]] = []
    for status, old_path, path in statuses[:MAX_CHANGED_FILES]:
        paths = [old_path, path] if old_path else [path]
        additions, deletions = _diff_stats(repo, merge_base, head_sha, [item for item in paths if item])
        patch_bytes, patch_truncated = _run_limited(
            repo,
            ["diff", "--no-ext-diff", "--no-textconv", "--unified=0", merge_base, head_sha, "--", *paths],
            2 * 1024 * 1024,
        )
        patch = patch_bytes.decode("utf-8", errors="replace")
        left, right, lines_truncated = _valid_lines(patch)
        old_entry = base_tree.get(old_path or path)
        new_entry = head_tree.get(path)
        binary = additions is None or deletions is None
        special = any(
            entry and entry["mode"] not in {"100644", "100755", "120000", "160000"} for entry in (old_entry, new_entry)
        )
        changes.append(
            {
                "status": status,
                "path": path,
                "old_path": old_path,
                "additions": additions,
                "deletions": deletions,
                "binary": binary,
                "submodule": any(entry and entry["mode"] == "160000" for entry in (old_entry, new_entry)),
                "symlink": any(entry and entry["mode"] == "120000" for entry in (old_entry, new_entry)),
                "special": special,
                "old": old_entry,
                "new": new_entry,
                "valid_left_lines": left,
                "valid_right_lines": right,
                "line_map_truncated": lines_truncated or patch_truncated,
            }
        )
        context_complete = context_complete and not special and not lines_truncated and not patch_truncated

    diff, diff_truncated = _run_limited(
        repo, ["diff", "--no-ext-diff", "--no-textconv", "--find-renames", merge_base, head_sha], MAX_DIFF_BYTES
    )
    (output_dir / "diff.patch").write_bytes(diff)
    context_complete = context_complete and not diff_truncated
    _archive(repo, base_sha, output_dir / "base.tar")
    _archive(repo, head_sha, output_dir / "head.tar")

    trusted_dir = output_dir / "trusted"
    trusted_dir.mkdir(exist_ok=True)
    trusted_paths = ["AGENTS.md", ".github/scripts/claude_review.py"] + sorted(
        path
        for path, entry in base_tree.items()
        if path.startswith("skills/") and path.endswith("/SKILL.md") and entry["mode"] == "100644"
    )
    trusted_manifest = []
    for path in trusted_paths:
        entry = base_tree.get(path)
        if not entry:
            continue
        record = {"path": path, **entry}
        if entry["mode"] == "100644" and (entry["size"] or 0) <= MAX_FILE_READ:
            content = _run(repo, ["show", f"{base_sha}:{path}"], text=False)
            assert isinstance(content, bytes)
            destination = trusted_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            record["available"] = True
        else:
            record["available"] = False
        trusted_manifest.append(record)
    (trusted_dir / "manifest.json").write_text(json.dumps(trusted_manifest, indent=2, sort_keys=True) + "\n")

    history_raw = str(_run(repo, ["log", "-100", "--format=%H%x00%P%x00%aI%x00%an%x00%s", base_sha]))
    history = []
    for line in history_raw.splitlines():
        fields = line.split("\0")
        if len(fields) == 5:
            history.append(dict(zip(("sha", "parents", "authored_at", "author", "subject"), fields)))

    metadata.update(
        {
            "merge_base_sha": merge_base,
            "context_complete": context_complete,
            "changed_files_total": len(statuses),
            "changed_files_in_context": len(changes),
            "diff_bytes_total": len(diff) + (1 if diff_truncated else 0),
            "diff_truncated": diff_truncated,
            "limits": {
                "max_changed_files": MAX_CHANGED_FILES,
                "max_diff_bytes": MAX_DIFF_BYTES,
                "max_file_read": MAX_FILE_READ,
                "max_search_bytes": MAX_SEARCH_BYTES,
            },
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "changes.json").write_text(json.dumps(changes, indent=2, sort_keys=True) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    (output_dir / "trees.json").write_text(json.dumps({"base": base_tree, "head": head_tree}, sort_keys=True) + "\n")
    return metadata


@dataclass
class ContextStore:
    root: Path

    def __post_init__(self) -> None:
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self.changes = json.loads((self.root / "changes.json").read_text())
        self.history = json.loads((self.root / "history.json").read_text())
        self.trees = json.loads((self.root / "trees.json").read_text())

    def _member(self, revision: str, path: str, offset: int, max_bytes: int) -> dict[str, Any]:
        path = _safe_path(path)
        if revision not in {"base", "head"}:
            raise ReviewError("revision must be base or head")
        if offset < 0 or max_bytes < 1 or max_bytes > MAX_FILE_READ:
            raise ReviewError("invalid file window")
        entry = self.trees[revision].get(path)
        if entry is None:
            raise ReviewError("path is absent at the selected revision")
        if entry["mode"] == "160000":
            return {"path": path, "kind": "submodule", "commit": entry["oid"], "content": None}
        archive = self.root / f"{revision}.tar"
        with tarfile.open(archive, "r:") as stream:
            try:
                member = stream.getmember(path)
            except KeyError as error:
                raise ReviewError("path is unavailable in the inert archive") from error
            if member.issym():
                return {"path": path, "kind": "symlink", "target": member.linkname, "content": None}
            if not member.isfile():
                raise ReviewError("special files cannot be read")
            extracted = stream.extractfile(member)
            assert extracted is not None
            extracted.seek(offset)
            raw = extracted.read(max_bytes + 1)
        if b"\0" in raw:
            return {"path": path, "kind": "binary", "size": entry.get("size"), "content": None}
        return {
            "path": path,
            "kind": "text",
            "offset": offset,
            "content": raw[:max_bytes].decode("utf-8", errors="replace"),
            "truncated": len(raw) > max_bytes or offset + max_bytes < (entry.get("size") or 0),
            "size": entry.get("size"),
        }

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "review_metadata":
            return self.metadata
        if name == "list_changes":
            offset = max(0, int(arguments.get("offset", 0)))
            limit = min(200, max(1, int(arguments.get("limit", 100))))
            return {"items": self.changes[offset : offset + limit], "offset": offset, "total": len(self.changes)}
        if name == "read_file":
            return self._member(
                str(arguments.get("revision")),
                str(arguments.get("path", "")),
                int(arguments.get("offset", 0)),
                min(int(arguments.get("max_bytes", 65536)), MAX_FILE_READ),
            )
        if name == "read_diff":
            offset = max(0, int(arguments.get("offset", 0)))
            limit = min(MAX_FILE_READ, max(1, int(arguments.get("max_bytes", 65536))))
            path = self.root / "diff.patch"
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(limit + 1)
            return {
                "offset": offset,
                "content": data[:limit].decode("utf-8", errors="replace"),
                "truncated": len(data) > limit,
                "next_offset": offset + min(len(data), limit),
                "context_complete": self.metadata["context_complete"],
            }
        if name == "trusted_instructions":
            manifest = json.loads((self.root / "trusted/manifest.json").read_text())
            selected = arguments.get("path")
            if selected is None:
                return {"base_sha": self.metadata["base_sha"], "inventory": manifest}
            selected = _safe_path(selected)
            allowed = next((item for item in manifest if item["path"] == selected and item.get("available")), None)
            if not allowed:
                raise ReviewError("trusted instruction is unavailable")
            raw = (self.root / "trusted" / selected).read_bytes()
            return {"path": selected, "base_sha": self.metadata["base_sha"], "content": raw.decode(errors="replace")}
        if name == "trusted_history":
            if self.metadata["mode"] != "strict":
                raise ReviewError("trusted history is available only in strict mode")
            limit = min(100, max(1, int(arguments.get("limit", 50))))
            return self.history[:limit]
        if name == "search_repository":
            revision = str(arguments.get("revision"))
            query = str(arguments.get("query", ""))
            if revision not in {"base", "head"} or not query or len(query) > 128:
                raise ReviewError("invalid search request")
            max_matches = min(100, max(1, int(arguments.get("max_matches", 40))))
            query_bytes = query.encode()
            scanned = 0
            matches = []
            with tarfile.open(self.root / f"{revision}.tar", "r:") as stream:
                for member in stream:
                    if not member.isfile() or member.size > MAX_FILE_READ:
                        continue
                    if scanned + member.size > MAX_SEARCH_BYTES:
                        return {"matches": matches, "truncated": True, "scanned_bytes": scanned}
                    extracted = stream.extractfile(member)
                    assert extracted is not None
                    raw = extracted.read(MAX_FILE_READ + 1)
                    scanned += len(raw)
                    if b"\0" in raw:
                        continue
                    for number, line in enumerate(raw.splitlines(), 1):
                        if query_bytes in line:
                            matches.append(
                                {"path": member.name, "line": number, "text": line[:1000].decode(errors="replace")}
                            )
                            if len(matches) >= max_matches:
                                return {"matches": matches, "truncated": True, "scanned_bytes": scanned}
            return {"matches": matches, "truncated": False, "scanned_bytes": scanned}
        raise ReviewError(f"unknown repository tool: {name}")


TOOLS = [
    {
        "name": "review_metadata",
        "description": "Read normalized immutable PR metadata and coverage limits.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_changes",
        "description": "List bounded changed-file metadata and valid inline line maps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read a bounded inert base/head file; symlinks, binaries, submodules and special files are never followed or executed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "revision": {"enum": ["base", "head"]},
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": MAX_FILE_READ},
            },
            "required": ["revision", "path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_diff",
        "description": "Retrieve the immutable three-dot diff incrementally by byte window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": MAX_FILE_READ},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_repository",
        "description": "Perform bounded literal repository-wide search in an inert revision archive.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "revision": {"enum": ["base", "head"]},
                "query": {"type": "string", "minLength": 1, "maxLength": 128},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["revision", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trusted_instructions",
        "description": "List or read instructions and actual mcore-prefixed skills captured only from BASE_SHA.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False},
    },
    {
        "name": "trusted_history",
        "description": "Read bounded trusted-base history in strict mode.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "additionalProperties": False,
        },
    },
]


def serve(context_dir: Path) -> None:
    store = ContextStore(context_dir)
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            method = request.get("method")
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "megatron-review-context", "version": "1"},
                }
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params") or {}
                value = store.call(str(params.get("name")), params.get("arguments") or {})
                result = {
                    "content": [{"type": "text", "text": json.dumps(value, separators=(",", ":"))}],
                    "isError": False,
                }
            else:
                raise ReviewError(f"unsupported MCP method: {method}")
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        except Exception as error:  # MCP must encode deterministic tool failures.
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32000, "message": str(error)[:1000]},
            }
        print(json.dumps(response, separators=(",", ":")), flush=True)


def _exact_keys(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReviewError(f"{name} has missing or unknown fields")
    return value


def _bounded_string(value: Any, name: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ReviewError(f"{name} has invalid length")
    return value


def validate_report(report: Any, context: dict[str, Any], changes: list[dict[str, Any]]) -> dict[str, Any]:
    report = _exact_keys(
        report,
        {
            "version",
            "mode",
            "base_sha",
            "merge_base_sha",
            "head_sha",
            "status",
            "coverage",
            "inline_findings",
            "general_findings",
            "summary",
            "clean",
            "failure_reason",
        },
        "report",
    )
    if report["version"] != REPORT_VERSION or report["mode"] != context["mode"]:
        raise ReviewError("report version or mode does not match context")
    for key in ("base_sha", "merge_base_sha", "head_sha"):
        if report[key] != context[key] or not SHA_RE.fullmatch(report[key]):
            raise ReviewError(f"report {key} does not match captured context")
    if report["status"] not in {"complete", "incomplete"} or not isinstance(report["clean"], bool):
        raise ReviewError("invalid status or clean flag")
    if report["failure_reason"] not in FAILURE_REASONS:
        raise ReviewError("invalid failure reason")

    coverage = _exact_keys(
        report["coverage"],
        {"changed_files_total", "changed_files_reviewed", "diff_bytes_total", "diff_bytes_reviewed", "skipped"},
        "coverage",
    )
    integers = ("changed_files_total", "changed_files_reviewed", "diff_bytes_total", "diff_bytes_reviewed")
    if any(not isinstance(coverage[key], int) or coverage[key] < 0 for key in integers):
        raise ReviewError("coverage counts must be non-negative integers")
    if (
        coverage["changed_files_total"] != context["changed_files_total"]
        or coverage["diff_bytes_total"] != context["diff_bytes_total"]
    ):
        raise ReviewError("coverage totals do not match immutable context")
    if (
        coverage["changed_files_reviewed"] > coverage["changed_files_total"]
        or coverage["diff_bytes_reviewed"] > coverage["diff_bytes_total"]
    ):
        raise ReviewError("coverage exceeds context totals")
    if not isinstance(coverage["skipped"], list) or len(coverage["skipped"]) > 200:
        raise ReviewError("invalid skipped coverage")
    for skipped in coverage["skipped"]:
        _exact_keys(skipped, {"path", "reason"}, "skipped coverage")
        _safe_path(skipped["path"])
        _bounded_string(skipped["reason"], "skip reason", 1, 300)

    if not isinstance(report["inline_findings"], list) or len(report["inline_findings"]) > MAX_INLINE_FINDINGS:
        raise ReviewError("too many inline findings")
    if not isinstance(report["general_findings"], list) or len(report["general_findings"]) > MAX_GENERAL_FINDINGS:
        raise ReviewError("too many general findings")
    by_path = {change["path"]: change for change in changes}
    for finding in report["inline_findings"]:
        _exact_keys(finding, {"path", "side", "line", "severity", "category", "body"}, "inline finding")
        path = _safe_path(finding["path"])
        change = by_path.get(path)
        if change is None:
            raise ReviewError("inline finding path is not in the captured diff")
        if finding["side"] not in SIDES or not isinstance(finding["line"], int) or finding["line"] < 1:
            raise ReviewError("invalid inline side or line")
        valid = change["valid_left_lines"] if finding["side"] == "LEFT" else change["valid_right_lines"]
        if finding["line"] not in valid:
            raise ReviewError("inline finding line is not a changed line on the selected side")
        if change["binary"] or change["submodule"] or change["special"] or change["line_map_truncated"]:
            raise ReviewError("inline finding targets an unsafe or incompletely mapped change")
        if finding["severity"] not in SEVERITIES:
            raise ReviewError("invalid finding severity")
        _bounded_string(finding["category"], "finding category", 1, 80)
        _bounded_string(finding["body"], "finding body", 1, 2_000)
    for finding in report["general_findings"]:
        _exact_keys(finding, {"severity", "category", "body"}, "general finding")
        if finding["severity"] not in SEVERITIES:
            raise ReviewError("invalid finding severity")
        _bounded_string(finding["category"], "finding category", 1, 80)
        _bounded_string(finding["body"], "finding body", 1, 2_000)
    _bounded_string(report["summary"], "summary", 1, 2_000)

    finding_count = len(report["inline_findings"]) + len(report["general_findings"])
    complete = (
        context["context_complete"]
        and coverage["changed_files_reviewed"] == coverage["changed_files_total"]
        and coverage["diff_bytes_reviewed"] == coverage["diff_bytes_total"]
        and not coverage["skipped"]
    )
    if report["status"] == "complete" and not complete:
        raise ReviewError("complete status does not satisfy coverage requirements")
    if report["status"] == "incomplete" and (report["clean"] or report["failure_reason"] == "none"):
        raise ReviewError("incomplete output cannot be clean or have no failure reason")
    if report["status"] == "complete" and report["failure_reason"] != "none":
        raise ReviewError("complete output must have failure_reason=none")
    if report["clean"] != (report["status"] == "complete" and finding_count == 0):
        raise ReviewError("clean flag is inconsistent with status and findings")
    return report


def _load_context(context_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return json.loads((context_dir / "metadata.json").read_text()), json.loads(
        (context_dir / "changes.json").read_text()
    )


def validate_report_file(report_path: Path, context_dir: Path, output_path: Path | None) -> dict[str, Any]:
    context, changes = _load_context(context_dir)
    report = validate_report(json.loads(report_path.read_text(encoding="utf-8")), context, changes)
    if output_path:
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _oidc_bot_api(context: dict[str, Any]) -> GitHubAPI:
    exchange_url = os.environ.get("CLAUDE_BOT_TOKEN_EXCHANGE_URL", "")
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
    audience = os.environ.get("CLAUDE_BOT_OIDC_AUDIENCE", "claude-review-publisher")
    if not exchange_url or not request_url or not request_token:
        raise ReviewError("claude[bot] OIDC exchange is not configured")
    separator = "&" if "?" in request_url else "?"
    oidc_request = urllib.request.Request(
        request_url + separator + urllib.parse.urlencode({"audience": audience}),
        headers={"Authorization": f"Bearer {request_token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(oidc_request, timeout=30) as response:
        oidc = json.loads(response.read(1024 * 1024))
    jwt = oidc.get("value")
    if not isinstance(jwt, str) or not jwt:
        raise ReviewError("GitHub did not issue an OIDC token")
    payload = {
        "repository": context["repository"],
        "audience": audience,
        "workflow": os.environ.get("GITHUB_WORKFLOW_REF", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
    }
    request = urllib.request.Request(
        exchange_url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {jwt}", "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        exchanged = json.loads(response.read(1024 * 1024))
    token = exchanged.get("token") or exchanged.get("github_token")
    if not isinstance(token, str) or not token:
        raise ReviewError("OIDC exchange did not return a GitHub token")
    api = GitHubAPI(token, os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    identity = api.request("GET", "/user")
    if identity.get("login") != "claude[bot]" or identity.get("type") != "Bot":
        raise ReviewError("OIDC token identity is not exactly claude[bot]")
    return api


def _fixed_status(kind: str, mode: str, head_sha: str) -> str:
    messages = {
        "stale": "Review not published because the pull request revision changed. Run a new review command for the current head.",
        "incomplete": "Review incomplete: bounded analysis did not cover the complete change. No LGTM was produced.",
        "invalid": "Review failed closed because structured analysis output was invalid. No model-authored content was published.",
        "failed": "Review analysis failed before producing valid structured output. No LGTM was produced.",
        "timeout": "Review timed out before complete coverage. No LGTM was produced.",
    }
    return f"### Claude {mode} review — {kind}\n\n{messages[kind]}\n\nCaptured head: `{head_sha}`"


def _status_body(report: dict[str, Any]) -> str:
    mode = report["mode"]
    head = report["head_sha"]
    if report["clean"]:
        return f"### Claude {mode} review — LGTM\n\nNo significant issues found with complete bounded coverage. This is a non-approving comment.\n\nReviewed head: `{head}`"
    counts = {severity: 0 for severity in SEVERITIES}
    for finding in report["inline_findings"] + report["general_findings"]:
        counts[finding["severity"]] += 1
    lines = [
        f"### Claude {mode} review — findings",
        "",
        report["summary"],
        "",
        f"Critical: {counts['critical']} · Important: {counts['important']} · Suggestions: {counts['suggestion']}",
    ]
    for finding in report["general_findings"]:
        lines.extend(["", f"- **[{finding['severity'].upper()} {finding['category']}]** {finding['body']}"])
    lines.extend(
        [
            "",
            f"Reviewed head: `{head}`",
            "",
            "This automated review is COMMENT-only and does not approve the pull request.",
        ]
    )
    return "\n".join(lines)


def publish(context_dir: Path, report_path: Path | None, analysis_result: str) -> str:
    context, changes = _load_context(context_dir)
    publisher = context_dir / "trusted" / ".github/scripts/claude_review.py"
    if not publisher.is_file() or publisher.read_bytes() != Path(__file__).read_bytes():
        raise ReviewError("publisher is not the script captured from BASE_SHA")
    api = _oidc_bot_api(context)
    repo = context["repository"]
    number = context["pr_number"]
    pr = api.request("GET", f"/repos/{repo}/pulls/{number}")
    live_base = (pr.get("base") or {}).get("sha")
    live_head = (pr.get("head") or {}).get("sha")
    if (
        live_head != context["head_sha"]
        or (pr.get("base") or {}).get("repo", {}).get("full_name") != context["base_repository"]
        or (pr.get("base") or {}).get("ref") != context["base_ref"]
    ):
        body = _fixed_status("stale", context["mode"], context["head_sha"])
        api.request("POST", f"/repos/{repo}/issues/{number}/comments", {"body": body})
        return "stale"

    if analysis_result == "timed_out":
        kind, report = "timeout", None
    elif analysis_result == "invalid":
        kind, report = "invalid", None
    elif analysis_result != "success" or report_path is None or not report_path.exists():
        kind, report = "failed", None
    else:
        try:
            report = validate_report(json.loads(report_path.read_text(encoding="utf-8")), context, changes)
            kind = "incomplete" if report["status"] == "incomplete" else "valid"
        except (ReviewError, ValueError, OSError, json.JSONDecodeError):
            kind, report = "invalid", None

    if kind != "valid":
        body = _fixed_status(kind, context["mode"], context["head_sha"])
        api.request("POST", f"/repos/{repo}/issues/{number}/comments", {"body": body})
        return kind

    assert report is not None
    if report["inline_findings"]:
        comments = [
            {
                "path": finding["path"],
                "side": finding["side"],
                "line": finding["line"],
                "body": f"**[{finding['severity'].upper()} {finding['category']}]** {finding['body']}",
            }
            for finding in report["inline_findings"]
        ]
        api.request(
            "POST",
            f"/repos/{repo}/pulls/{number}/reviews",
            {
                "commit_id": context["head_sha"],
                "event": "COMMENT",
                "body": "Validated inline findings from the isolated Claude review.",
                "comments": comments,
            },
        )
    api.request("POST", f"/repos/{repo}/issues/{number}/comments", {"body": _status_body(report)})
    return "clean" if report["clean"] else "findings"


def report_schema() -> dict[str, Any]:
    finding = {
        "type": "object",
        "properties": {
            "severity": {"enum": sorted(SEVERITIES)},
            "category": {"type": "string", "minLength": 1, "maxLength": 80},
            "body": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "required": ["severity", "category", "body"],
        "additionalProperties": False,
    }
    inline = json.loads(json.dumps(finding))
    inline["properties"].update(
        {"path": {"type": "string"}, "side": {"enum": sorted(SIDES)}, "line": {"type": "integer", "minimum": 1}}
    )
    inline["required"] = ["path", "side", "line", "severity", "category", "body"]
    return {
        "type": "object",
        "properties": {
            "version": {"const": REPORT_VERSION},
            "mode": {"enum": ["light", "strict"]},
            "base_sha": {"type": "string", "pattern": "^[0-9a-fA-F]{40}$"},
            "merge_base_sha": {"type": "string", "pattern": "^[0-9a-fA-F]{40}$"},
            "head_sha": {"type": "string", "pattern": "^[0-9a-fA-F]{40}$"},
            "status": {"enum": ["complete", "incomplete"]},
            "coverage": {
                "type": "object",
                "properties": {
                    "changed_files_total": {"type": "integer", "minimum": 0},
                    "changed_files_reviewed": {"type": "integer", "minimum": 0},
                    "diff_bytes_total": {"type": "integer", "minimum": 0},
                    "diff_bytes_reviewed": {"type": "integer", "minimum": 0},
                    "skipped": {
                        "type": "array",
                        "maxItems": 200,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "reason": {"type": "string", "minLength": 1, "maxLength": 300},
                            },
                            "required": ["path", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "changed_files_total",
                    "changed_files_reviewed",
                    "diff_bytes_total",
                    "diff_bytes_reviewed",
                    "skipped",
                ],
                "additionalProperties": False,
            },
            "inline_findings": {"type": "array", "maxItems": MAX_INLINE_FINDINGS, "items": inline},
            "general_findings": {"type": "array", "maxItems": MAX_GENERAL_FINDINGS, "items": finding},
            "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
            "clean": {"type": "boolean"},
            "failure_reason": {"enum": sorted(FAILURE_REASONS)},
        },
        "required": [
            "version",
            "mode",
            "base_sha",
            "merge_base_sha",
            "head_sha",
            "status",
            "coverage",
            "inline_findings",
            "general_findings",
            "summary",
            "clean",
            "failure_reason",
        ],
        "additionalProperties": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    trigger_parser = subparsers.add_parser("parse-trigger")
    trigger_parser.add_argument("body")
    prepare_parser = subparsers.add_parser("prepare-event")
    prepare_parser.add_argument("--event", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--acknowledge", action="store_true")
    context_parser = subparsers.add_parser("build-context")
    context_parser.add_argument("--repo", type=Path, required=True)
    context_parser.add_argument("--metadata", type=Path, required=True)
    context_parser.add_argument("--output-dir", type=Path, required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--context-dir", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--context-dir", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--context-dir", type=Path, required=True)
    publish_parser.add_argument("--report", type=Path)
    publish_parser.add_argument(
        "--analysis-result", choices=["success", "failed", "invalid", "timed_out"], default="success"
    )
    subparsers.add_parser("schema")
    args = parser.parse_args(argv)
    try:
        if args.command == "parse-trigger":
            print(json.dumps(parse_trigger(args.body)))
        elif args.command == "prepare-event":
            prepare_event(args.event, args.output, args.acknowledge)
        elif args.command == "build-context":
            print(json.dumps(build_context(args.repo, args.metadata, args.output_dir), sort_keys=True))
        elif args.command == "serve":
            serve(args.context_dir)
        elif args.command == "validate":
            validate_report_file(args.report, args.context_dir, args.output)
        elif args.command == "publish":
            print(publish(args.context_dir, args.report, args.analysis_result))
        elif args.command == "schema":
            print(json.dumps(report_schema(), separators=(",", ":")))
    except (ReviewError, OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
