# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build and serve inert, bounded repository context."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    MAX_ARCHIVE_BYTES,
    MAX_CHANGED_FILES,
    MAX_DIFF_BYTES,
    MAX_FILE_READ,
    MAX_SEARCH_BYTES,
    MAX_VALID_LINES,
    TRUSTED_CODE_PATHS,
    ReviewError,
    _full_sha,
    _safe_path,
)


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
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    data = process.stdout.read(limit + 1)
    truncated = len(data) > limit
    if truncated:
        process.kill()
    _, stderr = process.communicate(timeout=30)
    if not truncated and process.returncode != 0:
        raise ReviewError(
            f"read-only git command failed: {args[0]}: {stderr[:1000].decode(errors='replace')}"
        )
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
        result[path] = {
            "mode": mode,
            "kind": kind,
            "oid": oid,
            "size": None if size == "-" else int(size),
        }
    return result


def _name_status(repo: Path, merge_base: str, head: str) -> list[tuple[str, str | None, str]]:
    raw = _run(
        repo, ["diff", "--name-status", "-z", "--find-renames", merge_base, head], text=False
    )
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


def _diff_stats(
    repo: Path, merge_base: str, head: str, paths: list[str]
) -> tuple[int | None, int | None]:
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
        additions, deletions = _diff_stats(
            repo, merge_base, head_sha, [item for item in paths if item]
        )
        patch_bytes, patch_truncated = _run_limited(
            repo,
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--unified=0",
                merge_base,
                head_sha,
                "--",
                *paths,
            ],
            2 * 1024 * 1024,
        )
        patch = patch_bytes.decode("utf-8", errors="replace")
        left, right, lines_truncated = _valid_lines(patch)
        old_entry = base_tree.get(old_path or path)
        new_entry = head_tree.get(path)
        binary = additions is None or deletions is None
        special = any(
            entry and entry["mode"] not in {"100644", "100755", "120000", "160000"}
            for entry in (old_entry, new_entry)
        )
        changes.append(
            {
                "status": status,
                "path": path,
                "old_path": old_path,
                "additions": additions,
                "deletions": deletions,
                "binary": binary,
                "submodule": any(
                    entry and entry["mode"] == "160000" for entry in (old_entry, new_entry)
                ),
                "symlink": any(
                    entry and entry["mode"] == "120000" for entry in (old_entry, new_entry)
                ),
                "special": special,
                "old": old_entry,
                "new": new_entry,
                "valid_left_lines": left,
                "valid_right_lines": right,
                "line_map_truncated": lines_truncated or patch_truncated,
            }
        )
        context_complete = (
            context_complete and not special and not lines_truncated and not patch_truncated
        )

    diff, diff_truncated = _run_limited(
        repo,
        ["diff", "--no-ext-diff", "--no-textconv", "--find-renames", merge_base, head_sha],
        MAX_DIFF_BYTES,
    )
    (output_dir / "diff.patch").write_bytes(diff)
    context_complete = context_complete and not diff_truncated
    _archive(repo, base_sha, output_dir / "base.tar")
    _archive(repo, head_sha, output_dir / "head.tar")

    trusted_dir = output_dir / "trusted"
    trusted_dir.mkdir(exist_ok=True)
    trusted_paths = ["AGENTS.md", *TRUSTED_CODE_PATHS] + sorted(
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
    (trusted_dir / "manifest.json").write_text(
        json.dumps(trusted_manifest, indent=2, sort_keys=True) + "\n"
    )

    history_raw = str(
        _run(repo, ["log", "-100", "--format=%H%x00%P%x00%aI%x00%an%x00%s", base_sha])
    )
    history = []
    for line in history_raw.splitlines():
        fields = line.split("\0")
        if len(fields) == 5:
            history.append(
                dict(zip(("sha", "parents", "authored_at", "author", "subject"), fields))
            )

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
    (output_dir / "trees.json").write_text(
        json.dumps({"base": base_tree, "head": head_tree}, sort_keys=True) + "\n"
    )
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
            return {
                "items": self.changes[offset : offset + limit],
                "offset": offset,
                "total": len(self.changes),
            }
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
            allowed = next(
                (item for item in manifest if item["path"] == selected and item.get("available")),
                None,
            )
            if not allowed:
                raise ReviewError("trusted instruction is unavailable")
            raw = (self.root / "trusted" / selected).read_bytes()
            return {
                "path": selected,
                "base_sha": self.metadata["base_sha"],
                "content": raw.decode(errors="replace"),
            }
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
                                {
                                    "path": member.name,
                                    "line": number,
                                    "text": line[:1000].decode(errors="replace"),
                                }
                            )
                            if len(matches) >= max_matches:
                                return {
                                    "matches": matches,
                                    "truncated": True,
                                    "scanned_bytes": scanned,
                                }
            return {"matches": matches, "truncated": False, "scanned_bytes": scanned}
        raise ReviewError(f"unknown repository tool: {name}")


from .server import TOOLS, serve  # noqa: E402,F401
