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
from .context import ContextStore

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
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
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
