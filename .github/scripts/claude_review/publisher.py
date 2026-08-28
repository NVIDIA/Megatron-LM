# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate structured analysis and publish only deterministic COMMENT feedback."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .common import (
    FAILURE_REASONS,
    MAX_GENERAL_FINDINGS,
    MAX_INLINE_FINDINGS,
    REPORT_VERSION,
    SEVERITIES,
    SHA_RE,
    SIDES,
    TRUSTED_CODE_PATHS,
    GitHubAPI,
    ReviewError,
    _safe_path,
)


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
    repository_root = Path(__file__).parents[3]
    for relative_path in TRUSTED_CODE_PATHS:
        trusted_file = context_dir / "trusted" / relative_path
        local_file = repository_root / relative_path
        if not trusted_file.is_file() or trusted_file.read_bytes() != local_file.read_bytes():
            raise ReviewError(f"publisher code is not captured from BASE_SHA: {relative_path}")
    api = _oidc_bot_api(context)
    repo = context["repository"]
    number = context["pr_number"]
    pr = api.request("GET", f"/repos/{repo}/pulls/{number}")
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
