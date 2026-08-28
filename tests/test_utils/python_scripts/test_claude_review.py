# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / ".github" / "scripts" / "claude_review.py"
SPEC = importlib.util.spec_from_file_location("claude_review", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review
SPEC.loader.exec_module(review)

A = "a" * 40
B = "b" * 40
C = "c" * 40


def context(*, complete=True):
    return {
        "mode": "light",
        "base_sha": A,
        "merge_base_sha": B,
        "head_sha": C,
        "context_complete": complete,
        "changed_files_total": 1,
        "diff_bytes_total": 10,
    }


def report(**updates):
    value = {
        "version": 1,
        "mode": "light",
        "base_sha": A,
        "merge_base_sha": B,
        "head_sha": C,
        "status": "complete",
        "coverage": {
            "changed_files_total": 1,
            "changed_files_reviewed": 1,
            "diff_bytes_total": 10,
            "diff_bytes_reviewed": 10,
            "skipped": [],
        },
        "inline_findings": [],
        "general_findings": [],
        "summary": "No significant issues.",
        "clean": True,
        "failure_reason": "none",
    }
    value.update(updates)
    return value


def changes():
    return [
        {
            "path": "src/new.py",
            "old_path": "src/old.py",
            "status": "R100",
            "binary": False,
            "submodule": False,
            "special": False,
            "line_map_truncated": False,
            "valid_left_lines": [3],
            "valid_right_lines": [4, 5],
        }
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("/claude review", ("light", None)),
        ("  /claude strict-review  ", ("strict", None)),
        (f"/claude review {A.upper()}\nignored text", ("light", A)),
        ("/claude review\n/claude strict-review", ("light", None)),
        ("/claude strict-review extra", None),
        ("/claude review deadbeef", None),
        ("/claude  review", None),
        ("/Claude review", None),
        ("> /claude review", None),
        ("`/claude review`", None),
        ("please /claude review", None),
        ("/claude review-now", None),
    ],
)
def test_parse_trigger_exact_first_line(body, expected):
    assert review.parse_trigger(body) == expected


@pytest.mark.parametrize("permission", ["admin", "maintain", "write", "triage"])
def test_human_repository_permission_policy(permission):
    assert review.actor_authorized({"login": "maintainer", "type": "User"}, permission)


@pytest.mark.parametrize("permission", ["read", "none", ""])
def test_human_without_review_permission_is_denied(permission):
    assert not review.actor_authorized({"login": "reader", "type": "User"}, permission)


def test_bots_denied_except_exact_allowlist():
    actor = {"login": "trusted-reviewer[bot]", "type": "Bot"}
    assert not review.actor_authorized(actor, "admin")
    assert review.actor_authorized(actor, "none", ["trusted-reviewer[bot]"])
    assert not review.actor_authorized(actor, "none", ["other[bot]"])


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "a/../b", "./a", "a\nname"])
def test_safe_path_rejects_escape_or_control(path):
    with pytest.raises(review.ReviewError):
        review._safe_path(path)


def test_clean_complete_report_is_valid():
    assert review.validate_report(report(), context(), changes())["clean"] is True


def test_unknown_report_field_is_rejected():
    value = report(extra="model controlled")
    with pytest.raises(review.ReviewError, match="unknown"):
        review.validate_report(value, context(), changes())


def test_incomplete_report_can_never_be_clean():
    value = report(status="incomplete", failure_reason="context_incomplete")
    with pytest.raises(review.ReviewError, match="incomplete"):
        review.validate_report(value, context(), changes())


def test_context_incomplete_can_never_claim_complete():
    with pytest.raises(review.ReviewError, match="coverage"):
        review.validate_report(report(), context(complete=False), changes())


def test_coverage_totals_and_counts_are_validated():
    value = report()
    value["coverage"]["changed_files_total"] = 2
    with pytest.raises(review.ReviewError, match="totals"):
        review.validate_report(value, context(), changes())


def finding(**updates):
    value = {
        "path": "src/new.py",
        "side": "RIGHT",
        "line": 4,
        "severity": "critical",
        "category": "Correctness",
        "body": "The value is wrong; compute it from the reduced tensor.",
    }
    value.update(updates)
    return value


def test_valid_changed_line_finding_is_accepted_and_not_clean():
    value = report(inline_findings=[finding()], clean=False, summary="One issue.")
    assert review.validate_report(value, context(), changes())["clean"] is False


@pytest.mark.parametrize(
    "update",
    [
        {"path": "unchanged.py"},
        {"side": "RIGHT", "line": 3},
        {"side": "LEFT", "line": 4},
        {"side": "INVALID"},
        {"line": 0},
    ],
)
def test_invalid_inline_location_is_rejected(update):
    value = report(inline_findings=[finding(**update)], clean=False)
    with pytest.raises(review.ReviewError):
        review.validate_report(value, context(), changes())


@pytest.mark.parametrize("flag", ["binary", "submodule", "special", "line_map_truncated"])
def test_unsafe_change_never_accepts_inline_finding(flag):
    changed = changes()
    changed[0][flag] = True
    value = report(inline_findings=[finding()], clean=False)
    with pytest.raises(review.ReviewError, match="unsafe"):
        review.validate_report(value, context(), changed)


def test_inline_and_general_limits_are_enforced():
    value = report(
        inline_findings=[finding() for _ in range(review.MAX_INLINE_FINDINGS + 1)], clean=False
    )
    with pytest.raises(review.ReviewError, match="too many"):
        review.validate_report(value, context(), changes())


def test_fixed_statuses_do_not_contain_model_output_or_approval():
    for kind in ("stale", "incomplete", "invalid", "failed", "timeout"):
        text = review._fixed_status(kind, "strict", C)
        assert C in text
        assert "APPROVE" not in text.upper()
        assert "model says" not in text


def test_clean_status_is_explicitly_non_approving():
    text = review._status_body(report())
    assert "LGTM" in text
    assert "non-approving comment" in text
    assert "--approve" not in text


def test_schema_rejects_unknown_fields_and_has_limits():
    schema = review.report_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["inline_findings"]["maxItems"] == review.MAX_INLINE_FINDINGS
    assert schema["properties"]["general_findings"]["maxItems"] == review.MAX_GENERAL_FINDINGS


def test_mcp_surface_is_read_only():
    names = {tool["name"] for tool in review.TOOLS}
    assert names == {
        "review_metadata",
        "list_changes",
        "read_file",
        "read_diff",
        "search_repository",
        "trusted_instructions",
        "trusted_history",
    }
    assert not names & {"shell", "write", "publish", "github"}


def test_workflow_has_separate_boundary_and_comment_only_publication():
    root = SCRIPT.parents[2]
    entry = (root / ".github/workflows/claude_review.yml").read_text()
    isolated = (root / ".github/workflows/_claude-review-isolated.yml").read_text()
    combined = entry + isolated
    assert "${{ needs.capture.outputs.mode }}" in entry
    assert "concurrency:" in entry
    assert "Read-only structured analysis" in isolated
    assert "Validated claude[bot] publication" in isolated
    assert '"event": "COMMENT"' not in isolated  # Fixed local publisher owns it.
    assert "--approve" not in combined
    assert "FW-CI-Templates" not in combined
    assert "FW-CI-templates" not in combined
    analyze = isolated.split("  analyze:", 1)[1].split("  publish:", 1)[0]
    publish = isolated.split("  publish:", 1)[1]
    assert "id-token: write" not in analyze
    assert "id-token: write" in publish
    assert "anthropics/claude-code-action" not in publish


def test_actual_mcore_skill_names_are_present_in_trusted_inventory_source():
    root = SCRIPT.parents[2]
    assert (root / "skills/mcore-testing/SKILL.md").is_file()
    assert (root / "skills/mcore-cicd/SKILL.md").is_file()
    assert (root / "skills/mcore-linting-and-formatting/SKILL.md").is_file()


def test_trusted_code_inventory_covers_every_local_review_module():
    root = SCRIPT.parents[2]
    actual = {
        str(path.relative_to(root))
        for path in (root / ".github/scripts/claude_review").glob("*.py")
    }
    actual.add(".github/scripts/claude_review.py")
    assert set(review.TRUSTED_CODE_PATHS) == actual


def test_review_implementation_is_split_into_bounded_components():
    root = SCRIPT.parents[2]
    assert sum(1 for _ in SCRIPT.open(encoding="utf-8")) < 100
    for path in (root / ".github/scripts/claude_review").glob("*.py"):
        assert sum(1 for _ in path.open(encoding="utf-8")) < 500, path
