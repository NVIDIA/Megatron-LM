# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_WORKFLOW = _REPO_ROOT / ".github/workflows/nightly-sync-main-to-dev.yml"
_SKILL = _REPO_ROOT / "skills/nightly-sync/SKILL.md"


def _workflow_hook() -> str:
    workflow = _WORKFLOW.read_text()
    return workflow.split("cat > .git/hooks/pre-push <<'HOOK'\n", 1)[1].split(
        "\n          HOOK", 1
    )[0]


def test_nightly_sync_hook_reports_findings_without_blocking_pushes():
    hook = _workflow_hook()

    assert "set +e" in hook
    assert "set -euo pipefail" in hook
    assert 'WARNING: .github/CODEOWNERS differs from dev' in hook
    assert 'if [ "$findings" -gt 0 ]; then' in hook
    assert "allowing the push to continue" in hook
    assert "exit 1" not in hook
    assert hook.rstrip().endswith("exit 0")


def test_nightly_sync_instructions_preserve_advisory_contract():
    workflow = _WORKFLOW.read_text()
    skill = _SKILL.read_text()

    assert "It is advisory and MUST NOT block a push" in workflow
    assert "must never block the push" in skill
    assert "All pre-push findings are advisory" in skill
    assert "A warning by itself is never a reason to stop" in skill
