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


def test_nightly_sync_hook_hard_fails_integrity_violations():
    hook = _workflow_hook()

    assert "set -euo pipefail" in hook
    assert "set +e" not in hook
    assert 'ABORT: no merge commit found' in hook
    assert 'ABORT: .github/CODEOWNERS differs from dev' in hook
    assert 'if [ "$violations" -gt 0 ]; then' in hook
    assert hook.count("exit 1") >= 3
    assert "allowing the push to continue" not in hook


def test_nightly_sync_instructions_do_not_authorize_bypass():
    workflow = _WORKFLOW.read_text()
    skill = _SKILL.read_text()

    assert "It MUST block when either invariant fails" in workflow
    assert "It must block the push when" in skill
    assert "All pre-push findings are advisory" not in skill
    assert "Never use `--no-verify`" in skill
