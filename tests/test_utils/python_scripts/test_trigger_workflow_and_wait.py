# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / ".github" / "scripts" / "trigger_workflow_and_wait.py"
SPEC = importlib.util.spec_from_file_location("trigger_workflow_and_wait", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_workflow_runs_url_scopes_event_branch_and_creation_time():
    url = MODULE.workflow_runs_url(
        "https://api.github.test",
        "owner",
        "repo",
        "cicd-main.yml",
        "branch/name",
        "2026-01-01T00:00:00+00:00",
    )

    assert url.startswith(
        "https://api.github.test/repos/owner/repo/actions/workflows/cicd-main.yml/runs?"
    )
    assert "event=workflow_dispatch" in url
    assert "branch=branch%2Fname" in url
    assert "created=%3E%3D2026-01-01T00%3A00%3A00%2B00%3A00" in url


def test_poll_workflow_returns_completed_conclusion():
    def request(token, method, url, payload):
        assert token == "downscoped-token"
        assert method == "GET"
        assert payload is None
        return {"status": "completed", "conclusion": "success"}

    assert MODULE.poll_workflow(
        "downscoped-token",
        "https://api.github.test",
        "owner",
        "repo",
        123,
        60,
        2100,
        request=request,
    ) == (True, "success")


def test_poll_workflow_returns_incomplete_at_token_rotation_boundary():
    times = iter([0.0, 2100.0])

    def request(token, method, url, payload):
        return {"status": "in_progress", "conclusion": None}

    assert MODULE.poll_workflow(
        "downscoped-token",
        "https://api.github.test",
        "owner",
        "repo",
        123,
        60,
        2100,
        request=request,
        clock=lambda: next(times),
        sleep=lambda _: None,
    ) == (False, None)


def test_validate_workflow_result_rejects_incomplete_run():
    with pytest.raises(TimeoutError, match="run 123"):
        MODULE.validate_workflow_result(False, None, 123)
