# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[3] / ".github" / "scripts" / "trigger_workflow_and_wait.py"
SPEC = importlib.util.spec_from_file_location("trigger_workflow_and_wait", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_installation_token_is_reused_then_refreshed_before_expiry():
    now = 1_800_000_000
    requests = []

    def request(api_url, method, path, token, payload=None):
        requests.append((method, path, token, payload))
        if path.endswith("/installation"):
            return {"id": 123}
        expiry = datetime.fromtimestamp(now + 3600, timezone.utc).isoformat()
        return {"token": f"token-{len(requests)}", "expires_at": expiry}

    with mock.patch.object(MODULE, "create_app_jwt", return_value="jwt"):
        auth = MODULE.InstallationAuth(
            "1", "key", "owner", "repo", "https://api.github.test", lambda: now, request
        )
        first = auth.token()
        assert auth.token() == first
        assert len(requests) == 2

        now += 3600 - MODULE.TOKEN_REFRESH_MARGIN_SECONDS
        refreshed = auth.token()
        assert refreshed != first
        assert len(requests) == 5
        assert requests[2] == ("DELETE", "/installation/token", first, None)

        auth.close()
        assert requests[-1] == ("DELETE", "/installation/token", refreshed, None)
        assert not auth._token


def test_installation_request_refreshes_once_after_unauthorized():
    auth = mock.Mock()
    auth.api_url = "https://api.github.test"
    auth.token.side_effect = ["expired", "fresh"]
    auth.request.side_effect = [MODULE.GitHubApiError(401, "expired"), {"status": "ok"}]

    assert MODULE.installation_request(auth, "GET", "/resource") == {"status": "ok"}
    auth.invalidate.assert_called_once_with()


def test_workflow_runs_path_scopes_event_branch_and_creation_time():
    path = MODULE.workflow_runs_path(
        "owner", "repo", "cicd-main.yml", "branch/name", "2026-01-01T00:00:00+00:00"
    )

    assert path.startswith("/repos/owner/repo/actions/workflows/cicd-main.yml/runs?")
    assert "event=workflow_dispatch" in path
    assert "branch=branch%2Fname" in path
    assert "created=%3E%3D2026-01-01T00%3A00%3A00%2B00%3A00" in path
