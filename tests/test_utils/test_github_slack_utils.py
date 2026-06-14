# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
from pathlib import Path

import pytest


def load_utils_module():
    module_path = Path(__file__).parents[2] / ".github" / "scripts" / "github_slack_utils.py"
    spec = importlib.util.spec_from_file_location("github_slack_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_get_user_email_uses_signed_off_by_fallback(monkeypatch):
    module = load_utils_module()
    requests_seen = []

    class FakeRequests:
        @staticmethod
        def get(url, headers, timeout):
            requests_seen.append((url, headers, timeout))
            if url.endswith("/users/alice"):
                return FakeResponse(200, {"email": None})
            return FakeResponse(
                200,
                [
                    {
                        "commit": {
                            "author": {"email": "12345+alice@users.noreply.github.com"},
                            "message": "Subject\n\nSigned-off-by: Alice <alice@nvidia.com>",
                        }
                    }
                ],
            )

    monkeypatch.setenv("GH_TOKEN", "token")
    monkeypatch.setattr(module, "requests", FakeRequests)

    assert module.get_user_email("alice") == "alice@nvidia.com"
    assert requests_seen[0][1]["Authorization"] == "Bearer token"
    assert requests_seen[0][1]["Accept"] == "application/vnd.github+json"
    assert requests_seen[0][1]["X-GitHub-Api-Version"] == "2022-11-28"
    assert requests_seen[0][2] == 30


def test_get_headers_requires_gh_token_without_github_token_fallback(monkeypatch):
    module = load_utils_module()

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    with pytest.raises(SystemExit):
        module.get_headers()


def test_get_headers_uses_requested_token_env(monkeypatch):
    module = load_utils_module()

    monkeypatch.setenv("ISSUE_COMMENT_TOKEN", "comment-token")

    headers = module.get_headers("ISSUE_COMMENT_TOKEN")

    assert headers["Authorization"] == "Bearer comment-token"


def test_get_slack_user_id_uses_lookup_by_email():
    module = load_utils_module()

    class FakeSlackClient:
        def users_lookupByEmail(self, email):
            assert email == "alice@nvidia.com"
            return {"user": {"id": "U123"}}

    assert module.get_slack_user_id(FakeSlackClient(), "alice@nvidia.com") == "U123"
