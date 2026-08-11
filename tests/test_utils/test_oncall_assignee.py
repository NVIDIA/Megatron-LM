# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.test_utils.python_scripts import resolve_oncall_assignee


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_current_oncall_uses_first_schedule_entry():
    schedule = [
        {"user": "janEbert", "date": "2026-08-05"},
        {"user": "maanug-nv", "date": "2026-08-12"},
    ]

    assert resolve_oncall_assignee.current_oncall(schedule) == "janEbert"


def test_current_oncall_rejects_invalid_first_entry():
    with pytest.raises(resolve_oncall_assignee.AssigneeResolutionError, match="must have a user"):
        resolve_oncall_assignee.current_oncall([{"date": "2026-08-05"}])


def test_resolve_nvidia_email_matches_github_login_case_insensitively():
    users = {"JanEbert": {"nvidia_email": "jan@nvidia.com"}}

    assert resolve_oncall_assignee.resolve_nvidia_email(users, "janebert") == "jan@nvidia.com"


@pytest.mark.parametrize(
    "users,error",
    [
        ({}, "found 0"),
        ({"janEbert": {"email": "jan@nvidia.com"}}, "has no nvidia_email"),
        ({"janEbert": {"nvidia_email": "jan@example.com"}}, "invalid NVIDIA email"),
    ],
)
def test_resolve_nvidia_email_rejects_missing_or_invalid_records(users, error):
    with pytest.raises(resolve_oncall_assignee.AssigneeResolutionError, match=error):
        resolve_oncall_assignee.resolve_nvidia_email(users, "janEbert")


def test_download_sso_users_reads_private_release_asset():
    asset_url = (
        "https://api.github.com/repos/NVIDIA-GitHub-Management/github-audits/releases/assets/1"
    )
    users = {"janEbert": {"nvidia_email": "jan@nvidia.com"}}
    responses = [
        FakeResponse({"assets": [{"name": "users_sso.json", "url": asset_url}]}),
        FakeResponse(users),
    ]
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0)

    assert resolve_oncall_assignee.download_sso_users("secret-token", fake_get) == users
    assert calls[0][0].endswith(
        "/repos/NVIDIA-GitHub-Management/github-audits/releases/tags/v0.1.0"
    )
    assert calls[0][1]["headers"]["Authorization"] == "Bearer secret-token"
    assert calls[1][0] == asset_url
    assert calls[1][1]["headers"]["Accept"] == "application/octet-stream"
    assert calls[1][1]["allow_redirects"] is True


def test_download_sso_users_rejects_missing_asset():
    def fake_get(_url, **_kwargs):
        return FakeResponse({"assets": []})

    with pytest.raises(resolve_oncall_assignee.AssigneeResolutionError, match="found 0"):
        resolve_oncall_assignee.download_sso_users("secret-token", fake_get)


def test_download_sso_users_retries_transport_failure_without_leaking_token(monkeypatch):
    calls = 0

    def fake_get(_url, **_kwargs):
        nonlocal calls
        calls += 1
        raise resolve_oncall_assignee.requests.ConnectionError("unavailable")

    monkeypatch.setattr(resolve_oncall_assignee.time, "sleep", lambda _seconds: None)

    with pytest.raises(
        resolve_oncall_assignee.AssigneeResolutionError, match="failed to download"
    ) as error:
        resolve_oncall_assignee.download_sso_users("secret-token", fake_get)

    assert "secret-token" not in str(error.value)
    assert calls == resolve_oncall_assignee.REQUEST_ATTEMPTS


def test_main_prints_only_resolved_email_to_stdout(monkeypatch, tmp_path, capsys):
    schedule = tmp_path / "schedule.json"
    schedule.write_text('[{"user": "janEbert", "date": "2026-08-05"}]')
    monkeypatch.setenv("NVIDIA_MANAGEMENT_ORG_PAT", "secret-token")
    monkeypatch.setattr(
        resolve_oncall_assignee,
        "download_sso_users",
        lambda _token: {"JANEBERT": {"nvidia_email": "jan@nvidia.com"}},
    )

    result = resolve_oncall_assignee.main(["--schedule-file", str(schedule)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "jan@nvidia.com\n"
    assert "secret-token" not in captured.err
    assert "JANEBERT" not in captured.err


def test_main_fails_cleanly_when_github_audits_token_is_missing(monkeypatch, tmp_path, capsys):
    schedule = tmp_path / "schedule.json"
    schedule.write_text('[{"user": "janEbert", "date": "2026-08-05"}]')
    monkeypatch.delenv("NVIDIA_MANAGEMENT_ORG_PAT", raising=False)

    result = resolve_oncall_assignee.main(["--schedule-file", str(schedule)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "NVIDIA_MANAGEMENT_ORG_PAT is missing or invalid" in captured.err
