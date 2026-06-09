# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ONCALL_MANAGER_PATH = REPO_ROOT / ".github" / "scripts" / "oncall_manager.py"


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class FakeRequests:
    def __init__(self, pr_data):
        self.pr_data = pr_data
        self.posts = []

    def get(self, url, headers=None):
        return FakeResponse(200, self.pr_data)

    def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(201)


class FakeSlackClient:
    def __init__(self, users_by_email=None, usergroups=None):
        self.users_by_email = users_by_email or {}
        self.usergroups = usergroups if usergroups is not None else []
        self.messages = []
        self.usergroup_updates = []

    def users_lookupByEmail(self, email):
        if email not in self.users_by_email:
            raise Exception("users_not_found")
        return {"user": {"id": self.users_by_email[email]}}

    def usergroups_list(self, include_users=True):
        return {"usergroups": self.usergroups}

    def usergroups_users_update(self, usergroup, users):
        self.usergroup_updates.append({"usergroup": usergroup, "users": users})

    def chat_postMessage(self, channel, text):
        self.messages.append({"channel": channel, "text": text})


@pytest.fixture
def oncall_manager(monkeypatch):
    slack_module = types.ModuleType("slack_sdk")
    slack_module.WebClient = object

    slack_errors_module = types.ModuleType("slack_sdk.errors")
    slack_errors_module.SlackApiError = Exception
    requests_module = types.ModuleType("requests")

    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "slack_sdk", slack_module)
    monkeypatch.setitem(sys.modules, "slack_sdk.errors", slack_errors_module)
    monkeypatch.setenv("GITHUB_REPOSITORY", "NVIDIA/Megatron-LM")
    monkeypatch.setenv("GH_TOKEN", "token")

    spec = importlib.util.spec_from_file_location("oncall_manager", ONCALL_MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_slack_usergroup_notifies_previous_oncall_when_new_oncall_lookup_fails(
    oncall_manager, monkeypatch
):
    client = FakeSlackClient(users_by_email={"previous@nvidia.com": "UOLD"})
    monkeypatch.setattr(oncall_manager, "get_slack_client", lambda: client)
    monkeypatch.setattr(
        oncall_manager,
        "get_user_email",
        lambda username: {
            "new-oncall": "new-oncall@users.noreply.github.com",
            "previous-oncall": "previous@nvidia.com",
        }[username],
    )

    oncall_manager.update_slack_usergroup("new-oncall", ["previous-oncall"])

    assert client.usergroup_updates == []
    assert len(client.messages) == 1
    assert client.messages[0]["channel"] == "UOLD"
    assert "new-oncall@users.noreply.github.com" in client.messages[0]["text"]
    assert "@mcore-oncall" in client.messages[0]["text"]
    assert "You should still be in @mcore-oncall" in client.messages[0]["text"]


def test_update_slack_usergroup_does_not_notify_fallback_when_previous_oncall_lookup_fails(
    oncall_manager, monkeypatch
):
    client = FakeSlackClient(users_by_email={"fallback@nvidia.com": "UFALLBACK"})
    monkeypatch.setattr(oncall_manager, "get_slack_client", lambda: client)
    monkeypatch.setattr(
        oncall_manager,
        "get_user_email",
        lambda username: {
            "new-oncall": "new-oncall@users.noreply.github.com",
            "previous-oncall": "previous-oncall@users.noreply.github.com",
        }[username],
    )

    oncall_manager.update_slack_usergroup("new-oncall", ["previous-oncall"])

    assert client.usergroup_updates == []
    assert client.messages == []


def test_update_slack_usergroup_updates_members_when_new_oncall_lookup_succeeds(
    oncall_manager, monkeypatch
):
    client = FakeSlackClient(
        users_by_email={"new-oncall@nvidia.com": "UNEW"},
        usergroups=[{"handle": "mcore-oncall", "id": "S123", "users": ["UOLD"]}],
    )
    monkeypatch.setattr(oncall_manager, "get_slack_client", lambda: client)
    monkeypatch.setattr(oncall_manager, "get_user_email", lambda username: "new-oncall@nvidia.com")

    oncall_manager.update_slack_usergroup("new-oncall", ["previous-oncall"])

    assert client.usergroup_updates == [
        {"usergroup": "S123", "users": ["UOLD", "UNEW"]},
        {"usergroup": "S123", "users": ["UNEW"]},
    ]
    assert client.messages == []


@pytest.mark.parametrize(
    "pr_data",
    [
        {
            "user": {"login": "maintainer"},
            "requested_reviewers": [{"login": "alice"}],
            "requested_teams": [],
        },
        {
            "user": {"login": "maintainer"},
            "requested_reviewers": [],
            "requested_teams": [{"slug": "mcore"}],
        },
        {
            "requested_reviewers": [{"login": "alice"}],
            "requested_teams": [{"slug": "mcore-oncall"}],
            "labels": [{"name": "community-request"}],
        },
    ],
)
def test_assign_reviewer_skips_when_oncall_is_not_needed(
    oncall_manager, monkeypatch, capsys, pr_data
):
    fake_requests = FakeRequests(pr_data)
    monkeypatch.setattr(oncall_manager, "requests", fake_requests)

    oncall_manager.assign_reviewer(123)

    assert fake_requests.posts == []
    assert "Skipping reviewer request" in capsys.readouterr().out


@pytest.mark.parametrize(
    "pr_data",
    [
        {"user": {"login": "maintainer"}, "requested_reviewers": [], "requested_teams": []},
        {
            "requested_reviewers": [{"login": "alice"}],
            "requested_teams": [{"slug": "mcore"}],
            "labels": [{"name": "community-request"}],
        },
    ],
)
def test_assign_reviewer_requests_oncall_when_needed(oncall_manager, monkeypatch, pr_data):
    fake_requests = FakeRequests(pr_data)
    monkeypatch.setattr(oncall_manager, "requests", fake_requests)

    oncall_manager.assign_reviewer(123)

    assert fake_requests.posts == [
        {
            "url": "https://api.github.com/repos/NVIDIA/Megatron-LM/pulls/123/requested_reviewers",
            "headers": {"Authorization": "token token", "Accept": "application/vnd.github.v3+json"},
            "json": {"team_reviewers": ["mcore-oncall"]},
        }
    ]


def test_get_headers_rejects_invalid_token(oncall_manager, monkeypatch, capsys):
    monkeypatch.setenv("GH_TOKEN", "not a token\nwith newline")

    with pytest.raises(SystemExit) as error:
        oncall_manager.get_headers()

    assert error.value.code == 1
    assert "GH_TOKEN or GITHUB_TOKEN is invalid" in capsys.readouterr().out


def test_get_rotation_order_uses_alphabetical_rotation_team(oncall_manager, monkeypatch):
    monkeypatch.setattr(
        oncall_manager,
        "get_team_members",
        lambda org, team_slug: {"charlie", "Alice", "bob", "svcnvidia-nemo-ci"},
    )

    assert oncall_manager.get_rotation_order("NVIDIA") == ["Alice", "bob", "charlie"]


def test_ensure_schedule_filled_uses_rotation_team_order(oncall_manager, monkeypatch):
    schedule = [{"user": "bob", "date": "2026-01-07"}]
    rotation_order = ["Alice", "bob", "charlie"]
    monkeypatch.setattr(oncall_manager, "TARGET_WEEKS", 5)
    monkeypatch.setattr(
        oncall_manager,
        "get_team_members",
        lambda *_args, **_kwargs: pytest.fail("team members should not determine oncall order"),
    )

    oncall_manager.ensure_schedule_filled(schedule, rotation_order)

    assert [entry["user"] for entry in schedule] == ["bob", "charlie", "Alice", "bob", "charlie"]
    assert [entry["date"] for entry in schedule[-4:]] == [
        "2026-01-14",
        "2026-01-21",
        "2026-01-28",
        "2026-02-04",
    ]


def test_validate_schedule_users_in_rotation_team_accepts_all_users(
    oncall_manager, monkeypatch, capsys
):
    schedule = [
        {"user": "charlie", "date": "2026-01-07"},
        {"user": "alice", "date": "2026-01-14"},
        {"user": "bob", "date": "2026-01-21"},
        {"user": "alice", "date": "2026-01-28"},
    ]
    monkeypatch.setattr(
        oncall_manager,
        "get_team_members",
        lambda org, team_slug: {"alice", "bob", "charlie", "dana"},
    )

    rotation_order = ["alice", "bob", "charlie", "dana"]

    oncall_manager.validate_schedule_users_in_rotation_team(schedule, rotation_order)

    assert "Validated 3 scheduled user(s) in mcore-oncall-rotation" in capsys.readouterr().out


def test_validate_schedule_users_in_rotation_team_rejects_missing_user(
    oncall_manager, monkeypatch, capsys
):
    schedule = [{"user": "charlie", "date": "2026-01-07"}, {"user": "alice", "date": "2026-01-14"}]
    with pytest.raises(SystemExit) as error:
        oncall_manager.validate_schedule_users_in_rotation_team(schedule, ["alice"])

    assert error.value.code == 1
    assert "charlie" in capsys.readouterr().out


def test_rotate_schedule_keeps_popped_user_in_rotation_order(oncall_manager, monkeypatch):
    schedule = [
        {"user": "charlie", "date": "2026-01-07"},
        {"user": "alice", "date": "2026-01-14"},
        {"user": "bob", "date": "2026-01-21"},
    ]
    saved_schedule = []
    real_datetime = oncall_manager.datetime

    class FakeDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 1, 14, tzinfo=tz)

    monkeypatch.setattr(oncall_manager, "TARGET_WEEKS", 3)
    monkeypatch.setattr(oncall_manager, "datetime", FakeDateTime)
    monkeypatch.setattr(
        oncall_manager, "load_schedule", lambda: [entry.copy() for entry in schedule]
    )
    monkeypatch.setattr(
        oncall_manager, "save_schedule", lambda new_schedule: saved_schedule.extend(new_schedule)
    )
    monkeypatch.setattr(
        oncall_manager, "get_team_members", lambda org, team_slug: {"alice", "bob", "charlie"}
    )
    monkeypatch.setattr(oncall_manager, "update_active_oncall_team", lambda *_args, **_kwargs: None)

    oncall_manager.rotate_schedule("NVIDIA")

    assert [entry["user"] for entry in saved_schedule] == ["alice", "bob", "charlie"]
    assert saved_schedule[-1]["date"] == "2026-01-28"
