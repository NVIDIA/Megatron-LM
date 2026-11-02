# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
MODULE_PATH = SCRIPTS_DIR / "re_review_reminder.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "re-review-reminder.yml"
EVENT_TIMESTAMP = "2026-09-03T12:00:00Z"


def load_reminder_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("re_review_reminder", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module():
    return load_reminder_module()


def make_event(*, requester="requester", reviewer="reviewer", updated_at=EVENT_TIMESTAMP):
    event = {
        "action": "review_requested",
        "number": 6402,
        "sender": {"login": requester},
        "repository": {"full_name": "NVIDIA/Megatron-LM"},
        "pull_request": {"number": 6402, "updated_at": updated_at, "base": {"ref": "main"}},
    }
    if reviewer is None:
        event["requested_team"] = {"slug": "mcore-engineers"}
    else:
        event["requested_reviewer"] = {"login": reviewer}
    return event


def make_review_request(module, *, requester="requester", reviewer="reviewer"):
    return module.ReviewRequest(
        number=6402,
        requester=requester,
        reviewer=reviewer,
        event_timestamp=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
    )


def set_eligible_github_checks(monkeypatch, module):
    monkeypatch.setattr(module, "_get_mcore_engineers", lambda: {"requester", "reviewer"})
    monkeypatch.setattr(module, "_reviewer_is_still_requested", lambda request: True)
    monkeypatch.setattr(module, "_has_prior_review", lambda request: True)


def test_eligible_event_sends_one_exact_slack_dm(monkeypatch, module):
    set_eligible_github_checks(monkeypatch, module)
    slack_client = Mock()
    slack_client.conversations_open.return_value = {"channel": {"id": "D6402"}}
    get_user_email = Mock(return_value="reviewer@nvidia.com")
    get_slack_client = Mock(return_value=slack_client)
    get_slack_user_id = Mock(return_value="U_REVIEWER")
    monkeypatch.setattr(module, "get_user_email", get_user_email)
    monkeypatch.setattr(module, "get_slack_client", get_slack_client)
    monkeypatch.setattr(module, "get_slack_user_id", get_slack_user_id)

    assert module.process_event(make_event()) is True

    get_user_email.assert_called_once_with("reviewer")
    get_slack_client.assert_called_once_with(require_slack=True)
    get_slack_user_id.assert_called_once_with(slack_client, "reviewer@nvidia.com")
    slack_client.conversations_open.assert_called_once_with(users="U_REVIEWER")
    slack_client.chat_postMessage.assert_called_once_with(
        channel="D6402",
        text=(
            "You have previously reviewed Megatron-LM PR "
            "<https://github.com/NVIDIA/Megatron-LM/pull/6402|#6402> and your review "
            "has been re-requested. Please take another look."
        ),
        unfurl_links=False,
        unfurl_media=False,
    )


@pytest.mark.parametrize("state", ["COMMENTED", "APPROVED", "CHANGES_REQUESTED", "DISMISSED"])
def test_submitted_prior_review_states_count(monkeypatch, module, state):
    review_request = make_review_request(module)
    monkeypatch.setattr(
        module,
        "_get_paginated_list",
        lambda path: [
            {"user": {"login": "reviewer"}, "state": state, "submitted_at": "2026-09-03T11:59:59Z"}
        ],
    )

    assert module._has_prior_review(review_request) is True


@pytest.mark.parametrize(
    "review",
    [
        {"user": {"login": "reviewer"}, "state": "PENDING", "submitted_at": None},
        {"user": {"login": "reviewer"}, "state": "PENDING", "submitted_at": "2026-09-03T11:59:59Z"},
        {"user": {"login": "reviewer"}, "state": "COMMENTED"},
    ],
    ids=["pending", "pending-with-submission-time", "missing-submitted-at"],
)
def test_unsubmitted_reviews_do_not_count(monkeypatch, module, review):
    monkeypatch.setattr(module, "_get_paginated_list", lambda path: [review])

    assert module._has_prior_review(make_review_request(module)) is False


@pytest.mark.parametrize("submitted_at", ["2026-09-03T12:00:00Z", "2026-09-03T12:00:01Z"])
def test_review_at_or_after_event_timestamp_does_not_count(monkeypatch, module, submitted_at):
    monkeypatch.setattr(
        module,
        "_get_paginated_list",
        lambda path: [
            {"user": {"login": "reviewer"}, "state": "COMMENTED", "submitted_at": submitted_at}
        ],
    )

    assert module._has_prior_review(make_review_request(module)) is False


def test_initial_review_request_is_ignored(monkeypatch, module):
    monkeypatch.setattr(module, "_get_mcore_engineers", lambda: {"requester", "reviewer"})
    monkeypatch.setattr(module, "_reviewer_is_still_requested", lambda request: True)
    monkeypatch.setattr(module, "_has_prior_review", lambda request: False)
    send_slack_dm = Mock()
    monkeypatch.setattr(module, "_send_slack_dm", send_slack_dm)

    assert module.process_event(make_event()) is False
    send_slack_dm.assert_not_called()


@pytest.mark.parametrize(
    "event",
    [make_event(reviewer=None), make_event(requester="Reviewer", reviewer="reviewer")],
    ids=["team-request", "self-request"],
)
def test_team_and_self_requests_are_ignored_before_api_calls(monkeypatch, module, event):
    get_members = Mock(side_effect=AssertionError("GitHub API should not be called"))
    monkeypatch.setattr(module, "_get_mcore_engineers", get_members)

    assert module.process_event(event) is False
    get_members.assert_not_called()


@pytest.mark.parametrize("members", [{"reviewer"}, {"requester"}], ids=["requester", "reviewer"])
def test_nonmember_requester_or_reviewer_is_ignored(monkeypatch, module, members):
    monkeypatch.setattr(module, "_get_mcore_engineers", lambda: members)
    still_requested = Mock()
    send_slack_dm = Mock()
    monkeypatch.setattr(module, "_reviewer_is_still_requested", still_requested)
    monkeypatch.setattr(module, "_send_slack_dm", send_slack_dm)

    assert module.process_event(make_event()) is False
    still_requested.assert_not_called()
    send_slack_dm.assert_not_called()


def test_github_login_matching_is_case_insensitive(monkeypatch, module):
    review_request = make_review_request(module, requester="ReQuEsTeR", reviewer="ReViEwEr")
    monkeypatch.setattr(
        module,
        "_request_json",
        lambda path: {"state": "open", "requested_reviewers": [{"login": "reviewer"}]},
    )

    assert module._reviewer_is_still_requested(review_request) is True

    def paginated_response(path):
        if path.endswith("/members"):
            return [{"login": "REQUESTER"}, {"login": "REVIEWER"}]
        return [
            {
                "user": {"login": "REVIEWER"},
                "state": "APPROVED",
                "submitted_at": "2026-09-03T11:00:00Z",
            }
        ]

    monkeypatch.setattr(module, "_get_paginated_list", paginated_response)

    assert module._get_mcore_engineers() == {"requester", "reviewer"}
    assert module._has_prior_review(review_request) is True


def test_closed_pull_request_is_no_longer_actionable(monkeypatch, module):
    review_request = make_review_request(module)
    monkeypatch.setattr(
        module,
        "_request_json",
        lambda path: {"state": "closed", "requested_reviewers": [{"login": "reviewer"}]},
    )

    assert module._reviewer_is_still_requested(review_request) is False


def test_reviewer_no_longer_pending_is_ignored(monkeypatch, module):
    monkeypatch.setattr(module, "_get_mcore_engineers", lambda: {"requester", "reviewer"})
    monkeypatch.setattr(module, "_reviewer_is_still_requested", lambda request: False)
    has_prior_review = Mock()
    send_slack_dm = Mock()
    monkeypatch.setattr(module, "_has_prior_review", has_prior_review)
    monkeypatch.setattr(module, "_send_slack_dm", send_slack_dm)

    assert module.process_event(make_event()) is False
    has_prior_review.assert_not_called()
    send_slack_dm.assert_not_called()


def test_paginated_list_fetches_all_pages(monkeypatch, module):
    monkeypatch.setattr(module, "PER_PAGE", 2)
    requests = []

    def request_json(path):
        requests.append(path)
        if path.endswith("page=1"):
            return [{"id": 1}, {"id": 2}]
        return [{"id": 3}]

    monkeypatch.setattr(module, "_request_json", request_json)

    assert module._get_paginated_list("repos/NVIDIA/Megatron-LM/pulls/6402/reviews") == [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]
    assert requests == [
        "repos/NVIDIA/Megatron-LM/pulls/6402/reviews?per_page=2&page=1",
        "repos/NVIDIA/Megatron-LM/pulls/6402/reviews?per_page=2&page=2",
    ]


def test_github_http_error_is_a_hard_failure(monkeypatch, module):
    response = Mock(status_code=503)
    monkeypatch.setattr(module, "get_headers", lambda: {"Authorization": "Bearer token"})
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(module.NotifierError, match="HTTP 503"):
        module._request_json("repos/NVIDIA/Megatron-LM/pulls/6402/reviews")


def test_github_transport_error_is_a_hard_failure(monkeypatch, module):
    monkeypatch.setattr(module, "get_headers", lambda: {"Authorization": "Bearer token"})

    def raise_transport_error(*args, **kwargs):
        raise module.requests.RequestException("connection failed")

    monkeypatch.setattr(module.requests, "get", raise_transport_error)

    with pytest.raises(module.NotifierError, match="GitHub API request failed"):
        module._request_json("repos/NVIDIA/Megatron-LM/pulls/6402/reviews")


def test_non_nvidia_identity_is_a_hard_failure(monkeypatch, module):
    monkeypatch.setattr(module, "get_user_email", lambda username: "reviewer@example.com")
    get_slack_client = Mock()
    monkeypatch.setattr(module, "get_slack_client", get_slack_client)

    with pytest.raises(module.NotifierError, match="NVIDIA Slack identity"):
        module._send_slack_dm(make_review_request(module))

    get_slack_client.assert_not_called()


def test_missing_slack_identity_is_a_hard_failure(monkeypatch, module):
    slack_client = Mock()
    monkeypatch.setattr(module, "get_user_email", lambda username: "reviewer@nvidia.com")
    monkeypatch.setattr(module, "get_slack_client", lambda require_slack: slack_client)
    monkeypatch.setattr(module, "get_slack_user_id", lambda client, email: None)

    with pytest.raises(module.NotifierError, match="Slack user"):
        module._send_slack_dm(make_review_request(module))

    slack_client.conversations_open.assert_not_called()


@pytest.mark.parametrize("operation", ["conversations_open", "chat_postMessage"])
def test_slack_api_errors_are_not_suppressed(monkeypatch, module, operation):
    slack_client = Mock()
    slack_client.conversations_open.return_value = {"channel": {"id": "D6402"}}
    getattr(slack_client, operation).side_effect = RuntimeError("Slack unavailable")
    monkeypatch.setattr(module, "get_user_email", lambda username: "reviewer@nvidia.com")
    monkeypatch.setattr(module, "get_slack_client", lambda require_slack: slack_client)
    monkeypatch.setattr(module, "get_slack_user_id", lambda client, email: "U_REVIEWER")

    with pytest.raises(RuntimeError, match="Slack unavailable"):
        module._send_slack_dm(make_review_request(module))


def test_run_loads_event_from_github_event_path(monkeypatch, module, tmp_path):
    event = make_event()
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    process_event = Mock(return_value=True)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setattr(module, "process_event", process_event)

    assert module.run() is True
    process_event.assert_called_once_with(event)


def test_workflow_uses_trusted_main_review_request_event_and_minimal_permissions():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert (
        "\non:\n"
        "  pull_request_target:\n"
        "    types: [review_requested]\n"
        "    branches:\n"
        "      - main\n"
    ) in workflow
    assert "\npermissions:\n  contents: read\n  pull-requests: read\n" in workflow
    assert "uses: actions/checkout@" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "github.event.pull_request.head" not in workflow
    assert "github.head_ref" not in workflow
    assert "refs/pull/" not in workflow
    assert "GH_TOKEN: ${{ secrets.PAT }}" in workflow
    assert "SLACK_TOKEN: ${{ secrets.ISSUE_BOT_SLACK_TOKEN }}" in workflow
