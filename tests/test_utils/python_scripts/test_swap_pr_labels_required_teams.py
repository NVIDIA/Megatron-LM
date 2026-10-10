# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
from github import GithubException

from tests.test_utils.python_scripts import swap_pr_labels


def make_tracker(monkeypatch, stage, team, members, reviews):
    tracker = swap_pr_labels.PRReviewTracker.__new__(swap_pr_labels.PRReviewTracker)
    labels = set() if stage == tracker.EXPERT_REVIEW else {stage}
    tracker.pr = mock.Mock(number=42, draft=False)
    tracker.pr.labels = [SimpleNamespace(name=label) for label in labels]
    tracker.pr.get_files.return_value = [SimpleNamespace(filename="megatron/core/example.py")]
    tracker.pr.add_to_labels.side_effect = labels.add
    tracker.pr.remove_from_labels.side_effect = labels.discard
    tracker.stage = tracker.get_stage(tracker.pr)
    tracker.token = "fixture"
    tracker.repo = SimpleNamespace(full_name="NVIDIA/Megatron-LM")
    tracker.org = mock.Mock()
    tracker._team_cache = {}
    tracker._codeowner_rules = [("megatron/core/", {team})]

    def get_team(slug):
        if slug == team and members is None:
            raise GithubException(403, {"message": "Team membership unavailable"}, {})
        team_members = members if slug == team else []
        result = mock.Mock()
        result.get_members.return_value = [SimpleNamespace(login=member) for member in team_members]
        return result

    tracker.org.get_team_by_slug.side_effect = get_team
    response = mock.Mock()
    response.json.return_value = {
        "data": {"repository": {"pullRequest": {"latestReviews": {"nodes": reviews}}}}
    }
    monkeypatch.setattr(swap_pr_labels.requests, "post", mock.Mock(return_value=response))
    return tracker, labels


@pytest.mark.parametrize("members", [None, []])
@pytest.mark.parametrize("team", ["ci", "core-adlr"])
@pytest.mark.parametrize("stage", ["Expert Review", "Final Review", "Approved"])
def test_unavailable_required_team_never_counts_as_approved(monkeypatch, stage, team, members):
    tracker, labels = make_tracker(monkeypatch, stage, team, members, [])

    tracker.swap_labels()

    assert tracker.APPROVED not in labels
    if team == "ci":
        assert tracker.FINAL_REVIEW not in labels
    elif stage != tracker.EXPERT_REVIEW:
        assert tracker.FINAL_REVIEW in labels


@pytest.mark.parametrize("team", ["ci", "core-adlr"])
def test_known_required_team_member_approval_completes_review(monkeypatch, team):
    tracker, labels = make_tracker(
        monkeypatch,
        "Expert Review",
        team,
        ["reviewer"],
        [{"author": {"login": "reviewer"}, "state": "APPROVED"}],
    )

    tracker.swap_labels()

    assert tracker.APPROVED in labels


def test_known_required_team_without_approval_stays_pending(monkeypatch):
    tracker, labels = make_tracker(monkeypatch, "Expert Review", "ci", ["reviewer"], [])

    tracker.swap_labels()

    assert tracker.APPROVED not in labels
    assert tracker.FINAL_REVIEW not in labels
