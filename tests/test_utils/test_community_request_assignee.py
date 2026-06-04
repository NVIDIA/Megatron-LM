# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
from pathlib import Path


def load_assignee_module():
    module_path = Path(__file__).parents[2] / ".github" / "scripts" / "community_request_assignee.py"
    spec = importlib.util.spec_from_file_location("community_request_assignee", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_should_use_candidate_requires_user_login_and_confidence():
    module = load_assignee_module()

    assert module.should_use_candidate(
        {"assignee": "@valid-user", "confidence": 0.75, "fallback_to_oncall": False}
    )
    assert not module.should_use_candidate(
        {"assignee": "@NVIDIA/mcore-oncall", "confidence": 0.95, "fallback_to_oncall": False}
    )
    assert not module.should_use_candidate(
        {"assignee": "valid-user", "confidence": 0.74, "fallback_to_oncall": False}
    )
    assert not module.should_use_candidate(
        {"assignee": "valid-user", "confidence": 0.95, "fallback_to_oncall": True}
    )
    assert not module.should_use_candidate(
        {"assignee": "svcnvidia-nemo-ci", "confidence": 0.95, "fallback_to_oncall": False}
    )


def test_human_members_excludes_service_accounts():
    module = load_assignee_module()

    assert module.human_members({"alice", "svc-test-account", "svcnvidia-nemo-ci", "bob"}) == [
        "alice",
        "bob",
    ]


def test_create_assignment_plan_uses_high_confidence_candidate(monkeypatch):
    module = load_assignee_module()
    issue = module.IssueContext(
        owner="NVIDIA",
        repo="Megatron-LM",
        number=123,
        title="Feature request",
        url="https://github.com/NVIDIA/Megatron-LM/issues/123",
        author="external-user",
    )

    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)

    plan = module.create_assignment_plan(
        {
            "assignee": "@alice",
            "confidence": 0.91,
            "fallback_to_oncall": False,
            "rationale": "CODEOWNERS and blame point to alice.",
            "relevant_paths": ["megatron/core/transformer/attention.py"],
        },
        issue,
    )

    assert plan.mode == "candidate"
    assert plan.assignees == ["alice"]
    assert plan.notify_users == ["alice"]
    assert plan.confidence == 0.91


def test_create_assignment_plan_falls_back_to_assignable_oncall(monkeypatch):
    module = load_assignee_module()
    issue = module.IssueContext(
        owner="NVIDIA",
        repo="Megatron-LM",
        number=124,
        title="Ambiguous request",
        url="https://github.com/NVIDIA/Megatron-LM/issues/124",
        author="external-user",
    )

    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: {"alice", "bob", "svcnvidia-nemo-ci"},
    )
    monkeypatch.setattr(module, "check_assignable", lambda issue, login: login == "bob")

    plan = module.create_assignment_plan(
        {
            "assignee": "carol",
            "confidence": 0.40,
            "fallback_to_oncall": True,
            "rationale": "The issue does not name a clear code area.",
            "relevant_paths": [],
        },
        issue,
    )

    assert plan.mode == "oncall"
    assert plan.assignees == ["bob"]
    assert plan.notify_users == ["alice", "bob"]
    assert plan.confidence == 0.40


def test_build_slack_message_uses_requested_candidate_copy():
    module = load_assignee_module()
    issue = module.IssueContext(
        owner="NVIDIA",
        repo="Megatron-LM",
        number=125,
        title="Transformer bug",
        url="https://github.com/NVIDIA/Megatron-LM/issues/125",
        author="external-user",
    )
    plan = module.AssignmentPlan(
        mode="candidate",
        assignees=["alice"],
        notify_users=["alice"],
        confidence=0.88,
        rationale="CODEOWNERS and recent blame point to alice.",
        relevant_paths=["megatron/core/transformer/attention.py"],
    )

    message = module.build_slack_message(issue, plan)

    assert message == (
        "You have been automatically assigned to community issue: "
        "<https://github.com/NVIDIA/Megatron-LM/issues/125|"
        "https://github.com/NVIDIA/Megatron-LM/issues/125>.\n\n"
        "Claude has determined that you are the best individual to answer this community issue. "
        "If Claude has made a mistake or if you are unsure how to proceed, please reach out to "
        "@mcore-oncall directly."
    )
