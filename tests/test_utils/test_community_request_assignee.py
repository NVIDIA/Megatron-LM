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
    monkeypatch.setattr(module, "get_team_members", lambda org, team_slug: set())

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


def test_create_assignment_plan_prioritizes_rotation_candidate(monkeypatch):
    module = load_assignee_module()
    issue = module.IssueContext(
        owner="NVIDIA",
        repo="Megatron-LM",
        number=126,
        title="Transformer bug",
        url="https://github.com/NVIDIA/Megatron-LM/issues/126",
        author="external-user",
    )

    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)
    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: {"bob"} if team_slug == module.PREFERRED_ASSIGNEE_TEAM_SLUG else set(),
    )

    plan = module.create_assignment_plan(
        {
            "assignee": "alice",
            "candidate_assignees": [
                {"login": "alice", "confidence": 0.93, "rationale": "Most recent blame."},
                {"login": "bob", "confidence": 0.88, "rationale": "CODEOWNERS and recent commits."},
            ],
            "confidence": 0.93,
            "fallback_to_oncall": False,
            "rationale": "Both candidates have relevant history.",
            "relevant_paths": ["megatron/core/transformer/attention.py"],
        },
        issue,
    )

    assert plan.mode == "candidate"
    assert plan.assignees == ["bob"]
    assert plan.notify_users == ["bob"]
    assert plan.confidence == 0.88


def test_create_assignment_plan_keeps_clearly_stronger_non_rotation_candidate(monkeypatch):
    module = load_assignee_module()
    issue = module.IssueContext(
        owner="NVIDIA",
        repo="Megatron-LM",
        number=127,
        title="Transformer bug",
        url="https://github.com/NVIDIA/Megatron-LM/issues/127",
        author="external-user",
    )

    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)
    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: {"bob"} if team_slug == module.PREFERRED_ASSIGNEE_TEAM_SLUG else set(),
    )

    plan = module.create_assignment_plan(
        {
            "assignee": "alice",
            "candidate_assignees": [
                {"login": "alice", "confidence": 0.95, "rationale": "Most recent blame."},
                {"login": "bob", "confidence": 0.80, "rationale": "Some relevant commits."},
            ],
            "confidence": 0.95,
            "fallback_to_oncall": False,
            "rationale": "Alice has substantially stronger ownership evidence.",
            "relevant_paths": ["megatron/core/transformer/attention.py"],
        },
        issue,
    )

    assert plan.mode == "candidate"
    assert plan.assignees == ["alice"]
    assert plan.notify_users == ["alice"]
    assert plan.confidence == 0.95


def test_create_assignment_plan_continues_when_rotation_team_unavailable(monkeypatch):
    module = load_assignee_module()
    issue = module.IssueContext(
        owner="NVIDIA",
        repo="Megatron-LM",
        number=128,
        title="Transformer bug",
        url="https://github.com/NVIDIA/Megatron-LM/issues/128",
        author="external-user",
    )

    def fail_team_lookup(org, team_slug):
        raise SystemExit(1)

    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)
    monkeypatch.setattr(module, "get_team_members", fail_team_lookup)

    plan = module.create_assignment_plan(
        {
            "assignee": "alice",
            "candidate_assignees": [{"login": "alice", "confidence": 0.92, "rationale": "Recent blame."}],
            "confidence": 0.92,
            "fallback_to_oncall": False,
            "rationale": "Alice has relevant history.",
            "relevant_paths": ["megatron/core/transformer/attention.py"],
        },
        issue,
    )

    assert plan.mode == "candidate"
    assert plan.assignees == ["alice"]
    assert plan.notify_users == ["alice"]


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
        "I (Megatron Issue Bot) have assigned you to the newly created community issue: "
        "<https://github.com/NVIDIA/Megatron-LM/issues/125|"
        "https://github.com/NVIDIA/Megatron-LM/issues/125>.\n\n"
        "I determined that you are the best individual to answer this community issue. "
        "Please take action at your earliest convenience, at latest within 1 business day. "
        "If I made a mistake or if you are unsure how to proceed, please reach out to "
        "<!subteam^S0A7B4U1T3P|mcore-oncall> directly."
    )
