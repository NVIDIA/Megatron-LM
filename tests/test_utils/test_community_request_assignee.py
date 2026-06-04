# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
from pathlib import Path


def load_assignee_module():
    module_path = Path(__file__).parents[2] / ".github" / "scripts" / "community_request_assignee.py"
    spec = importlib.util.spec_from_file_location("community_request_assignee", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_issue(module, number=123, title="Community issue"):
    return module.IssueContext(
        owner="NVIDIA",
        repo="Megatron-LM",
        number=number,
        title=title,
        url=f"https://github.com/NVIDIA/Megatron-LM/issues/{number}",
        author="external-user",
    )


def make_analysis(**overrides):
    analysis = {
        "assignee": "alice",
        "confidence": 0.91,
        "fallback_to_oncall": False,
        "issue_type": "bug",
        "feature_topic": None,
        "root_cause_pr": None,
        "rationale": "A recent PR and blame both point to alice.",
        "slack_context": "The issue reports a transformer regression. PR #42 changed the affected path.",
        "relevant_paths": ["megatron/core/transformer/attention.py"],
    }
    analysis.update(overrides)
    return analysis


def test_human_members_excludes_service_accounts():
    module = load_assignee_module()

    assert module.human_members({"alice", "svc-test-account", "svcnvidia-nemo-ci", "bob"}) == [
        "alice",
        "bob",
    ]


def test_create_assignment_plan_uses_engineer_candidate(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module)

    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)
    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: {"alice", "bob"} if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG else set(),
    )

    plan = module.create_assignment_plan(make_analysis(), issue)

    assert plan.mode == "candidate"
    assert plan.assignees == ["alice"]
    assert plan.notify_users == ["alice"]
    assert plan.confidence == 0.91
    assert plan.issue_type == "bug"
    assert plan.context.startswith("The issue reports a transformer regression.")


def test_create_assignment_plan_rejects_non_engineer_candidate(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=124, title="Feature request")

    def fake_team_members(org, team_slug):
        if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG:
            return {"bob"}
        if team_slug == module.ACTIVE_ONCALL_TEAM_SLUG:
            return {"bob", "carol", "svcnvidia-nemo-ci"}
        return set()

    monkeypatch.setattr(module, "get_team_members", fake_team_members)
    monkeypatch.setattr(module, "check_assignable", lambda issue, login: login == "bob")

    plan = module.create_assignment_plan(make_analysis(assignee="alice"), issue)

    assert plan.mode == "oncall"
    assert plan.assignees == ["bob"]
    assert plan.notify_users == ["bob"]


def test_create_assignment_plan_falls_back_to_engineer_oncall_when_uncertain(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=125, title="Ambiguous request")

    def fake_team_members(org, team_slug):
        if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG:
            return {"alice", "bob"}
        if team_slug == module.ACTIVE_ONCALL_TEAM_SLUG:
            return {"alice", "bob", "svcnvidia-nemo-ci"}
        return set()

    monkeypatch.setattr(module, "get_team_members", fake_team_members)
    monkeypatch.setattr(module, "check_assignable", lambda issue, login: login == "bob")

    plan = module.create_assignment_plan(
        make_analysis(
            assignee=None,
            confidence=0.40,
            fallback_to_oncall=True,
            issue_type="feature_request",
            feature_topic="unknown",
            rationale="The request does not match a known feature topic.",
            slack_context="This is a new feature request, but it does not match the configured topic map.",
            relevant_paths=[],
        ),
        issue,
    )

    assert plan.mode == "oncall"
    assert plan.assignees == ["bob"]
    assert plan.notify_users == ["alice", "bob"]
    assert plan.confidence == 0.40


def test_build_slack_message_includes_candidate_context():
    module = load_assignee_module()
    issue = make_issue(module, number=126, title="Transformer bug")
    plan = module.AssignmentPlan(
        mode="candidate",
        assignees=["alice"],
        notify_users=["alice"],
        confidence=0.88,
        rationale="PR #42 likely introduced the regression.",
        relevant_paths=["megatron/core/transformer/attention.py"],
        issue_type="bug",
        context="The issue reports a transformer regression. PR #42 changed the affected path and may be the root cause.",
    )

    message = module.build_slack_message(issue, plan)

    assert "I (Megatron Issue Bot) have assigned you to the newly created community issue" in message
    assert "Context from my analysis:" in message
    assert "PR #42 changed the affected path and may be the root cause." in message
    assert "Please take action at your earliest convenience, at latest within 1 business day." in message
    assert "<!subteam^S0A7B4U1T3P|mcore-oncall>" in message


def test_build_slack_message_includes_oncall_uncertainty_context():
    module = load_assignee_module()
    issue = make_issue(module, number=127, title="Unknown feature request")
    plan = module.AssignmentPlan(
        mode="oncall",
        assignees=["bob"],
        notify_users=["alice", "bob"],
        confidence=0.35,
        rationale="The request does not match the configured feature map.",
        relevant_paths=[],
        issue_type="feature_request",
        context="This is a new community issue, but I am not sure who should own it.",
    )

    message = module.build_slack_message(issue, plan)

    assert "needs on-call triage" in message
    assert "I found a new community issue, but I am not confident who should own it." in message
    assert "This is a new community issue, but I am not sure who should own it." in message
    assert "Issue type: feature_request" in message
