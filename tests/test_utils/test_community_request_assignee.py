# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_assignee_module():
    scripts_dir = Path(__file__).parents[2] / ".github" / "scripts"
    module_path = scripts_dir / "community_request_assignee.py"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
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
        "potential_assignee": None,
        "potential_assignee_reason": None,
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
        lambda org, team_slug: (
            {"alice", "bob"} if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG else set()
        ),
    )

    plan = module.create_assignment_plan(make_analysis(), issue)

    assert plan.mode == "candidate"
    assert plan.assignees == ["alice"]
    assert plan.notify_users == ["alice"]
    assert plan.confidence == 0.91
    assert plan.issue_type == "bug"
    assert plan.context.startswith("The issue reports a transformer regression.")


def test_create_assignment_plan_accepts_topic_mapped_other_candidate(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=129, title="FSDP memory question")

    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)
    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: (
            {"wujingyue"} if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG else set()
        ),
    )

    plan = module.create_assignment_plan(
        make_analysis(
            assignee="wujingyue",
            confidence=0.86,
            fallback_to_oncall=False,
            issue_type="other",
            feature_topic="FSDP",
            rationale="FSDP questions should use the FSDP topic mapping.",
            slack_context="This FSDP question maps to wujingyue under the topic mapping.",
            relevant_paths=["megatron/core/distributed/fsdp/"],
        ),
        issue,
    )

    assert plan.mode == "candidate"
    assert plan.assignees == ["wujingyue"]
    assert plan.notify_users == ["wujingyue"]
    assert plan.issue_type == "other"


def test_requested_assignee_override_uses_manual_candidate(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=130, title="Manual assignment")

    monkeypatch.setenv("REQUESTED_ASSIGNEE", "@bob")
    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)
    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: (
            {"alice", "bob"} if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG else set()
        ),
    )

    analysis = module.apply_requested_assignee_override(
        make_analysis(
            assignee="alice",
            confidence=0.20,
            fallback_to_oncall=True,
            rationale="Claude was unsure who should own this.",
        )
    )
    plan = module.create_assignment_plan(analysis, issue)

    assert plan.mode == "candidate"
    assert plan.assignees == ["bob"]
    assert plan.notify_users == ["bob"]
    assert plan.confidence == 1.0
    assert plan.assignment_source == "manual"
    assert plan.rationale.startswith("Assignee was requested explicitly by /claude assign.")


def test_requested_assignee_requires_exact_login_match(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=134, title="Manual assignment casing")

    monkeypatch.setenv("REQUESTED_ASSIGNEE", "@phlip79")
    monkeypatch.setattr(
        module,
        "check_assignable",
        lambda issue, login: (_ for _ in ()).throw(
            AssertionError("wrong-case login should be rejected before assignability check")
        ),
    )
    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: (
            {"Phlip79"} if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG else set()
        ),
    )

    analysis = module.apply_requested_assignee_override(make_analysis(assignee=None))
    plan = module.create_assignment_plan(analysis, issue)

    assert plan.mode == "manual_rejected"
    assert plan.assignees == []
    assert plan.notify_users == []
    assert plan.rejected_candidate == "phlip79"
    assert (
        module.manual_assignee_rejection_comment(plan.rejected_candidate)
        == "User @phlip79 does not exist or is not part of mcore-engineers"
    )


def test_requested_assignee_rejection_does_not_fallback_to_oncall(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=135, title="Invalid manual assignment")

    monkeypatch.setenv("REQUESTED_ASSIGNEE", "@mallory")

    def fake_team_members(org, team_slug):
        if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG:
            return {"bob"}
        if team_slug == module.ACTIVE_ONCALL_TEAM_SLUG:
            return {"bob"}
        return set()

    monkeypatch.setattr(module, "get_team_members", fake_team_members)
    monkeypatch.setattr(module, "check_assignable", lambda issue, login: True)

    analysis = module.apply_requested_assignee_override(make_analysis(assignee=None))
    plan = module.create_assignment_plan(analysis, issue)

    assert plan.mode == "manual_rejected"
    assert plan.assignees == []
    assert plan.notify_users == []
    assert plan.rejected_candidate == "mallory"
    assert (
        module.manual_assignee_rejection_comment(plan.rejected_candidate)
        == "User @mallory does not exist or is not part of mcore-engineers"
    )


def test_run_comments_and_exits_for_invalid_requested_assignee(monkeypatch):
    module = load_assignee_module()
    comments = []

    monkeypatch.setenv("GITHUB_REPOSITORY", "NVIDIA/Megatron-LM")
    monkeypatch.setenv("ISSUE_NUMBER", "136")
    monkeypatch.setenv("ISSUE_TITLE", "Invalid manual assignment")
    monkeypatch.setenv("ISSUE_URL", "https://github.com/NVIDIA/Megatron-LM/issues/136")
    monkeypatch.setenv("ISSUE_AUTHOR", "external-user")
    monkeypatch.setenv("REQUESTED_ASSIGNEE", "@mallory")
    monkeypatch.setenv("ANALYSIS_JSON", json.dumps(make_analysis(assignee=None)))
    monkeypatch.setattr(
        module,
        "get_team_members",
        lambda org, team_slug: (
            {"bob"} if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG else set()
        ),
    )
    monkeypatch.setattr(
        module,
        "post_issue_comment",
        lambda issue, body, dry_run: comments.append((issue.number, body, dry_run)),
    )
    monkeypatch.setattr(
        module,
        "assign_issue",
        lambda issue, assignees, dry_run=False: (_ for _ in ()).throw(
            AssertionError("manual rejection must not assign the issue")
        ),
    )
    monkeypatch.setattr(
        module,
        "send_slack_notifications",
        lambda issue, plan, dry_run, require_slack: (_ for _ in ()).throw(
            AssertionError("manual rejection must not send Slack notifications")
        ),
    )

    with pytest.raises(SystemExit):
        module.run(dry_run=False, require_slack=True)

    assert comments == [
        (136, "User @mallory does not exist or is not part of mcore-engineers", False)
    ]


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
    assert plan.rejected_candidate == "alice"
    assert plan.rejected_candidate_confidence == 0.91
    assert plan.rejected_candidate_reason == "they are not in mcore-engineers"


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


def test_create_assignment_plan_records_low_confidence_potential_candidate(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=128, title="Pipeline P2P bug")

    def fake_team_members(org, team_slug):
        if team_slug == module.ASSIGNEE_ALLOWED_TEAM_SLUG:
            return {"bob", "yashaswikarnati"}
        if team_slug == module.ACTIVE_ONCALL_TEAM_SLUG:
            return {"bob", "yashaswikarnati"}
        return set()

    monkeypatch.setattr(module, "get_team_members", fake_team_members)
    monkeypatch.setattr(module, "check_assignable", lambda issue, login: login == "bob")

    plan = module.create_assignment_plan(
        make_analysis(
            assignee=None,
            potential_assignee="yashaswikarnati",
            potential_assignee_reason="They recently updated the affected pipeline-parallel area.",
            confidence=0.62,
            fallback_to_oncall=True,
            rationale="No recent merged root-cause PR was identified.",
            slack_context="The issue appears to be an older unresolved pipeline P2P ordering bug.",
            relevant_paths=["megatron/core/pipeline_parallel/p2p_communication.py"],
        ),
        issue,
    )

    assert plan.mode == "oncall"
    assert plan.assignees == ["bob"]
    assert plan.rejected_candidate == "yashaswikarnati"
    assert plan.rejected_candidate_confidence == 0.62
    assert plan.rejected_candidate_reason == "confidence 0.62 is below the 0.75 threshold"


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

    assert (
        "I (Megatron Issue Bot) have assigned you to the newly created community issue" in message
    )
    assert "Context from my analysis:" in message
    assert "PR #42 changed the affected path and may be the root cause." in message
    assert (
        "Please take action at your earliest convenience, at latest within 1 business day."
        in message
    )
    assert "<!subteam^S0A7B4U1T3P|mcore-oncall>" in message


def test_build_slack_message_uses_manual_assignment_wording():
    module = load_assignee_module()
    issue = make_issue(module, number=131, title="Manual assignment")
    plan = module.AssignmentPlan(
        mode="candidate",
        assignees=["bob"],
        notify_users=["bob"],
        confidence=1.0,
        rationale="Assignee was requested explicitly by /claude assign.",
        relevant_paths=[],
        issue_type="other",
        context="The issue was manually assigned for follow-up.",
        assignment_source="manual",
    )

    message = module.build_slack_message(issue, plan)

    assert "I was asked to assign this community issue to you." in message
    assert "I determined that you are the best individual" not in message


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
        rejected_candidate="yashaswikarnati",
        rejected_candidate_confidence=0.62,
        rejected_candidate_reason="confidence 0.62 is below the 0.75 threshold",
    )

    message = module.build_slack_message(issue, plan)

    assert "needs on-call triage" in message
    assert "I found a new community issue, but I am not confident who should own it." in message
    assert "This is a new community issue, but I am not sure who should own it." in message
    assert "Potential assignee considered: yashaswikarnati (confidence: 0.62)." in message
    assert "Not assigned because confidence 0.62 is below the 0.75 threshold." in message
    assert "Issue type: feature_request" in message


def test_send_slack_notifications_skips_non_nvidia_email_without_failing(monkeypatch, capsys):
    module = load_assignee_module()
    issue = make_issue(module, number=132, title="Missing Slack mapping")
    comments = []
    plan = module.AssignmentPlan(
        mode="candidate",
        assignees=["alice"],
        notify_users=["alice"],
        confidence=0.91,
        rationale="Alice owns the affected feature area.",
        relevant_paths=[],
        issue_type="bug",
        context="Alice owns the affected feature area.",
    )

    monkeypatch.setattr(module, "get_slack_client", lambda require_slack: object())
    monkeypatch.setattr(module, "get_user_email", lambda username: "alice@example.com")
    monkeypatch.setattr(
        module,
        "post_issue_comment",
        lambda issue, body, dry_run: comments.append((issue.number, body, dry_run)),
    )

    def fail_slack_lookup(slack_client, email):
        raise AssertionError("non-NVIDIA emails should not be sent to Slack lookup")

    monkeypatch.setattr(module, "get_slack_user_id", fail_slack_lookup)

    module.send_slack_notifications(issue, plan, dry_run=False, require_slack=True)

    output = capsys.readouterr().out
    assert module.NON_NVIDIA_EMAIL_SLACK_FALLBACK in output
    assert "alice@example.com" in output
    assert comments == [(132, module.NON_NVIDIA_EMAIL_SLACK_FALLBACK, False)]


def test_post_issue_comment_uses_issue_comment_token(monkeypatch):
    module = load_assignee_module()
    issue = make_issue(module, number=133, title="Fallback comment")
    requests_seen = []

    class FakeResponse:
        status_code = 201
        text = ""

    class FakeRequests:
        @staticmethod
        def post(url, headers, json, timeout):
            requests_seen.append((url, headers, json, timeout))
            return FakeResponse()

    monkeypatch.setenv("ISSUE_COMMENT_TOKEN", "comment-token")
    monkeypatch.setattr(module, "requests", FakeRequests)

    module.post_issue_comment(issue, module.NON_NVIDIA_EMAIL_SLACK_FALLBACK, dry_run=False)

    assert requests_seen == [
        (
            "https://api.github.com/repos/NVIDIA/Megatron-LM/issues/133/comments",
            {
                "Authorization": "Bearer comment-token",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            {"body": module.NON_NVIDIA_EMAIL_SLACK_FALLBACK},
            30,
        )
    ]
