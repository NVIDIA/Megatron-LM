# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner

from tests.test_utils.python_scripts import generate_jet_trigger_job, linear_ci, recipe_parser


def _mock_llm_reporting(monkeypatch):
    summarize = Mock(
        side_effect=lambda failures, _prompt: [
            linear_ci._fallback_summary(failure) for failure in failures
        ]
    )

    def group_failures(failures):
        grouped = {}
        for failure in failures:
            grouped.setdefault((failure["category"], failure["summary"]), []).append(
                failure["test_name"]
            )
        return (
            [
                {"label": f"test-bucket-{index}", "rationale": summary, "tests": tests}
                for index, ((_, summary), tests) in enumerate(grouped.items(), 1)
            ],
            None,
        )

    subcategorize = Mock(side_effect=group_failures)
    digest = Mock(return_value="LLM pipeline digest")
    monkeypatch.setattr(linear_ci.summarizer, "_summarize_failures", summarize)
    monkeypatch.setattr(linear_ci.summarizer, "_subcategorize", subcategorize)
    monkeypatch.setattr(linear_ci.summarizer, "_digest", digest)
    return summarize, subcategorize, digest


@pytest.fixture
def notify_module(monkeypatch):
    pytest.importorskip("nemo_ci_triage.slack_notification")
    monkeypatch.setenv("GITLAB_ENDPOINT", "ci.example.com")
    from tests.test_utils.python_scripts import notify

    return notify


def test_build_test_script_preserves_workload_exit_code():
    script = generate_jet_trigger_job.build_test_script("python workload.py")

    assert "set +e" in script
    assert "set -o pipefail" in script
    assert "python workload.py 2>&1 | tee jet_workload.log" in script
    assert "exit_code=${PIPESTATUS[0]}" in script
    assert "set -e" in script
    assert "extract-errors jet_workload.log" in script
    assert "--output error_report.json" in script
    assert 'exit "$exit_code"' in script


@pytest.mark.parametrize("enable_error_extraction", [False, True])
def test_error_extraction_is_opt_in_for_generated_jobs(
    monkeypatch, tmp_path, enable_error_extraction
):
    workload = recipe_parser.dotdict(
        type="basic",
        spec=recipe_parser.dotdict(model="gpt", environment="dev", test_case="triage-test"),
    )
    monkeypatch.setattr(
        generate_jet_trigger_job.recipe_parser, "load_workloads", lambda **_kwargs: [workload]
    )
    output_path = tmp_path / "pipeline.yml"
    args = [
        "--scope",
        "mr",
        "--environment",
        "dev",
        "--n-repeat",
        "1",
        "--time-limit",
        "60",
        "--test-cases",
        "all",
        "--platform",
        "dgx_h100",
        "--cluster",
        "ghci",
        "--output-path",
        str(output_path),
        "--container-image",
        "utility",
        "--container-tag",
        "test",
        "--dependent-job",
        "functional:configure",
        "--record-checkpoints",
        "false",
        "--slurm-account",
        "mcore",
        "--no-enable-warmup",
    ]
    if enable_error_extraction:
        args.append("--enable-error-extraction")

    result = CliRunner().invoke(generate_jet_trigger_job.main, args)

    assert result.exit_code == 0, result.output
    job = yaml.safe_load(output_path.read_text())["triage-test"]
    if enable_error_extraction:
        assert "extract-errors jet_workload.log" in job["script"][0]
        assert job["artifacts"]["paths"] == ["results/", "jet_workload.log", "error_report.json"]
    else:
        assert "extract-errors" not in job["script"][0]
        assert job["artifacts"]["paths"] == ["results/"]


def test_notification_rules_cover_all_enabled_test_runs():
    unit = yaml.safe_load(Path(".gitlab/stages/02.test.yml").read_text())
    functional = yaml.safe_load(Path(".gitlab/stages/04.functional-tests.yml").read_text())
    triage = yaml.safe_load(Path(".gitlab/stages/06.triage.yml").read_text())

    unit_conditions = [
        rule["if"] for rule in unit["test:unit_tests_notify"]["rules"] if "if" in rule
    ]
    assert unit_conditions == [
        "$UNIT_TEST == 'yes' && $CI_MERGE_REQUEST_EVENT_TYPE == 'merged_result' && "
        '$CI_MERGE_REQUEST_TARGET_BRANCH_PROTECTED != "true"',
        "$UNIT_TEST == 'yes' && $UNIT_TEST_REPEAT != '0'",
    ]

    smoke_condition = functional["functional:smoke_notify"]["rules"][1]["if"]
    assert smoke_condition == (
        '$FUNCTIONAL_TEST == "yes" && ' "$FUNCTIONAL_TEST_SCOPE =~ /^(mr|nightly|weekly|release)$/"
    )
    assert functional["functional:x_notify"]["rules"][0]["if"] == ('$FUNCTIONAL_TEST == "yes"')

    triage_jobs = (".linear_reconcile_rules", "triage:linear_write", "triage:slack_linear_followup")
    for job_name in triage_jobs:
        condition = triage[job_name]["rules"][0]["if"]
        assert '$FUNCTIONAL_TEST == "yes"' in condition
        assert "CI_PIPELINE_SOURCE" not in condition
        assert "CI_COMMIT_BRANCH" not in condition


def test_all_generated_test_types_enable_error_extraction():
    unit = Path(".gitlab/stages/02.test.yml").read_text()
    integration = Path(".gitlab/stages/03.integration-tests.yml").read_text()
    functional = Path(".gitlab/stages/04.functional-tests.yml").read_text()

    assert unit.count('"--enable-error-extraction"') >= 1
    assert integration.count('"--enable-error-extraction"') >= 1
    assert functional.count('"--enable-error-extraction"') >= 2


def test_get_pipeline_jobs_uses_triage_collector(monkeypatch, notify_module):
    notify = notify_module
    bridge = SimpleNamespace(
        name="functional:run_dev_dgx_h100", attributes={"downstream_pipeline": {"id": 101}}
    )
    root_pipeline = Mock()
    root_pipeline.bridges.list.return_value = [bridge]
    project = Mock()
    project.pipelines.get.return_value = root_pipeline
    handle = Mock()
    handle.projects.get.return_value = project
    jobs = [{"status": "failed", "gpu": "Unknown"}]

    monkeypatch.setattr(notify, "get_gitlab_handle", lambda: handle)
    collector = Mock(return_value=jobs)
    monkeypatch.setattr(notify.notification, "get_jobs_from_pipeline", collector)

    assert notify.get_pipeline_jobs(123, "functional:run_") == [
        ("functional:run_dev_dgx_h100", 101, [{"status": "failed", "gpu": "H100"}])
    ]
    collector.assert_called_once_with(project, 101)


def test_build_linear_reports_groups_matching_failures(monkeypatch):
    summarize, subcategorize, digest = _mock_llm_reporting(monkeypatch)
    pipeline_jobs = [
        (
            "functional:run_dev_dgx_h100",
            101,
            [
                {
                    "config_name": "gpt_pass",
                    "id": 1,
                    "status": "success",
                    "allow_failure": False,
                    "error_type": None,
                },
                {
                    "config_name": "gpt_fail_a",
                    "id": 2,
                    "status": "failed",
                    "allow_failure": False,
                    "error_type": "CUDA OOM",
                },
            ],
        ),
        (
            "functional:run_lts_dgx_h100",
            102,
            [
                {
                    "config_name": "gpt_fail_b",
                    "id": 3,
                    "status": "failed",
                    "allow_failure": True,
                    "error_type": None,
                }
            ],
        ),
    ]
    reports = {
        2: {
            "exit_code_training": 1,
            "category": "CUDA OOM",
            "error_subtype": "torch.OutOfMemoryError",
            "excerpt": "CUDA out of memory",
        },
        3: {
            "exit_code_training": 1,
            "category": "CUDA OOM",
            "error_subtype": "torch.OutOfMemoryError",
            "excerpt": "CUDA out of memory",
        },
    }

    summaries, buckets = linear_ci.build_pipeline_reports(
        123, "nightly", pipeline_jobs, reports.get, "https://ci.example.com/ADLR/megatron-lm"
    )

    stats = summaries["modules"][linear_ci.LINEAR_MODULE]
    assert stats == {"passed": 1, "failed": 2, "passed_tests": ["gpt_pass@dev-dgx-h100"]}
    assert len(buckets["buckets"]) == 1
    bucket = buckets["buckets"][0]
    assert bucket["category"] == "CUDA OOM"
    assert bucket["rationale"] == "CUDA OOM: torch.OutOfMemoryError"
    assert bucket["tests"] == [
        {
            "name": "gpt_fail_a@dev-dgx-h100",
            "job_url": "https://ci.example.com/ADLR/megatron-lm/-/jobs/2",
        },
        {
            "name": "gpt_fail_b@lts-dgx-h100",
            "job_url": "https://ci.example.com/ADLR/megatron-lm/-/jobs/3",
        },
    ]
    summarize.assert_called_once()
    subcategorize.assert_called_once()
    digest.assert_called_once()


def test_allow_failure_without_report_is_not_counted_as_passed(monkeypatch):
    _mock_llm_reporting(monkeypatch)
    pipeline_jobs = [
        (
            "functional:run_dev_dgx_h100",
            101,
            [
                {
                    "config_name": "ambiguous",
                    "id": 4,
                    "status": "success",
                    "allow_failure": True,
                    "error_type": None,
                }
            ],
        )
    ]

    summaries, buckets = linear_ci.build_pipeline_reports(
        123,
        "nightly",
        pipeline_jobs,
        lambda _job_id: None,
        "https://ci.example.com/ADLR/megatron-lm",
    )

    stats = summaries["modules"][linear_ci.LINEAR_MODULE]
    assert stats["passed_tests"] == []
    assert stats["failed"] == 0
    assert buckets["buckets"] == []


def test_failed_job_without_report_still_creates_a_safe_bucket(monkeypatch):
    _mock_llm_reporting(monkeypatch)
    pipeline_jobs = [
        (
            "functional:run_dev_dgx_h100",
            101,
            [
                {
                    "config_name": "missing_report",
                    "id": 5,
                    "status": "failed",
                    "allow_failure": False,
                    "error_type": None,
                }
            ],
        )
    ]

    summaries, buckets = linear_ci.build_pipeline_reports(
        123,
        "nightly",
        pipeline_jobs,
        lambda _job_id: None,
        "https://ci.example.com/ADLR/megatron-lm",
    )

    assert summaries["modules"][linear_ci.LINEAR_MODULE]["failed"] == 1
    assert buckets["buckets"][0]["tests"][0]["name"] == "missing_report@dev-dgx-h100"
    assert "No structured error report" in buckets["buckets"][0]["rationale"]


def test_triage_config_selects_megatron_and_enables_write_actions():
    linear_status = pytest.importorskip("nemo_ci_triage.linear.linear_status")
    linear_write = pytest.importorskip("nemo_ci_triage.linear.linear_write")
    config = Path(".gitlab/nemo-ci-triage.yml")

    assert linear_status.modules_for_regex("^megatron-lm$", config) == [
        (
            linear_ci.LINEAR_MODULE,
            {
                "build_module": "megatron-lm",
                "team_key": "MCORE",
                "project_template": "MCore CI Testing",
                "enable_linear_open": True,
                "enable_linear_modify": True,
                "enable_linear_close": True,
            },
        )
    ]
    assert linear_write.write_gates(config) == {
        linear_ci.LINEAR_MODULE: {"open": True, "modify": True, "close": True}
    }


def test_notification_delegates_to_triage_package(monkeypatch, notify_module):
    notify = notify_module
    pipeline_jobs = [("functional:run_dev_dgx_h100", 101, [{"status": "failed"}])]
    sender = Mock()

    monkeypatch.setattr(notify, "WEBHOOK_URL", "https://slack.invalid/webhook")
    monkeypatch.setattr(notify, "SLACK_BOT_TOKEN", "")
    monkeypatch.setattr(notify, "SLACK_CHANNEL_ID", "")
    monkeypatch.setattr(notify, "PROJECT_URL", "https://ci.example.com/ADLR/megatron-lm")
    monkeypatch.setattr(notify, "get_project", Mock())
    monkeypatch.setattr(notify, "get_pipeline_jobs", lambda *_args, **_kwargs: pipeline_jobs)
    monkeypatch.setattr(notify.notification, "send_slack_notification", sender)

    result = CliRunner().invoke(
        notify.main,
        [
            "--pipeline-id",
            "123",
            "--check-for",
            "functional-tests",
            "--pipeline-context",
            "mr",
            "--pipeline-created-at",
            "2026-07-12T00:00:00Z",
        ],
    )

    assert result.exit_code == 0, result.output
    sender.assert_called_once_with(
        "megatron-lm",
        "mr",
        pipeline_jobs,
        None,
        webhook_url="https://slack.invalid/webhook",
        slack_bot_token=None,
        slack_channel_id=None,
        config=notify.TRIAGE_CONFIG,
    )


def test_notification_records_bot_thread_context(monkeypatch, tmp_path, notify_module):
    notify = notify_module
    pipeline_jobs = [("functional:run_dev_dgx_h100", 101, [{"status": "failed"}])]
    sender = Mock(return_value="1712345678.000100")
    slack_output = tmp_path / "slack_notification.json"

    monkeypatch.setattr(notify, "WEBHOOK_URL", "")
    monkeypatch.setattr(notify, "SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr(notify, "SLACK_CHANNEL_ID", "C0123456789")
    monkeypatch.setattr(notify, "get_project", Mock())
    monkeypatch.setattr(notify, "get_pipeline_jobs", lambda *_args, **_kwargs: pipeline_jobs)
    monkeypatch.setattr(notify.notification, "send_slack_notification", sender)

    result = CliRunner().invoke(
        notify.main,
        [
            "--pipeline-id",
            "123",
            "--check-for",
            "functional-tests",
            "--pipeline-context",
            "mr",
            "--pipeline-created-at",
            "2026-07-12T00:00:00Z",
            "--slack-output",
            str(slack_output),
        ],
    )

    assert result.exit_code == 0, result.output
    sender.assert_called_once_with(
        "megatron-lm",
        "mr",
        pipeline_jobs,
        None,
        webhook_url=None,
        slack_bot_token="xoxb-test",
        slack_channel_id="C0123456789",
        config=notify.TRIAGE_CONFIG,
    )
    assert json.loads(slack_output.read_text()) == {
        "channel_id": "C0123456789",
        "thread_timestamp": "1712345678.000100",
    }


def test_notification_writes_linear_inputs_without_webhook(monkeypatch, tmp_path, notify_module):
    notify = notify_module
    project = Mock()
    pipeline_jobs = [("functional:run_dev_dgx_h100", 101, [])]
    writer = Mock()

    monkeypatch.setattr(notify, "WEBHOOK_URL", "")
    monkeypatch.setattr(notify, "SLACK_BOT_TOKEN", "")
    monkeypatch.setattr(notify, "SLACK_CHANNEL_ID", "")
    monkeypatch.setattr(notify, "get_project", lambda: project)
    monkeypatch.setattr(notify, "get_pipeline_jobs", lambda *_args, **_kwargs: pipeline_jobs)
    monkeypatch.setattr(notify.linear_ci, "write_pipeline_reports", writer)
    summaries = tmp_path / "pipeline_summaries.json"
    buckets = tmp_path / "failure_buckets.json"

    result = CliRunner().invoke(
        notify.main,
        [
            "--pipeline-id",
            "123",
            "--check-for",
            "functional-tests",
            "--pipeline-context",
            "nightly",
            "--pipeline-created-at",
            "2026-07-12T00:00:00Z",
            "--summary-output",
            str(summaries),
            "--failure-buckets-output",
            str(buckets),
        ],
    )

    assert result.exit_code == 0, result.output
    writer.assert_called_once_with(
        123, "nightly", pipeline_jobs, project, notify.PROJECT_URL, summaries, buckets
    )
