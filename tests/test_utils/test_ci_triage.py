# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner

from tests.test_utils.python_scripts import generate_jet_trigger_job, recipe_parser


@pytest.fixture
def notify_module():
    pytest.importorskip("nemo_ci_triage.slack_notification")
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


def test_notification_delegates_to_triage_package(monkeypatch, notify_module):
    notify = notify_module
    pipeline_jobs = [("functional:run_dev_dgx_h100", 101, [{"status": "failed"}])]
    sender = Mock()

    monkeypatch.setattr(notify, "WEBHOOK_URL", "https://slack.invalid/webhook")
    monkeypatch.setattr(notify, "PROJECT_URL", "https://ci.example.com/ADLR/megatron-lm")
    monkeypatch.setattr(notify, "get_pipeline_jobs", lambda *_args: pipeline_jobs)
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
    assert (
        notify.notification.JOB_URL_TEMPLATE == "https://ci.example.com/ADLR/megatron-lm/-/jobs/{}"
    )
    assert (
        notify.notification.PIPELINE_URL_TEMPLATE
        == "https://ci.example.com/ADLR/megatron-lm/-/pipelines/{}"
    )
    sender.assert_called_once_with(
        "megatron-lm", "mr", pipeline_jobs, None, webhook_url="https://slack.invalid/webhook"
    )
