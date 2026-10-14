# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise failure selection and notification errors without live services."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

pytest.importorskip("cerno.health")

from tests.test_utils.python_scripts import notify_ci_failures

CONFIG = Path(__file__).resolve().parents[2] / ".gitlab/cerno-health.yml"


def job(job_id, name, stage, status, **extra):
    """Build a GitLab job with realistic status and retry metadata."""
    return SimpleNamespace(
        id=job_id,
        name=name,
        stage=stage,
        status=status,
        web_url=f"https://ci.example.com/jobs/{job_id}",
        **extra,
    )


@pytest.fixture
def pipeline(monkeypatch):
    """Replace metadata reads while retaining Cerno's actual dry-run delivery."""
    root = Mock(web_url="https://ci.example.com/pipelines/123")
    root.jobs.list.return_value = []
    root.bridges.list.return_value = []
    client = Mock()
    client.projects.get.return_value.pipelines.get.return_value = root
    monkeypatch.setattr(notify_ci_failures.gitlab, "Gitlab", Mock(return_value=client))
    monkeypatch.setenv("RO_API_TOKEN", "test-token")
    return root


def run_observer(tmp_path, *extra):
    """Run the CLI and return its persisted report."""
    output = tmp_path / "health.json"
    result = CliRunner().invoke(
        notify_ci_failures.main,
        ["--pipeline-id", "123", "--config", str(CONFIG), "--output", str(output), *extra],
    )
    return result, json.loads(output.read_text())


def test_failed_non_test_stages_consolidate_into_one_alert(pipeline, tmp_path):
    pipeline.jobs.list.return_value = [
        job(1, "pre:setup", ".pre", "failed"),
        job(2, "test:pre_build_image: [dev, arm64]", "build", "failed"),
        job(3, "publish:docs", "publish", "failed", allow_failure=True),
        job(4, "triage:linear_write", "triage", "failed", allow_failure=True),
        job(5, "future:cleanup", ".post", "failed"),
        job(6, "test:unit_tests", "test", "failed"),
        job(7, "integration:configure", "integration_tests", "failed"),
        job(8, "functional:configure", "functional_tests", "failed"),
    ]
    pipeline.bridges.list.return_value = [
        job(9, "functional:run_dev", "functional_tests", "failed"),
        job(10, "publish:downstream", "publish", "failed"),
    ]
    result, report = run_observer(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert report["status"] == "violation"
    assert {j["id"] for check in report["checks"] for j in check["jobs"]} == {1, 2, 3, 4, 5, 10}
    assert len(report["deliveries"]) == 1
    alert = report["deliveries"][0]
    assert alert["status"] == "preview"
    assert alert["target"] == "C074W9J7S0N"
    assert "<!subteam^S06GU680R3N>" in alert["text"]
    assert "https://ci.example.com/jobs/2" in alert["text"]
    pipeline.jobs.list.assert_called_once_with(get_all=True)
    pipeline.bridges.list.assert_called_once_with(get_all=True)


def test_recovered_retries_and_nonfailure_statuses_do_not_alert(pipeline, tmp_path):
    pipeline.jobs.list.return_value = [
        job(1, "build", "build", "failed", retried=True),
        job(2, "build", "build", "success"),
        job(3, "test", "test", "failed"),
        *[
            job(index, status, "publish", status)
            for index, status in enumerate(["skipped", "manual", "canceled", "running"], 4)
        ],
    ]
    # Bridge listings may contain old attempts without a retried marker.
    pipeline.bridges.list.return_value = [
        job(20, "publish:child", "publish", "success"),
        job(19, "publish:child", "publish", "failed"),
    ]
    result, report = run_observer(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert report["status"] == "healthy"
    assert report["deliveries"] == []


@pytest.mark.parametrize("has_failure", [False, True])
def test_api_errors_remain_visible_without_hiding_known_failures(pipeline, tmp_path, has_failure):
    if has_failure:
        pipeline.jobs.list.return_value = [job(1, "build", "build", "failed")]
    pipeline.bridges.list.side_effect = notify_ci_failures.gitlab.GitlabListError("secret response")
    result, report = run_observer(tmp_path, "--dry-run")
    assert result.exit_code == 1
    assert report["status"] == ("violation" if has_failure else "incomplete")
    assert len(report["deliveries"]) == int(has_failure)
    assert report["errors"] == ["Cannot read bridges: GitlabListError"]
    assert "secret response" not in json.dumps(report)


def test_delivery_failure_preserves_evidence_and_exits_nonzero(pipeline, tmp_path, monkeypatch):
    pipeline.jobs.list.return_value = [job(1, "build", "build", "failed")]

    def fail_delivery(report, _config, **_kwargs):
        report["deliveries"].append({"status": "failed", "target": "test-channel"})

    monkeypatch.setattr(notify_ci_failures, "deliver", fail_delivery)
    result, report = run_observer(tmp_path)
    assert result.exit_code == 1
    assert report["checks"][0]["jobs"][0]["id"] == 1
    assert report["deliveries"][0]["status"] == "failed"


def test_missing_credentials_write_an_error_report(pipeline, tmp_path, monkeypatch):
    monkeypatch.delenv("RO_API_TOKEN")
    result, report = run_observer(tmp_path, "--dry-run")
    assert result.exit_code == 1
    assert report["status"] == "error"
    assert report["deliveries"] == []
    pipeline.jobs.list.assert_not_called()
