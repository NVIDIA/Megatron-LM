# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Report failed non-test jobs in the root GitLab pipeline through Cerno."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import gitlab
from cerno.config import load_config
from cerno.health.cli import write_report
from cerno.health.config import notification as validate_notification
from cerno.health.gitlab import latest_attempts
from cerno.health.notify import deliver

TEST_STAGES = {"test", "integration_tests", "functional_tests"}


def collect_failures(pipeline: Any, report: dict[str, Any], route: dict[str, Any]) -> None:
    """Collect latest failed root jobs, retaining failures if another API read fails."""
    failures = defaultdict(list)
    # Do not follow test bridges: their descendants have their own stage names.
    for resource, manager in (("jobs", pipeline.jobs), ("bridges", pipeline.bridges)):
        try:
            items = latest_attempts(manager.list(get_all=True))
        except gitlab.GitlabError as exc:
            report["errors"].append(f"Cannot read {resource}: {type(exc).__name__}")
            continue
        for job in items:
            if job.status == "failed" and job.stage not in TEST_STAGES:
                failures[job.stage].append(
                    {"name": job.name, "status": job.status, "web_url": job.web_url, "id": job.id}
                )

    report["checks"] = [
        {
            "name": f"{stage} failures",
            "status": "violation",
            "pipeline_path": "root",
            "violations": [f"{len(jobs)} job(s) failed in stage {stage}"],
            "jobs": jobs,
            "notify": route,
        }
        for stage, jobs in sorted(failures.items())
    ]
    report["status"] = "violation" if failures else "incomplete" if report["errors"] else "healthy"


@click.command()
@click.option("--pipeline-id", required=True, type=int)
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default="pipeline_health.json")
@click.option("--dry-run", is_flag=True, help="Preview alerts without contacting Slack.")
def main(pipeline_id: int, config_path: Path, output: Path, dry_run: bool) -> None:
    """Alert on actual failures without treating skipped or absent stages as failures."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "module": "megatron_lm",
        "pipeline_id": pipeline_id,
        "status": "error",
        "checks": [],
        "deliveries": [],
        "errors": [],
    }
    try:
        config = load_config(config_path, required=True)
        route = validate_notification(config["modules"]["megatron_lm"]["health"]["notify"])
        token = os.getenv("RO_API_TOKEN")
        if not token:
            raise ValueError("RO_API_TOKEN is required")
        client = gitlab.Gitlab(
            os.getenv("CI_SERVER_URL", "https://gitlab-master.nvidia.com"),
            private_token=token,
            timeout=30,
            retry_transient_errors=True,
        )
        project_id = config["gitlab"]["project_id"]
        pipeline = client.projects.get(project_id).pipelines.get(pipeline_id)
        report.update(project_id=project_id, pipeline_url=pipeline.web_url)
        collect_failures(pipeline, report, route)
        write_report(output, report)
        deliver(report, config, dry_run=dry_run)
    except Exception as exc:  # Persist sanitized diagnostics even when setup or delivery fails.
        report["status"] = "error"
        report["errors"].append(type(exc).__name__)
    finally:
        write_report(output, report)

    click.echo(f"Pipeline health: {report['status']}; report: {output}")
    if report["errors"] or any(item["status"] == "failed" for item in report["deliveries"]):
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    main()
