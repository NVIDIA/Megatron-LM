import logging
import os

import click
import gitlab
import pandas as pd
import requests

PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
DASHBOARD_ENDPOINT = os.getenv("DASHBOARD_ENDPOINT")

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle():
    return gitlab.Gitlab(
        f"https://{os.getenv('GITLAB_ENDPOINT')}", private_token=os.getenv("RO_API_TOKEN")
    )


def get_build_analytics(pipeline_id: int) -> pd.DataFrame:
    pipeline_jobs = (
        get_gitlab_handle()
        .projects.get(PROJECT_ID)
        .pipelines.get(pipeline_id)
        .jobs.list(get_all=True)
    )

    return pd.DataFrame(
        [
            {
                "name": pipeline_job.name,
                "started_at": pipeline_job.attributes['started_at'],
                "finished_at": pipeline_job.attributes['finished_at'],
            }
            for pipeline_job in pipeline_jobs
            if pipeline_job.name.startswith("test:build_image: [CI")
        ]
    )


def get_unit_test_analytics(pipeline_id: int) -> pd.DataFrame:
    pipeline = get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id)
    unit_test_pipeline_bridges = [
        pipeline_bridge
        for pipeline_bridge in pipeline.bridges.list()
        if pipeline_bridge.name.startswith("test:unit_tests")
        and pipeline_bridge.downstream_pipeline is not None
    ]

    return pd.DataFrame(
        [
            {
                "name": pipeline_bridge.name,
                "started_at": pipeline_bridge.attributes['started_at'],
                "finished_at": pipeline_bridge.attributes['finished_at'],
            }
            for pipeline_bridge in unit_test_pipeline_bridges
        ]
    )


def get_functional_test_analytics(pipeline_id: int, type: str) -> pd.DataFrame:
    pipeline = get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id)
    functional_test_pipeline_bridges = [
        pipeline_bridge
        for pipeline_bridge in pipeline.bridges.list()
        if pipeline_bridge.name.startswith(type) and pipeline_bridge.downstream_pipeline is not None
    ]

    return pd.DataFrame(
        [
            {
                "name": pipeline_bridge.name,
                "started_at": pipeline_bridge.attributes['started_at'],
                "finished_at": pipeline_bridge.attributes['finished_at'],
            }
            for pipeline_bridge in functional_test_pipeline_bridges
        ]
    )


def get_failed_stage(pipeline_id: int) -> pd.DataFrame:
    pipeline_bridges = (
        get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id).bridges.list()
    )

    pipeline_jobs = (
        get_gitlab_handle()
        .projects.get(PROJECT_ID)
        .pipelines.get(pipeline_id)
        .jobs.list(get_all=True)
    ) + [
        pipeline_bridge
        for pipeline_bridge in pipeline_bridges
        if pipeline_bridge.downstream_pipeline is not None
    ]

    failed_stages = list(
        set(
            [
                job.stage
                for job in pipeline_jobs
                if job.status == "failed" and job.allow_failure is False
            ]
        )
    )

    if "build" in failed_stages:
        return "build"

    elif "test" in failed_stages:
        return "test"

    elif "integration_tests" in failed_stages:
        return "integration_tests"

    elif "functional_tests" in failed_stages:
        return "functional_tests"

    return


def get_analytics_per_pipeline(pipeline_id: int) -> pd.DataFrame:
    build_analytics = get_build_analytics(pipeline_id)
    unit_tests_analytics = get_unit_test_analytics(pipeline_id)
    integration_tests_analytics = get_functional_test_analytics(pipeline_id, "integration")
    functional_tests_analytics = get_functional_test_analytics(pipeline_id, "functional")
    failed_stage = get_failed_stage(pipeline_id)

    analytics = {"mcore_analytics": "v0.2", "pipeline_id": pipeline_id}
    analytics["build_stage_failed"] = int(failed_stage == "build")
    analytics["unit_tests_stage_failed"] = int(failed_stage == "test")
    analytics["integration_tests_stage_failed"] = int(failed_stage == "integration_tests")

    if not build_analytics.empty:
        analytics["ci_started_at"] = build_analytics['started_at'].min()
        analytics["build_started_at"] = build_analytics['started_at'].min()
        analytics["build_finished_at"] = build_analytics['finished_at'].max()
        analytics["build_duration_total"] = (
            pd.Timestamp(build_analytics['finished_at'].max())
            - pd.Timestamp(build_analytics['started_at'].min())
        ).total_seconds()
        analytics["functional_tests_stage_failed"] = int(failed_stage == "functional_tests")

    if not unit_tests_analytics.empty:
        analytics["unit_tests_started_at"] = unit_tests_analytics['started_at'].min()
        analytics["unit_tests_finished_at"] = unit_tests_analytics['finished_at'].max()
        analytics["unit_tests_duration_total"] = (
            pd.Timestamp(unit_tests_analytics['finished_at'].max())
            - pd.Timestamp(unit_tests_analytics['started_at'].min())
        ).total_seconds()

    if not integration_tests_analytics.empty:

        analytics["integration_tests_started_at"] = integration_tests_analytics['started_at'].min()
        analytics["integration_tests_finished_at"] = integration_tests_analytics[
            'finished_at'
        ].max()
        analytics["integration_tests_duration_total"] = (
            pd.Timestamp(integration_tests_analytics['finished_at'].max())
            - pd.Timestamp(integration_tests_analytics['started_at'].min())
        ).total_seconds()

    if not functional_tests_analytics.empty:
        analytics["functional_tests_started_at"] = functional_tests_analytics['started_at'].min()
        analytics["functional_tests_finished_at"] = functional_tests_analytics['finished_at'].max()
        analytics["functional_tests_duration_total"] = (
            pd.Timestamp(functional_tests_analytics['finished_at'].max())
            - pd.Timestamp(functional_tests_analytics['started_at'].min())
        ).total_seconds()

    return pd.DataFrame([analytics])


@click.command()
@click.option("--pipeline-id", required=True, type=int, help="PipelineID")
def upload_statistics(pipeline_id: int):
    payload = get_analytics_per_pipeline(pipeline_id).to_json(orient="records")

    logger.info(payload)

    res = requests.post(
        DASHBOARD_ENDPOINT,
        data=payload,
        headers={'Content-Type': 'application/json', 'Accept-Charset': 'UTF-8'},
    )

    if not res.ok:
        raise requests.exceptions.HTTPError(
            f"Failed to make POST request. Received response: {res.status_code}"
        )


if __name__ == "__main__":
    upload_statistics()
