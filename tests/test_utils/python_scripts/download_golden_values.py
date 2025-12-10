# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import os
import pathlib
import shutil
import zipfile

import click
import gitlab

BASE_PATH = pathlib.Path(__file__).parent.resolve()
PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))

logger = logging.getLogger(__name__)


@click.command()
@click.option("--pipeline-id", required=True, type=int, help="Pipeline ID")
@click.option(
    "--only-failing/--no-only-failing",
    default=False,
    help="Only download artifacts from failing jobs",
)
def main(pipeline_id: int, only_failing: bool):
    logging.basicConfig(level=logging.INFO)
    logger.info('Started')

    gitlab_endpoint = os.getenv('GITLAB_ENDPOINT')
    ro_api_token = os.getenv('RO_API_TOKEN')

    if not gitlab_endpoint or not ro_api_token:
        raise Exception(
            "Environment variables {GITLAB_ENDPOINT} and {RO_API_TOKEN} have not been set. ie. GITLAB_ENDPOINT=<gitlab-endpoint>, RO_API_TOKEN=<gitlab-token>"
        )

    gl = gitlab.Gitlab(f"https://{gitlab_endpoint}", private_token=ro_api_token)
    logger.info("Setting only_failing to %s", only_failing)

    project = gl.projects.get(PROJECT_ID)
    pipeline = project.pipelines.get(pipeline_id)
    print(pipeline.bridges.list(get_all=True))

    pipeline_bridges = [
        pipeline_bridge
        for pipeline_bridge in pipeline.bridges.list(get_all=True)
        if pipeline_bridge.name.startswith("functional")
        and pipeline_bridge.downstream_pipeline is not None
    ]

    ASSETS_DIR = pathlib.Path("tmp") / "results" / "iteration=0"
    for pipeline_bridge in pipeline_bridges:
        functional_pipeline = project.pipelines.get(pipeline_bridge.downstream_pipeline['id'])
        environment = pipeline_bridge.name[len("functional:run_") :]
        functional_pipeline_jobs = functional_pipeline.jobs.list(get_all=True)
        logger.info("Starting with pipeline %s", pipeline_bridge.name)
        for functional_pipeline_job in functional_pipeline_jobs:
            job = project.jobs.get(functional_pipeline_job.id)
            logger.info("Starting with job %s", job.name)
            if only_failing and job.status == "success":
                logger.info("Job %s is successful. Skipping.", job.name)
                continue

            try:
                file_name = '__artifacts.zip'
                with open(file_name, "wb") as f:
                    job.artifacts(streamed=True, action=f.write)
                zip = zipfile.ZipFile(file_name)
                zip.extractall("tmp")
                logger.info("Downloaded artifacts of job %s", job.name)
            except Exception as e:
                logger.error("Failed to download artifacts of job %s due to %s", job.name, e)
                continue

            os.unlink(file_name)
            restart_dir = os.listdir(pathlib.Path("tmp") / "results" / "iteration=0")[-1]
            golden_values_sources = list(
                (
                    pathlib.Path(ASSETS_DIR)
                    / f"{restart_dir}"
                    / "assets"
                    / "basic"
                    / f"{job.name.replace('_', '-').lower()}-{environment.replace('_', '-')}"
                ).glob("g*.json")
            )

            if len(golden_values_sources) == 1:
                golden_values_source = golden_values_sources[0]
            else:
                logger.info(
                    "Golden values for %s does not exist. Skip.", str(golden_values_sources)
                )
                continue

            golden_values_source_name = golden_values_source.name
            golden_values_source_name = golden_values_source_name.replace(
                "generations", "golden_values"
            )

            golden_values_target = (
                pathlib.Path("tests")
                / "functional_tests"
                / 'test_cases'
                / job.stage
                / job.name
                / golden_values_source_name
            )

            if golden_values_source.exists():
                pathlib.Path(golden_values_target.parent).mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Move artifacts from %s to %s", golden_values_source, golden_values_target
                )

                shutil.move(golden_values_source, golden_values_target)
            else:
                logger.info("Golden values for %s does not exist. Skip.", str(golden_values_source))

            shutil.rmtree("tmp")

    logger.info("beep boop: All done!")


if __name__ == "__main__":
    main()
