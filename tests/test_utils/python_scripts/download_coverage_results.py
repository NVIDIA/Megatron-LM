import glob
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
def main(pipeline_id: int):
    logging.basicConfig(level=logging.INFO)
    logger.info('Started')

    gl = gitlab.Gitlab(
        f"https://{os.getenv('GITLAB_ENDPOINT')}", private_token=os.getenv("RO_API_TOKEN")
    )

    project = gl.projects.get(PROJECT_ID)
    pipeline = project.pipelines.get(pipeline_id)
    print(pipeline.bridges.list())

    pipeline_bridges = [
        pipeline_bridge
        for pipeline_bridge in pipeline.bridges.list()
        if pipeline_bridge.name.startswith("test:unit_tests")
        and pipeline_bridge.downstream_pipeline is not None
    ]

    ASSETS_DIR = pathlib.Path("tmp") / "results" / "iteration=0"
    for pipeline_bridge in pipeline_bridges:
        functional_pipeline = project.pipelines.get(pipeline_bridge.downstream_pipeline['id'])

        functional_pipeline_jobs = functional_pipeline.jobs.list(get_all=True)
        if "legacy" in pipeline_bridge.name:
            continue

        logger.info("Starting with pipeline %s", pipeline_bridge.name)
        for functional_pipeline_job in functional_pipeline_jobs:
            job = project.jobs.get(functional_pipeline_job.id)
            logger.info("Starting with job %s", job.name)

            try:
                file_name = '__artifacts.zip'
                with open(file_name, "wb") as f:
                    job.artifacts(streamed=True, action=f.write)
                zip = zipfile.ZipFile(file_name)
                zip.extractall("tmp")
                logger.info("Downloaded artifacts of job %s", job.name)
            except Exception:
                continue

            os.unlink(file_name)
            restart_dir = os.listdir(pathlib.Path("tmp") / "results" / "iteration=0")[-1]
            coverage_report_source = list(
                glob.glob(
                    str(
                        pathlib.Path(ASSETS_DIR)
                        / f"{restart_dir}"
                        / "assets"
                        / "basic"
                        / "*"
                        / "coverage_report"
                    )
                )
            )[0]

            coverage_report_target = (
                pathlib.Path("coverage_results") / job.name.replace("/", "-") / "coverage_report"
            )

            if pathlib.Path(coverage_report_source).exists():
                pathlib.Path(coverage_report_target.parent).mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Move artifacts from %s to %s", coverage_report_source, coverage_report_target
                )

                shutil.move(coverage_report_source, coverage_report_target)
            else:
                logger.info(
                    "coverage_report for %s does not exist. Skip.", str(f"{job.stage} / {job.name}")
                )

            shutil.rmtree("tmp")

    logger.info("beep boop: All done!")


if __name__ == "__main__":
    main()
