# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import logging
import os
import pathlib
import re
import shutil
import zipfile

import click
import gitlab
import requests

BASE_PATH = pathlib.Path(__file__).parent.resolve()
PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY", "NVIDIA/Megatron-LM")

logger = logging.getLogger(__name__)


def download_from_gitlab(pipeline_id: int, only_failing: bool):
    """Download golden values from Gitlab pipeline."""
    gitlab_endpoint = os.getenv("GITLAB_ENDPOINT")
    ro_api_token = os.getenv("RO_API_TOKEN")

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
        functional_pipeline = project.pipelines.get(pipeline_bridge.downstream_pipeline["id"])
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
                file_name = "__artifacts.zip"
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

            if len(golden_values_sources) < 1:
                logger.info(
                    "Golden values for %s does not exist. Skip.", str(golden_values_sources)
                )
                continue

            for golden_values_source in golden_values_sources:
                golden_values_source_name = golden_values_source.name
                golden_values_source_name = golden_values_source_name.replace(
                    "generations", "golden_values"
                )

                golden_values_target = (
                    pathlib.Path("tests")
                    / "functional_tests"
                    / "test_cases"
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
                    logger.info(
                        "Golden values for %s does not exist. Skip.", str(golden_values_source)
                    )

            shutil.rmtree("tmp")

    logger.info("beep boop: All done!")


def _setup_github_headers(github_token: str) -> dict:
    """Set up headers for GitHub API requests."""
    return {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _fetch_workflow_run(workflow_id: int, headers: dict, api_base: str) -> dict:
    """Fetch workflow run details from GitHub."""
    logger.info("Fetching workflow run %d from %s", workflow_id, GITHUB_REPO)
    workflow_run_url = f"{api_base}/repos/{GITHUB_REPO}/actions/runs/{workflow_id}"
    response = requests.get(workflow_run_url, headers=headers)
    response.raise_for_status()
    workflow_run = response.json()

    logger.info(
        "Workflow run status: %s, conclusion: %s",
        workflow_run["status"],
        workflow_run["conclusion"],
    )
    return workflow_run


def _fetch_all_jobs(workflow_id: int, headers: dict, api_base: str) -> list:
    """Fetch all jobs from the workflow run with pagination."""
    logger.info("Fetching all jobs from workflow run...")
    all_jobs = []
    jobs_url = f"{api_base}/repos/{GITHUB_REPO}/actions/runs/{workflow_id}/jobs"
    page = 1
    per_page = 100

    while True:
        response = requests.get(
            jobs_url, headers=headers, params={"page": page, "per_page": per_page}
        )
        response.raise_for_status()
        jobs_data = response.json()
        jobs = jobs_data["jobs"]

        if not jobs:
            break

        all_jobs.extend(jobs)
        logger.info("Fetched page %d with %d jobs", page, len(jobs))

        if len(jobs) < per_page:
            break

        page += 1

    logger.info("Total jobs fetched: %d", len(all_jobs))
    return all_jobs


def _filter_cicd_jobs(all_jobs: list) -> list:
    """Filter jobs that belong to cicd-integration-tests-latest."""
    # Try to filter by workflow_name field
    cicd_jobs = [
        job
        for job in all_jobs
        if "workflow_name" in job
        and "cicd-integration-tests-latest" in job.get("workflow_name", "").lower()
    ]

    # If workflow_name field doesn't exist, try filtering by job labels
    if not cicd_jobs:
        logger.info("No jobs found with workflow_name field. Checking job labels...")
        cicd_jobs = [
            job
            for job in all_jobs
            if "labels" in job
            and any(
                "cicd-integration-tests-latest" in str(label).lower() for label in job["labels"]
            )
        ]

    # If still no jobs found, use all jobs as fallback
    if not cicd_jobs:
        logger.warning(
            "Could not identify jobs belonging to cicd-integration-tests-latest. "
            "Using all jobs from workflow run as fallback."
        )
        cicd_jobs = all_jobs
    else:
        logger.info("Filtered %d jobs belonging to cicd-integration-tests-latest", len(cicd_jobs))

    return cicd_jobs


def _create_job_name_map(cicd_jobs: list) -> dict:
    """Create a mapping of normalized job names to job data for matching with artifacts."""
    job_name_map = {}
    for job in cicd_jobs:
        job_name = job["name"]

        # Extract test name from job name (between '/' and ' - ')
        if "/" in job_name and " - " in job_name:
            # Format: "stage/test_name - version"
            test_name = job_name.split("/", 1)[1].split(" - ", 1)[0]
        elif "/" in job_name:
            # Format: "stage/test_name"
            test_name = job_name.split("/", 1)[1]
        else:
            # No stage prefix, use the whole name
            test_name = job_name.split(" - ", 1)[0] if " - " in job_name else job_name

        # Normalize for matching (lowercase, preserve underscores)
        normalized_name = test_name.lower()
        job_name_map[normalized_name] = job
        logger.info("Job '%s' normalized to '%s'", job["name"], normalized_name)

    return job_name_map


def _match_artifact_to_job(artifact: dict, workflow_id: int, job_name_map: dict) -> dict:
    """Match an artifact to a job from the job name map."""
    artifact_job_name = artifact["name"].lower().replace("logs-", "").split(f"-{workflow_id}")[0]
    for normalized_job_name, job in job_name_map.items():
        if normalized_job_name == artifact_job_name:
            logger.info("Artifact '%s' matched to job '%s'", artifact["name"], job["name"])
            return job

    logger.info(
        "Artifact '%s' does not match any cicd-integration-tests-latest jobs", artifact["name"]
    )
    return None


def _fetch_and_filter_artifacts(
    workflow_id: int, job_name_map: dict, only_failing: bool, headers: dict, api_base: str
) -> list:
    """Fetch and filter artifacts from the workflow run."""
    logger.info("Fetching and filtering artifacts from workflow run...")
    filtered_artifacts = []
    artifacts_url = f"{api_base}/repos/{GITHUB_REPO}/actions/runs/{workflow_id}/artifacts"
    page = 1
    per_page = 100

    while True:
        response = requests.get(
            artifacts_url, headers=headers, params={"page": page, "per_page": per_page}
        )
        response.raise_for_status()
        artifacts_data = response.json()
        artifacts = artifacts_data["artifacts"]

        if not artifacts:
            break

        logger.info("Fetched artifacts page %d with %d artifacts", page, len(artifacts))

        # Filter artifacts that match cicd jobs
        for artifact in artifacts:
            logger.info("Checking artifact: %s", artifact["name"])

            matched_job = _match_artifact_to_job(artifact, workflow_id, job_name_map)

            if matched_job:
                # Check if we should skip based on only_failing flag
                if only_failing and matched_job.get("conclusion") == "success":
                    logger.info(
                        "Job '%s' succeeded and only-failing is set. Skipping artifact '%s'",
                        matched_job["name"],
                        artifact["name"],
                    )
                    continue

                artifact["_matched_job"] = matched_job
                filtered_artifacts.append(artifact)

        if len(artifacts) < per_page:
            break

        page += 1

    logger.info(
        "Filtered %d artifacts matching cicd-integration-tests-latest jobs", len(filtered_artifacts)
    )
    return filtered_artifacts


def _extract_stage_and_test_name(job_name: str) -> tuple:
    """Extract stage and test name from job name."""
    if "/" in job_name:
        # Format: "stage/test_name - version" or "stage/test_name"
        stage, rest = job_name.split("/", 1)
        # Remove version suffix if present (e.g., " - latest")
        test_name = rest.split(" - ", 1)[0] if " - " in rest else rest
    else:
        # No stage in job name - this shouldn't happen for integration tests
        raise ValueError(
            f"Job name '{job_name}' does not contain a stage (expected format: 'stage/test_name - version')"
        )

    logger.info(f"Extracted stage: {stage}, test name: {test_name} from job '{job_name}'")
    return stage, test_name


def _download_and_extract_artifact(
    artifact: dict, headers: dict, api_base: str, file_name: str = "__artifacts.zip"
) -> list:
    """Download and extract an artifact, returning list of golden values files."""
    artifact_download_url = f"{api_base}/repos/{GITHUB_REPO}/actions/artifacts/{artifact['id']}/zip"

    logger.info("Downloading artifact from %s", artifact_download_url)
    response = requests.get(artifact_download_url, headers=headers, stream=True)
    response.raise_for_status()

    with open(file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the artifact
    zip_file = zipfile.ZipFile(file_name)
    zip_file.extractall("tmp")
    logger.info("Downloaded and extracted artifact %s", artifact["name"])

    os.unlink(file_name)

    # For GitHub, golden values are always in the top level of tmp
    tmp_dir = pathlib.Path("tmp")
    golden_values_in_tmp = list(tmp_dir.glob("golden_values*.json"))

    if golden_values_in_tmp:
        logger.info(
            "Found %d golden values files in artifact %s",
            len(golden_values_in_tmp),
            artifact["name"],
        )
    else:
        logger.info("No golden values found in artifact %s. Skipping.", artifact["name"])

    return golden_values_in_tmp


def _move_golden_values(golden_values_files: list, stage: str, test_name: str) -> int:
    """Move golden values files to their target location."""
    golden_values_moved = 0

    for golden_values_file in golden_values_files:
        golden_values_target = (
            pathlib.Path("tests")
            / "functional_tests"
            / "test_cases"
            / stage
            / test_name
            / golden_values_file.name
        )

        pathlib.Path(golden_values_target.parent).mkdir(parents=True, exist_ok=True)
        logger.info(f"Moving golden values from {golden_values_file} to {golden_values_target}")
        shutil.move(golden_values_file, golden_values_target)
        golden_values_moved += 1

    return golden_values_moved


def _cleanup_tmp():
    """Clean up temporary directory."""
    if pathlib.Path("tmp").exists():
        shutil.rmtree("tmp")


def _process_artifact(artifact: dict, headers: dict, api_base: str) -> int:
    """Process a single artifact (download, extract, and move golden values)."""
    matched_job = artifact.get("_matched_job")
    logger.info(
        "Processing artifact: %s (matched to job: %s)",
        artifact["name"],
        matched_job["name"] if matched_job else "unknown",
    )

    file_name = "__artifacts.zip"

    try:
        # Download and extract the artifact
        golden_values_files = _download_and_extract_artifact(artifact, headers, api_base, file_name)

        if not golden_values_files:
            _cleanup_tmp()
            return 0

        # Extract stage and test name from the matched job name
        stage, test_name = _extract_stage_and_test_name(matched_job["name"])

        # Move golden values to target location
        golden_values_moved = _move_golden_values(golden_values_files, stage, test_name)

        logger.info(
            f"Successfully moved {golden_values_moved} golden values for test: {stage}/{test_name}"
        )

        # Clean up tmp directory after processing this artifact
        _cleanup_tmp()

        return golden_values_moved

    except Exception as e:
        logger.error("Failed to download/process artifact %s due to %s", artifact["name"], e)
        # Clean up on error
        if os.path.exists(file_name):
            os.unlink(file_name)
        _cleanup_tmp()
        return 0


def download_from_github(workflow_id: int, only_failing: bool):
    """Download golden values from Github workflow run."""
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        raise Exception(
            "Environment variable {GITHUB_TOKEN} has not been set. ie. GITHUB_TOKEN=<github-token>"
        )

    api_base = "https://api.github.com"
    headers = _setup_github_headers(github_token)

    # Fetch workflow run details
    _fetch_workflow_run(workflow_id, headers, api_base)
    logger.info("Setting only_failing to %s", only_failing)

    # Fetch and filter jobs
    all_jobs = _fetch_all_jobs(workflow_id, headers, api_base)
    cicd_jobs = _filter_cicd_jobs(all_jobs)
    job_name_map = _create_job_name_map(cicd_jobs)

    # Fetch and filter artifacts
    artifacts = _fetch_and_filter_artifacts(
        workflow_id, job_name_map, only_failing, headers, api_base
    )

    # Process each artifact
    total_golden_values = 0
    total_tests_with_golden_values = 0

    for artifact in artifacts:
        golden_values_moved = _process_artifact(artifact, headers, api_base)

        if golden_values_moved > 0:
            total_golden_values += golden_values_moved
            total_tests_with_golden_values += 1

    logger.info(f"Total tests with golden values: {total_tests_with_golden_values}")
    logger.info(f"Total golden values found: {total_golden_values}")
    logger.info("beep boop: All done!")


@click.command()
@click.option(
    "--source",
    type=click.Choice(["gitlab", "github"], case_sensitive=False),
    required=True,
    help="Source platform (gitlab or github)",
)
@click.option(
    "--pipeline-id", type=int, help="Gitlab Pipeline ID or Github Workflow Run ID", required=True
)
@click.option(
    "--only-failing/--no-only-failing",
    default=False,
    help="Only download artifacts from failing jobs",
)
def main(source: str, pipeline_id: int, only_failing: bool):
    """Download golden values from Gitlab or Github."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Started")

    source = source.lower()

    if source == "gitlab":
        download_from_gitlab(pipeline_id, only_failing)
    elif source == "github":
        download_from_github(pipeline_id, only_failing)
    else:
        raise click.UsageError(f"Unknown source: {source}")


if __name__ == "__main__":
    main()
