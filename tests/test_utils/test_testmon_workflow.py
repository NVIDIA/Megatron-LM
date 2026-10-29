# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def _step(workflow: str, job: str, step_id: str) -> dict:
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return next(step for step in document["jobs"][job]["steps"] if step.get("id") == step_id)


def _run(
    script: str, directory: Path, environment: dict[str, str]
) -> tuple[subprocess.CompletedProcess, dict]:
    output = directory / "outputs"
    output.unlink(missing_ok=True)
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input=script,
        cwd=directory,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(directory / "summary"),
            **environment,
        },
        text=True,
        capture_output=True,
    )
    values = (
        dict(line.split("=", 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, values


@pytest.fixture
def shell_environment(tmp_path: Path) -> dict[str, str]:
    if shutil.which("jq") is None:
        pytest.skip("Workflow shell validation requires jq")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # CI provides yq. Use PyYAML for that extraction locally and execute the
    # actual producer/build shell and jq programs unchanged.
    yq = binaries / "yq"
    yq.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, yaml\n"
        "assert sys.argv[1:3] == ['-o', 'json']\n"
        "if len(sys.argv) == 5:\n"
        "    assert sys.argv[3] == '[.products[].test_case[]]'\n"
        "else:\n"
        "    assert sys.argv[3] == "
        "'[.products[].test_case[] | { \"bucket\": .}] | sort_by(.model, .test_case)'\n"
        "with open(sys.argv[4]) if len(sys.argv) == 5 else sys.stdin as stream:\n"
        "    recipe = yaml.safe_load(stream)\n"
        "cases = [case for product in recipe['products'] for case in product['test_case']]\n"
        "print(json.dumps(cases if len(sys.argv) == 5 else [{'bucket': case} for case in cases]))\n"
    )
    yq.chmod(0o755)
    for platform in ("h100", "gb200"):
        relative = Path("tests/test_utils/recipes") / platform / "unit-tests.yaml"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True)
        destination.symlink_to(ROOT / relative)
    return {"PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}"}


@pytest.mark.parametrize(
    ("maintainer", "gb_enabled", "expected_platforms"),
    [
        ("true", "false", {"dgx_h100"}),
        ("false", "true", {"dgx_h100"}),
        ("true", "true", {"dgx_h100", "dgx_gb200"}),
    ],
)
def test_producer_consumes_the_actual_build_matrix(
    tmp_path: Path,
    shell_environment: dict[str, str],
    maintainer: str,
    gb_enabled: str,
    expected_platforms: set[str],
) -> None:
    build_step = _step("_build_ci_container.yml", "cicd-compute-build-matrix", "compute")
    build, build_outputs = _run(
        build_step["run"],
        tmp_path,
        {
            "IS_MAINTAINER": maintainer,
            "ENABLE_GB_TESTING": gb_enabled,
            "REGISTRY_AWS": "h100.example.test/team",
            "REGISTRY_GCP_H100": "gcp.example.test/team",
            "REGISTRY_GB_GPU": "gb200.example.test/team",
            "SELECTED_RUNNER": "h100-runner",
            "SELECTED_RUNNER_GCP_H100": "gcp-build-runner",
            "SELECTED_RUNNER_GB_GPU": "gb200-runner",
            "USE_GCP_H100": "false",
            "INCLUDE_GCP_H100": "false",
        },
    )
    assert build.returncode == 0, build.stderr
    sha = "1234567890abcdef" * 2 + "12345678"
    producer_step = _step("populate-build-cache.yml", "parse-unit-tests", "matrix")
    producer, outputs = _run(
        producer_step["run"],
        tmp_path,
        {**shell_environment, "BUILDS": build_outputs["matrix"], "SOURCE_SHA": sha},
    )
    assert producer.returncode == 0, producer.stderr
    matrix = json.loads(outputs["matrix"])
    assert {entry["platform"] for entry in matrix} == expected_platforms
    for platform, cloud, registry, runner in (
        ("dgx_h100", "aws-h100", "h100.example.test/team", "h100-runner"),
        ("dgx_gb200", "gb-gpu", "gb200.example.test/team", "gb200-runner"),
    ):
        entries = [entry for entry in matrix if entry["platform"] == platform]
        if platform not in expected_platforms:
            assert entries == []
            continue
        recipe = yaml.safe_load(
            (
                ROOT
                / "tests/test_utils/recipes"
                / platform.removeprefix("dgx_")
                / "unit-tests.yaml"
            ).read_text()
        )
        buckets = [case for product in recipe["products"] for case in product["test_case"]]
        assert [entry["bucket"] for entry in entries] == buckets
        assert all(
            entry["runner"] == runner and entry["image"] == f"{registry}/megatron-lm:{sha}-{cloud}"
            for entry in entries
        )


def test_producer_rejects_an_unsupported_built_platform(
    tmp_path: Path, shell_environment: dict[str, str]
) -> None:
    step = _step("populate-build-cache.yml", "parse-unit-tests", "matrix")
    result, outputs = _run(
        step["run"],
        tmp_path,
        {
            **shell_environment,
            "BUILDS": json.dumps({"include": [{"cloud": "unsupported"}]}),
            "SOURCE_SHA": "a" * 40,
        },
    )
    assert result.returncode != 0
    assert "matrix" not in outputs


def test_producer_publishes_one_h100_testmon_baseline_for_aws_and_gcp_builds(
    tmp_path: Path, shell_environment: dict[str, str]
) -> None:
    build_step = _step("_build_ci_container.yml", "cicd-compute-build-matrix", "compute")
    build, build_outputs = _run(
        build_step["run"],
        tmp_path,
        {
            "IS_MAINTAINER": "true",
            "ENABLE_GB_TESTING": "true",
            "REGISTRY_AWS": "h100.example.test/team",
            "REGISTRY_GCP_H100": "gcp.example.test/team",
            "REGISTRY_GB_GPU": "gb200.example.test/team",
            "SELECTED_RUNNER": "h100-runner",
            "SELECTED_RUNNER_GCP_H100": "gcp-build-runner",
            "SELECTED_RUNNER_GB_GPU": "gb200-runner",
            "USE_GCP_H100": "false",
            "INCLUDE_GCP_H100": "true",
        },
    )
    assert build.returncode == 0, build.stderr
    builds = json.loads(build_outputs["matrix"])["include"]
    assert {entry["cloud"] for entry in builds} == {"aws-h100", "gcp-h100", "gb-gpu"}

    source_sha = "1234567890abcdef" * 2 + "12345678"
    producer_step = _step("populate-build-cache.yml", "parse-unit-tests", "matrix")
    producer, outputs = _run(
        producer_step["run"],
        tmp_path,
        {**shell_environment, "BUILDS": build_outputs["matrix"], "SOURCE_SHA": source_sha},
    )
    assert producer.returncode == 0, producer.stderr
    matrix = json.loads(outputs["matrix"])
    assert {entry["platform"] for entry in matrix} == {"dgx_h100", "dgx_gb200"}

    for platform, bucket_file, cloud, registry, runner in (
        ("dgx_h100", "h100", "aws-h100", "h100.example.test/team", "h100-runner"),
        ("dgx_gb200", "gb200", "gb-gpu", "gb200.example.test/team", "gb200-runner"),
    ):
        recipe = yaml.safe_load(
            (ROOT / "tests/test_utils/recipes" / bucket_file / "unit-tests.yaml").read_text()
        )
        expected_buckets = [case for product in recipe["products"] for case in product["test_case"]]
        entries = [entry for entry in matrix if entry["platform"] == platform]
        assert [entry["bucket"] for entry in entries] == expected_buckets
        assert all(
            entry["runner"] == runner
            and entry["image"] == f"{registry}/megatron-lm:{source_sha}-{cloud}"
            for entry in entries
        )


@pytest.mark.parametrize(
    "source_ref", ["refs/heads/main", "refs/heads/pull-request/6934", "refs/tags/main"]
)
def test_producer_requires_the_main_branch(tmp_path: Path, source_ref: str) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/populate-build-cache.yml").read_text())
    guard = workflow["jobs"]["validate-source"]["steps"][0]["run"]
    result, _ = _run(guard, tmp_path, {"SOURCE_REF": source_ref})
    assert (result.returncode == 0) == (source_ref == "refs/heads/main")


@pytest.fixture
def pr_files_environment(tmp_path: Path, shell_environment: dict[str, str]) -> dict[str, str]:
    gh = tmp_path / "bin/gh"
    gh.write_text(
        '#!/bin/sh\n[ "$1" = "api" ] || exit 2\n'
        'if [ "$2" = "--paginate" ]; then\n'
        '  [ "$#" -eq 3 ] && '
        '[ "$3" = "repos/NVIDIA/Megatron-LM/pulls/7824/files?per_page=100" ] || exit 2\n'
        '  echo files >> "$GH_LOG"\n'
        '  cat "$TEST_PR_RESPONSE"\n'
        '  exit "$TEST_API_EXIT"\n'
        'fi\n'
        '[ "$#" -eq 4 ] && [ "$2" = "repos/NVIDIA/Megatron-LM/pulls/7824" ] && '
        '[ "$3" = "--jq" ] && [ "$4" = ".merge_commit_sha" ] || exit 2\n'
        'echo sha >> "$GH_LOG"\n'
        'printf "%s\\n" "$TEST_CURRENT_PR_SHA"\n'
        'exit "$TEST_SHA_EXIT"\n'
    )
    gh.chmod(0o755)
    response = tmp_path / "api-response"
    response.write_text(
        json.dumps(
            [
                {"status": "modified", "filename": "megatron/core/z.py"},
                {"status": "added", "filename": "tests/unit_tests/test_a.py"},
            ]
        )
        + "\n"
        + json.dumps([{"status": "removed", "filename": "megatron/core/removed.py"}])
        + "\n"
    )
    return {
        **shell_environment,
        "GH_LOG": str(tmp_path / "gh-calls"),
        "TEST_PR_RESPONSE": str(response),
        "TEST_API_EXIT": "0",
        "TEST_SHA_EXIT": "0",
        "TEST_CURRENT_PR_SHA": "c" * 40,
        "RUNNER_TEMP": str(tmp_path / "runner"),
        "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
        "PR_NUMBER": "7824",
        "PR_CHANGED_FILES": "3",
        "PR_MERGE_SHA": "c" * 40,
        "TESTED_SHA": "c" * 40,
    }


def _assert_pr_files_artifact(tmp_path, outputs, record_count, paths):
    assert outputs == {"ready": "true"}
    artifact = tmp_path / "runner/unit-test-pr-files"
    assert {path.name for path in artifact.iterdir()} == {"changed-files", "metadata.json"}
    expected_paths = sorted(set(paths))
    assert (artifact / "changed-files").read_text().splitlines() == expected_paths
    assert json.loads((artifact / "metadata.json").read_text()) == {
        "tested_sha": "c" * 40,
        "changed_files": record_count,
        "changed_paths": len(expected_paths),
    }


@pytest.mark.parametrize(
    "scenario",
    [
        "multi-page",
        "empty",
        "api-error",
        "recheck-error",
        "wrong-count",
        "too-many-files",
        "initial-sha-mismatch",
        "current-sha-mismatch",
        "invalid-pr-number",
        "invalid-tested-sha",
        "invalid-file-count",
        "invalid-json",
        "invalid-page",
        "missing-filename",
        "missing-rename-source",
        "newline-path",
    ],
)
def test_pr_files_producer_requires_complete_data_for_the_tested_commit(
    tmp_path: Path, pr_files_environment: dict[str, str], scenario: str
) -> None:
    environment = pr_files_environment
    response = Path(environment["TEST_PR_RESPONSE"])
    overrides = {
        "empty": {"PR_CHANGED_FILES": "0"},
        "api-error": {"TEST_API_EXIT": "1"},
        "recheck-error": {"TEST_SHA_EXIT": "1"},
        "wrong-count": {"PR_CHANGED_FILES": "4"},
        "too-many-files": {"PR_CHANGED_FILES": "3001"},
        "initial-sha-mismatch": {"PR_MERGE_SHA": "d" * 40},
        "current-sha-mismatch": {"TEST_CURRENT_PR_SHA": "d" * 40},
        "invalid-pr-number": {"PR_NUMBER": "invalid"},
        "invalid-tested-sha": {"TESTED_SHA": "invalid"},
        "invalid-file-count": {"PR_CHANGED_FILES": "invalid"},
    }
    environment.update(overrides.get(scenario, {}))
    if scenario == "empty":
        response.write_text("[]\n")
    elif scenario == "invalid-json":
        response.write_text("[")
    elif scenario == "invalid-page":
        response.write_text("{}")
    elif scenario in {"missing-filename", "missing-rename-source", "newline-path"}:
        record = {
            "missing-filename": {"status": "modified"},
            "missing-rename-source": {"status": "renamed", "filename": "new.py"},
            "newline-path": {"status": "modified", "filename": "split\npath.py"},
        }[scenario]
        response.write_text(json.dumps([record]))
        environment["PR_CHANGED_FILES"] = "1"
    step = _step("cicd-main.yml", "configure", "unit-test-pr-files")
    result, outputs = _run(step["run"], tmp_path, environment)
    assert result.returncode == 0, result.stderr
    if scenario in {"multi-page", "empty"}:
        paths = (
            []
            if scenario == "empty"
            else ["megatron/core/z.py", "tests/unit_tests/test_a.py", "megatron/core/removed.py"]
        )
        _assert_pr_files_artifact(tmp_path, outputs, len(paths), paths)
    else:
        assert "ready" not in outputs
        assert "::notice::" in result.stdout
        assert not (tmp_path / "runner/unit-test-pr-files/metadata.json").exists()
    no_api = {
        "too-many-files",
        "initial-sha-mismatch",
        "invalid-pr-number",
        "invalid-tested-sha",
        "invalid-file-count",
    }
    expected_calls = [] if scenario in no_api else ["files"]
    if scenario in {"multi-page", "empty", "recheck-error", "current-sha-mismatch"}:
        expected_calls.append("sha")
    calls = Path(environment["GH_LOG"])
    assert (calls.read_text().splitlines() if calls.exists() else []) == expected_calls


@pytest.mark.parametrize("direction", ["into", "out", "within"])
def test_pr_rename_endpoints_trigger_mandatory_selection(
    tmp_path: Path, pr_files_environment: dict[str, str], direction: str
) -> None:
    source = "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental"
    old = "outside/old.py" if direction == "into" else f"{source}/old.py"
    new = "outside/new.py" if direction == "out" else f"{source}/new.py"
    records = [{"status": "renamed", "filename": new, "previous_filename": old}]
    if direction == "within":
        # One rename's old path is another's new path: publish each path only once.
        records.append(
            {"status": "renamed", "filename": f"{source}/other.py", "previous_filename": new}
        )
    Path(pr_files_environment["TEST_PR_RESPONSE"]).write_text(json.dumps(records))
    pr_files_environment["PR_CHANGED_FILES"] = str(len(records))
    step = _step("cicd-main.yml", "configure", "unit-test-pr-files")
    result, outputs = _run(step["run"], tmp_path, pr_files_environment)
    assert result.returncode == 0, result.stderr
    paths = [
        path for record in records for path in (record["filename"], record["previous_filename"])
    ]
    _assert_pr_files_artifact(tmp_path, outputs, len(records), paths)

    test_file = "tests/unit_tests/required/test_case.py"
    (tmp_path / test_file).parent.mkdir(parents=True)
    (tmp_path / test_file).write_text("def test_case(): pass\n")
    config = tmp_path / "mandatory.yaml"
    config.write_text(
        yaml.safe_dump(
            {"mappings": [{"source_dirs": [source], "test_buckets": {"dgx_h100": [test_file]}}]}
        )
    )
    selection = tmp_path / "selected-tests"
    selection.write_text(f"{test_file}::test_case\n")
    selected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/unit_tests/testmon_mandatory.py"),
            "--config",
            str(config),
            "--changed-files",
            str(tmp_path / "runner/unit-test-pr-files/changed-files"),
            "--bucket",
            "tests/unit_tests/**/*.py",
            "--platform",
            "dgx_h100",
            "--selection",
            str(selection),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert selected.returncode == 0, selected.stderr
    assert selection.read_text().splitlines() == [test_file]
    assert (tmp_path / "mandatory-tests").read_text().splitlines() == [test_file]


def test_pr_files_producer_accepts_3000_renames_with_6000_paths(
    tmp_path: Path, pr_files_environment: dict[str, str]
) -> None:
    records = [
        {"status": "renamed", "filename": f"new/{i:04}.py", "previous_filename": f"old/{i:04}.py"}
        for i in range(3000)
    ]
    Path(pr_files_environment["TEST_PR_RESPONSE"]).write_text(
        "\n".join(json.dumps(records[start : start + 100]) for start in range(0, 3000, 100))
    )
    pr_files_environment["PR_CHANGED_FILES"] = "3000"
    step = _step("cicd-main.yml", "configure", "unit-test-pr-files")
    result, outputs = _run(step["run"], tmp_path, pr_files_environment)
    assert result.returncode == 0, result.stderr
    paths = [
        path for record in records for path in (record["filename"], record["previous_filename"])
    ]
    _assert_pr_files_artifact(tmp_path, outputs, 3000, paths)


@pytest.fixture
def configure_environment(
    tmp_path: Path, shell_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    monkeypatch.delenv("ENABLE_SELECTIVE_UNIT_TESTS", raising=False)
    gh = tmp_path / "bin/gh"
    gh.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$TEST_PR_LABELS"\n' 'exit "$TEST_PR_LABEL_LOOKUP_EXIT"\n'
    )
    gh.chmod(0o755)
    return {
        **shell_environment,
        "IS_CI_WORKLOAD": "false",
        "IS_MERGE_GROUP": "false",
        "EVENT_NAME": "push",
        "REF": "refs/heads/pull-request/6934",
        "FORCE_RUN_ALL": "false",
        "TEST_PR_LABEL_LOOKUP_EXIT": "0",
        "H100_TRAFFIC_MODE": "aws",
        "PREFLIGHT_IS_MEMBER": "true",
        "IS_MAINTAINER": "true",
        "EPHEMERAL_RUNNER": "nvidia-ci-aws-gpu-x8-ephemeral",
        "RUN_ATTEMPT": "1",
        "REGISTRY_AWS": "h100.example.test/team",
        "REGISTRY_GCP_H100": "gcp.example.test/team",
    }


def _run_configure(
    tmp_path: Path, environment: dict[str, str], labels: list[str], random_seed: int | None = None
) -> tuple[subprocess.CompletedProcess, dict]:
    script = _step("cicd-main.yml", "configure", "configure")["run"]
    script = script.replace(
        "${{ fromJSON(steps.get-pr-info.outputs.pr-info || '{}').number }}", "6934"
    )
    script = script.replace("${{ github.repository }}", "NVIDIA/Megatron-LM")
    effective_environment = {
        **environment,
        "IS_PR": str(
            environment["EVENT_NAME"] == "push"
            and environment["REF"].startswith("refs/heads/pull-request/")
        ).lower(),
    }
    if random_seed is not None:
        # Seed Bash's RANDOM before the actual workflow script for repeatable
        # coverage of each 50-50 routing outcome.
        script = f"RANDOM={random_seed}\n{script}"
    return _run(script, tmp_path, {**effective_environment, "TEST_PR_LABELS": json.dumps(labels)})


@pytest.mark.parametrize(
    ("overrides", "random_seed", "cloud", "build_runner", "gpu_runner", "registry", "use_gcp"),
    [
        pytest.param(
            {},
            None,
            "aws-h100",
            "nvidia-ci-aws-cpu",
            {"group": "nemo-ci-gpu-x8", "labels": "nvidia-ci-aws-gpu-x8"},
            "h100.example.test/team",
            "false",
            id="member-pr-aws",
        ),
        pytest.param(
            {"H100_TRAFFIC_MODE": "gcp"},
            None,
            "gcp-h100",
            "nvidia-ci-gcp-cpu-h100",
            {"group": "nemo-ci-gpu-x8", "labels": "nvidia-ci-gcp-h100-x8"},
            "gcp.example.test/team",
            "true",
            id="member-pr-gcp",
        ),
        pytest.param(
            {"H100_TRAFFIC_MODE": "50-50"},
            3,
            "gcp-h100",
            "nvidia-ci-gcp-cpu-h100",
            {"group": "nemo-ci-gpu-x8", "labels": "nvidia-ci-gcp-h100-x8"},
            "gcp.example.test/team",
            "true",
            id="member-pr-50-50-gcp",
        ),
        pytest.param(
            {"H100_TRAFFIC_MODE": "50-50"},
            2,
            "aws-h100",
            "nvidia-ci-aws-cpu",
            {"group": "nemo-ci-gpu-x8", "labels": "nvidia-ci-aws-gpu-x8"},
            "h100.example.test/team",
            "false",
            id="member-pr-50-50-aws",
        ),
        pytest.param(
            {
                "PREFLIGHT_IS_MEMBER": "false",
                "IS_MAINTAINER": "false",
                "H100_TRAFFIC_MODE": "gcp",
                "EPHEMERAL_RUNNER": "ephe-v2-example",
                "RUN_ATTEMPT": "2",
            },
            None,
            "aws-h100",
            "ephe-v2-example-a2",
            "ephe-v2-example-a2",
            "h100.example.test/team",
            "false",
            id="external-pr-ephemeral-aws",
        ),
        pytest.param(
            {
                "EVENT_NAME": "merge_group",
                "REF": "refs/heads/gh-readonly-queue/main/pr-6934-abc",
                "H100_TRAFFIC_MODE": "gcp",
            },
            None,
            "aws-h100",
            "nvidia-ci-aws-cpu",
            {"group": "nemo-ci-gpu-x8", "labels": "nvidia-ci-aws-gpu-x8"},
            "h100.example.test/team",
            "false",
            id="merge-queue-stays-aws",
        ),
        pytest.param(
            {
                "EVENT_NAME": "workflow_dispatch",
                "REF": "refs/heads/main",
                "H100_TRAFFIC_MODE": "gcp",
            },
            None,
            "aws-h100",
            "nvidia-ci-aws-cpu",
            {"group": "nemo-ci-gpu-x8", "labels": "nvidia-ci-aws-gpu-x8"},
            "h100.example.test/team",
            "false",
            id="dispatch-stays-aws",
        ),
    ],
)
def test_h100_runner_traffic_routing(
    tmp_path: Path,
    configure_environment: dict[str, str],
    overrides: dict[str, str],
    random_seed: int | None,
    cloud: str,
    build_runner: str,
    gpu_runner: dict[str, str] | str,
    registry: str,
    use_gcp: str,
) -> None:
    result, outputs = _run_configure(
        tmp_path, {**configure_environment, **overrides}, [], random_seed=random_seed
    )
    assert result.returncode == 0, result.stderr
    assert outputs["h100_cloud"] == cloud
    assert outputs["h100_build_runner"] == build_runner
    assert json.loads(outputs["h100_gpu_runs_on"]) == gpu_runner
    assert outputs["h100_registry"] == registry
    assert outputs["h100_use_gcp"] == use_gcp


def test_nemo_cicd_validator_declares_every_needs_reference_directly() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())
    job = workflow["jobs"]["Nemo_CICD_Test"]
    direct_dependencies = set(job["needs"])
    referenced_dependencies = set(re.findall(r"needs\.([A-Za-z0-9_-]+)", yaml.safe_dump(job)))
    assert "configure" in direct_dependencies
    assert (
        job["steps"][-1]["env"]["RUN_H100_FUNCTIONAL"]
        == "${{ needs.configure.outputs.run_h100_functional }}"
    )
    assert referenced_dependencies <= direct_dependencies


@pytest.mark.parametrize(
    ("run_h100_functional", "h100_result", "expected_success"),
    [("true", "skipped", False), ("false", "skipped", True), ("true", "success", True)],
)
def test_nemo_cicd_validator_checks_h100_functional_result(
    tmp_path: Path, run_h100_functional: str, h100_result: str, expected_success: bool
) -> None:
    gh = tmp_path / "bin/gh"
    gh.parent.mkdir(parents=True)
    gh.write_text("#!/bin/sh\nprintf '0\\n'\n")
    gh.chmod(0o755)
    step = _step("cicd-main.yml", "Nemo_CICD_Test", "result")
    result, _ = _run(
        step["run"],
        tmp_path,
        {
            "PATH": f"{gh.parent}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "test-token",
            "GITHUB_RUN_ID": "1",
            "DOCS_ONLY": "false",
            "IS_DEPLOYMENT": "false",
            "IS_MAINTAINER": "true",
            "IS_CI_WORKLOAD": "false",
            "FORCE_RUN_ALL": "false",
            "RUN_H100_FUNCTIONAL": run_h100_functional,
            "GB200_SHOULD_RUN": "false",
            "ENABLE_GB200_TESTING": "false",
            "UNIT_RESULT": "success",
            "UNIT_GB200_RESULT": "skipped",
            "H100_RESULT": h100_result,
            "GB200_RESULT": "skipped",
        },
    )
    assert (result.returncode == 0) == expected_success, result.stdout + result.stderr


def test_selective_default_uses_the_repository_variable() -> None:
    step = _step("cicd-main.yml", "configure", "configure")
    assert step["env"]["ENABLE_SELECTIVE_UNIT_TESTS"] == "${{ vars.ENABLE_SELECTIVE_UNIT_TESTS }}"


@pytest.mark.parametrize(
    ("default_value", "labels", "expected_default", "expected_requested"),
    [
        pytest.param(None, [], False, False, id="unset"),
        pytest.param("", [], False, False, id="empty"),
        pytest.param("false", [], False, False, id="disabled"),
        pytest.param("true", [], True, True, id="enabled"),
        pytest.param("TRUE", [], False, False, id="uppercase-is-not-enabled"),
        pytest.param("true ", [], False, False, id="whitespace-is-not-enabled"),
        pytest.param("1", [], False, False, id="numeric-is-not-enabled"),
        pytest.param(None, ["Run selective unit tests"], False, True, id="label-without-default"),
        pytest.param("false", ["Run selective unit tests"], False, True, id="label-enables"),
        pytest.param("invalid", ["Run selective unit tests"], False, True, id="label-with-invalid"),
        pytest.param("true", ["Run selective unit tests"], True, True, id="both-enable"),
        pytest.param("true", ["Disable selective unit tests"], True, False, id="label-disables"),
        pytest.param(
            "false", ["Disable selective unit tests"], False, False, id="already-disabled"
        ),
        pytest.param(
            "true",
            ["Run selective unit tests", "Disable selective unit tests"],
            True,
            False,
            id="disable-wins-over-default-and-enable-label",
        ),
        pytest.param(
            None,
            ["Run selective unit tests", "Disable selective unit tests"],
            False,
            False,
            id="disable-wins-over-enable-label",
        ),
    ],
)
def test_selective_variable_and_label_precedence(
    tmp_path: Path,
    configure_environment: dict[str, str],
    default_value: str | None,
    labels: list[str],
    expected_default: bool,
    expected_requested: bool,
) -> None:
    if default_value is not None:
        configure_environment["ENABLE_SELECTIVE_UNIT_TESTS"] = default_value
    result, outputs = _run_configure(tmp_path, configure_environment, labels)
    assert result.returncode == 0, result.stderr
    assert outputs["unit_testmon_eligible"] == str(expected_requested).lower()
    summary = (tmp_path / "summary").read_text()
    for name, expected in {
        "unit_testmon_default_enabled": expected_default,
        "unit_testmon_enable_label": "Run selective unit tests" in labels,
        "unit_testmon_disable_label": "Disable selective unit tests" in labels,
        "unit_testmon_requested": expected_requested,
        "unit_testmon_eligible": expected_requested,
    }.items():
        assert f"| `{name}` | `{str(expected).lower()}` |" in summary


@pytest.mark.parametrize("request_source", ["default", "label"])
@pytest.mark.parametrize(
    ("labels", "environment"),
    [
        pytest.param(["Run tests"], {}, id="run-tests-label"),
        pytest.param(["Run functional tests"], {}, id="run-functional-tests-label"),
        pytest.param(["force-run-all"], {}, id="force-run-all-label"),
        pytest.param(["container::lts"], {}, id="lts-label"),
        pytest.param([], {"FORCE_RUN_ALL": "true"}, id="preflight-forces-full"),
        pytest.param([], {"TEST_PR_LABEL_LOOKUP_EXIT": "1"}, id="label-lookup-fails"),
        pytest.param([], {"REF": "refs/heads/main"}, id="main-push"),
        pytest.param([], {"REF": "refs/heads/pull-request/not-a-number"}, id="non-pr-branch"),
        pytest.param([], {"REF": "refs/heads/deploy-release/1.0"}, id="release-push"),
        pytest.param([], {"EVENT_NAME": "workflow_dispatch"}, id="manual-dispatch"),
        pytest.param([], {"EVENT_NAME": "schedule", "IS_CI_WORKLOAD": "true"}, id="schedule"),
        pytest.param([], {"IS_MERGE_GROUP": "true"}, id="preflight-merge-group"),
        pytest.param(
            [],
            {
                "EVENT_NAME": "merge_group",
                "IS_MERGE_GROUP": "true",
                "REF": "refs/heads/gh-readonly-queue/main/pr-6934-abc",
            },
            id="merge-queue",
        ),
    ],
)
def test_full_test_guards_override_selective_requests(
    tmp_path: Path,
    configure_environment: dict[str, str],
    request_source: str,
    labels: list[str],
    environment: dict[str, str],
) -> None:
    if request_source == "default":
        configure_environment["ENABLE_SELECTIVE_UNIT_TESTS"] = "true"
    else:
        labels = [*labels, "Run selective unit tests"]
    result, outputs = _run_configure(tmp_path, {**configure_environment, **environment}, labels)
    assert result.returncode == 0, result.stderr
    assert outputs["unit_testmon_eligible"] == "false"
    # Eligibility guards must not obscure a requested selection in the summary.
    # If label lookup failed, only the repository default can establish a request.
    requested = request_source == "default" or environment.get("TEST_PR_LABEL_LOOKUP_EXIT") != "1"
    summary = (tmp_path / "summary").read_text()
    assert f"| `unit_testmon_requested` | `{str(requested).lower()}` |" in summary
    assert "| `unit_testmon_eligible` | `false` |" in summary


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_unit_tests_share_gpu_selection_before_the_existing_bucket_runners(
    tmp_path, shell_environment, platform
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/_unit-tests.yml").read_text())
    assert set(workflow["jobs"]) == {"select", "run"}
    selection = workflow["jobs"]["select"]
    assert selection["runs-on"] == "${{ fromJSON(inputs.runner) }}"
    assert "strategy" not in selection
    planning = _step("_unit-tests.yml", "select", "plan")
    assert planning["env"]["CONTAINER_IMAGE"] == "${{ inputs.container-image }}"
    assert "'uv', ['run', '--no-sync', 'python'" in planning["with"]["script"]
    job = workflow["jobs"]["run"]
    assert job["needs"] == "select"
    assert job["runs-on"] == "${{ fromJSON(inputs.runner) }}"
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["include"] == (
        "${{ fromJSON(needs.select.result == 'success' && needs.select.outputs.matrix || inputs.buckets) }}"
    )
    action = job["steps"][-1]["with"]
    assert action["container-image"] == "${{ inputs.container-image }}"
    assert action["test_case"] == "${{ matrix.bucket }}"
    assert action["timeout"] == "${{ matrix.timeout || 30 }}"
    assert (
        action["unit_test_pr_files_artifact_id"] == "${{ inputs.unit_test_pr_files_artifact_id }}"
    )
    assert action["unit_testmon_plan_artifact"] == "${{ matrix.plan_artifact }}"
    assert action["unit_testmon_mode"] == "${{ matrix.mode || 'full' }}"

    callers = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())["jobs"]
    suffix = "" if platform == "h100" else "-gb200"
    caller = callers[f"cicd-unit-tests-latest{suffix}"]
    parser_job = f"cicd-parse-unit-tests{suffix}"
    output = f"unit-tests{suffix}"
    assert caller["uses"] == "./.github/workflows/_unit-tests.yml"
    assert "strategy" not in caller
    assert parser_job in caller["needs"]
    assert caller["with"]["buckets"] == "${{ needs." + parser_job + ".outputs." + output + " }}"
    assert caller["with"]["runner"] == (
        "${{ needs.configure.outputs.h100_gpu_runs_on }}"
        if platform == "h100"
        else "${{ toJSON(needs.is-not-external-contributor.outputs.selected_runner_gb200) }}"
    )
    parser = _step("cicd-main.yml", parser_job, "parse-unit-tests")
    result, parsed = _run(parser["run"], tmp_path, shell_environment)
    assert result.returncode == 0, result.stderr
    recipe = yaml.safe_load(
        (ROOT / "tests/test_utils/recipes" / platform / "unit-tests.yaml").read_text()
    )
    assert json.loads(parsed[output]) == [
        {"bucket": case} for product in recipe["products"] for case in product["test_case"]
    ]


@pytest.mark.parametrize("mode", ["enforce", "full"])
@pytest.mark.parametrize("branch", ["main", "dev", "core_r0.20.0", "pull-request/8523", ""])
def test_grouped_gpu_selection_preserves_main_only_baselines_and_full_requests(mode, branch):
    workflow = yaml.safe_load((ROOT / ".github/workflows/_unit-tests.yml").read_text())
    expression = workflow["jobs"]["select"]["if"].replace("&&", "and")
    inputs = SimpleNamespace(unit_testmon_mode=mode, unit_testmon_target_branch=branch)
    assert eval(expression, {"__builtins__": {}}, {"inputs": inputs}) == (
        mode == "enforce" and branch == "main"
    )


@pytest.mark.parametrize(
    "result,has_tests,cancelled,expected",
    [
        ("success", "true", False, True),
        ("success", "false", False, False),
        ("failure", "false", False, True),
        ("skipped", "", False, True),
        ("success", "", False, True),
        ("success", "true", True, False),
    ],
)
def test_gpu_bucket_dispatch_only_skips_verified_empty_selection(
    result, has_tests, cancelled, expected
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/_unit-tests.yml").read_text())
    expression = workflow["jobs"]["run"]["if"].removeprefix("${{").removesuffix("}}")
    expression = (
        expression.replace("!cancelled()", "not cancelled").replace("&&", "and").replace("||", "or")
    )
    needs = SimpleNamespace(
        select=SimpleNamespace(result=result, outputs=SimpleNamespace(has_tests=has_tests))
    )
    assert (
        eval(expression, {"__builtins__": {}}, {"needs": needs, "cancelled": cancelled}) == expected
    )


@pytest.mark.parametrize("has_tests", [False, True], ids=["empty", "selected"])
def test_coverage_requires_artifacts_or_verified_empty_selections(tmp_path, has_tests):
    bucket = "tests/unit_tests/test_example.py"
    callers = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())["jobs"]
    coverage = _step("cicd-main.yml", "Coverage", "coverage")
    for variable, job, output in (
        ("H100_BUCKETS", "cicd-parse-unit-tests", "unit-tests"),
        ("GB200_BUCKETS", "cicd-parse-unit-tests-gb200", "unit-tests-gb200"),
    ):
        assert job in callers["Coverage"]["needs"]
        assert coverage["env"][variable] == (
            "${{ needs." + job + ".outputs." + output + " || '[]' }}"
        )

    for platform in ("dgx_h100", "dgx_gb200"):
        directory = tmp_path / f"unit-test-plan-{platform}"
        directory.mkdir()
        (directory / "plan.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "source_sha": "a" * 40,
                    "bucket": bucket,
                    "platform": platform,
                    "mode": "selected",
                    "phases": {"prod": [bucket] if has_tests else [], "experimental": []},
                }
            )
        )
    result, outputs = _run(
        coverage["run"],
        tmp_path,
        {
            "SELECTIVE": "true",
            "H100_BUCKETS": json.dumps([{"bucket": bucket}]),
            "GB200_BUCKETS": json.dumps([{"bucket": bucket}]),
            "SOURCE_SHA": "a" * 40,
        },
    )
    # Missing coverage is valid only when every bucket selected no tests.
    assert (result.returncode == 0) == (not has_tests), result.stdout + result.stderr
    assert outputs == ({} if has_tests else {"has_coverage": "false"})


@pytest.mark.parametrize(
    "scenario",
    [
        "mixed",
        "empty",
        "cache-error",
        "selector-error",
        "missing-output",
        "invalid-pr",
        "upload-error",
    ],
)
def test_grouped_native_gpu_selection(
    tmp_path: Path, scenario: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("testmon")
    buckets = [
        {"bucket": f"tests/unit_tests/{name}/**/*.py", "timeout": 40}
        for name in (["empty"] if scenario == "empty" else ["selected", "empty", "full"])
    ]
    unit_tests = tmp_path / "tests/unit_tests"
    unit_tests.mkdir(parents=True)
    for script in (
        "testmon_cache.py",
        "testmon_selector.py",
        "testmon_plan.py",
        "testmon_mandatory.py",
        "find_test_cases.py",
    ):
        (unit_tests / script).symlink_to(ROOT / "tests/unit_tests" / script)
    (unit_tests / "testmon_mandatory_tests.yaml").write_text("mappings: []\n")
    dataset = tmp_path / "tests/test_utils/python_scripts/download_unit_tests_dataset.py"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("")
    recipe = tmp_path / "tests/test_utils/recipes/h100/unit-tests.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        json.dumps({"products": [{"test_case": [entry["bucket"] for entry in buckets]}]})
    )
    selected = unit_tests / "selected/test_case.py"
    selected.parent.mkdir()
    selected.write_text("def test_case(): pass\n")
    pr_files = tmp_path / "pr-files"
    pr_files.mkdir()
    (pr_files / "changed-files").write_text("")
    (pr_files / "metadata.json").write_text(
        json.dumps({"tested_sha": "a" * 40, "changed_files": 0, "changed_paths": 0})
    )
    if scenario == "invalid-pr":
        (pr_files / "metadata.json").write_text(
            json.dumps({"tested_sha": "a" * 40, "changed_files": 3001, "changed_paths": 0})
        )
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(unit_tests),
        "BUCKETS": json.dumps(buckets),
        "RECIPE_PLATFORM": "dgx_h100",
        "SOURCE_SHA": "a" * 40,
        "TARGET_BRANCH": "main",
        "PR_FILES_DIR": str(pr_files),
        "CONTAINER_IMAGE": "ci-test-image",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }
    # Empty recorded databases are sufficient: a new file is selected, and a
    # bucket without files is skipped. Use real cache validation and selection.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os
from pathlib import Path
from testmon.db import DB
from testmon_cache import PHASES, cache_identity, record_phase, finalize
for index, entry in enumerate(json.loads(os.environ['BUCKETS'])):
    if '/full/' in entry['bucket']:
        continue
    directory = Path('saved') / str(index)
    for phase in PHASES:
        database = directory / phase / '.testmondata'
        database.parent.mkdir(parents=True)
        DB(str(database)).con.close()
        record_phase(directory, phase)
    identity = cache_identity(Path.cwd(), entry['bucket'], 'dgx_h100')
    finalize(directory, identity, 'b' * 40, '111-1')
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location(
        "select_unit_tests", ROOT / ".github/scripts/select_unit_tests.py"
    )
    assert spec is not None and spec.loader is not None
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    monkeypatch.chdir(tmp_path)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    uploads = []
    restores = 0

    def actions(operation, *args):
        nonlocal restores
        if operation == "restore":
            path, key, prefix, key_file = args
            assert path == "assets_dir/testmon"
            assert key == f"{prefix}lookup-123-2"
            assert not Path(path).exists(), "previous bucket cache was not cleared"
            index = restores
            restores += 1
            if scenario == "cache-error" and index == 0:
                raise subprocess.CalledProcessError(1, [operation])
            saved = tmp_path / "saved" / str(index)
            if saved.exists():
                shutil.copytree(saved, path)
            Path(key_file).write_text(f"{prefix}111-1" if saved.exists() else "")
        else:
            assert operation == "upload"
            if scenario == "upload-error":
                raise subprocess.CalledProcessError(1, [operation])
            name, plan, root = args
            assert root == "assets_dir/testmon-plan"
            assert plan == f"{root}/plan.json"
            uploads.append({"name": name, "plan": json.loads(Path(plan).read_text())})

    monkeypatch.setattr(selector, "_actions", actions)

    launches = []

    def launch(bucket):
        launches.append(bucket)
        if scenario == "selector-error" and "/selected/" in bucket:
            raise subprocess.CalledProcessError(1, ["native-gpu-collection"])
        for phase in ("prod", "experimental"):
            directory = Path("assets_dir/testmon/.testmon-work") / phase
            directory.mkdir(parents=True)
            (directory / "selected-tests").write_text(
                "tests/unit_tests/selected/test_case.py::test_case\n"
                if "/selected/" in bucket
                else ""
            )
            (directory / "mandatory-tests").write_text("")
        if scenario == "missing-output" and "/selected/" in bucket:
            (Path("assets_dir/testmon/.testmon-work/experimental/selected-tests")).unlink()

    monkeypatch.setattr(selector, "_launch_gpu_selection", launch)
    monkeypatch.setattr(
        selector, "_clear_cache", lambda path: shutil.rmtree(path) if path.exists() else None
    )
    if scenario == "upload-error":
        with pytest.raises(subprocess.CalledProcessError):
            selector.main()
        assert not (tmp_path / "outputs").exists()
        return
    selector.main()
    assert restores == len(buckets)
    outputs = dict(line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines())
    matrix = json.loads(outputs["matrix"])
    assert outputs["has_tests"] == str(scenario != "empty").lower()
    assert [entry["bucket"] for entry in matrix] == [
        entry["bucket"]
        for entry in buckets
        if scenario == "invalid-pr" or entry["bucket"] != "tests/unit_tests/empty/**/*.py"
    ]
    assert [entry["mode"] for entry in matrix] == (
        ["full", "full", "full"]
        if scenario == "invalid-pr"
        else (
            []
            if scenario == "empty"
            else [
                (
                    "full"
                    if scenario in {"cache-error", "selector-error", "missing-output"}
                    else "preselected"
                ),
                "full",
            ]
        )
    )
    assert all(entry["timeout"] == 40 for entry in matrix)
    assert bool(caplog.records) == (scenario != "empty")
    for upload in uploads:
        plan = upload["plan"]
        assert re.fullmatch(r"unit-test-plan-dgx_h100-[0-9a-f]{16}-123-2", upload["name"])
        expected = (
            ["tests/unit_tests/selected/test_case.py::test_case"]
            if "/selected/" in plan["bucket"]
            else []
        )
        assert plan["phases"] == {"prod": expected, "experimental": expected}
        assert plan["source_sha"] == "a" * 40
        matches = [entry for entry in matrix if entry["bucket"] == plan["bucket"]]
        assert bool(matches) == bool(expected)
        if matches:
            assert matches[0]["plan_artifact"] == upload["name"]
    assert bool(uploads) == (scenario != "invalid-pr")
    if scenario == "invalid-pr":
        assert not launches
    else:
        assert any("/empty/" in upload["plan"]["bucket"] for upload in uploads)
    assert all(entry["plan_artifact"] == "" for entry in matrix if entry["mode"] == "full")


@pytest.mark.parametrize(
    "bucket",
    [
        "tests/unit_tests/inference/test_batch_invariant_fp8.py",
        "tests/unit_tests/inference/test_disagg_transport.py",
    ],
)
def test_grouped_gpu_selection_preserves_always_full_buckets(tmp_path, monkeypatch, bucket):
    spec = importlib.util.spec_from_file_location(
        "select_unit_tests", ROOT / ".github/scripts/select_unit_tests.py"
    )
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "TARGET_BRANCH": "main",
        "RECIPE_PLATFORM": "dgx_h100",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "BUCKETS": json.dumps([{"bucket": bucket}]),
    }.items():
        monkeypatch.setenv(key, value)

    def unexpected(*args):
        pytest.fail("Always-full buckets must not restore caches or run collection")

    monkeypatch.setattr(selector, "_actions", unexpected)
    monkeypatch.setattr(selector, "_python", unexpected)
    monkeypatch.setattr(selector, "_launch_gpu_selection", unexpected)
    selector.main()
    output = dict(line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines())
    assert output["has_tests"] == "true"
    assert json.loads(output["matrix"]) == [{"bucket": bucket, "mode": "full", "plan_artifact": ""}]


@pytest.mark.parametrize("platform", ["dgx_h100", "dgx_gb200"])
def test_grouped_selector_launches_native_selection_without_execution(
    tmp_path, monkeypatch, platform
):
    spec = importlib.util.spec_from_file_location(
        "select_unit_tests", ROOT / ".github/scripts/select_unit_tests.py"
    )
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RECIPE_PLATFORM", platform)
    monkeypatch.setenv("CONTAINER_IMAGE", "existing-ci-image")
    calls = []
    monkeypatch.setattr(
        selector.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    bucket = "tests/unit_tests/generalized_tensor_parallel/**/*.py"
    selector._launch_gpu_selection(bucket)
    command, options = calls[0]
    assert command[:2] == [
        sys.executable,
        "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
    ]
    assert command[command.index("--unit-testmon-mode") + 1] == "select-only"
    assert command[command.index("--test-case") + 1] == bucket
    assert command[command.index("--container-image") + 1] == "existing-ci-image"
    assert command[command.index("--platform") + 1] == platform
    assert options["check"] is True
    assert options["env"]["PYTHONPATH"] == str(tmp_path)
