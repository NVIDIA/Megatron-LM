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
    "source_ref",
    [
        "refs/heads/main",
        "refs/heads/dev",
        "refs/heads/pull-request/6934",
        "refs/heads/dev-feature",
        "refs/tags/main",
        "refs/tags/dev",
    ],
)
def test_producer_requires_a_baseline_branch(tmp_path: Path, source_ref: str) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/populate-build-cache.yml").read_text())
    guard = workflow["jobs"]["validate-source"]["steps"][0]["run"]
    result, _ = _run(guard, tmp_path, {"SOURCE_REF": source_ref})
    assert (result.returncode == 0) == (source_ref in {"refs/heads/main", "refs/heads/dev"})


def _expression(value: str) -> str:
    """Translate a GitHub expression with dotted contexts into Python syntax."""
    expression = value.strip().removeprefix("${{").removesuffix("}}").strip()
    expression = re.sub(
        r"\b(steps|inputs|github|needs|matrix|vars)((?:\.[A-Za-z0-9_-]+)+)",
        lambda match: f"context[{match.group(0)!r}]",
        expression,
    )
    expression = expression.replace("!cancelled()", "True")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return expression.replace("&&", " and ").replace("||", " or ")


def _evaluate(value: str, context: dict[str, str]):
    return eval(_expression(value), {"__builtins__": {}}, {"context": context})


@pytest.mark.parametrize(
    ("event", "ref", "enabled", "records"),
    [
        ("push", "refs/heads/dev", "true", True),
        ("push", "refs/heads/dev", "false", False),
        ("push", "refs/heads/dev", "", False),
        ("workflow_dispatch", "refs/heads/dev", "false", True),
        ("schedule", "refs/heads/main", "false", True),
        ("workflow_dispatch", "refs/heads/main", "true", True),
        ("push", "refs/heads/main", "true", False),
    ],
)
def test_dev_producer_records_each_dev_push(
    event: str, ref: str, enabled: str, records: bool
) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/populate-build-cache.yml").read_text())
    triggers = workflow[True]
    # Scheduled workflows run only from main; dev records every push without path filters.
    assert triggers["push"] == {"branches": ["dev"]}
    assert "schedule" not in triggers and "workflow_dispatch" in triggers
    assert workflow["concurrency"] == {
        "group": "populate-ci-caches-${{ github.ref_name }}",
        "cancel-in-progress": False,
    }
    context = {
        "github.event_name": event,
        "github.ref": ref,
        "vars.ENABLE_SELECTIVE_UNIT_TESTS": enabled,
    }
    assert _evaluate(workflow["jobs"]["parse-unit-tests"]["if"], context) is records
    producer = workflow["jobs"]["populate-testmon-cache"]["steps"][-1]["with"]
    assert producer["unit_testmon_mode"] == "baseline"
    assert producer["unit_testmon_target_branch"] == "${{ github.ref_name }}"
    assert producer["sha"] == "${{ github.sha }}"


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
        "TARGET_BRANCH": "main",
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
    assert "| `unit_testmon_target_branch` | `main` |" in summary
    assert "| `unit_testmon_baseline` | `main-actions-cache` |" in summary


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


@pytest.mark.parametrize(
    ("target_branch", "baseline", "eligible"),
    [
        ("main", "main-actions-cache", True),
        ("dev", "dev-producer-artifacts", True),
        ("core_r0.16.0", "none", False),
        ("devel", "none", False),
        ("", "none", False),
    ],
)
def test_selective_requests_need_a_trusted_baseline_branch(
    tmp_path: Path,
    configure_environment: dict[str, str],
    target_branch: str,
    baseline: str,
    eligible: bool,
) -> None:
    step = _step("cicd-main.yml", "configure", "configure")
    assert step["env"]["TARGET_BRANCH"] == (
        "${{ fromJSON(steps.get-pr-info.outputs.pr-info || '{}').base.ref }}"
    )
    configure_environment.update(ENABLE_SELECTIVE_UNIT_TESTS="true", TARGET_BRANCH=target_branch)
    result, outputs = _run_configure(tmp_path, configure_environment, [])
    assert result.returncode == 0, result.stderr
    assert outputs["unit_testmon_eligible"] == str(eligible).lower()
    summary = (tmp_path / "summary").read_text()
    assert "| `unit_testmon_requested` | `true` |" in summary
    assert f"| `unit_testmon_target_branch` | `{target_branch or 'none'}` |" in summary
    assert f"| `unit_testmon_baseline` | `{baseline}` |" in summary
    assert "requests CPU Testmon planning only" in summary


@pytest.mark.parametrize("has_tests", [False, True], ids=["empty", "selected"])
def test_cpu_selection_controls_gpu_execution_and_coverage(tmp_path, shell_environment, has_tests):
    # Hosted runners provide zstd, but the existing GPU baselines use gzip.
    zstd = tmp_path / "zstd"
    zstd.write_text("#!/bin/sh\nprintf '1.5.7\\n'\n")
    zstd.chmod(0o755)
    environment = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    result, compression = _run(
        _step("_unit-tests.yml", "select", "cache-compression")["run"],
        tmp_path,
        {**environment, "RUNNER_TEMP": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    restore = _step("_unit-tests.yml", "select", "plan")
    assert restore["env"]["PATH"] == "${{ steps.cache-compression.outputs.path }}"
    workflow = yaml.safe_load((ROOT / ".github/workflows/_unit-tests.yml").read_text())
    assert all(
        "PATH" not in step.get("env", {})
        for step in workflow["jobs"]["select"]["steps"]
        if step.get("id") != "plan"
    )
    probe = subprocess.run(
        ["zstd", "--quiet", "--version"],
        env={**environment, "PATH": compression["path"]},
        text=True,
        capture_output=True,
    )
    # The pinned cache action uses gzip when the version probe emits no output.
    assert probe.stdout == probe.stderr == ""
    assert probe.returncode != 0
    assert (
        subprocess.check_output(["zstd", "--quiet", "--version"], env=environment, text=True)
        == "1.5.7\n"
    )

    bucket = "tests/unit_tests/test_example.py"
    outputs = {"has_tests": str(has_tests).lower()}
    assert workflow["jobs"]["select"]["runs-on"] == "ubuntu-latest"
    assert workflow["jobs"]["select"]["name"] == "unit tests selection"
    supported = (
        "(inputs.unit_testmon_target_branch == 'main' || "
        "inputs.unit_testmon_target_branch == 'dev')"
    )
    assert workflow["jobs"]["select"]["if"] == (
        f"inputs.unit_testmon_mode == 'enforce' && {supported}"
    )
    for target, selects in (("main", True), ("dev", True), ("core_r0.16.0", False), ("", False)):
        context = {
            "inputs.unit_testmon_mode": "enforce",
            "inputs.unit_testmon_target_branch": target,
        }
        assert _evaluate(workflow["jobs"]["select"]["if"], context) is selects
    for step in workflow["jobs"]["select"]["steps"]:
        assert "unit_testmon_mode" not in step.get("if", "")
    # Dev baselines are artifacts of other workflow runs.
    assert workflow["permissions"]["actions"] == "read"
    assert workflow["jobs"]["select"]["permissions"] == {"actions": "read", "contents": "read"}
    assert restore["env"]["GH_TOKEN"] == "${{ github.token }}"
    for platform, parser_job, output in (
        ("h100", "cicd-parse-unit-tests", "unit-tests"),
        ("gb200", "cicd-parse-unit-tests-gb200", "unit-tests-gb200"),
    ):
        parser = _step("cicd-main.yml", parser_job, "parse-unit-tests")
        result, parsed = _run(parser["run"], tmp_path, shell_environment)
        assert result.returncode == 0, result.stderr
        recipe = yaml.safe_load(
            (ROOT / "tests/test_utils/recipes" / platform / "unit-tests.yaml").read_text()
        )
        assert json.loads(parsed[output]) == [
            {"bucket": case} for product in recipe["products"] for case in product["test_case"]
        ]
    assert restore["env"]["BUCKETS"] == "${{ inputs.buckets }}"
    assert workflow[True]["workflow_call"]["inputs"]["buckets"]["required"] is True
    assert workflow[True]["workflow_call"]["inputs"]["buckets"]["type"] == "string"
    assert workflow["jobs"]["run"]["needs"] == "select"
    expression = workflow["jobs"]["run"]["if"].removeprefix("${{").removesuffix("}}")
    expression = expression.replace("!cancelled()", "True").replace("&&", "and").replace("||", "or")
    needs = SimpleNamespace(
        select=SimpleNamespace(result="success", outputs=SimpleNamespace(**outputs))
    )
    inputs = SimpleNamespace(unit_testmon_mode="enforce", unit_testmon_target_branch="main")
    context = {"needs": needs, "inputs": inputs}
    for target in ("main", "dev"):
        inputs.unit_testmon_target_branch = target
        assert eval(expression, {"__builtins__": {}}, context) == has_tests
    assert workflow["jobs"]["run"]["strategy"]["matrix"]["include"] == (
        "${{ fromJSON(needs.select.outputs.matrix || inputs.buckets) }}"
    )
    run_inputs = workflow["jobs"]["run"]["steps"][-1]["with"]
    assert run_inputs["unit_testmon_mode"] == (
        "${{ inputs.unit_testmon_mode == 'enforce' && " + supported + " && 'enforce' || 'full' }}"
    )
    assert "unit_testmon_plan_artifact" not in run_inputs
    for name in ("run_id", "artifact_id", "key", "source_sha"):
        assert run_inputs[f"unit_testmon_baseline_{name}"] == f"${{{{ matrix.baseline_{name} }}}}"

    # CPU failures stop GPU scheduling; full-suite requests bypass selection.
    needs.select.result = "failure"
    assert not eval(expression, {"__builtins__": {}}, context)
    needs.select.result = "skipped"
    assert not eval(expression, {"__builtins__": {}}, context)
    # Targets without a trusted baseline skip selection and run every bucket in full.
    inputs.unit_testmon_target_branch = "core_r0.16.0"
    assert eval(expression, {"__builtins__": {}}, context)
    inputs.unit_testmon_target_branch = "dev"
    inputs.unit_testmon_mode = "full"
    assert eval(expression, {"__builtins__": {}}, context)
    callers = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())["jobs"]
    for caller, parser, output in (
        ("cicd-unit-tests-latest", "cicd-parse-unit-tests", "unit-tests"),
        ("cicd-unit-tests-latest-gb200", "cicd-parse-unit-tests-gb200", "unit-tests-gb200"),
    ):
        assert "strategy" not in callers[caller]
        # A called workflow cannot elevate the caller's token permissions.
        assert callers[caller]["permissions"] == {
            "actions": "read",
            "contents": "read",
            "id-token": "write",
        }
        assert parser in callers[caller]["needs"]
        assert callers[caller]["with"]["buckets"] == (
            "${{ needs." + parser + ".outputs." + output + " }}"
        )
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
    ("requested_mode", "target_branch", "cpu_mode", "expected_mode"),
    [
        pytest.param("enforce", "main", "preselected", "enforce", id="main-cpu-selected"),
        pytest.param("enforce", "main", "full", "enforce", id="main-cpu-fallback"),
        pytest.param("enforce", "dev", "preselected", "enforce", id="dev-cpu-selected"),
        pytest.param("enforce", "dev", "full", "enforce", id="dev-cpu-fallback"),
        pytest.param("full", "main", "", "full", id="explicit-full"),
        pytest.param("full", "dev", "", "full", id="dev-explicit-full"),
        pytest.param("full", "", "", "full", id="merge-group-full"),
        pytest.param("enforce", "core_r0.16.0", "", "full", id="unsupported-target"),
    ],
)
def test_gpu_reselection_respects_full_run_guards(
    requested_mode, target_branch, cpu_mode, expected_mode
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/_unit-tests.yml").read_text())
    mode = workflow["jobs"]["run"]["steps"][-1]["with"]["unit_testmon_mode"]
    context = {
        "inputs.unit_testmon_mode": requested_mode,
        "inputs.unit_testmon_target_branch": target_branch,
        "matrix.mode": cpu_mode,
    }
    # Every dispatched bucket reselects natively, including CPU full fallbacks.
    assert _evaluate(mode, context) == expected_mode


@pytest.mark.parametrize(
    "scenario",
    [
        "mixed",
        "empty",
        "cache-error",
        "selector-error",
        "upload-error",
        "pinned",
        "pinned-miss",
        "pinned-other-generation",
        "pinned-source-mismatch",
    ],
)
def test_grouped_cpu_selection(
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
        json.dumps({"tested_sha": "a" * 40, "changed_paths": 0})
    )
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(unit_tests),
        "BUCKETS": json.dumps(buckets),
        "RECIPE_PLATFORM": "dgx_h100",
        "SOURCE_SHA": "a" * 40,
        "TARGET_BRANCH": "main",
        "TESTMON_GENERATION": "111-1" if scenario.startswith("pinned") else "",
        "TESTMON_BASE_SHA": "b" * 40 if scenario.startswith("pinned") else "",
        "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
        "PR_FILES_DIR": str(pr_files),
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
            cache_prefix = json.loads((tmp_path / "unit-testmon-identity.json").read_text())[
                "cache_prefix"
            ]
            if scenario.startswith("pinned"):
                assert key == f"{cache_prefix}111-1"
                assert prefix == "", "a pinned baseline must not use any restore-key fallback"
            else:
                assert prefix == cache_prefix
                assert key == f"{prefix}lookup-123-2"
            assert not Path(path).exists(), "previous bucket cache was not cleared"
            index = restores
            restores += 1
            if scenario == "selector-error" and index == 1:
                (unit_tests / "testmon_selector.py").unlink()
            if scenario == "cache-error" and index == 0:
                raise subprocess.CalledProcessError(1, [operation])
            saved = tmp_path / "saved" / str(index)
            if scenario == "pinned-miss" and index == 0:
                Path(key_file).write_text("")
                return
            if saved.exists():
                shutil.copytree(saved, path)
            matched_generation = "111-1"
            if scenario == "pinned-other-generation" and index == 0:
                # Even a complete compatible generation must not substitute for
                # the baseline used to construct the tested checkout.
                matched_generation = "222-1"
                manifest_path = Path(path) / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["generation"] = matched_generation
                manifest_path.write_text(json.dumps(manifest))
            if index == 0 and scenario == "pinned-source-mismatch":
                manifest_path = Path(path) / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["source_sha"] = "c" * 40
                manifest_path.write_text(json.dumps(manifest))
            Path(key_file).write_text(
                f"{cache_prefix}{matched_generation}" if saved.exists() else ""
            )
        else:
            assert operation == "upload"
            if scenario == "upload-error":
                raise subprocess.CalledProcessError(1, [operation])
            name, plan, root = args
            assert root == "assets_dir/testmon-plan"
            assert plan == f"{root}/plan.json"
            uploads.append({"name": name, "plan": json.loads(Path(plan).read_text())})

    monkeypatch.setattr(selector, "_actions", actions)
    if scenario in {"selector-error", "upload-error"}:
        with pytest.raises(subprocess.CalledProcessError):
            selector.main()
        assert not (tmp_path / "outputs").exists()
        assert restores == (2 if scenario == "selector-error" else 1)
        assert len(uploads) == (1 if scenario == "selector-error" else 0)
        return
    selector.main()
    assert restores == len(buckets)
    outputs = dict(line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines())
    matrix = json.loads(outputs["matrix"])
    assert outputs["has_tests"] == str(scenario != "empty").lower()
    assert [entry["bucket"] for entry in matrix] == [
        entry["bucket"] for entry in buckets if entry["bucket"] != "tests/unit_tests/empty/**/*.py"
    ]
    fallback = scenario in {
        "cache-error",
        "pinned-miss",
        "pinned-other-generation",
        "pinned-source-mismatch",
    }
    assert [entry["mode"] for entry in matrix] == (
        [] if scenario == "empty" else [("full" if fallback else "preselected"), "full"]
    )
    assert all(entry["timeout"] == 40 for entry in matrix)
    assert bool(caplog.records) == (fallback and scenario != "pinned-miss")
    for upload in uploads:
        plan = upload["plan"]
        assert re.fullmatch(r"unit-test-plan-dgx_h100-[0-9a-f]{16}-123-2", upload["name"])
        expected = (
            ["tests/unit_tests/selected/test_case.py"] if "/selected/" in plan["bucket"] else []
        )
        assert plan["phases"] == {"prod": expected, "experimental": expected}
        assert plan["source_sha"] == "a" * 40
        matches = [entry for entry in matrix if entry["bucket"] == plan["bucket"]]
        assert bool(matches) == bool(expected)
        if matches:
            assert matches[0]["plan_artifact"] == upload["name"]
    assert any("/empty/" in upload["plan"]["bucket"] for upload in uploads)
    assert all(entry["plan_artifact"] == "" for entry in matrix if entry["mode"] == "full")


def test_baseline_branch_lists_agree_with_the_cache_helper() -> None:
    spec = importlib.util.spec_from_file_location(
        "branch_testmon_cache", ROOT / "tests/unit_tests/testmon_cache.py"
    )
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    branches = set(helper.BASELINE_BRANCHES)
    assert branches == {"main", "dev"}
    unit = yaml.safe_load((ROOT / ".github/workflows/_unit-tests.yml").read_text())["jobs"]
    for expression in (
        unit["select"]["if"],
        unit["run"]["if"],
        unit["run"]["steps"][-1]["with"]["unit_testmon_mode"],
    ):
        assert set(re.findall(r"unit_testmon_target_branch [!=]= '([^']*)'", expression)) == (
            branches
        )
    configure = _step("cicd-main.yml", "configure", "configure")["run"]
    assert set(re.findall(r"^\s+([a-z]+)\) UNIT_TESTMON_BASELINE=", configure, re.M)) == branches
    action = yaml.safe_load((ROOT / ".github/actions/action.yml").read_text())
    guard = next(
        step["run"]
        for step in action["runs"]["steps"]
        if step["name"] == "Validate unit Testmon baseline producer"
    )
    assert set(re.findall(r'"\$TARGET_BRANCH" == "([^"]+)"', guard)) == branches
    producer = yaml.safe_load((ROOT / ".github/workflows/populate-build-cache.yml").read_text())
    source_guard = producer["jobs"]["validate-source"]["steps"][0]["run"]
    assert set(re.findall(r'"refs/heads/([^"]+)"', source_guard)) == branches


@pytest.mark.parametrize("transport", ["cache", "artifact", ""])
def test_action_restores_and_publishes_each_transport_once(transport: str) -> None:
    action = yaml.safe_load((ROOT / ".github/actions/action.yml").read_text())
    steps = {step.get("id") or step["name"]: step for step in action["runs"]["steps"]}
    branch = "dev" if transport == "artifact" else "main"
    context = {
        "inputs.is_unit_test": "true",
        "inputs.unit_testmon_mode": "enforce",
        "inputs.unit_testmon_target_branch": branch,
        "inputs.test_case": "tests/unit_tests/models/**/*.py",
        "inputs.unit_testmon_baseline_artifact_id": "13",
        "inputs.unit_testmon_baseline_run_id": "250",
        "steps.unit-testmon-identity.outcome": "success",
        "steps.unit-testmon-identity.outputs.baseline_transport": transport,
    }

    def evaluate(expression: str, overrides: dict[str, str] | None = None) -> bool:
        functions = {
            "startsWith(": "_starts(",
            "contains(": "_has(",
            "format('refs/heads/{0}', ": "_ref(",
        }
        for name, replacement in functions.items():
            expression = expression.replace(name, replacement)
        return eval(
            _expression(expression),
            {
                "__builtins__": {},
                "_starts": lambda value, prefix: value.startswith(prefix),
                "_has": lambda value, part: part in value,
                "_ref": lambda value: f"refs/heads/{value}",
            },
            {"context": {**context, **(overrides or {})}},
        )

    # Consumers: main restores Actions caches; dev downloads the CPU-resolved artifact.
    # Helpers that predate dev baselines print no transport; they only knew main's cache.
    assert evaluate(steps["restore-unit-testmon"]["if"]) is (transport != "artifact")
    assert evaluate(steps["download-unit-testmon-baseline"]["if"]) is (transport == "artifact")
    assert not evaluate(
        steps["download-unit-testmon-baseline"]["if"],
        {"inputs.unit_testmon_baseline_artifact_id": ""},
    )
    download = steps["download-unit-testmon-baseline"]
    assert download["continue-on-error"] is True
    assert download["with"] == {
        "artifact-ids": "${{ inputs.unit_testmon_baseline_artifact_id }}",
        "run-id": "${{ inputs.unit_testmon_baseline_run_id }}",
        "github-token": "${{ github.token }}",
        "path": "assets_dir/testmon",
        "merge-multiple": True,
    }
    resolve = steps["unit-testmon"]["env"]
    assert resolve["MATCHED_KEY"] == "${{ steps.unit-testmon-baseline.outputs.matched_key }}"
    assert resolve["BASELINE_SOURCE_SHA"] == "${{ inputs.unit_testmon_baseline_source_sha }}"

    # Producers: main saves caches; dev uploads the validated generation under its key.
    context.update(
        {
            "steps.unit-testmon.outputs.mode": "baseline",
            "steps.prepare-unit-testmon-cache.outcome": "success",
            "steps.validate-unit-testmon-artifact.outcome": "success",
            "github.repository": "NVIDIA/Megatron-LM",
            "github.event_name": "push" if transport == "artifact" else "schedule",
            "github.ref": f"refs/heads/{branch}",
        }
    )
    for step, cache_only in (
        ("Save unit Testmon baseline", True),
        ("verify-unit-testmon-cache", True),
        ("confirm-unit-testmon-cache", True),
        ("validate-unit-testmon-artifact", False),
        ("upload-unit-testmon-artifact", False),
    ):
        assert evaluate(steps[step]["if"]) is ((transport != "artifact") == cache_only), step
    key = (
        "${{ steps.unit-testmon-identity.outputs.cache_prefix }}"
        "${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert steps["upload-unit-testmon-artifact"]["with"] == {
        "name": key,
        "path": "assets_dir/testmon",
        "include-hidden-files": True,
        "if-no-files-found": "error",
        "retention-days": 30,
    }
    assert steps["validate-unit-testmon-artifact"]["env"] == {
        "EXPECTED_KEY": key,
        "SOURCE_SHA": "${{ github.sha }}",
    }
    assert steps["confirm-unit-testmon-artifact"]["env"]["ARTIFACT_NAME"] == key
    if transport == "artifact":
        for untrusted in (
            {"github.ref": "refs/heads/pull-request/6934"},
            {"github.ref": "refs/heads/main"},
            {"github.repository": "fork/Megatron-LM"},
            {"github.event_name": "pull_request"},
            {"github.event_name": "merge_group"},
            {"steps.validate-unit-testmon-artifact.outcome": "failure"},
        ):
            assert not evaluate(steps["upload-unit-testmon-artifact"]["if"], untrusted), untrusted
    assert steps["check"]["env"]["CACHE_PUBLICATION"] == (
        "${{ steps.unit-testmon-identity.outputs.baseline_transport == 'artifact' && "
        "steps.confirm-unit-testmon-artifact.outcome || steps.confirm-unit-testmon-cache.outcome }}"
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "trusted",
        "listing-error",
        "download-error",
        "pinned",
        "pinned-untrusted",
        "pinned-source-mismatch",
    ],
)
def test_grouped_cpu_selection_reads_only_trusted_dev_artifacts(
    tmp_path: Path, scenario: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Dev buckets select from dev producer artifacts; anything untrusted runs in full."""
    pytest.importorskip("testmon")
    names = ["selected", "empty", "missing", "main-only", "wrong-commit"]
    buckets = [{"bucket": f"tests/unit_tests/{name}/**/*.py"} for name in names]
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
    for name in names:
        if name != "empty":
            test_file = unit_tests / name / "test_case.py"
            test_file.parent.mkdir()
            test_file.write_text("def test_case(): pass\n")
    pr_files = tmp_path / "pr-files"
    pr_files.mkdir()
    (pr_files / "changed-files").write_text("")
    (pr_files / "metadata.json").write_text(
        json.dumps({"tested_sha": "a" * 40, "changed_paths": 0})
    )
    tip, older, newer, outside = "e" * 40, "1" * 40, "2" * 40, "3" * 40
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(unit_tests),
        "BUCKETS": json.dumps(buckets),
        "RECIPE_PLATFORM": "dgx_h100",
        "SOURCE_SHA": "a" * 40,
        "TARGET_BRANCH": "dev",
        "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
        "PR_FILES_DIR": str(pr_files),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_RUN_ID": "900",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        # A prepared checkout pins the generation of one producer run.
        "TESTMON_GENERATION": {"pinned-untrusted": "310-1"}.get(
            scenario, "250-1" if scenario.startswith("pinned") else ""
        ),
        "TESTMON_BASE_SHA": {"pinned-source-mismatch": newer}.get(
            scenario, older if scenario.startswith("pinned") else ""
        ),
    }
    # Record real generations exactly as the dev producer does. "main-only" has
    # only a main-namespace artifact; "wrong-commit" claims a different source.
    producer = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, sys
from pathlib import Path
from testmon.db import DB
from testmon_cache import PHASES, cache_identity, record_phase, finalize
keys = {}
for name, branch, sha, generation in json.loads(sys.argv[1]):
    bucket = f'tests/unit_tests/{name}/**/*.py'
    identity = cache_identity(Path.cwd(), bucket, 'dgx_h100', target_branch=branch)
    key = identity['cache_prefix'] + generation
    directory = Path('saved') / key
    for phase in PHASES:
        database = directory / phase / '.testmondata'
        database.parent.mkdir(parents=True)
        DB(str(database)).con.close()
        record_phase(directory, phase)
    finalize(directory, identity, sha, generation)
    keys[f'{name}/{branch}/{generation}'] = key
print(json.dumps(keys))
""",
            json.dumps(
                [
                    ["selected", "dev", older, "250-1"],
                    ["selected", "dev", older, "250-2"],
                    ["selected", "dev", outside, "270-1"],
                    ["empty", "dev", newer, "260-1"],
                    ["empty", "dev", older, "250-1"],
                    ["main-only", "main", older, "250-1"],
                    ["wrong-commit", "dev", "f" * 40, "250-1"],
                ]
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    keys = json.loads(producer.stdout)

    def run(run_id, sha, **overrides):
        return {
            "id": run_id,
            "event": "push",
            "head_branch": "dev",
            "head_sha": sha,
            "path": ".github/workflows/populate-build-cache.yml",
            "repository": {"full_name": "NVIDIA/Megatron-LM"},
            "head_repository": {"full_name": "NVIDIA/Megatron-LM"},
            **overrides,
        }

    def artifact(artifact_id, key, run_id, sha, **overrides):
        return {
            "id": artifact_id,
            "name": key,
            "expired": False,
            "digest": f"sha256:{artifact_id:064x}",
            "workflow_run": {"id": run_id, "head_sha": sha},
            **overrides,
        }

    selected = keys["selected/dev/250-2"]
    runs = [
        # Untrusted or unrelated runs, newest first, even with plausible artifacts.
        run(310, newer, event="pull_request"),
        run(300, newer, head_repository={"full_name": "fork/Megatron-LM"}),
        run(290, newer, head_branch="dev-feature"),
        run(280, newer, path=".github/workflows/other.yml"),
        run(270, outside),
        run(260, newer),
        run(250, older, event="workflow_dispatch"),
        run(240, older, id="240"),
    ]
    artifacts = {
        270: [artifact(27, keys["selected/dev/270-1"], 270, outside)],
        260: [
            artifact(26, keys["empty/dev/260-1"], 260, newer),
            artifact(31, selected.replace("-250-2", "-260-1"), 250, older),
        ],
        250: [
            artifact(21, keys["selected/dev/250-1"], 250, older),
            artifact(22, selected, 250, older),
            artifact(23, selected.replace("-250-2", "-250-3"), 250, older, expired=True),
            artifact(24, keys["empty/dev/250-1"], 250, older),
            artifact(25, keys["main-only/main/250-1"], 250, older),
            artifact(28, keys["wrong-commit/dev/250-1"], 250, older),
            artifact(29, "logs-tests-unit_tests-selected-250", 250, older),
        ],
    }
    compare = {newer: "ahead", older: "ahead", outside: "diverged"}
    downloads = []

    def gh(*args):
        if args[0] == "api":
            if scenario == "listing-error":
                raise subprocess.CalledProcessError(1, ["gh", *args])
            endpoint = next(arg for arg in args[1:] if arg != "--paginate")
            query = args[-1]
            if endpoint == "repos/NVIDIA/Megatron-LM/git/ref/heads/dev":
                assert query == ".object.sha"
                return tip + "\n"
            if endpoint.startswith(
                "repos/NVIDIA/Megatron-LM/actions/workflows/populate-build-cache.yml/runs?"
                "branch=dev&"
            ):
                assert not scenario.startswith("pinned"), "a pinned run needs no discovery"
                assert query == ".workflow_runs[] | @json" and "--paginate" not in args
                return "".join(json.dumps(entry) + "\n" for entry in runs)
            if endpoint.startswith("repos/NVIDIA/Megatron-LM/compare/"):
                base, head = endpoint.split("/compare/")[1].split("?")[0].split("...")
                assert head == tip and query == ".status"
                return compare[base] + "\n"
            if re.fullmatch(r"repos/NVIDIA/Megatron-LM/actions/runs/[0-9]+", endpoint):
                # A pinned generation names one run; it gets the same trust checks.
                assert scenario.startswith("pinned") and query == ". | @json"
                pinned = next(
                    entry for entry in runs if str(entry["id"]) == endpoint.split("/")[-1]
                )
                return json.dumps(pinned) + "\n"
            run_id = int(endpoint.split("/actions/runs/")[1].split("/")[0])
            assert query == ".artifacts[] | @json" and "--paginate" in args
            assert run_id in {270, 260, 250}, f"untrusted run {run_id} was inspected"
            return "".join(json.dumps(entry) + "\n" for entry in artifacts[run_id])
        raise AssertionError(f"unexpected gh command: {args}")

    spec = importlib.util.spec_from_file_location(
        "select_dev_unit_tests", ROOT / ".github/scripts/select_unit_tests.py"
    )
    assert spec is not None and spec.loader is not None
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    monkeypatch.chdir(tmp_path)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    uploads = []
    artifact_names = {
        entry["id"]: entry["name"]
        for run_artifacts in artifacts.values()
        for entry in run_artifacts
    }

    def actions(operation, *args):
        # Dev baselines never use the Actions cache.
        assert operation in {"upload", "download"}, args
        if operation == "upload":
            uploads.append(json.loads(Path(args[1]).read_text()))
            return
        artifact_id, directory, run_id, repository, digest = args
        assert directory == "assets_dir/testmon" and repository == "NVIDIA/Megatron-LM"
        assert digest == f"sha256:{int(artifact_id):064x}"
        assert not Path(directory).exists(), "previous bucket baseline was not cleared"
        name = artifact_names[int(artifact_id)]
        downloads.append((run_id, name))
        if scenario == "download-error" and name == selected:
            raise subprocess.CalledProcessError(1, ["node", operation, *args])
        shutil.copytree(tmp_path / "saved" / name, directory)

    monkeypatch.setattr(selector, "_actions", actions)
    monkeypatch.setattr(selector, "_gh", gh)
    selector.main()

    outputs = dict(line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines())
    matrix = {entry["bucket"].split("/")[2]: entry for entry in json.loads(outputs["matrix"])}
    plans = {plan["bucket"].split("/")[2]: plan for plan in uploads}
    summary = (tmp_path / "summary").read_text()
    assert outputs["has_tests"] == "true"
    assert "Target branch: `dev`" in summary
    if scenario == "listing-error":
        # Without a verified baseline even an apparently empty bucket runs in full.
        assert downloads == [] and plans == {}
        assert set(matrix) == set(names)
        assert all(
            entry["mode"] == "full" and "baseline_key" not in entry for entry in matrix.values()
        )
        assert "Unable to list trusted dev baselines" in caplog.text
        assert summary.count("no trusted `dev` artifact") == len(names)
        return
    if scenario in {"pinned-untrusted", "pinned-source-mismatch"}:
        # An untrusted pinned run, or a source that differs from the prepared
        # checkout's baseline, yields no baseline: every bucket runs in full.
        assert downloads == [] and plans == {}
        assert set(matrix) == set(names)
        assert all(
            entry["mode"] == "full" and "baseline_key" not in entry for entry in matrix.values()
        )
        return
    if scenario == "pinned":
        # Only the pinned run and generation count, even when a newer run exists.
        assert downloads == [
            ("250", keys["selected/dev/250-1"]),
            ("250", keys["empty/dev/250-1"]),
            ("250", keys["wrong-commit/dev/250-1"]),
        ]
        assert matrix["selected"]["baseline_key"] == keys["selected/dev/250-1"]
        assert matrix["selected"]["mode"] == "preselected"
        assert plans["empty"]["phases"] == {"prod": [], "experimental": []}
        assert matrix["wrong-commit"]["mode"] == "full"
        assert "expected producer commit" in summary
        return
    # The newest trusted run wins per bucket, then the newest attempt in that run.
    assert downloads == [
        ("250", selected),
        ("260", keys["empty/dev/260-1"]),
        ("250", keys["wrong-commit/dev/250-1"]),
    ]
    assert set(matrix) == {"selected", "missing", "main-only", "wrong-commit"}
    baseline = {
        "baseline_run_id": "250",
        "baseline_artifact_id": "22",
        "baseline_key": selected,
        "baseline_source_sha": older,
    }
    assert {key: matrix["selected"][key] for key in baseline} == baseline
    if scenario == "trusted":
        assert matrix["selected"]["mode"] == "preselected"
        assert plans["selected"]["phases"] == {
            "prod": ["tests/unit_tests/selected/test_case.py"],
            "experimental": ["tests/unit_tests/selected/test_case.py"],
        }
        assert plans["selected"]["baseline_source_sha"] == older
        assert "| 1 affected file(s), 0 mandatory | Testmon reselects |" in summary
    else:
        assert matrix["selected"]["mode"] == "full"
        assert "download failed | full bucket: no baseline | Testmon reselects |" in summary
    # A dev baseline with no affected tests skips the GPU entirely.
    assert plans["empty"]["phases"] == {"prod": [], "experimental": []}
    assert (
        f"| `{keys['empty/dev/260-1']}` (`{newer[:12]}`) | no affected tests | not allocated |"
        in summary
    )
    # Missing, main-only, and unverifiable artifacts run full; never silently skip.
    for name in ("missing", "main-only"):
        assert matrix[name]["mode"] == "full" and "baseline_key" not in matrix[name]
    assert matrix["wrong-commit"]["mode"] == "full"
    assert matrix["wrong-commit"]["baseline_key"] == keys["wrong-commit/dev/250-1"]
    assert "expected producer commit" in summary
    assert summary.count("no trusted `dev` artifact") == 2


def test_grouped_cpu_selection_rejects_targets_without_a_trusted_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit_tests = tmp_path / "tests/unit_tests"
    unit_tests.mkdir(parents=True)
    (unit_tests / "testmon_cache.py").symlink_to(ROOT / "tests/unit_tests/testmon_cache.py")
    spec = importlib.util.spec_from_file_location(
        "select_release_unit_tests", ROOT / ".github/scripts/select_unit_tests.py"
    )
    assert spec is not None and spec.loader is not None
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "BUCKETS": json.dumps([{"bucket": "tests/unit_tests/example/**/*.py"}]),
        "RECIPE_PLATFORM": "dgx_h100",
        "TARGET_BRANCH": "core_r0.16.0",
        "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(key, value)

    def unexpected(*args):
        raise AssertionError(f"unexpected baseline access: {args}")

    monkeypatch.setattr(selector, "_actions", unexpected)
    monkeypatch.setattr(selector, "_gh", unexpected)
    with pytest.raises(subprocess.CalledProcessError):
        selector.main()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("mismatch", [False, True], ids=["verified", "digest-mismatch"])
def test_selection_sdk_adapter_downloads_artifacts_from_other_runs(
    tmp_path: Path, mismatch: bool
) -> None:
    node = shutil.which("node") or ""
    if not node:
        pytest.skip("Adapter validation requires Node.js")
    script = _step("_unit-tests.yml", "select", "plan")["with"]["script"]
    client = script.split("const client = `", 1)[1].split("`;", 1)[0]
    modules = tmp_path / "testmon/node_modules/@actions"
    calls = tmp_path / "calls.jsonl"
    for package, body in (
        (
            "artifact",
            "class DefaultArtifactClient {\n"
            "  async downloadArtifact(id, options) {\n"
            "    record(['download', id, options]);\n"
            "    return {digestMismatch: process.env.TEST_MISMATCH === 'true'};\n"
            "  }\n"
            "  async uploadArtifact(...args) { record(['upload', ...args]); }\n"
            "}\n"
            "module.exports = {DefaultArtifactClient};\n",
        ),
        (
            "cache",
            "module.exports = {restoreCache: async (...args) => {\n"
            "  record(['restore', ...args]); return args[1] + '-matched';\n"
            "}};\n",
        ),
    ):
        (modules / package).mkdir(parents=True)
        (modules / package / "index.js").write_text(
            "const record = entry => require('node:fs').appendFileSync(\n"
            "  process.env.TEST_CALLS, JSON.stringify(entry) + '\\n');\n" + body
        )
    environment = {
        **os.environ,
        "RUNNER_TEMP": str(tmp_path),
        "GH_TOKEN": "token-value",
        "TEST_CALLS": str(calls),
        "TEST_MISMATCH": str(mismatch).lower(),
    }

    def adapter(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [node, "-e", client, *args],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )

    download = adapter("download", "13", "assets_dir/testmon", "250", "NVIDIA/Megatron-LM", "")
    assert (download.returncode != 0) is mismatch, download.stderr
    assert ("Artifact 13 digest mismatch" in download.stderr) is mismatch
    restored = adapter("restore", "assets_dir/testmon", "prefix-lookup-1-1", "prefix-", "key")
    assert restored.returncode == 0, restored.stderr
    assert (tmp_path / "key").read_text() == "prefix-lookup-1-1-matched"
    uploaded = adapter("upload", "plan-artifact", "plan.json", "plan-root")
    assert uploaded.returncode == 0, uploaded.stderr
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        [
            "download",
            13,
            {
                "path": "assets_dir/testmon",
                "findBy": {
                    "token": "token-value",
                    "workflowRunId": 250,
                    "repositoryOwner": "NVIDIA",
                    "repositoryName": "Megatron-LM",
                },
            },
        ],
        ["restore", ["assets_dir/testmon"], "prefix-lookup-1-1", ["prefix-"]],
        ["upload", "plan-artifact", ["plan.json"], "plan-root"],
    ]


@pytest.mark.parametrize(
    ("base_ref", "ready", "use_cache_base", "prepares"),
    [
        ("main", "true", "", True),
        ("dev", "true", "", True),
        ("dev", "true", "false", False),
        ("main", "true", "false", False),
        ("dev", "", "", False),
        ("core_r0.16.0", "true", "", False),
    ],
)
def test_configure_prepares_baseline_checkouts_for_trusted_targets(
    base_ref: str, ready: str, use_cache_base: str, prepares: bool
) -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())["jobs"][
        "configure"
    ]["steps"]
    checkout = next(
        step for step in steps if step.get("name") == "Checkout PR for Testmon baseline preparation"
    )
    prepare = next(step for step in steps if step.get("id") == "prepare-testmon-checkout")
    assert checkout["if"] == prepare["if"]
    base = "fromJSON(steps.get-pr-info.outputs.pr-info || '{}').base.ref"
    context = {
        "steps.unit-test-pr-files.outputs.ready": ready,
        "vars.UNIT_TESTMON_USE_CACHE_BASE": use_cache_base,
        "base": base_ref,
    }
    assert _evaluate(prepare["if"].replace(base, "context['base']"), context) is prepares
    # The producer lookup follows the PR target: dev pins a trusted dev producer run.
    assert prepare["env"]["TARGET_BRANCH"] == "${{ " + base + " }}"
    assert '--target-branch "$TARGET_BRANCH"' in prepare["run"]
