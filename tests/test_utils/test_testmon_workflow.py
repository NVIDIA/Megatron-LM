# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def test_mhc_kernel_file_has_one_recipe_bucket(monkeypatch, capsys):
    recipe = yaml.safe_load((ROOT / "tests/test_utils/recipes/h100/unit-tests.yaml").read_text())
    buckets = [case for product in recipe["products"] for case in product["test_case"]]
    kernel_file = "tests/unit_tests/fusions/test_fused_mhc_kernels.py"
    assert buckets.count(kernel_file) == 1

    spec = importlib.util.spec_from_file_location(
        "mhc_test_case_finder", ROOT / "tests/unit_tests/find_test_cases.py"
    )
    assert spec is not None and spec.loader is not None
    finder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finder)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(finder, "get_test_cases", lambda _: buckets)

    for bucket, excluded in [("tests/unit_tests/**/*.py", True), (kernel_file, False)]:
        monkeypatch.setattr(sys, "argv", ["find_test_cases.py", bucket, "h100"])
        finder.main()
        ignored = capsys.readouterr().out.splitlines()
        assert (f"--ignore={kernel_file}" in ignored) is excluded


@pytest.mark.parametrize("mode", ["full", "baseline", "enforce"])
@pytest.mark.parametrize(
    "bucket,workers",
    [
        ("tests/unit_tests/fusions/test_fused_mhc_kernels.py", 1),
        ("tests/unit_tests/**/*.py", 8),
        ("tests/unit_tests/generalized_tensor_parallel/**/*.py", 4),
    ],
)
def test_unit_recipe_worker_count_is_shared_by_baseline_and_consumers(
    tmp_path, mode, bucket, workers
):
    recipe = yaml.safe_load((ROOT / "tests/test_utils/recipes/h100/unit-tests.yaml").read_text())
    script = recipe["spec"]["script"].format(
        tag="latest", environment="dev", test_case=bucket, n_repeat=1, assets_dir=tmp_path
    )
    # Execute the recipe, intercepting filesystem operations and the GPU harness.
    stubs = '\n'.join(
        (
            'rm() { :; }',
            'ls() { :; }',
            'cd() { :; }',
            'bash() { printf "workers=%s\\n" "$GPUS_PER_NODE" >> "$GITHUB_OUTPUT"; }',
        )
    )
    result, outputs = _run(
        stubs + "\n" + script, tmp_path, {"UNIT_TESTMON_MODE": mode, "GPUS_PER_NODE": "8"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["workers"] == str(workers)


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
        "assert sys.argv[1:4] == ['-o', 'json', '[.products[].test_case[]]']\n"
        "with open(sys.argv[4]) as stream:\n"
        "    recipe = yaml.safe_load(stream)\n"
        "print(json.dumps([case for product in recipe['products'] for case in product['test_case']]))\n"
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
            "REGISTRY_GB_GPU": "gb200.example.test/team",
            "SELECTED_RUNNER": "h100-runner",
            "SELECTED_RUNNER_GB_GPU": "gb200-runner",
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
    }


def _run_configure(
    tmp_path: Path, environment: dict[str, str], labels: list[str]
) -> tuple[subprocess.CompletedProcess, dict]:
    script = _step("cicd-main.yml", "configure", "configure")["run"]
    script = script.replace(
        "${{ fromJSON(steps.get-pr-info.outputs.pr-info || '{}').number }}", "6934"
    )
    script = script.replace("${{ github.repository }}", "NVIDIA/Megatron-LM")
    return _run(script, tmp_path, {**environment, "TEST_PR_LABELS": json.dumps(labels)})


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
