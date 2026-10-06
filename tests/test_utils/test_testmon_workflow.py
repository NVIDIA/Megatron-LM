# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "empty-pr",
        "large-pr",
        "special-characters",
        "missing-json",
        "malformed-json",
        "object-json",
        "non-string-path",
        "newline-path",
        "count-mismatch",
        "invalid-tested-sha",
        "invalid-file-count",
    ],
)
def test_shared_pr_files_producer_packages_changed_files_output(
    tmp_path: Path, shell_environment: dict[str, str], scenario: str
) -> None:
    gh = tmp_path / "bin/gh"
    gh.write_text('#!/bin/sh\necho "$*" >> "$GH_LOG"\nexit 99\n')
    gh.chmod(0o755)
    gh_log = tmp_path / "gh.log"
    runner_temp = tmp_path / "runner"
    expected_files = (
        [] if scenario == "empty-pr" else ["megatron/core/a.py", "tests/unit_tests/test_b.py"]
    )
    if scenario == "large-pr":
        expected_files = [f"megatron/core/source_{index}.py" for index in range(3001)]
    elif scenario == "special-characters":
        expected_files = [
            "megatron/core/a b.py",
            'megatron/core/quoted"file.py',
            "megatron/core/unicode_é.py",
            r"megatron/core/back\slash.py",
        ]
    elif scenario == "newline-path":
        expected_files = ["megatron/core/a\nb.py"]
    output_dir = runner_temp / "unit-test-changed-files"
    output_dir.mkdir(parents=True)
    changed_files_json = output_dir / "all_changed_and_modified_files.json"
    if scenario != "missing-json":
        changed_files_json.write_text(
            {
                "malformed-json": "{",
                "object-json": "{}",
                "non-string-path": '["megatron/core/a.py", 1]',
            }.get(scenario, json.dumps(expected_files))
        )
    step = _step("cicd-main.yml", "configure", "unit-test-pr-files")
    result, outputs = _run(
        step["run"],
        tmp_path,
        {
            **shell_environment,
            "GH_LOG": str(gh_log),
            "RUNNER_TEMP": str(runner_temp),
            "PR_CHANGED_FILES": {
                "count-mismatch": str(len(expected_files) + 1),
                "invalid-file-count": "invalid",
            }.get(scenario, str(len(expected_files))),
            "TESTED_SHA": "invalid" if scenario == "invalid-tested-sha" else "c" * 40,
        },
    )
    assert result.returncode == 0, result.stderr
    valid = scenario in {"valid", "empty-pr", "large-pr", "special-characters"}
    artifact_dir = runner_temp / "unit-test-pr-files"
    if valid:
        assert outputs == {"ready": "true"}
        assert (artifact_dir / "changed-files").read_text().splitlines() == expected_files
        assert json.loads((artifact_dir / "metadata.json").read_text()) == {
            "tested_sha": "c" * 40,
            "changed_files": len(expected_files),
        }
    else:
        assert outputs.get("ready") != "true"
        assert "::notice::" in result.stdout
        assert not (artifact_dir / "metadata.json").exists()
    assert not gh_log.exists()


def test_unit_test_matrices_consume_one_shared_pr_files_artifact() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())
    configure = workflow["jobs"]["configure"]
    checkout = _step("cicd-main.yml", "configure", "unit-test-pr-checkout")
    changed_files = _step("cicd-main.yml", "configure", "changed-files")
    producer = _step("cicd-main.yml", "configure", "unit-test-pr-files")
    upload = _step("cicd-main.yml", "configure", "upload-unit-test-pr-files")
    assert checkout["if"] == "steps.configure.outputs.unit_testmon_eligible == 'true'"
    assert checkout["continue-on-error"] is True
    assert checkout["with"]["ref"] == "${{ steps.resolve-sha.outputs.sha }}"
    assert checkout["with"]["fetch-depth"] == 2
    assert checkout["with"]["persist-credentials"] is False
    assert changed_files["if"] == "steps.unit-test-pr-checkout.outcome == 'success'"
    assert changed_files["continue-on-error"] is True
    assert changed_files["uses"].startswith("step-security/changed-files@")
    assert len(changed_files["uses"].split("@", 1)[1]) == 40
    assert changed_files["with"]["base_sha"] == "${{ steps.resolve-sha.outputs.sha }}^1"
    assert changed_files["with"]["sha"] == "${{ steps.resolve-sha.outputs.sha }}"
    for option in (
        "skip_initial_fetch",
        "fail_on_initial_diff_error",
        "output_renamed_files_as_deleted_and_added",
        "json",
        "write_output_files",
    ):
        assert changed_files["with"][option] is True
    for option in ("escape_json", "safe_output", "quotepath"):
        assert changed_files["with"][option] is False
    assert changed_files["with"]["output_dir"] == "${{ runner.temp }}/unit-test-changed-files"
    assert producer["if"] == "steps.changed-files.outcome == 'success'"
    assert producer["continue-on-error"] is True
    assert producer["env"]["TESTED_SHA"] == "${{ steps.resolve-sha.outputs.sha }}"
    assert producer["env"]["PR_CHANGED_FILES"] == (
        "${{ steps.changed-files.outputs.all_changed_and_modified_files_count }}"
    )
    assert upload["if"] == "steps.unit-test-pr-files.outputs.ready == 'true'"
    assert upload["continue-on-error"] is True
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert configure["outputs"]["unit_test_pr_files_artifact_id"] == (
        "${{ steps.upload-unit-test-pr-files.outcome == 'success' "
        "&& steps.upload-unit-test-pr-files.outputs.artifact-id || '' }}"
    )
    for job in ("cicd-unit-tests-latest", "cicd-unit-tests-latest-gb200"):
        steps = workflow["jobs"][job]["steps"]
        consumer = next(step for step in steps if step.get("uses") == "./.github/actions")
        assert consumer["with"]["unit_test_pr_files_artifact_id"] == (
            "${{ needs.configure.outputs.unit_test_pr_files_artifact_id }}"
        )
        assert consumer["with"]["sha"] == "${{ needs.configure.outputs.sha }}"
    action = yaml.safe_load((ROOT / ".github/actions/action.yml").read_text())
    steps = action["runs"]["steps"]
    download = next(step for step in steps if step.get("id") == "download-unit-test-pr-files")
    assert download["uses"].startswith("actions/download-artifact@")
    assert download["with"]["artifact-ids"] == "${{ inputs.unit_test_pr_files_artifact_id }}"
    assert download["continue-on-error"] is True
    assert not any(step.get("uses", "").startswith("nv-gha-runners/get-pr-info@") for step in steps)
    resolver = next(step for step in steps if step.get("id") == "unit-testmon")
    assert "gh api" not in resolver["run"]


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
