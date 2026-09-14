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


@pytest.mark.parametrize("force_label", [False, True])
def test_synthetic_pr_force_label_overrides_selective_testing(
    tmp_path: Path, shell_environment: dict[str, str], force_label: bool
) -> None:
    gh = tmp_path / "bin/gh"
    gh.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_PR_LABELS"\n')
    gh.chmod(0o755)
    labels = ["Run selective unit tests"] + (["force-run-all"] if force_label else [])
    script = _step("cicd-main.yml", "configure", "configure")["run"]
    script = script.replace(
        "${{ fromJSON(steps.get-pr-info.outputs.pr-info || '{}').number }}", "6934"
    )
    script = script.replace("${{ github.repository }}", "NVIDIA/Megatron-LM")
    result, outputs = _run(
        script,
        tmp_path,
        {
            **shell_environment,
            "TEST_PR_LABELS": json.dumps(labels),
            "IS_CI_WORKLOAD": "false",
            "IS_MERGE_GROUP": "false",
            "EVENT_NAME": "push",
            "REF": "refs/heads/pull-request/6934",
            "FORCE_RUN_ALL": "false",
        },
    )
    assert result.returncode == 0, result.stderr
    assert outputs["unit_testmon_eligible"] == ("false" if force_label else "true")


def _testmon_cache(cache_id: int, created_at: str, **overrides) -> dict:
    return {
        "id": cache_id,
        "key": f"unit-testmon-v1-main-dgx_h100-bucket-{cache_id}-1",
        "ref": "refs/heads/main",
        "created_at": created_at,
        "last_accessed_at": "2026-09-18T12:00:00Z",
        **overrides,
    }


def _run_cleanup(pages: list[list[dict]], failures: dict | None = None) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Workflow JavaScript validation requires Node.js")
    script = _step("populate-build-cache.yml", "cleanup-testmon-cache", "cleanup")["with"]["script"]
    harness = r"""
const {readFileSync} = require('node:fs');
const input = JSON.parse(readFileSync(0, 'utf8'));
Date.now = () => Date.parse('2026-09-18T12:00:00Z');
const calls = [];
const messages = [];
const summary = {};
for (const method of ['addHeading', 'addRaw']) {
  summary[method] = (...args) => { messages.push({method, args}); return summary; };
}
summary.write = async () => summary;
const core = {summary};
core.info = (...args) => messages.push({method: 'info', args});
const github = {
  paginate: async (route, parameters) => {
    calls.push({method: 'paginate', route, parameters});
    return input.pages.flat();
  },
  request: async (route, parameters) => {
    calls.push({method: 'request', route, parameters});
    const status = input.failures[String(parameters.cache_id)];
    if (status) throw Object.assign(new Error(`delete failed: ${status}`), {status});
    return {status: 204};
  },
};
const context = {repo: {owner: 'NVIDIA', repo: 'Megatron-LM'}};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
(async () => {
  let error = null;
  try {
    await new AsyncFunction('github', 'context', 'core', input.script)(github, context, core);
  } catch (failure) {
    error = failure.message;
  }
  process.stdout.write(JSON.stringify({calls, messages, error}));
})();
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=json.dumps({"script": script, "pages": pages, "failures": failures or {}}),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def _deleted_ids(result: dict) -> list[int]:
    requests = [call for call in result["calls"] if call["method"] == "request"]
    assert all(
        call["route"] == "DELETE /repos/{owner}/{repo}/actions/caches/{cache_id}"
        and call["parameters"]["owner"] == "NVIDIA"
        and call["parameters"]["repo"] == "Megatron-LM"
        for call in requests
    )
    return [call["parameters"]["cache_id"] for call in requests]


def _cleanup_summary(result: dict) -> str:
    return "".join(
        message["args"][0] for message in result["messages"] if message["method"] == "addRaw"
    )


def test_cleanup_uses_creation_time_and_exact_two_day_boundary() -> None:
    result = _run_cleanup(
        [
            [
                _testmon_cache(1, "2026-09-15T12:00:00Z"),
                _testmon_cache(2, "2026-09-17T12:00:00Z", last_accessed_at="2026-09-01T00:00:00Z"),
                _testmon_cache(3, "2026-09-16T12:00:00Z"),
                _testmon_cache(4, "2026-09-16T11:59:59.999Z"),
                _testmon_cache(5, "2026-09-16T12:00:00.001Z"),
            ]
        ]
    )
    assert result["error"] is None
    assert _deleted_ids(result) == [1, 4]
    assert _cleanup_summary(result).startswith("Deleted 2 main Testmon generations ")


def test_cleanup_filters_namespace_and_branch_and_lists_all_pages_before_deleting() -> None:
    result = _run_cleanup(
        [
            [
                _testmon_cache(1, "2026-09-01T00:00:00Z"),
                _testmon_cache(2, "invalid", key="build-cache-123"),
                _testmon_cache(3, "invalid", ref="refs/heads/pull-request/6934"),
                _testmon_cache(4, "invalid", key="unit-testmon-v1-mainish-123"),
            ],
            [_testmon_cache(5, "2026-09-01T00:00:00Z")],
        ]
    )
    assert result["error"] is None
    assert result["calls"][0] == {
        "method": "paginate",
        "route": "GET /repos/{owner}/{repo}/actions/caches",
        "parameters": {
            "owner": "NVIDIA",
            "repo": "Megatron-LM",
            "ref": "refs/heads/main",
            "key": "unit-testmon-v1-main-",
            "per_page": 100,
            "sort": "created_at",
            "direction": "asc",
        },
    }
    assert [call["method"] for call in result["calls"]] == ["paginate", "request", "request"]
    assert _deleted_ids(result) == [1, 5]


def test_cleanup_empty_listing_is_successful() -> None:
    result = _run_cleanup([[]])
    assert result["error"] is None
    assert _deleted_ids(result) == []
    assert _cleanup_summary(result).startswith("Deleted 0 main Testmon generations ")


def test_cleanup_tolerates_an_already_deleted_cache_and_repeated_execution() -> None:
    caches = [_testmon_cache(cache_id, "2026-09-01T00:00:00Z") for cache_id in (1, 2)]
    result = _run_cleanup([caches], {1: 404})
    assert result["error"] is None
    assert _deleted_ids(result) == [1, 2]
    assert _cleanup_summary(result).startswith("Deleted 1 main Testmon generations ")
    repeated = _run_cleanup([caches], {1: 404, 2: 404})
    assert repeated["error"] is None
    assert _deleted_ids(repeated) == [1, 2]
    assert _cleanup_summary(repeated).startswith("Deleted 0 main Testmon generations ")


@pytest.mark.parametrize("status", [403, 500])
def test_cleanup_propagates_delete_failures(status: int) -> None:
    caches = [_testmon_cache(cache_id, "2026-09-01T00:00:00Z") for cache_id in (1, 2)]
    result = _run_cleanup([caches], {1: status})
    assert result["error"] == f"delete failed: {status}"
    assert _deleted_ids(result) == [1]


def test_cleanup_rejects_invalid_creation_time_before_any_deletion() -> None:
    result = _run_cleanup(
        [[_testmon_cache(1, "2026-09-01T00:00:00Z"), _testmon_cache(2, "invalid")]]
    )
    assert result["error"]
    assert _deleted_ids(result) == []
