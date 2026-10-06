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
TESTED_SHA = "a" * 40


def _workflow() -> dict:
    return yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())


def _step(job: str, step_id: str) -> dict:
    return next(step for step in _workflow()["jobs"][job]["steps"] if step.get("id") == step_id)


def _run(script: str, directory: Path, environment: dict[str, str]) -> dict[str, str]:
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
            "RUNNER_TEMP": str(directory),
            **environment,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return (
        dict(line.split("=", 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )


@pytest.fixture
def shell_environment(tmp_path: Path) -> dict[str, str]:
    if shutil.which("jq") is None:
        pytest.skip("Workflow shell validation requires jq")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").symlink_to(sys.executable)
    # Run the real workflow jq expression on generated YAML without requiring
    # Mike Farah's yq on developers' hosts.
    yq = binaries / "yq"
    yq.write_text(
        f"#!{sys.executable}\n"
        "import json, subprocess, sys, yaml\n"
        "assert sys.argv[1:3] == ['-o', 'json']\n"
        "result = subprocess.run(['jq', sys.argv[3]], "
        "input=json.dumps(yaml.safe_load(sys.stdin)), text=True)\n"
        "sys.exit(result.returncode)\n"
    )
    yq.chmod(0o755)
    gh = binaries / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['pr', 'view']:\n"
        "    print(os.environ.get('TEST_LABELS', '[]'))\n"
        "elif any('/files?' in arg for arg in args):\n"
        "    assert '--paginate' in args\n"
        "    if os.environ.get('TEST_API_FAILURE') == 'true':\n"
        "        sys.exit(1)\n"
        "    pages = json.loads(os.environ['TEST_FILE_PAGES'])\n"
        "    result = subprocess.run(['jq', args[args.index('--jq') + 1]], "
        "input='\\n'.join(json.dumps(page) for page in pages), text=True)\n"
        "    sys.exit(result.returncode)\n"
        "else:\n"
        "    print(os.environ['TEST_CURRENT_SHA'])\n"
    )
    gh.chmod(0o755)
    return {"PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}"}


def _file(test_case: str, status: str = "modified") -> dict[str, str]:
    return {
        "filename": f"tests/functional_tests/test_cases/gpt/{test_case}/model_config.yaml",
        "status": status,
    }


@pytest.mark.parametrize("failure", [None, "count", "initial-sha", "current-sha", "api", "limit"])
def test_pr_file_producer_preserves_status_and_requires_complete_current_data(
    tmp_path: Path, shell_environment: dict[str, str], failure: str | None
) -> None:
    files = [_file("new", "added"), _file("deleted", "removed"), _file("renamed", "renamed")]
    pages = [[files[0]], [files[1], {**files[2], "previous_filename": "old/name.yaml"}]]
    environment = {
        **shell_environment,
        "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
        "PR_NUMBER": "7840",
        "PR_CHANGED_FILES": "3",
        "PR_MERGE_SHA": TESTED_SHA,
        "TESTED_SHA": TESTED_SHA,
        "TEST_CURRENT_SHA": TESTED_SHA,
        "TEST_FILE_PAGES": json.dumps(pages),
    }
    environment.update(
        {
            "count": {"PR_CHANGED_FILES": "4"},
            "initial-sha": {"PR_MERGE_SHA": "b" * 40},
            "current-sha": {"TEST_CURRENT_SHA": "b" * 40},
            "api": {"TEST_API_FAILURE": "true"},
            "limit": {"PR_CHANGED_FILES": "3001"},
        }.get(failure, {})
    )
    outputs = _run(_step("configure", "pr-files")["run"], tmp_path, environment)
    if failure is not None:
        assert outputs.get("ready") != "true"
        assert not (tmp_path / "pr-files/metadata.json").exists()
        return
    assert outputs["ready"] == "true"
    artifact = tmp_path / "pr-files"
    assert json.loads((artifact / "files.json").read_text()) == files
    assert (artifact / "changed-files").read_text().splitlines() == [
        item["filename"] for item in files
    ]
    assert json.loads((artifact / "metadata.json").read_text()) == {
        "tested_sha": TESTED_SHA,
        "changed_files": len(files),
    }


@pytest.mark.parametrize(
    ("labels", "event", "ref", "expected"),
    [
        ([], "push", "refs/heads/pull-request/7840", "true"),
        (["Disable selective unit tests"], "push", "refs/heads/pull-request/7840", "true"),
        (["Run tests"], "push", "refs/heads/pull-request/7840", "false"),
        (["Run functional tests"], "push", "refs/heads/pull-request/7840", "false"),
        ([], "push", "refs/heads/main", "false"),
        ([], "push", "refs/heads/pull-request/not-a-number", "false"),
        ([], "workflow_dispatch", "refs/heads/pull-request/7840", "false"),
        ([], "merge_group", "refs/heads/gh-readonly-queue/main/pr-7840-abc", "false"),
    ],
)
def test_changed_functional_selection_is_independent_of_testmon(
    tmp_path: Path,
    shell_environment: dict[str, str],
    labels: list[str],
    event: str,
    ref: str,
    expected: str,
) -> None:
    script = (
        _step("configure", "configure")["run"]
        .replace("${{ fromJSON(steps.get-pr-info.outputs.pr-info || '{}').number }}", "7840")
        .replace("${{ github.repository }}", "NVIDIA/Megatron-LM")
    )
    outputs = _run(
        script,
        tmp_path,
        {
            **shell_environment,
            "IS_CI_WORKLOAD": "false",
            "IS_MERGE_GROUP": str(event == "merge_group").lower(),
            "EVENT_NAME": event,
            "REF": ref,
            "FORCE_RUN_ALL": "false",
            "ENABLE_SELECTIVE_UNIT_TESTS": "false",
            "TEST_LABELS": json.dumps(labels),
        },
    )
    assert outputs["changed_functional_tests"] == expected
    assert outputs["unit_testmon_eligible"] == "false"


@pytest.fixture
def functional_recipes(tmp_path: Path, shell_environment: dict[str, str]) -> dict[str, str]:
    scripts = tmp_path / "tests/test_utils/python_scripts"
    scripts.mkdir(parents=True)
    (tmp_path / "tests/__init__.py").touch()
    for name in ("generate_jet_trigger_job.py", "recipe_parser.py", "changed_functional_tests.py"):
        shutil.copyfile(ROOT / "tests/test_utils/python_scripts" / name, scripts / name)
    # Use real generator/parser code and controlled recipes so changes to the
    # production test inventory do not silently change the regression's meaning.
    recipes = tmp_path / "tests/test_utils/recipes"
    recipes.mkdir()
    products = []
    for name, scopes, cadence in (
        ("baseline", ["L0", "L1"], ["pr", "nightly"]),
        ("l0_nightly", ["L0"], ["nightly"]),
        ("changed", ["L1"], ["nightly"]),
        ("untouched", ["L1"], ["nightly"]),
        ("deleted", ["L1"], ["nightly"]),
        ("l2_only", ["L2"], ["nightly"]),
    ):
        products.append(
            {
                "test_case": [name],
                "products": [
                    {
                        "scope": scopes,
                        "cadence": cadence,
                        "environment": ["dev"],
                        "platforms": ["dgx_h100", "dgx_gb200"],
                    }
                ],
            }
        )
    (recipes / "functional.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "basic",
                "spec": {"model": "gpt", "build": "mcore-pyt-{environment}"},
                "products": products,
            }
        )
    )
    return shell_environment


@pytest.mark.parametrize("platform", ["h100", "gb200"])
@pytest.mark.parametrize(
    "artifact_state",
    ["valid", "empty", "deleted-only", "missing", "stale", "download-failed", "disabled", "labels"],
)
def test_functional_matrix_uses_artifact_or_conservative_fallback(
    tmp_path: Path, functional_recipes: dict[str, str], platform: str, artifact_state: str
) -> None:
    files = [_file(name) for name in ("baseline", "l0_nightly", "changed", "l2_only")]
    files.append(_file("deleted", "removed"))
    if artifact_state == "empty":
        files = []
    elif artifact_state == "deleted-only":
        files = [_file("deleted", "removed")]
    artifact = tmp_path / "artifact"
    if artifact_state != "missing":
        artifact.mkdir()
        (artifact / "files.json").write_text(json.dumps(files))
        (artifact / "changed-files").write_text("".join(f"{item['filename']}\n" for item in files))
        (artifact / "metadata.json").write_text(
            json.dumps(
                {
                    "tested_sha": "b" * 40 if artifact_state == "stale" else TESTED_SHA,
                    "changed_files": len(files),
                }
            )
        )
    outputs = _run(
        _step(f"cicd-parse-integration-tests-{platform}", "main")["run"],
        tmp_path,
        {
            **functional_recipes,
            "SCOPE": "L1" if artifact_state == "labels" else "L0",
            "LIGHTWEIGHT": "false",
            "CADENCE": "" if artifact_state == "labels" else "pr",
            "SELECT_CHANGED_TESTS": "false" if artifact_state in {"disabled", "labels"} else "true",
            "PR_FILES_OUTCOME": "failure" if artifact_state == "download-failed" else "success",
            "PR_FILES_DIR": str(artifact),
            "TESTED_SHA": TESTED_SHA,
        },
    )
    matrix = json.loads(outputs[f"integration-tests-{platform}"])
    expected = [{"model": "gpt", "test_case": "baseline", "scope": "L0", "cadence": "pr"}]
    if artifact_state == "valid":
        additions = {"changed": "L1", "l0_nightly": "L0"}
    elif artifact_state in {"missing", "stale", "download-failed"}:
        additions = {"changed": "L1", "deleted": "L1", "l0_nightly": "L0", "untouched": "L1"}
    elif artifact_state == "labels":
        expected = []
        additions = {"baseline": "L1", "changed": "L1", "deleted": "L1", "untouched": "L1"}
    else:
        additions = {}
    expected.extend(
        {"model": "gpt", "test_case": name, "scope": scope, "cadence": ""}
        for name, scope in additions.items()
    )
    assert matrix == sorted(expected, key=lambda entry: (entry["model"], entry["test_case"]))


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_artifact_wiring_uses_immutable_id_and_private_download_directory(
    tmp_path: Path, platform: str
) -> None:
    job = f"cicd-parse-integration-tests-{platform}"
    prepare = _step(job, "pr-files-directory")
    first = _run(prepare["run"], tmp_path, {})["path"]
    second = _run(prepare["run"], tmp_path, {})["path"]
    assert first != second
    assert Path(first).parent == tmp_path
    assert Path(second).parent == tmp_path
    download = _step(job, "download-pr-files")
    assert download["continue-on-error"] is True
    assert download["with"]["artifact-ids"] == "${{ needs.configure.outputs.pr_files_artifact_id }}"
    assert "name" not in download["with"]
    assert download["with"]["path"] == "${{ steps.pr-files-directory.outputs.path }}"
    assert (
        _step(job, "main")["env"]["SELECT_CHANGED_TESTS"]
        == "${{ needs.configure.outputs.changed_functional_tests }}"
    )
    producer = _step("configure", "pr-files")
    assert "unit_testmon_eligible == 'true' ||" in producer["if"]
    assert "changed_functional_tests == 'true'" in producer["if"]
