# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise release guards and uploaded-distribution checks without publishing."""

import hashlib
import importlib.util
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release.yaml").read_text())
SPEC = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).with_name("verify_pypi_release.py")
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def evaluate(expression: str, context: dict, failed=False, cancelled=False):
    """Evaluate workflow guards against supplied event and job states."""
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("!cancelled()", str(not cancelled)).replace(
        "!failure()", str(not failed)
    )
    expression = re.sub(
        r"\b(?:github|inputs|needs)\.[\w.-]+", lambda match: repr(context[match[0]]), expression
    )
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = expression.replace("== true", "== True")
    expression = re.sub(r"!(?!=)", "not ", expression)
    return eval(" ".join(expression.split()), {"__builtins__": {}}, {})


def release_context(event="workflow_dispatch", dry_run=False, target="pypi") -> dict:
    """Build event context with successful upstream jobs."""
    return {
        "github.repository": "NVIDIA/Megatron-LM",
        "github.event_name": event,
        "inputs.dry-run": dry_run,
        "inputs.publish-target": target,
        "needs.bump.result": "success",
        "needs.build-test-wheels.result": "success",
        "needs.verify-published-wheels.result": "success",
    }


@pytest.mark.parametrize("event", ["push", "merge_group", "workflow_dispatch"])
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("target", ["pypi", "testpypi"])
def test_only_explicit_uploads_can_publish(event, dry_run, target):
    """Pushes, merge groups, dry runs, failures and cancellation cannot publish."""
    context = release_context(event, dry_run, target)
    guard = WORKFLOW["jobs"]["publish-wheels"]["if"]
    assert bool(evaluate(guard, context)) == (event == "workflow_dispatch" and not dry_run)
    assert not evaluate(guard, context, failed=True)
    assert not evaluate(guard, context, cancelled=True)


@pytest.mark.parametrize("job", ["bump", "build-test-wheels"])
@pytest.mark.parametrize("state", ["failure", "cancelled", "skipped"])
def test_missing_success_blocks_publication(job, state):
    """Publishing requires successful completion of all required producers."""
    context = release_context()
    context[f"needs.{job}.result"] = state
    assert not evaluate(WORKFLOW["jobs"]["publish-wheels"]["if"], context)


@pytest.mark.parametrize("state", ["failure", "cancelled", "skipped"])
def test_real_release_requires_verified_upload(state):
    """An absent verification cannot finalize a real release; dry runs may continue."""
    context = release_context()
    context["needs.verify-published-wheels.result"] = state
    guard = WORKFLOW["jobs"]["finalize"]["if"]
    assert not evaluate(guard, context)
    context["inputs.dry-run"] = True
    assert evaluate(guard, context)
    assert not evaluate(guard, context, failed=True)


def test_testpypi_has_no_release_side_effects():
    """TestPyPI uploads keep both shared release phases in validation mode."""
    context = release_context(target="testpypi")
    for name in ("bump", "finalize"):
        for key in ("validate-only", "dry-run"):
            assert evaluate(WORKFLOW["jobs"][name]["with"][key], context)
    context["inputs.publish-docs"] = True
    assert not evaluate(WORKFLOW["jobs"]["finalize"]["with"]["publish-docs"], context)


@pytest.mark.parametrize(
    "event,requested,expected",
    [("push", "", True), ("workflow_dispatch", False, False), ("workflow_dispatch", True, True)],
)
def test_finalize_receives_boolean_for_release_creation(event, requested, expected):
    """Keep implicit push inputs boolean while respecting a manual false value."""
    context = release_context(event=event)
    context["inputs.create-gh-release"] = requested
    actual = evaluate(WORKFLOW["jobs"]["finalize"]["with"]["create-gh-release"], context)
    assert actual is expected


def test_builder_has_no_upload_credentials():
    """Builds cannot inherit publishing secrets or mint OIDC credentials."""
    caller = WORKFLOW["jobs"]["build-test-wheels"]
    assert "secrets" not in caller
    assert caller["permissions"] == {"contents": "read"}
    builder = yaml.safe_load((ROOT / ".github/workflows/_build_test_wheel.yml").read_text())
    assert "secrets" not in builder[True]["workflow_call"]
    assert builder["permissions"] == {"contents": "read"}
    publisher = WORKFLOW["jobs"]["publish-wheels"]
    assert publisher["permissions"] == {"id-token": "write"}
    assert not any(
        step.get("uses", "").startswith("actions/checkout@") for step in publisher["steps"]
    )


@pytest.mark.parametrize("state", ["success", "failure", "cancelled", "skipped"])
def test_verification_requires_successful_publish(state):
    """Run verification after publishing even when optional ancestors were skipped."""
    context = {"needs.publish-wheels.result": state}
    guard = WORKFLOW["jobs"]["verify-published-wheels"]["if"]
    assert bool(evaluate(guard, context)) == (state == "success")
    assert not evaluate(guard, context, failed=True)
    assert not evaluate(guard, context, cancelled=True)


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("state", ["success", "failure", "cancelled", "skipped"])
def test_summary_rejects_missing_required_verification(tmp_path, required, state):
    """Execute the summary guard and reject false success after a skipped verifier."""
    jobs = {
        name: {"result": "success"}
        for name in ("publish-wheels", "verify-published-wheels", "finalize")
    }
    jobs["verify-published-wheels"]["result"] = state
    script = tmp_path / "summary.sh"
    script.write_text(WORKFLOW["jobs"]["release-summary"]["steps"][0]["run"])
    env = dict(os.environ, NEEDS_JSON=json.dumps(jobs), REQUIRE_UPLOAD=str(required).lower())
    result = subprocess.run(
        ["bash", "-euo", "pipefail", str(script)], env=env, capture_output=True, check=False
    )
    expected_success = state == "success" or (state == "skipped" and not required)
    assert (result.returncode == 0) == expected_success


@pytest.mark.parametrize(
    "ref,production,branch,valid",
    [
        ("a" * 40, True, "refs/heads/main", True),
        ("main", True, "refs/heads/main", False),
        ("a" * 40, True, "refs/heads/pull-request/123", False),
        ("a" * 40, False, "refs/heads/pull-request/123", True),
    ],
)
def test_release_input_validation(tmp_path, ref, production, branch, valid):
    """Execute the release SHA and workflow branch guard."""
    script = tmp_path / "validate.sh"
    script.write_text(WORKFLOW["jobs"]["validate-inputs"]["steps"][0]["run"])
    env = dict(
        os.environ, RELEASE_REF=ref, IS_PRODUCTION=str(production).lower(), GITHUB_REF=branch
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", str(script)], env=env, check=False, capture_output=True
    )
    assert (result.returncode == 0) == valid


@pytest.mark.parametrize("mutation", ["none", "missing", "changed", "yanked", "extra"])
def test_distribution_inventory_and_hashes(tmp_path, mutation):
    """Reject missing, extra, changed, or yanked uploads against local hashes."""
    names = ["megatron_core-1.0-py3-none-any.whl", "megatron_core-1.0.tar.gz"]
    for name in names:
        (tmp_path / name).write_bytes(b"fixture")
    version, files = VERIFY.release_files(tmp_path, "megatron-core")
    assert version == "1.0"
    assert files == {name: hashlib.sha256(b"fixture").hexdigest() for name in names}
    metadata = {
        "urls": [
            {"filename": name, "digests": {"sha256": digest}} for name, digest in files.items()
        ]
    }
    if mutation == "missing":
        metadata["urls"].pop()
    elif mutation == "changed":
        metadata["urls"][0]["digests"]["sha256"] = "wrong"
    elif mutation == "yanked":
        metadata["urls"][0]["yanked"] = True
    elif mutation == "extra":
        metadata["urls"].append({"filename": "unexpected"})
    if mutation == "none":
        VERIFY.check_release(metadata, files)
    else:
        with pytest.raises(ValueError):
            VERIFY.check_release(metadata, files)


@pytest.mark.parametrize("package", ["megatron-core", "megatron-fsdp"])
def test_rehearsal_version_survives_partial_matrix_retry(tmp_path, monkeypatch, package):
    """Build one version across retries while independent workflow runs stay distinct."""
    builder = yaml.safe_load((ROOT / ".github/workflows/_build_test_wheel.yml").read_text())
    step = next(
        step
        for step in builder["jobs"]["build-and-test-wheels"]["steps"]
        if step.get("id") == "build-wheel"
    )
    mutation = step["run"].split("pushd $BUILD_DIR", 1)[0]
    relative = (
        "megatron/core/package_info.py"
        if package == "megatron-core"
        else "megatron/core/distributed/fsdp/src/megatron_fsdp/package_info.py"
    )
    monkeypatch.setenv("NO_VCS_VERSION", "1")
    versions = []
    for run_id, attempt in (("123456", "1"), ("123456", "2"), ("123457", "1")):
        checkout = tmp_path / f"{run_id}-{attempt}"
        info = checkout / relative
        info.parent.mkdir(parents=True)
        info.write_text((ROOT / relative).read_text())
        subprocess.run(
            ["bash", "-euo", "pipefail", "-c", mutation],
            cwd=checkout,
            env=dict(
                os.environ,
                PACKAGE=package,
                PUBLISH_DRYRUN="true",
                GITHUB_RUN_ID=run_id,
                GITHUB_RUN_ATTEMPT=attempt,
            ),
            check=True,
            capture_output=True,
        )
        versions.append(runpy.run_path(str(info))["__version__"])
    assert versions[0] == versions[1]
    assert versions[0] != versions[2]
    merged = tmp_path / "merged"
    merged.mkdir()
    name = package.replace("-", "_")
    if package == "megatron-core":
        for version, architecture in zip(versions[:2], ("x86_64", "aarch64")):
            for python in ("cp311", "cp312", "cp313"):
                (
                    merged / f"{name}-{version}-{python}-{python}-manylinux_2_28_{architecture}.whl"
                ).touch()
    else:
        (merged / f"{name}-{versions[1]}-py3-none-any.whl").touch()
    (merged / f"{name}-{versions[0]}.tar.gz").touch()
    version, files = VERIFY.release_files(merged, package)
    assert version == versions[0]
    assert len(files) == (7 if package == "megatron-core" else 2)


@pytest.mark.parametrize("index", ["pypi", "testpypi"])
@pytest.mark.parametrize(
    "provenance_state",
    ["valid", "missing", "empty", "wrong-workflow", "wrong-kind", "bad-signature"],
)
def test_main_requires_verified_attestations(tmp_path, monkeypatch, index, provenance_state):
    """Only verified provenance may produce the version consumed by install checks."""
    directory = tmp_path / "dist"
    directory.mkdir()
    for name in ("megatron_core-1.0-py3-none-any.whl", "megatron_core-1.0.tar.gz"):
        (directory / name).write_bytes(b"fixture")
    _, files = VERIFY.release_files(directory, "megatron-core")
    metadata = {
        "urls": [
            {"filename": name, "digests": {"sha256": digest}} for name, digest in files.items()
        ]
    }
    publisher = {"kind": "GitHub", "workflow": "release.yaml"}
    provenance = {"attestation_bundles": [{"publisher": publisher, "attestations": [{}]}]}
    if provenance_state == "missing":
        provenance["attestation_bundles"] = []
    elif provenance_state == "empty":
        provenance["attestation_bundles"][0]["attestations"] = []
    elif provenance_state == "wrong-workflow":
        publisher["workflow"] = "untrusted.yaml"
    elif provenance_state == "wrong-kind":
        publisher["kind"] = "Untrusted"
    fetch = Mock(side_effect=lambda url: provenance if "/integrity/" in url else metadata)
    command = Mock()
    if provenance_state == "bad-signature":
        command.side_effect = subprocess.CalledProcessError(1, "pypi_attestations")
    monkeypatch.setattr(VERIFY, "fetch_json", fetch)
    monkeypatch.setattr(VERIFY.subprocess, "run", command)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--index", index, "--package", "megatron-core", "--dist-dir", str(directory)],
    )
    output = tmp_path / "github-output"
    output.write_text("existing=1\n")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    if provenance_state == "valid":
        VERIFY.main()
        assert output.read_text() == "existing=1\nversion=1.0\n"
        assert command.call_count == len(files)
        for call, name in zip(command.call_args_list, files):
            args = call.args[0]
            assert args[:5] == [sys.executable, "-m", "pypi_attestations", "verify", "pypi"]
            assert args[5:7] == ["--repository", "https://github.com/NVIDIA/Megatron-LM"]
            assert args[7] == "--provenance-file"
            assert args[-1] == str(directory / name)
            assert call.kwargs == {"check": True, "timeout": 180}
        assert all(call.args[0].startswith(VERIFY.INDEXES[index]) for call in fetch.call_args_list)
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            VERIFY.main()
        assert output.read_text() == "existing=1\n"
        assert command.call_count == (1 if provenance_state == "bad-signature" else 0)
