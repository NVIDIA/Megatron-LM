# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GB300 convergence scheduling without changing asynchronous nightly behavior."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from tests.test_utils.python_scripts import generate_jet_trigger_job, recipe_parser


@pytest.mark.parametrize(
    "platform,scope,expected_timeout",
    [
        ("dgx_gb300", "nightly", "24 hours"),
        ("dgx_gb300", "weekly", "7 days"),
        ("dgx_gb300", "L3", "7 days"),
        ("dgx_gb300", "release", "7 days"),
        ("dgx_gb200", "nightly", "7 days"),
    ],
)
def test_generated_job_allows_convergence_restarts(
    monkeypatch, tmp_path, platform, scope, expected_timeout
):
    """A long resumable run must outlive the GB300 nightly job deadline."""
    workload = recipe_parser.dotdict(
        type="basic",
        spec=recipe_parser.dotdict(model="mixtral", environment="dev", test_case="convergence"),
    )
    monkeypatch.setattr(
        generate_jet_trigger_job.recipe_parser, "load_workloads", lambda **_kwargs: [workload]
    )
    output = tmp_path / "jobs.yaml"
    cluster = "dgxgb300_oci-jhb" if platform == "dgx_gb300" else "dgxgb200_oci-hsg"
    args = [
        "--scope",
        scope,
        "--environment",
        "dev",
        "--test-cases",
        "all",
        "--platform",
        platform,
        "--cluster",
        cluster,
        "--output-path",
        str(output),
        "--container-image",
        "utility",
        "--container-tag",
        "test",
        "--dependent-job",
        "functional:configure",
        "--record-checkpoints",
        "false",
        "--slurm-account",
        "mcore",
        "--no-enable-warmup",
    ]
    result = CliRunner().invoke(generate_jet_trigger_job.main, args)
    assert result.exit_code == 0, result.output
    job = yaml.safe_load(output.read_text())["convergence"]
    assert job["timeout"] == expected_timeout
    assert f"--platform {platform}" in job["script"][0]
    assert job["allow_failure"] is False


@pytest.mark.parametrize("scope", ["nightly", "weekly", "release", "mr"])
def test_configure_generates_gb300_with_convergence_run_identity(tmp_path, scope):
    """Execute the CI shell blocks and capture the generated command's arguments."""
    pipeline = yaml.safe_load(Path(".gitlab/stages/04.functional-tests.yml").read_text())
    scripts = pipeline["functional:configure"]["script"]
    release_args = next(script for script in scripts if "RELEASE_ARGS=(" in script)
    gb300 = next(script for script in scripts if "--platform dgx_gb300" in script)
    capture = tmp_path / "args.txt"
    environment = {
        "PATH": os.environ["PATH"],
        "FUNCTIONAL_TEST_SCOPE": scope,
        "FUNCTIONAL_TEST_NAME": "dsv4/convergence",
        "CLUSTER_GB300": "dgxgb300_oci-jhb",
        "CAPTURE": str(capture),
    }
    script = (
        'set -e\nARGS=()\npython() { printf "%s\\n" "$@" > "$CAPTURE"; }\n'
        + release_args
        + "\n"
        + gb300
    )
    subprocess.run(["bash", "-c", script], env=environment, check=True)
    if scope == "mr":
        assert not capture.exists()
        return
    args = capture.read_text().splitlines()
    assert args[args.index("--platform") + 1] == "dgx_gb300"
    assert args[args.index("--cluster") + 1] == "dgxgb300_oci-jhb"
    if scope in ("weekly", "release"):
        assert args[args.index("--run-name") + 1] == "dsv4/convergence"
        assert args[args.index("--wandb-experiment") + 1] == "dsv4-convergence"
    else:
        assert "--run-name" not in args
        assert "--wandb-experiment" not in args


@pytest.mark.parametrize(
    "scope,build,functional_test,expected_jobs",
    [
        ("nightly", "yes", "yes", [True, False]),
        ("weekly", "yes", "yes", [False, True]),
        ("release", "yes", "yes", [False, True]),
        ("mr", "yes", "yes", [False, False]),
        ("weekly", "no", "yes", [False, False]),
        ("release", "yes", "no", [False, False]),
    ],
)
def test_convergence_waits_for_results_without_changing_nightly(
    scope, build, functional_test, expected_jobs
):
    """Notifications must wait for convergence but not the nightly GPU matrix."""
    pipeline = yaml.safe_load(Path(".gitlab/stages/04.functional-tests.yml").read_text())
    nightly = pipeline["functional:run_dev_dgx_gb300"]
    convergence = pipeline["functional:run_dev_dgx_gb300_convergence"]
    assert nightly["allow_failure"] is True
    assert nightly["trigger"]["strategy"] is None
    assert convergence["extends"] == [".functional_run"]
    assert "trigger" not in convergence
    assert pipeline[".functional_run"]["trigger"]["strategy"] == "depend"
    assert convergence.get("allow_failure", False) is False
    assert {"job": "functional:run_dev_dgx_gb300_convergence", "optional": True} in pipeline[
        "functional:x_notify"
    ]["needs"]
    environment = {
        "PATH": os.environ["PATH"],
        "FUNCTIONAL_TEST_SCOPE": scope,
        "BUILD": build,
        "FUNCTIONAL_TEST": functional_test,
    }
    selected_jobs = []
    for job in (nightly, convergence):
        for rule in job["rules"]:
            if "if" in rule:
                result = subprocess.run(["bash", "-c", f'[[ {rule["if"]} ]]'], env=environment)
                assert result.returncode in (0, 1)
                if result.returncode:
                    continue
            selected_jobs.append(rule["when"] != "never")
            break
    assert selected_jobs == expected_jobs
