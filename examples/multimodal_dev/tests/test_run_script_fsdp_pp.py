# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Launcher tests for the FSDP/PP interaction in ``run_qwen35_vl.sh``.

FSDP and pipeline parallelism are mutually exclusive on Megatron's standard
path, so the script has to decide what to do when both are requested.  The
contract these tests pin down is:

* ``USE_FSDP`` unset and ``PP>1`` -- default it to 0 so a plain PP smoke run
  works out of the box;
* ``USE_FSDP=0`` and ``PP>1`` -- run without FSDP, no complaints;
* ``USE_FSDP=1`` and ``PP>1`` -- fail, rather than quietly downgrading to a
  non-FSDP run that the caller could mistake for the FSDP+PP validation they
  asked for;
* ``PP=1`` -- FSDP stays on, which is the script's long-standing default.

These run the script with ``DRY_RUN=1``, which assembles the full torchrun
command line, echoes it and exits before launching anything, so no GPU or
dataset is required.
"""

import os
import subprocess

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_SCRIPT = os.path.join(_REPO_ROOT, "examples/multimodal_dev/scripts/run_qwen35_vl.sh")

# Present only when FSDP is actually enabled; --use-distributed-optimizer is
# emitted unconditionally elsewhere in the script, so it cannot be used as the
# probe here.
FSDP_FLAG = "--use-megatron-fsdp"


def _run(tmp_path, **env_overrides):
    """Dry-run the launcher and return the completed process."""
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "MODEL_VARIANT": "proxy",
            # Keep the script's mkdir side effects inside the test's tmpdir.
            "ROOT_DIR": f"{tmp_path}/root/",
            "TENSORBOARD_LOGS_PATH": f"{tmp_path}/tb",
        }
    )
    # Let each case decide whether USE_FSDP is set at all -- the unset case is
    # exactly what distinguishes the default from an explicit request.
    env.pop("USE_FSDP", None)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", _SCRIPT], env=env, capture_output=True, text=True, cwd=_REPO_ROOT
    )


@pytest.mark.skipif(not os.path.exists(_SCRIPT), reason="launcher script not found")
@pytest.mark.parametrize(
    "use_fsdp",
    [
        pytest.param(None, id="unset"),
        # USE_FSDP= (empty) is a natural way to say "no FSDP"; the script probes
        # with ${USE_FSDP:+x} so it is treated as unset rather than as an
        # explicit request for FSDP.
        pytest.param("", id="empty"),
        pytest.param("0", id="explicit-0"),
    ],
)
def test_pp_gt_1_runs_without_fsdp(tmp_path, use_fsdp):
    """PP>1 proceeds without FSDP when FSDP was not explicitly requested."""
    overrides = {"PP": "2"}
    if use_fsdp is not None:
        overrides["USE_FSDP"] = use_fsdp
    proc = _run(tmp_path, **overrides)

    assert proc.returncode == 0, f"launcher failed:\n{proc.stdout}\n{proc.stderr}"
    assert FSDP_FLAG not in proc.stdout, "PP>1 must not emit FSDP flags"


@pytest.mark.skipif(not os.path.exists(_SCRIPT), reason="launcher script not found")
def test_explicit_fsdp_with_pp_is_rejected(tmp_path):
    """An explicit USE_FSDP=1 with PP>1 fails instead of being downgraded."""
    proc = _run(tmp_path, PP="2", USE_FSDP="1")

    assert proc.returncode != 0, (
        "explicitly requesting FSDP with PP>1 must fail rather than silently "
        f"running without FSDP:\n{proc.stdout}"
    )
    # The caller needs to learn which mode was refused, not just that something
    # went wrong.
    assert "USE_FSDP=1" in proc.stderr
    assert FSDP_FLAG not in proc.stdout, "the rejected run must not launch"


@pytest.mark.skipif(not os.path.exists(_SCRIPT), reason="launcher script not found")
def test_pp_1_still_defaults_to_fsdp(tmp_path):
    """The PP>1 guard does not disturb the ordinary PP=1 FSDP default."""
    proc = _run(tmp_path, PP="1")

    assert proc.returncode == 0, f"launcher failed:\n{proc.stdout}\n{proc.stderr}"
    assert FSDP_FLAG in proc.stdout, "PP=1 must keep FSDP enabled by default"
