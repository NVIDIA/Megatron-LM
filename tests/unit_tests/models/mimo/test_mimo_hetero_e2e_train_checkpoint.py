# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end: the hetero MIMO 20L mock trains and round-trips a checkpoint.

This drives the training launcher, which spawns its own 8-rank ``torch.distributed.run``,
so it must run as a single plain pytest process (not under the multi-rank unit runner).
Invoke directly on an 8-GPU node, e.g. ``pytest <this file>``; it skips otherwise.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[4]
_LAUNCHER = _REPO_ROOT / "examples/mimo/scripts/run_hetero_nemotron_20l_mock_train.sh"

# The launcher spawns its own torchrun; skip when this file is collected under a
# multi-rank runner to avoid nesting torch.distributed.run.
_UNDER_TORCHRUN = int(os.environ.get("WORLD_SIZE", "1")) > 1


def _run_launcher(base, train_iters, extra_args, name):
    """Run the 20L launcher saving under ``base``; return the completed process."""
    env = {
        **os.environ,
        "TRAIN_ITERS": str(train_iters),
        "TORCHRUN_LOG_DIR": str(base / f"torchrun-{name}"),
    }
    # conftest's autouse set_env fixture disables TE flash/fused attention; the 20L model
    # at seq 8192 needs them (unfused attention OOMs), so let the launcher use TE defaults.
    env.pop("NVTE_FLASH_ATTN", None)
    env.pop("NVTE_FUSED_ATTN", None)
    # Shrink the MoE for the round-trip: the full 128-expert config trains but its
    # optimizer-state load on resume exceeds 80 GiB; fewer experts exercises the same
    # save/load path (grouped-GEMM experts, mamba, attention, Float16Module wrap) within memory.
    cmd = [
        "bash",
        str(_LAUNCHER),
        "--save",
        str(base / "ckpt"),
        "--save-interval",
        "10",
        "--num-experts",
        "8",
        *extra_args,
    ]
    return subprocess.run(
        cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800
    )


def _tail(result):
    """Both streams tailed: the launcher tees per-rank tracebacks to stdout."""
    return f"--- stdout ---\n{result.stdout[-6000:]}\n--- stderr ---\n{result.stderr[-3000:]}"


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
@pytest.mark.skipif(
    _UNDER_TORCHRUN, reason="launcher spawns its own torchrun; run as a plain process"
)
def test_hetero_mimo_20l_trains_and_checkpoint_round_trips():
    # The 128-expert MoE checkpoint is large; save under the repo workspace (a roomy
    # shared filesystem on the cluster) rather than pytest's node-local /tmp tmp_path.
    scratch = Path(tempfile.mkdtemp(prefix="mimo_e2e_", dir=_REPO_ROOT))
    ckpt = scratch / "ckpt"
    try:
        # Train 10 iterations and save a checkpoint.
        train = _run_launcher(scratch, train_iters=10, extra_args=[], name="train")
        assert train.returncode == 0, f"training run failed:\n{_tail(train)}"
        assert (ckpt / "latest_checkpointed_iteration.txt").exists(), "no checkpoint written"
        assert (ckpt / "iter_0000010").is_dir(), "iter_0000010 checkpoint dir missing"

        # Resume from the checkpoint and train two more iterations.
        resume = _run_launcher(
            scratch, train_iters=12, extra_args=["--load", str(ckpt)], name="resume"
        )
        assert resume.returncode == 0, f"resume run failed:\n{_tail(resume)}"
        assert (
            "successfully loaded checkpoint" in (resume.stdout + resume.stderr).lower()
        ), f"resume did not load the checkpoint:\n{_tail(resume)}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
