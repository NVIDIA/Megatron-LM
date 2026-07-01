# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end: the hetero MIMO 20L mock trains and round-trips a checkpoint.

This drives the training launcher, which spawns its own 8-rank ``torch.distributed.run``,
so it must run as a single plain pytest process (not under the multi-rank unit runner).
Invoke directly on an 8-GPU node, e.g. ``pytest <this file>``; it skips otherwise.
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[4]
_LAUNCHER = _REPO_ROOT / "examples/mimo/scripts/run_hetero_nemotron_20l_mock_train.sh"

# The launcher spawns its own torchrun; skip when this file is collected under a
# multi-rank runner to avoid nesting torch.distributed.run.
_UNDER_TORCHRUN = int(os.environ.get("WORLD_SIZE", "1")) > 1


def _run_launcher(tmp_path, train_iters, extra_args, name):
    """Run the 20L launcher with a shared checkpoint dir; return the completed process."""
    env = {
        **os.environ,
        "TRAIN_ITERS": str(train_iters),
        "TORCHRUN_LOG_DIR": str(tmp_path / f"torchrun-{name}"),
    }
    cmd = [
        "bash",
        str(_LAUNCHER),
        "--save",
        str(tmp_path / "ckpt"),
        "--save-interval",
        "10",
        *extra_args,
    ]
    return subprocess.run(
        cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800
    )


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
@pytest.mark.skipif(_UNDER_TORCHRUN, reason="launcher spawns its own torchrun; run as a plain process")
def test_hetero_mimo_20l_trains_and_checkpoint_round_trips(tmp_path):
    ckpt = tmp_path / "ckpt"

    # Train 10 iterations and save a checkpoint.
    train = _run_launcher(tmp_path, train_iters=10, extra_args=[], name="train")
    assert train.returncode == 0, f"training run failed:\n{train.stderr[-4000:]}"
    assert (ckpt / "latest_checkpointed_iteration.txt").exists(), "no checkpoint written"
    assert (ckpt / "iter_0000010").is_dir(), "iter_0000010 checkpoint dir missing"

    # Resume from the checkpoint and train two more iterations.
    resume = _run_launcher(tmp_path, train_iters=12, extra_args=["--load", str(ckpt)], name="resume")
    assert resume.returncode == 0, f"resume run failed:\n{resume.stderr[-4000:]}"
    assert "loading checkpoint from" in (resume.stdout + resume.stderr).lower(), "resume did not load"
