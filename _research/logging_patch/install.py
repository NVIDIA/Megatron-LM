"""Entry point wired into pretrain_gpt.py.

``install()`` is called once, before ``pretrain(...)``. It:

    1. applies deterministic flags (if ``APERTUS_DETERMINISTIC=1``)
    2. generates a run name + writes the meta to the JSON log
    3. monkey-patches ``setup_model_and_optimizer`` and ``training_log``
       so each log-interval call appends one row to the JSON log

All configuration is via environment variables so we don't have to modify
Megatron's argparse. See ``README.md`` for the full list.
"""

from __future__ import annotations

import atexit
import os
import time
from pathlib import Path

from . import determinism, env_capture, hooks, phase_timer
from .writer import JsonLogger

_INSTALLED = False


def _destroy_process_group_on_exit() -> None:
    # Megatron never calls torch.distributed.destroy_process_group(), so NCCL's
    # ProcessGroup destructor fires during Python finalization and every rank
    # logs a "destroy_process_group() was not called before program exit"
    # warning. Running this via atexit cleans up deterministically and silences
    # the noise without touching upstream training code.
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_run_name(repo: Path) -> str:
    info = env_capture.git_info(repo)
    branch = (info.get("git_branch") or "detached").replace("/", "-")
    sha = info.get("git_short_sha") or "nosha"
    ts = time.strftime("%Y%m%d-%H%M%S")
    job = os.environ.get("SLURM_JOB_ID", "local")
    return f"{branch}-{sha}-{ts}-{job}"


def _seed_model_shape_from_argv() -> dict[str, int]:
    """Best-effort parse of --num-layers / --hidden-size / --seq-length from argv.

    We need these for MFU before Megatron's argparse has run. Returns an empty
    dict if any flag is missing; the hook will just skip the MFU computation.
    """
    import sys
    argv = sys.argv
    out: dict[str, int] = {}
    keys = {
        "--num-layers": "n_layers",
        "--hidden-size": "hidden",
        "--seq-length": "seq_len",
    }
    for flag, key in keys.items():
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 < len(argv):
                try:
                    out[key] = int(argv[idx + 1])
                except ValueError:
                    pass
    return out


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    phase_timer.stamp("install_called")

    repo = _repo_dir()

    if os.environ.get("APERTUS_DETERMINISTIC") == "1":
        determinism.set_deterministic()

    log_dir = Path(
        os.environ.get("APERTUS_LOG_DIR", repo / "_research" / "results" / "performance")
    )
    run_name = os.environ.get("APERTUS_RUN_NAME") or _default_run_name(repo)

    meta = {
        "name": run_name,
        "feature": os.environ.get("APERTUS_FEATURE") or env_capture.git_info(repo).get("git_branch"),
        "track": os.environ.get("APERTUS_TRACK"),
        "git_sha": env_capture.git_info(repo).get("git_sha"),
        "start_time": time.time(),
        "env": env_capture.capture(repo, args=None),
    }

    writer = JsonLogger(log_dir, run_name, meta)
    phase_timer.install(writer)

    shape = _seed_model_shape_from_argv()
    hooks._STATE["n_layers"] = shape.get("n_layers")
    hooks._STATE["hidden"] = shape.get("hidden")
    hooks._STATE["seq_len"] = shape.get("seq_len")

    hooks.configure(writer, {
        "log_per_layer_grads": os.environ.get("APERTUS_LOG_PER_LAYER_GRADS") == "1",
        "log_act_stats": os.environ.get("APERTUS_LOG_ACT_STATS", "1") != "0",
        "log_loss_spikes": os.environ.get("APERTUS_LOG_LOSS_SPIKES") == "1",
        "log_top1_acc": os.environ.get("APERTUS_LOG_TOP1_ACC", "1") != "0",
    })
    hooks.patch_setup_model_and_optimizer()
    hooks.patch_training_log()
    hooks.patch_compute_language_model_loss()
    hooks.patch_train_step()

    atexit.register(_destroy_process_group_on_exit)
