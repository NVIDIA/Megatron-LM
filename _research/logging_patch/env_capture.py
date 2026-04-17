"""Capture the run environment: git, host, python/torch/cuda, slurm, config."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_info(repo_dir: str | Path) -> dict[str, Any]:
    cwd = str(repo_dir)
    sha = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    short_sha = _run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    status = _run(["git", "status", "--porcelain"], cwd=cwd)
    worktree = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    return {
        "git_sha": sha,
        "git_short_sha": short_sha,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_status": status,
        "git_worktree": worktree,
    }


def host_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER"),
        "pwd": os.getcwd(),
        "python": sys.version.split()[0],
    }


def framework_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformer_engine
        info["transformer_engine"] = transformer_engine.__version__
    except (ImportError, AttributeError):
        pass
    return info


def slurm_info() -> dict[str, Any]:
    keys = [
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NNODES", "SLURM_NTASKS",
        "SLURM_GPUS_PER_NODE", "SLURM_CPUS_PER_TASK", "SLURM_NODELIST",
        "SLURM_JOB_RESERVATION", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
    ]
    return {k.lower(): os.environ[k] for k in keys if k in os.environ}


def args_snapshot(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    snap: dict[str, Any] = {}
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue
        try:
            import json as _json
            _json.dumps(v)
            snap[k] = v
        except (TypeError, ValueError):
            snap[k] = repr(v)
    return snap


def capture(repo_dir: str | Path, args: Any = None) -> dict[str, Any]:
    """Gather a full environment snapshot for the run JSON meta field."""
    return {
        "git": git_info(repo_dir),
        "host": host_info(),
        "framework": framework_info(),
        "slurm": slurm_info(),
        "argv": sys.argv,
        "config": args_snapshot(args),
    }
