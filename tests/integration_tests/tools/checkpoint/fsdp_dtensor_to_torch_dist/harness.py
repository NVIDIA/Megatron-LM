# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Controller-side orchestration for the reverse-converter end-to-end suite.

Launches the real training / resume / reshard stages as ``torchrun`` subprocesses
running ``pretrain_gpt.py``, converts checkpoints via the converter CLI, and parses
their output into structured Python values (per-iteration loss + LR, load
confirmations, bit-exact verdicts) that the tests assert on.

It is **controller-only**: it never imports ``torch`` / ``megatron`` and never
initializes CUDA or NCCL. All model work happens in the subprocess children (the
torchrun jobs and the bit-exact worker), so this suite is safe to run as plain
pytest — each test spawns its own torchrun children.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import config

_WORKER = Path(__file__).parent / "_bitexact_worker.py"

# torch.distributed / torchelastic vars we must not leak into a child torchrun,
# otherwise a child inherits a stale rendezvous (e.g. if pytest itself was, against
# advice, launched under torch.distributed.run).
_DIST_ENV_KEYS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "LOCAL_WORLD_SIZE",
)  # fmt: skip


@dataclass(frozen=True)
class IterMetrics:
    """The two numbers we compare across the convert/resume boundary."""

    lm_loss: float
    learning_rate: float


@dataclass(frozen=True)
class BitexactVerdict:
    """Parsed JSON verdict emitted by ``_bitexact_worker.py``."""

    family: str
    loaded_iteration: int
    verdict: str
    weight_mismatches: Tuple[str, ...]
    optim_mismatches: Tuple[str, ...]
    unexpected_extra: Tuple[str, ...]

    @property
    def is_bit_exact(self) -> bool:
        return not (self.weight_mismatches or self.optim_mismatches or self.unexpected_extra)


# --- env + subprocess plumbing ----------------------------------------------
def _strip_dist_env(env: dict) -> dict:
    for k in list(env):
        if k in _DIST_ENV_KEYS or k.startswith("TORCHELASTIC") or k.startswith("RDZV"):
            env.pop(k, None)
    return env


def _free_port() -> int:
    """Grab a currently-free TCP port. Small TOCTOU race is fine — torchrun retries."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _torchrun_env(extra: Optional[dict] = None) -> dict:
    """Env for a torchrun child: strip inherited rendezvous, add deterministic vars."""
    env = _strip_dist_env(dict(os.environ))
    env.update(config.DETERMINISTIC_ENV)
    if extra:
        env.update(extra)
    return env


def _single_rank_env(port: int, extra: Optional[dict] = None) -> dict:
    """Env for a plain (non-torchrun) single-rank child: convert CLI / bit-exact worker."""
    env = _strip_dist_env(dict(os.environ))
    env.update(
        {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
         "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}
    )  # fmt: skip
    env.update(config.DETERMINISTIC_ENV)
    if extra:
        env.update(extra)
    return env


def _run(argv: Sequence, env: dict, log_path: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a child, tee combined stdout+stderr to ``log_path``, return the result.

    Runs with ``cwd=REPO_ROOT`` so ``pretrain_gpt.py``'s repo-root imports
    (``gpt_builders`` / ``model_provider``) resolve. The child is started in its own
    session (process group) so that on timeout the whole tree — torchrun *and* the
    worker processes it forks — is killed, rather than leaking orphaned GPU
    processes.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(a) for a in argv],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(config.REPO_ROOT),
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        out, _ = proc.communicate()
        log_path.write_text(out or "")
        raise
    log_path.write_text(out or "")
    return subprocess.CompletedProcess(argv, proc.returncode, stdout=out)


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _check(proc: subprocess.CompletedProcess, log_path: Path, what: str) -> None:
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(
            f"{what} failed (rc={proc.returncode}); full log at {log_path}\n--- tail ---\n{tail}"
        )


def _torchrun_argv(nproc: int, port: int, script: Path, args: Sequence[str]) -> list:
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(nproc), "--master_port", str(port),
        str(script), *args,
    ]  # fmt: skip


def _data_cache_for(log_path: Path) -> Path:
    """A per-stage data-cache path so concurrent/sequential stages never collide."""
    return log_path.parent / f"dc_{log_path.stem}"


# --- stages -----------------------------------------------------------------
def run_training(
    fam,
    out_dir: Path,
    *,
    nproc: int = 1,
    src_parallel: Sequence[str] = (),
    train_iters: int = config.TRAIN_ITERS,
    save_interval: int = config.SAVE_INTERVAL,
    timeout: int = 3600,
) -> str:
    """Train real Megatron-FSDP to ``train_iters`` (save every ``save_interval``).

    Produces ``out_dir/fsdp/iter_XXXXXXX`` fsdp_dtensor checkpoints. Returns the
    combined log text (also written to ``out_dir/train_fsdp.log``).

    Wipes ``out_dir`` first so a reused ``RESULTS_DIR`` never carries stale
    checkpoints into a fresh run.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    log_path = out_dir / "train_fsdp.log"
    extra = dict(fam.extra_env)
    if nproc >= 2:
        extra.update(config.MULTI_GPU_TRAIN_ENV)
    args = [
        *config.FSDP_TRAIN_FLAGS, *config.COMMON_ARGS,
        "--num-layers", str(fam.num_layers), *fam.arch, *src_parallel,
        "--train-iters", str(train_iters), "--save-interval", str(save_interval),
        "--save", str(out_dir / "fsdp"), "--data-cache-path", str(_data_cache_for(log_path)),
    ]  # fmt: skip
    proc = _run(
        _torchrun_argv(nproc, _free_port(), config.PRETRAIN_GPT, args),
        _torchrun_env(extra),
        log_path,
        timeout,
    )
    _check(proc, log_path, f"[{fam.name}] FSDP training")
    return proc.stdout or ""


def convert(fsdp_iter_dir: Path, td_dir: Path, iteration: int, *, timeout: int = 900) -> Path:
    """Reverse-convert one fsdp_dtensor checkpoint to torch_dist under ``td_dir``.

    Writes ``td_dir/iter_XXXXXXX`` and the ``latest_checkpointed_iteration.txt``
    marker mcore's loader needs. Returns ``td_dir``.
    """
    out_iter = td_dir / f"iter_{iteration:07d}"
    log_path = td_dir / f"convert_{iteration}.log"
    argv = [
        sys.executable, str(config.INSPECTOR),
        "convert-fsdp-dtensor-to-torch-dist", str(fsdp_iter_dir), str(out_iter),
    ]  # fmt: skip
    proc = _run(argv, _single_rank_env(_free_port()), log_path, timeout)
    _check(proc, log_path, f"convert iter {iteration}")
    (td_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    return td_dir


def resume_classic(
    fam,
    td_dir: Path,
    iteration: int,
    log_path: Path,
    *,
    target_parallel: Sequence[str] = (),
    with_optimizer: bool = True,
    nproc: int = 1,
    timeout: int = 1800,
) -> str:
    """Resume a classic (non-FSDP) job from a converted checkpoint for a few iters.

    Runs ``config.RESUME_EXTRA_ITERS`` iterations past the load point. ``target_parallel``
    is appended AFTER COMMON_ARGS so it overrides the TP1 there (later wins) — this is
    how the reshard sweep changes the load-side layout. Returns the combined log text
    (also written to ``log_path``).
    """
    end = iteration + config.RESUME_EXTRA_ITERS
    extra = dict(fam.extra_env)
    if nproc >= 2:
        extra.update(config.MULTI_GPU_LOAD_ENV)
    load_flags = list(config.CLASSIC_LOAD_FLAGS)
    if not with_optimizer:
        load_flags.append("--no-load-optim")
    args = [
        *load_flags, *config.COMMON_ARGS,
        "--num-layers", str(fam.num_layers), *fam.arch, *target_parallel,
        "--train-iters", str(end), "--save-interval", "1000",
        "--load", str(td_dir), "--data-cache-path", str(_data_cache_for(log_path)),
    ]  # fmt: skip
    proc = _run(
        _torchrun_argv(nproc, _free_port(), config.PRETRAIN_GPT, args),
        _torchrun_env(extra),
        log_path,
        timeout,
    )
    _check(proc, log_path, f"[{fam.name}] classic resume from {td_dir.name}")
    return proc.stdout or ""


def reshard_load(fam, td_dir: Path, reshard, log_path: Path, iteration: int = 80, **kw) -> str:
    """Load ``td_dir`` (converted) into a 2-GPU classic job under a target layout."""
    return resume_classic(
        fam, td_dir, iteration, log_path,
        target_parallel=config.target_parallel_flags(reshard.layout),
        with_optimizer=reshard.with_optimizer,
        nproc=2,
        **kw,
    )  # fmt: skip


def run_bitexact_worker(
    fam, td_dir: Path, iteration: int, *, timeout: int = 1800
) -> BitexactVerdict:
    """Run the per-family bit-exact worker in its own process, return its JSON verdict."""
    log_path = td_dir / f"bitexact_{iteration}.log"
    argv = [sys.executable, str(_WORKER), fam.name, "--iter", str(iteration), "--td", str(td_dir)]
    proc = _run(argv, _single_rank_env(_free_port()), log_path, timeout)
    _check(proc, log_path, f"[{fam.name}] bit-exact worker")
    return _parse_bitexact_json(fam.name, proc.stdout or "", log_path)


# --- parsing + assertions ---------------------------------------------------
_ITER_RE = re.compile(r"\biteration\s+(\d+)\s*/")
_LOSS_RE = re.compile(r"lm loss:\s*([0-9.eE+\-]+)")
_LR_RE = re.compile(r"learning rate:\s*([0-9.eE+\-]+)")
_LOADED_RE = re.compile(r"successfully loaded checkpoint from .* at iteration\s+(\d+)")
_JSON_BEGIN = "===BITEXACT_JSON_BEGIN==="
_JSON_END = "===BITEXACT_JSON_END==="


def parse_iter_metrics(text: str) -> Dict[int, IterMetrics]:
    """Extract {iteration -> (lm loss, learning rate)} from training-log lines.

    Keys off the ``training_log`` line format (megatron/training/training.py: the
    ` iteration N/M |`, ` learning rate: E |`, ` lm loss: E |` fields).
    """
    out: Dict[int, IterMetrics] = {}
    for line in text.splitlines():
        m_it = _ITER_RE.search(line)
        m_loss = _LOSS_RE.search(line)
        m_lr = _LR_RE.search(line)
        if m_it and m_loss and m_lr:
            out[int(m_it.group(1))] = IterMetrics(float(m_loss.group(1)), float(m_lr.group(1)))
    return out


def assert_loaded_at(text: str, iteration: int) -> None:
    """Assert the classic job reported loading the checkpoint at ``iteration``."""
    loaded = [int(m.group(1)) for m in _LOADED_RE.finditer(text)]
    assert iteration in loaded, (
        f"expected 'successfully loaded checkpoint ... at iteration {iteration}', "
        f"saw loads at {loaded or 'none'}"
    )


def assert_loss_lr(
    ref: IterMetrics, got: IterMetrics, *, loss_rtol: float, lr_exact: bool = True
) -> None:
    """Compare a resumed iteration against the FSDP reference at the same iteration.

    Loss must match within ``loss_rtol`` (weights loaded correctly); LR must match
    ~exactly (optimizer + LR-scheduler bookkeeping converted correctly).
    """
    loss_rel = abs(got.lm_loss - ref.lm_loss) / max(abs(ref.lm_loss), 1e-12)
    assert loss_rel <= loss_rtol, (
        f"lm loss {got.lm_loss:.6f} vs FSDP {ref.lm_loss:.6f} "
        f"(rel {loss_rel:.2e} > tol {loss_rtol:.2e})"
    )
    if lr_exact:
        lr_rel = abs(got.learning_rate - ref.learning_rate) / max(abs(ref.learning_rate), 1e-12)
        assert lr_rel <= 1e-6, (
            f"learning rate {got.learning_rate:.6e} vs FSDP {ref.learning_rate:.6e} "
            f"(rel {lr_rel:.2e}); expected exact match"
        )


def _parse_bitexact_json(family: str, text: str, log_path: Path) -> BitexactVerdict:
    if _JSON_BEGIN not in text or _JSON_END not in text:
        tail = "\n".join(text.splitlines()[-40:])
        raise RuntimeError(
            f"[{family}] bit-exact worker emitted no JSON verdict; log at {log_path}\n"
            f"--- tail ---\n{tail}"
        )
    blob = text.split(_JSON_BEGIN, 1)[1].split(_JSON_END, 1)[0].strip()
    d = json.loads(blob)
    return BitexactVerdict(
        family=d["family"],
        loaded_iteration=int(d["loaded_iteration"]),
        verdict=d["verdict"],
        weight_mismatches=tuple(d.get("weight_mismatches", ())),
        optim_mismatches=tuple(d.get("optim_mismatches", ())),
        unexpected_extra=tuple(d.get("unexpected_extra", ())),
    )
