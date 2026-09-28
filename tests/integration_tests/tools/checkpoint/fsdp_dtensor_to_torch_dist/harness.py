# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Controller-side orchestration for the reverse-converter end-to-end suite.

Launches the real training / resume / reshard stages as ``torchrun`` subprocesses
running each family's entrypoint (``pretrain_gpt.py`` / ``pretrain_hybrid.py``),
converts checkpoints via the converter CLI, and parses their output into structured
Python values (per-iteration loss + LR, load confirmations, bit-exact verdicts) that
the tests assert on.

It is **controller-only**: it never imports ``megatron`` and never initializes CUDA
or NCCL (``diff_torch_dist`` reads CPU tensors only). All model work happens in the
subprocess children (the torchrun jobs and the bit-exact worker), so this suite is
safe to run as plain pytest — each test spawns its own torchrun children.
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
    counts: Dict[str, int]  # tensors / param groups compared, per section
    mismatches: Tuple[str, ...]
    missing: Tuple[str, ...]  # in the FSDP source, not held by the classic job
    unexpected: Tuple[str, ...]  # held by the classic job, not in the FSDP source


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
    extra = config.MULTI_GPU_TRAIN_ENV if nproc >= 2 else None
    args = [
        *config.FSDP_TRAIN_FLAGS, *config.COMMON_ARGS,
        "--num-layers", str(fam.num_layers), *fam.arch, *src_parallel,
        "--train-iters", str(train_iters), "--save-interval", str(save_interval),
        "--save", str(out_dir / "fsdp"), "--data-cache-path", str(_data_cache_for(log_path)),
    ]  # fmt: skip
    proc = _run(
        _torchrun_argv(nproc, _free_port(), config.REPO_ROOT / fam.entrypoint, args),
        _torchrun_env(extra),
        log_path,
        timeout,
    )
    _check(proc, log_path, f"[{fam.name}] FSDP training")
    return proc.stdout or ""


def convert(
    fsdp_iter_dir: Path, td_dir: Path, iteration: int, *, nproc: int = 1, timeout: int = 900
) -> Path:
    """Reverse-convert one fsdp_dtensor checkpoint to torch_dist under ``td_dir``.

    Writes ``td_dir/iter_XXXXXXX`` and the ``latest_checkpointed_iteration.txt``
    marker mcore's loader needs. ``nproc > 1`` runs the converter's multi-process
    (CPU, gloo) mode under torchrun. Returns ``td_dir``.
    """
    out_iter = td_dir / f"iter_{iteration:07d}"
    log_path = td_dir / f"convert_{iteration}.log"
    cli = ["convert-fsdp-dtensor-to-torch-dist", str(fsdp_iter_dir), str(out_iter)]
    if nproc > 1:
        argv, env = _torchrun_argv(nproc, _free_port(), config.INSPECTOR, cli), _torchrun_env()
    else:
        argv, env = [sys.executable, str(config.INSPECTOR), *cli], _single_rank_env(_free_port())
    proc = _run(argv, env, log_path, timeout)
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
    extra = config.MULTI_GPU_LOAD_ENV if nproc >= 2 else None
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
        _torchrun_argv(nproc, _free_port(), config.REPO_ROOT / fam.entrypoint, args),
        _torchrun_env(extra),
        log_path,
        timeout,
    )
    _check(proc, log_path, f"[{fam.name}] classic resume from {td_dir.name}")
    return proc.stdout or ""


def run_bitexact_worker(
    fam, fsdp_dir: Path, td_dir: Path, iteration: int, *, timeout: int = 1800
) -> BitexactVerdict:
    """Load ``td_dir`` into a classic model in its own process; diff it against ``fsdp_dir``."""
    log_path = td_dir / f"bitexact_{iteration}.log"
    argv = [
        sys.executable, str(_WORKER), fam.name, "--iter", str(iteration),
        "--td", str(td_dir), "--fsdp", str(fsdp_dir),
    ]  # fmt: skip
    proc = _run(argv, _single_rank_env(_free_port()), log_path, timeout)
    _check(proc, log_path, f"[{fam.name}] bit-exact worker")
    return _parse_bitexact_json(fam.name, proc.stdout or "", log_path)


def diff_torch_dist(dir_a: Path, dir_b: Path) -> list:
    """Differences between two torch_dist checkpoint dirs: tensor keys, tensor values
    (``torch.equal``) and the ``common.pt`` payload. Empty list means identical.

    Imports torch lazily and only touches CPU tensors (no CUDA / NCCL init).
    """
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    def tensors(path):
        md = FileSystemReader(str(path)).read_metadata().state_dict_metadata
        out = {
            k: torch.empty(m.size, dtype=m.properties.dtype)
            for k, m in md.items()
            if isinstance(m, TensorStorageMetadata)
        }
        dcp.load(out, storage_reader=FileSystemReader(str(path)), no_dist=True)
        return out

    a, b = tensors(dir_a), tensors(dir_b)
    diffs = [f"only in {dir_a.parent.name}: {k}" for k in sorted(set(a) - set(b))]
    diffs += [f"only in {dir_b.parent.name}: {k}" for k in sorted(set(b) - set(a))]
    diffs += [f"values differ: {k}" for k in sorted(set(a) & set(b)) if not torch.equal(a[k], b[k])]
    common_a, common_b = (torch.load(d / "common.pt", weights_only=False) for d in (dir_a, dir_b))
    if common_a != common_b:
        diffs.append("common.pt differs")
    return diffs


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


def compare_loss_lr(ref: IterMetrics, got: IterMetrics, *, loss_rtol: float) -> Optional[str]:
    """Compare a resumed iteration against the FSDP reference at the same iteration.

    Loss must match within ``loss_rtol``; LR must match ~exactly (optimizer +
    LR-scheduler bookkeeping converted correctly). Returns a failure reason or None.
    """
    loss_rel = abs(got.lm_loss - ref.lm_loss) / max(abs(ref.lm_loss), 1e-12)
    lr_rel = abs(got.learning_rate - ref.learning_rate) / max(abs(ref.learning_rate), 1e-12)
    if loss_rel > loss_rtol:
        return f"lm loss rel diff {loss_rel:.2e} > tol {loss_rtol:.2e}"
    if lr_rel > 1e-6:
        return f"learning rate rel diff {lr_rel:.2e}; expected an exact match"
    return None


def check_resume(
    fam, reference: Dict[int, IterMetrics], td_dir: Path, iteration: int, log_path: Path, **kw
) -> None:
    """Resume a classic job from ``td_dir`` and assert it continues the FSDP run.

    Every resumed iteration is compared, not just the first: the first post-load
    loss is a forward pass on the loaded weights, while the later ones also depend
    on the Adam step taken from the loaded masters, moments and step count. A
    weights-only load (``with_optimizer=False``) starts from fresh moments, so only
    its first iteration can match. Every comparison is printed before asserting.
    """
    text = resume_classic(fam, td_dir, iteration, log_path, **kw)
    assert_loaded_at(text, iteration)
    resumed = parse_iter_metrics(text)
    n_iters = config.RESUME_EXTRA_ITERS if kw.get("with_optimizer", True) else 1
    failures = []
    for it in range(iteration + 1, iteration + 1 + n_iters):
        assert it in resumed, f"[{fam.name}] no iteration {it} in {log_path}"
        ref, got = reference[it], resumed[it]
        failure = compare_loss_lr(ref, got, loss_rtol=fam.loss_rtol)
        print(
            f"[{fam.name}] {log_path.stem} iter {it}: lm loss FSDP {ref.lm_loss:.6f} -> "
            f"resumed {got.lm_loss:.6f}, lr {ref.learning_rate:.6e} -> "
            f"{got.learning_rate:.6e}  {failure or 'ok'}"
        )
        if failure:
            failures.append(f"iter {it}: {failure}")
    assert not failures, f"[{fam.name}] resume from {td_dir.name} diverged: {failures}"


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
        counts=dict(d["counts"]),
        mismatches=tuple(d["mismatches"]),
        missing=tuple(d["missing"]),
        unexpected=tuple(d["unexpected"]),
    )
