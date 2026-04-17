"""Per-run JSON log loader for the apertus-2-ablations analyse notebook.

Mirrors the optimiser-playground pattern: Megatron writes one JSON per run to
`_research/results/runs/<run_name>.json`, this module loads them and exposes
helpers for comparison plots.

The JSON schema is defined by the logging patch on `main` (see `_apertus/log.md`
and the patch description). Minimal expected fields:

    {
        "name": str,
        "feature": str,           # branch name
        "track": "throughput" | "performance",
        "git_sha": str,
        "config": {...},          # resolved argparse namespace
        "n_params_total": int,
        "n_params_active": int,
        "wall_time_seconds": float,
        "final_val_loss": float | null,
        "tokens_per_sec_per_gpu": float,
        "mfu": float,
        "steps": int,
        "tokens": int,
        "series": {
            "step":         [int, ...],
            "train_loss":   [float, ...],
            "val_loss":     [float, ...],     # may be sparse
            "tput":         [float, ...],
            "grad_norm":    [float, ...],
            "act_max":      [float, ...],     # or per-layer dict
            ...
        }
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_VAL_LOSS_RE = re.compile(
    r"validation loss at iteration (\d+) on validation set \| lm loss value: ([0-9.eE+-]+)"
)
_JOBID_RE = re.compile(r"-(\d{6,})\.json$")


def _jobid(path: Path) -> str | None:
    m = _JOBID_RE.search(path.name)
    return m.group(1) if m else None


def _parse_val_loss_from_log(log_path: Path) -> tuple[int, float] | None:
    """Return (iteration, val_loss) of the LAST validation line in a slurm log."""
    if not log_path.exists():
        return None
    last: tuple[int, float] | None = None
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _VAL_LOSS_RE.search(line)
            if m:
                last = (int(m.group(1)), float(m.group(2)))
    return last


def load_runs(
    log_dir: str | Path = "_research/results/performance",
    names: list[str] | None = None,
    runs_dir: str | Path | None = None,
    feature: str | None = None,
) -> dict[str, dict]:
    """Load every `<name>.json` under `log_dir`.

    If `runs_dir` points to a directory of slurm `.log` files, a final val loss is
    extracted (by matching jobid suffix) and attached to each run as
    `final_val_loss` / `final_val_loss_iter`.

    If `feature` is set, it overrides the per-run `feature` field so the notebook
    can group runs by their source worktree.
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return {}
    if names is None:
        files = sorted(log_dir.glob("*.json"))
    else:
        files = [log_dir / f"{n}.json" for n in names]
    runs_dir_p = Path(runs_dir) if runs_dir is not None else None
    log_index: dict[str, Path] = {}
    if runs_dir_p is not None and runs_dir_p.exists():
        for p in runs_dir_p.glob("*.log"):
            m = re.search(r"-(\d{6,})\.log$", p.name)
            if m:
                log_index[m.group(1)] = p
    runs: dict[str, dict] = {}
    for f in files:
        if not f.exists():
            print(f"missing: {f}")
            continue
        with f.open() as fh:
            run = json.load(fh)
        if feature is not None:
            run["feature"] = feature
        jid = _jobid(f)
        if jid and jid in log_index:
            parsed = _parse_val_loss_from_log(log_index[jid])
            if parsed is not None:
                run["final_val_loss"] = parsed[1]
                run["final_val_loss_iter"] = parsed[0]
        runs[f.stem] = run
    return runs


def load_runs_tree(
    perf_root: str | Path,
    worktrees_root: str | Path,
    subdir: str = "performance",
    runs_dir: str | Path | None = None,
) -> dict[str, dict]:
    """Load runs from both the main clone and every worktree feature dir.

    Layout expected locally:
        perf_root/<*.json>                          # main clone mirror
        worktrees_root/<feature>/<*.json>           # per-feature mirrors
        runs_dir/<jobname>-<jobid>.log              # slurm stdout mirror

    Returns a dict keyed by run name (JSON stem), deduped if the same file was
    rsynced twice. Each run carries its `feature` field set from the directory
    it was loaded from.
    """
    out: dict[str, dict] = {}
    out.update(load_runs(perf_root, runs_dir=runs_dir, feature="main"))
    worktrees_root = Path(worktrees_root)
    if worktrees_root.exists():
        for d in sorted(worktrees_root.iterdir()):
            if not d.is_dir():
                continue
            sub = d / subdir if (d / subdir).exists() else d
            if not sub.exists():
                continue
            out.update(load_runs(sub, runs_dir=runs_dir, feature=d.name))
    return out


def rank_by(runs: dict[str, dict], metric: str = "final_val_loss", ascending: bool = True) -> list[tuple[str, float]]:
    """Rank runs by a scalar metric. Default: lowest final val loss first."""
    rows = [(name, run.get(metric)) for name, run in runs.items() if run.get(metric) is not None]
    rows.sort(key=lambda x: x[1], reverse=not ascending)
    return rows


def summary_table(runs: dict[str, dict]) -> list[dict]:
    """One row per run with the scalar fields for quick display in a DataFrame."""
    cols = [
        "name", "feature", "track", "git_sha",
        "final_val_loss", "tokens_per_sec_per_gpu", "mfu",
        "wall_time_seconds", "n_params_active", "n_params_total",
        "steps", "tokens",
    ]
    rows = []
    for name, run in runs.items():
        row = {c: run.get(c) for c in cols}
        row["name"] = name
        rows.append(row)
    return rows
