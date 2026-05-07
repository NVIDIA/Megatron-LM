# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3

"""Compare old vs. new golden-value JSONs using KL divergence.

The golden-value JSON files produced by `download_golden_values.py` look like:

    {
        "lm loss": {
            "start_step": 1, "end_step": 50, "step_interval": 1,
            "values": {"1": 10.93, "2": 10.92, ...}
        },
        "num-zeros": { ... }
    }

For each metric we treat the per-step value series as a discrete distribution
(after restricting to steps present in both files and applying a strictly
positive shift) and compute KL divergence plus a couple of summary stats.

The printed table and the CSV both emit the same 6 columns, each answering
a distinct question:

  * file / metric          -- which test, which metric
  * sym_KL                 -- did the distribution shape change, and by how
                              much?
  * median(old)/median(new) -- typical value before/after (matches what the
                               framework's median-based check uses, e.g. for
                               iteration-time)
  * max rel|d|             -- worst-step relative shift; catches single bad
                               steps that approximate-equality checks reject

By default the script compares the working-tree version of each modified
golden-value file against its `git show HEAD:<path>` version, so the typical
flow is:

    # after running download_golden_values.py, before committing
    python tests/test_utils/python_scripts/compare_golden_values_kl.py

You can also point at explicit files / pairs:

    # compare a single working-tree file against HEAD
    python .../compare_golden_values_kl.py --file path/to/golden.json

    # compare two arbitrary files
    python .../compare_golden_values_kl.py --old old.json --new new.json

    # write a CSV summary
    python .../compare_golden_values_kl.py --csv kl_summary.csv

    # only print rows whose symmetric KL exceeds a threshold
    python .../compare_golden_values_kl.py --threshold 1e-4
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

import click

logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    ).stdout.strip()
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_json_text(text: str) -> dict:
    return json.loads(text)


def load_from_path(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_from_git(rev: str, path: pathlib.Path) -> dict | None:
    """Load the contents of `path` at git revision `rev`. None if absent."""
    rel = path.resolve().relative_to(REPO_ROOT)
    try:
        out = subprocess.run(
            ["git", "show", f"{rev}:{rel.as_posix()}"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return _load_json_text(out)


def list_modified_golden_files() -> list[pathlib.Path]:
    """Files under tests/functional_tests/test_cases changed vs HEAD (tracked + untracked)."""
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "tests/functional_tests/test_cases"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "tests/functional_tests/test_cases",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.splitlines()

    paths: list[pathlib.Path] = []
    for rel in tracked + untracked:
        if not rel.endswith(".json"):
            continue
        p = REPO_ROOT / rel
        if p.exists():
            paths.append(p)
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def _normalize(values: list[float]) -> list[float]:
    """Shift to be strictly positive and normalize so they sum to 1.

    Loss-style trajectories are not probability distributions, so we treat the
    series as positive masses. Shifting by `min - eps` keeps the relative
    differences between adjacent steps well-defined while making KL finite.
    """
    eps = 1e-12
    m = min(values)
    shift = 0.0 if m > eps else (-m + eps)
    shifted = [v + shift for v in values]
    s = sum(shifted)
    if s <= 0:
        return [1.0 / len(shifted)] * len(shifted)
    return [v / s for v in shifted]


def kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(P || Q) in nats. Assumes p, q are probability vectors of equal length."""
    eps = 1e-12
    out = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        out += pi * math.log(pi / max(qi, eps))
    return out


# ---------------------------------------------------------------------------
# Per-metric comparison
# ---------------------------------------------------------------------------


@dataclass
class MetricStats:
    file: str
    metric: str
    sym_kl: float
    median_old: float
    median_new: float
    max_rel_diff: float


def _median(xs: list[float]) -> float:
    n = len(xs)
    s = sorted(xs)
    mid = n // 2
    return s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])


def _extract_series(metric_block: dict) -> dict[int, float]:
    values = metric_block.get("values", {})
    out: dict[int, float] = {}
    for k, v in values.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        out[int(k)] = fv
    return out


def compare_metric(
    file_label: str, metric_name: str, old_block: dict, new_block: dict
) -> MetricStats | None:
    old_series = _extract_series(old_block)
    new_series = _extract_series(new_block)
    shared = sorted(set(old_series) & set(new_series))
    if len(shared) < 2:
        return None

    old_vals = [old_series[s] for s in shared]
    new_vals = [new_series[s] for s in shared]

    p = _normalize(old_vals)
    q = _normalize(new_vals)

    rel_diffs = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(old_vals, new_vals)]

    sym_kl = kl_divergence(p, q) + kl_divergence(q, p)

    return MetricStats(
        file=file_label,
        metric=metric_name,
        sym_kl=sym_kl,
        median_old=_median(old_vals),
        median_new=_median(new_vals),
        max_rel_diff=max(rel_diffs),
    )


def compare_files(file_label: str, old_doc: dict, new_doc: dict) -> list[MetricStats]:
    metrics = sorted(set(old_doc) & set(new_doc))
    rows: list[MetricStats] = []
    for metric in metrics:
        stats = compare_metric(file_label, metric, old_doc[metric], new_doc[metric])
        if stats is not None:
            rows.append(stats)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Six columns, each answers a distinct question:
#   file / metric        -- which test, which metric
#   sym_KL               -- did the distribution shape change, and by how much?
#   median(old)/median(new) -- typical value before/after (matches the
#                              framework's median-based check, e.g.
#                              iteration-time)
#   max rel|d|           -- worst-step relative shift; catches single bad
#                            steps that approximate-equality checks reject
_HEADERS = ["file", "metric", "sym_KL", "median(old)", "median(new)", "max rel|d|"]


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e4:
        return f"{x:.3e}"
    return f"{x:.6f}"


def _row(r: MetricStats) -> list:
    return [r.file, r.metric, r.sym_kl, r.median_old, r.median_new, r.max_rel_diff]


def print_table(rows: Iterable[MetricStats]) -> None:
    rows = list(rows)
    if not rows:
        print("(no comparable metrics)")
        return

    table = [_HEADERS] + [[c if isinstance(c, str) else _fmt(c) for c in _row(r)] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(_HEADERS))]
    for i, row in enumerate(table):
        line = "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        print(line)
        if i == 0:
            print("  ".join("-" * w for w in widths))


def write_csv(rows: Iterable[MetricStats], path: pathlib.Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_HEADERS)
        for r in rows:
            w.writerow(_row(r))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--rev",
    default="HEAD",
    show_default=True,
    help="Git revision to use as the 'old' side when comparing the working tree.",
)
@click.option(
    "--file",
    "files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Specific working-tree golden-value JSON to compare against --rev. Repeatable.",
)
@click.option(
    "--old",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Compare an arbitrary 'old' file against --new. Bypasses git.",
)
@click.option(
    "--new",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Compare an arbitrary 'new' file against --old. Bypasses git.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="Only print metrics with sym_KL >= threshold.",
)
@click.option(
    "--top",
    type=int,
    default=0,
    help="If >0, also print the top-N metrics by sym_KL across all files.",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help="Write the full comparison table to this CSV path.",
)
def main(
    rev: str,
    files: tuple[pathlib.Path, ...],
    old: pathlib.Path | None,
    new: pathlib.Path | None,
    threshold: float,
    top: int,
    csv_path: pathlib.Path | None,
):
    """Compare old vs new golden-value JSONs using KL divergence."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    all_rows: list[MetricStats] = []

    if (old is None) != (new is None):
        raise click.UsageError("--old and --new must be provided together")

    if old is not None and new is not None:
        old_doc = load_from_path(old)
        new_doc = load_from_path(new)
        label = f"{old.name} vs {new.name}"
        all_rows.extend(compare_files(label, old_doc, new_doc))
    else:
        target_files = list(files) if files else list_modified_golden_files()
        if not target_files:
            print(
                "No modified golden-value files found under "
                "tests/functional_tests/test_cases. Pass --file or --old/--new explicitly."
            )
            sys.exit(0)

        logger.info("Comparing %d file(s) against %s", len(target_files), rev)
        for p in target_files:
            p = p.resolve()
            old_doc = load_from_git(rev, p)
            if old_doc is None:
                logger.warning("Skipping %s: not present at %s (new file).", p, rev)
                continue
            try:
                new_doc = load_from_path(p)
            except Exception as e:
                logger.warning("Skipping %s: failed to read working-tree copy (%s).", p, e)
                continue
            label = str(p.relative_to(REPO_ROOT))
            all_rows.extend(compare_files(label, old_doc, new_doc))

    rows = [r for r in all_rows if r.sym_kl >= threshold]
    rows.sort(key=lambda r: r.sym_kl, reverse=True)

    print_table(rows)

    if top > 0:
        print()
        print(f"Top {min(top, len(all_rows))} metrics by symmetric KL:")
        top_rows = sorted(all_rows, key=lambda r: r.sym_kl, reverse=True)[:top]
        print_table(top_rows)

    if csv_path is not None:
        write_csv(all_rows, csv_path)
        logger.info("Wrote %d rows to %s", len(all_rows), csv_path)


if __name__ == "__main__":
    main()
