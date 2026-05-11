# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3

"""Compare old vs. new golden-value JSONs by average normalized relative difference.

The golden-value JSON files produced by `download_golden_values.py` look like:

    {
        "lm loss": {
            "start_step": 1, "end_step": 50, "step_interval": 1,
            "values": {"1": 10.93, "2": 10.92, ...}
        },
        "num-zeros": { ... }
    }

For each (file, metric) we compute a single number:

    avg_rel_diff = mean( (old_v - new_v) / old_v )    over shared steps

Steps where `|old_v|` is below a small epsilon are skipped to avoid a
division blow-up. The result is **signed**: a positive value means the
new run is smaller than the old run at the typical step (e.g. loss went
down); negative means it went up. The magnitude `|avg_rel_diff|` is the
average fractional shift, so it's directly comparable across metrics of
wildly different absolute scales.

By default the script compares the working-tree version of each modified
golden-value file against its `git show HEAD:<path>` version, so the
typical flow is:

    # after running download_golden_values.py, before committing
    python tests/test_utils/python_scripts/compare_golden_values_kl.py

You can also point at explicit files / pairs:

    # compare a single working-tree file against HEAD
    python .../compare_golden_values_kl.py --file path/to/golden.json

    # compare two arbitrary files
    python .../compare_golden_values_kl.py --old old.json --new new.json

    # write a CSV summary
    python .../compare_golden_values_kl.py --csv summary.csv

    # only print rows whose |avg_rel_diff| exceeds a threshold
    python .../compare_golden_values_kl.py --threshold 1e-3
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

# Steps where |old| is below this are skipped — (old - new) / old is not
# meaningful when old ≈ 0 (e.g. num-zeros on dense models, mem-* at step 0).
ZERO_EPS = 1e-12


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
# Per-metric comparison
# ---------------------------------------------------------------------------


@dataclass
class MetricStats:
    file: str
    metric: str
    n_steps: int  # number of shared steps that contributed (after skipping old≈0)
    avg_rel_diff: float  # mean( (old - new) / old )


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

    rel_diffs: list[float] = []
    for s in shared:
        old_v = old_series[s]
        new_v = new_series[s]
        if abs(old_v) < ZERO_EPS:
            continue
        rel_diffs.append((old_v - new_v) / old_v)

    if not rel_diffs:
        return None

    return MetricStats(
        file=file_label,
        metric=metric_name,
        n_steps=len(rel_diffs),
        avg_rel_diff=sum(rel_diffs) / len(rel_diffs),
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

_HEADERS = ["file", "metric", "n_steps", "avg_rel_diff"]


def _fmt(x: float) -> str:
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e4:
        return f"{x:.3e}"
    return f"{x:.6f}"


def _row(r: MetricStats) -> list:
    return [r.file, r.metric, r.n_steps, r.avg_rel_diff]


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
    help="Only print metrics whose |avg_rel_diff| >= threshold.",
)
@click.option(
    "--top",
    type=int,
    default=0,
    help="If >0, also print the top-N metrics by |avg_rel_diff| across all files.",
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
    """Compare old vs new golden-value JSONs by average normalized relative difference."""
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

    rows = [r for r in all_rows if abs(r.avg_rel_diff) >= threshold]
    rows.sort(key=lambda r: abs(r.avg_rel_diff), reverse=True)

    print_table(rows)

    if top > 0:
        print()
        print(f"Top {min(top, len(all_rows))} metrics by |avg_rel_diff|:")
        top_rows = sorted(all_rows, key=lambda r: abs(r.avg_rel_diff), reverse=True)[:top]
        print_table(top_rows)

    if csv_path is not None:
        write_csv(all_rows, csv_path)
        logger.info("Wrote %d rows to %s", len(all_rows), csv_path)


if __name__ == "__main__":
    main()
