#!/usr/bin/env python3
"""Summarize existing grad-norm logs across sharding schemes.

The script scans a directory for JSONL files produced via
`--log-global-grad-norms`, associates each log with its `(tp, pp, cp, dp)`
configuration (based on the filename convention `tp{T}_pp{P}_cp{C}_dp{D}.jsonl`),
extracts the final gradient norm, and prints a comparison table. A CSV copy
is optional.

Example:

    python tools/summarize_grad_norm_logs.py \
        --log-dir logs/grad_norm_sweep \
        --baseline tp1_pp1_cp1_dp8.jsonl

If no baseline is specified, the script uses the first log found as reference.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class LogEntry:
    path: Path
    tp: int
    pp: int
    cp: int
    dp: int
    grad_norm: float | None

    @property
    def tag(self) -> str:
        return f"tp{self.tp}_pp{self.pp}_cp{self.cp}_dp{self.dp}"


@dataclass
class SummaryRow:
    entry: LogEntry
    abs_diff: float | None
    rel_diff: float | None


def parse_config_from_name(filename: str) -> Optional[tuple[int, int, int, int]]:
    stem = Path(filename).stem
    parts = stem.split('_')

    tp = pp = cp = dp = None

    for part in parts:
        if part.startswith('tp') and part[2:].isdigit():
            tp = int(part[2:])
        elif part.startswith('pp') and part[2:].isdigit():
            pp = int(part[2:])
        elif part.startswith('cp') and part[2:].isdigit():
            cp = int(part[2:])
        elif part.startswith('dp') and part[2:].isdigit():
            dp = int(part[2:])
        elif part.startswith('ws') and part[2:].isdigit():
            dp = int(part[2:])  # fallback label from earlier runs

    # Default any missing dimensions to 1 except dp, which defaults to 1 if still None.
    tp = tp or 1
    pp = pp or 1
    cp = cp or 1
    dp = dp or 1

    return tp, pp, cp, dp


def list_logs(directory: Path) -> List[LogEntry]:
    entries: List[LogEntry] = []
    for path in sorted(directory.glob("*.jsonl")):
        config = parse_config_from_name(path.name)
        if config is None:
            continue
        tp, pp, cp, dp = config
        grad_norm = extract_grad_norm(path)
        entries.append(LogEntry(path=path, tp=tp, pp=pp, cp=cp, dp=dp, grad_norm=grad_norm))
    return entries


def extract_grad_norm(path: Path) -> float | None:
    try:
        with path.open() as handle:
            last_record = None
            for line in handle:
                stripped = line.strip()
                if stripped:
                    last_record = stripped
    except OSError:
        return None

    if last_record is None:
        return None

    try:
        data = json.loads(last_record)
    except json.JSONDecodeError:
        return None

    value = data.get("grad_norm")
    return float(value) if value is not None else None


def compute_summary(entries: List[LogEntry], baseline_tag: Optional[str]) -> List[SummaryRow]:
    if not entries:
        return []

    baseline_entry = None
    if baseline_tag:
        baseline_tag = baseline_tag.strip()
        for entry in entries:
            if entry.tag == baseline_tag or entry.path.name == baseline_tag:
                baseline_entry = entry
                break
    if baseline_entry is None:
        baseline_entry = entries[0]

    baseline_grad = baseline_entry.grad_norm if baseline_entry else None

    rows: List[SummaryRow] = []
    for entry in entries:
        if baseline_grad is None or entry.grad_norm is None:
            abs_diff = None
            rel_diff = None
        else:
            abs_diff = entry.grad_norm - baseline_grad
            denom = max(abs(baseline_grad), 1e-12)
            rel_diff = abs_diff / denom
        rows.append(SummaryRow(entry=entry, abs_diff=abs_diff, rel_diff=rel_diff))
    return rows


def print_table(rows: List[SummaryRow]) -> None:
    header_cols = [
        ("tp", 4),
        ("pp", 4),
        ("cp", 4),
        ("dp", 4),
        ("grad_norm", 14),
        ("abs_diff", 12),
        ("rel_diff", 12),
        ("log_path", None),
    ]

    def fmt(col_name: str, width: int | None, value: str) -> str:
        if width is None:
            return value
        return f"{value:>{width}}"

    header = " ".join(
        fmt(name, width, name) for name, width in header_cols
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        entry = row.entry
        grad_str = "-" if entry.grad_norm is None else f"{entry.grad_norm:.6e}"
        abs_str = "-" if row.abs_diff is None else f"{row.abs_diff:.3e}"
        rel_str = "-" if row.rel_diff is None else f"{row.rel_diff:.3e}"

        values = [
            fmt("tp", header_cols[0][1], str(entry.tp)),
            fmt("pp", header_cols[1][1], str(entry.pp)),
            fmt("cp", header_cols[2][1], str(entry.cp)),
            fmt("dp", header_cols[3][1], str(entry.dp)),
            fmt("grad_norm", header_cols[4][1], grad_str),
            fmt("abs_diff", header_cols[5][1], abs_str),
            fmt("rel_diff", header_cols[6][1], rel_str),
            entry.path.as_posix(),
        ]
        print(" ".join(values))


def write_csv(rows: List[SummaryRow], csv_path: Path) -> None:
    import csv

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tp", "pp", "cp", "dp", "grad_norm", "abs_diff", "rel_diff", "log_path"])
        for row in rows:
            entry = row.entry
            writer.writerow(
                [
                    entry.tp,
                    entry.pp,
                    entry.cp,
                    entry.dp,
                    entry.grad_norm,
                    row.abs_diff,
                    row.rel_diff,
                    str(entry.path),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory containing grad norm JSONL logs")
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline log filename or tag (e.g. tp1_pp1_cp1_dp8.jsonl). Defaults to the first entry.",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional path to write CSV summary")

    args = parser.parse_args()

    entries = list_logs(args.log_dir)
    if not entries:
        parser.error(f"No matching JSONL logs found under {args.log_dir}")

    rows = compute_summary(entries, args.baseline)
    print_table(rows)

    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"\nWrote CSV summary to {args.output_csv}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
