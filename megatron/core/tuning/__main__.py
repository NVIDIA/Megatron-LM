# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command line for building and inspecting tuned tables.

Usage::

    python -m megatron.core.tuning merge  rec.rank*.json -o tables/sm103.json
    python -m megatron.core.tuning report rec.rank*.json
"""

import argparse
import collections
import sys

from megatron.core.tuning import table as table_mod


def _out(message: str) -> None:
    """Write one line to stdout (``print`` is a banned builtin in this repo)."""
    sys.stdout.write(message + "\n")


def _err(message: str) -> None:
    """Write one line to stderr."""
    sys.stderr.write(message + "\n")


def _load_votes(records):
    """Count votes across recordings, or report which file could not be used."""
    try:
        votes = table_mod.count_votes(records)
    except (OSError, ValueError) as exc:
        _err(f"cannot read recordings: {exc}")
        return None
    if not votes:
        _err("no records found")
        return None
    return votes


def _merge(args) -> int:
    """Merge per-rank recordings into a single table file."""
    votes = _load_votes(args.records)
    if votes is None:
        return 1
    merged = table_mod.merge_votes(votes)
    if args.arch and args.arch not in merged:
        _err(f"no records for architecture {args.arch!r}; recordings cover {sorted(merged)}")
        return 1
    if len(merged) > 1 and not args.arch:
        _err(f"records span several architectures {sorted(merged)}; pass --arch")
        return 1

    arch = args.arch or next(iter(merged))
    kernels = merged[arch]
    disagreements = [k for k in table_mod.disagreements(votes) if k[0] == arch]
    table_mod.write(arch, kernels, args.output, source=args.source)

    entries = sum(len(v) for v in kernels.values())
    _out(f"wrote {args.output}: {arch}, {len(kernels)} kernels, {entries} entries")
    if disagreements:
        _out(
            f"{len(disagreements)} entries had ranks disagreeing; the majority vote resolved "
            "them, and that disagreement is the variance this table removes"
        )
    return 0


def _report(args) -> int:
    """Summarize recordings without writing a table."""
    votes = _load_votes(args.records)
    if votes is None:
        return 1
    merged = table_mod.merge_votes(votes)
    disagreed = collections.Counter((a, k) for a, k, _ in table_mod.disagreements(votes))
    for arch, kernels in sorted(merged.items()):
        entries = sum(len(v) for v in kernels.values())
        _out(f"{arch}: {len(kernels)} kernels, {entries} entries")
        for kernel in sorted(kernels):
            count = disagreed[(arch, kernel)]
            note = f"  <- {count} disagreed across ranks" if count else ""
            _out(f"   {kernel}: {len(kernels[kernel])} entries{note}")
    return 0


def main(argv=None) -> int:
    """Entry point for ``python -m megatron.core.tuning``."""
    parser = argparse.ArgumentParser(prog="megatron.core.tuning")
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge", help="merge per-rank recordings into a table file")
    merge.add_argument("records", nargs="+")
    merge.add_argument("-o", "--output", required=True)
    merge.add_argument("--arch", help="architecture to write when records span several")
    merge.add_argument("--source", default="", help="note where the recording came from")
    merge.set_defaults(func=_merge)

    report = sub.add_parser("report", help="summarize recordings without writing a table")
    report.add_argument("records", nargs="+")
    report.set_defaults(func=_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
