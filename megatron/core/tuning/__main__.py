# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command line for building and inspecting tuned tables.

    python -m megatron.core.tuning merge  rec.rank*.json -o tables/sm103.json
    python -m megatron.core.tuning report rec.rank*.json
"""

import argparse
import sys

from megatron.core.tuning import table as table_mod


def _merge(args) -> int:
    merged = table_mod.merge_records(args.records)
    if not merged:
        print("no records found", file=sys.stderr)
        return 1
    if len(merged) > 1 and not args.arch:
        print(f"records span several architectures {sorted(merged)}; pass --arch", file=sys.stderr)
        return 1
    arch = args.arch or next(iter(merged))
    kernels = merged[arch]
    disagreements = table_mod.disagreement_report(args.records)
    table_mod.write(arch, kernels, args.output, source=args.source)
    print(
        f"wrote {args.output}: {arch}, {len(kernels)} kernels, "
        f"{sum(len(v) for v in kernels.values())} entries"
    )
    if disagreements:
        print(
            f"{len(disagreements)} entries had ranks disagreeing; the majority vote "
            "resolved them, and that disagreement is the variance this table removes"
        )
    return 0


def _report(args) -> int:
    merged = table_mod.merge_records(args.records)
    disagreements = table_mod.disagreement_report(args.records)
    for arch, kernels in sorted(merged.items()):
        print(f"{arch}: {len(kernels)} kernels, {sum(len(v) for v in kernels.values())} entries")
        for kernel in sorted(kernels):
            marks = [k for (a, kn, k) in disagreements if a == arch and kn == kernel]
            note = (
                f"  <- {len(marks)} entr{'y' if len(marks) == 1 else 'ies'} disagreed"
                if marks
                else ""
            )
            print(f"   {kernel}: {len(kernels[kernel])} entries{note}")
    if not merged:
        print("no records found", file=sys.stderr)
        return 1
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
