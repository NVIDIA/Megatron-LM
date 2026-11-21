# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command-line interface: ``python -m tools.pipeline_layout_planner <command> ...``.

Commands:
    size             parameters, FLOPs and memory per unit and per stage
    baselines        Megatron's default split and the count-based split
    profile-layouts  two layouts whose timings identify every unit cost
    fit              fit unit costs from profiling logs
    calibrate        re-estimate the per-step overhead from timer-free logs
    plan             rank layouts by predicted step time
    score            predict one layout, optionally against a measured step time

The model shape comes from ``--shape`` (JSON of ``ModelShape`` fields) or ``--megatron-args``
(a file with the training command line).
"""

import argparse
import json
import sys
from dataclasses import asdict
from typing import List, Optional

from . import layout as layouts
from .cost_model import CostModel, calibrate_overhead, fit, profiling_run_from_summary
from .model import ModelShape, shape_from_megatron_args
from .planner import plan, profiling_layouts, score
from .timings import parse_log, summarize


def _load_shape(args: argparse.Namespace) -> ModelShape:
    if args.shape:
        return ModelShape.from_json(args.shape)
    with open(args.megatron_args) as handle:
        return shape_from_megatron_args(
            handle.read(), vocab_size=args.vocab_size, data_parallel_size=args.data_parallel_size
        )


def _load_cost_model(args: argparse.Namespace, shape: ModelShape) -> CostModel:
    if args.costs:
        return CostModel.from_json(args.costs)
    if args.tflops:
        return CostModel.from_flops(shape, args.tflops)
    raise SystemExit("Give --costs (fitted) or --tflops (FLOP-only estimate).")


def _layout_from_arg(value: str, shape: ModelShape, pp_size: int) -> layouts.Stages:
    if value == "default":
        return layouts.megatron_default(shape.num_layers, pp_size, shape.mtp_num_layers)
    return layouts.parse(value)


def _summaries(specs: List[str], warmup: int):
    for spec in specs:
        layout_str, path = spec.split("=", 1)
        with open(path) as handle:
            yield layout_str, summarize(parse_log(handle), warmup=warmup)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the planner CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.pipeline_layout_planner",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(cmd: argparse.ArgumentParser, needs_pp: bool = True) -> None:
        source = cmd.add_mutually_exclusive_group(required=True)
        source.add_argument("--shape", help="JSON file of ModelShape fields")
        source.add_argument("--megatron-args", help="file with the training command line")
        cmd.add_argument("--vocab-size", type=int, default=128256, help="padded vocabulary size")
        cmd.add_argument("--data-parallel-size", type=int, default=1)
        cmd.add_argument("--pp-size", type=int, required=needs_pp)

    def add_costs(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--costs", help="cost model JSON written by fit or calibrate")
        cmd.add_argument("--tflops", type=float, help="FLOP-only estimate at this TFLOP/s per GPU")
        cmd.add_argument("--hbm-gb", type=float, help="memory limit per GPU in GB")

    size = sub.add_parser("size", help="parameters, FLOPs and memory")
    add_common(size, needs_pp=False)
    size.add_argument("--layout", help="also report per-stage parameters and memory")
    add_common(sub.add_parser("baselines", help="default and count-based layouts"))
    add_common(sub.add_parser("profile-layouts", help="layouts to profile"))

    fit_cmd = sub.add_parser("fit", help="fit unit costs from profiling logs")
    add_common(fit_cmd, needs_pp=False)
    fit_cmd.add_argument("--run", action="append", required=True, help="LAYOUT=path/to/log")
    fit_cmd.add_argument("--warmup", type=int, default=5, help="leading iterations to skip")
    fit_cmd.add_argument("--ranks-per-stage", type=int, default=1)
    fit_cmd.add_argument("--p2p-ms", type=float, default=0.2)
    fit_cmd.add_argument("-o", "--output", required=True)

    cal = sub.add_parser("calibrate", help="overhead from timer-free baseline logs")
    add_common(cal)
    cal.add_argument("--costs", required=True)
    cal.add_argument("--run", action="append", required=True, help="LAYOUT=path/to/log")
    cal.add_argument("--warmup", type=int, default=5)
    cal.add_argument("-o", "--output", required=True)

    plan_cmd = sub.add_parser("plan", help="rank layouts")
    add_common(plan_cmd)
    add_costs(plan_cmd)
    plan_cmd.add_argument("--top", type=int, default=5)
    plan_cmd.add_argument(
        "--include-standalone-mtp",
        action="store_true",
        help="also rank layouts with MTP on a stage without the loss",
    )

    score_cmd = sub.add_parser("score", help="predict one layout")
    add_common(score_cmd)
    add_costs(score_cmd)
    score_cmd.add_argument("--layout", required=True, help='layout string or "default"')
    score_cmd.add_argument("--measured-ms", type=float, help="measured step time to compare")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Run the planner CLI and write JSON to stdout."""
    args = build_parser().parse_args(argv)
    shape = _load_shape(args)
    if args.command == "size":
        out = {
            "total_params_b": shape.total_params() / 1e9,
            "active_params_b": shape.active_params() / 1e9,
            "units": {
                unit: {
                    "params_m": shape.unit_params(unit) / 1e6,
                    "gflop_per_microbatch": [
                        f / 1e9 for f in shape.unit_flops_per_microbatch(unit)
                    ],
                }
                for unit in layouts.UNITS
                if unit != "m" or shape.mtp_num_layers
            },
        }
        if args.layout:
            stages = layouts.parse(args.layout)
            first = [sum(s.count("t") for s in stages[:i]) for i in range(len(stages))]
            out["stages"] = [
                {
                    "stage": layouts.to_string([stage]),
                    "params_b": shape.stage_params(stage, f) / 1e9,
                    "memory_gb": shape.stage_memory_gb(stage, f),
                }
                for stage, f in zip(stages, first)
            ]
    elif args.command == "baselines":
        out = {
            "count_based": layouts.to_string(
                layouts.count_based(shape.num_layers, args.pp_size, shape.mtp_num_layers)
            )
        }
        if shape.num_layers % args.pp_size == 0:
            out["megatron_default"] = layouts.to_string(
                layouts.megatron_default(shape.num_layers, args.pp_size, shape.mtp_num_layers)
            )
    elif args.command == "profile-layouts":
        out = {
            "profiling_layouts": profiling_layouts(
                shape.num_layers, args.pp_size, shape.mtp_num_layers
            )
        }
    elif args.command == "fit":
        runs = [
            profiling_run_from_summary(
                layouts.parse(layout_str), summary, shape.num_microbatches, args.ranks_per_stage
            )
            for layout_str, summary in _summaries(args.run, args.warmup)
        ]
        model = fit(shape, runs, p2p_ms=args.p2p_ms)
        with open(args.output, "w") as handle:
            handle.write(model.to_json())
        out = {
            "output": args.output,
            "forward_ms": model.forward_ms,
            "backward_ms": model.backward_ms,
            "report": model.fit_report,
        }
    elif args.command == "calibrate":
        model = CostModel.from_json(args.costs)
        timer_free = [
            (_layout_from_arg(layout_str, shape, args.pp_size), summary.step_ms)
            for layout_str, summary in _summaries(args.run, args.warmup)
        ]
        model = calibrate_overhead(shape, model, timer_free)
        with open(args.output, "w") as handle:
            handle.write(model.to_json())
        out = {"output": args.output, "overhead_ms": model.overhead_ms, "report": model.fit_report}
    elif args.command == "plan":
        result = plan(
            shape,
            _load_cost_model(args, shape),
            args.pp_size,
            hbm_gb=args.hbm_gb,
            top=args.top,
            exclude_standalone_mtp=not args.include_standalone_mtp,
        )
        out = asdict(result)
    else:  # score
        stages = _layout_from_arg(args.layout, shape, args.pp_size)
        layouts.validate(stages, args.pp_size, shape.num_layers, shape.mtp_num_layers)
        out = asdict(score(shape, _load_cost_model(args, shape), stages, args.hbm_gb))
        if args.measured_ms:
            out["measured_ms"] = args.measured_ms
            out["prediction_error"] = (out["step_ms"] - args.measured_ms) / args.measured_ms
    sys.stdout.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
