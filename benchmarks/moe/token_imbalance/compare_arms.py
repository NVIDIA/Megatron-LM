# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare MoE-layer arms (dispatchers, load balancers, schedulers) on identical workloads.

Run with torchrun on one NVLink host. Every arm builds the same MoE layer (``arms.py``) from
the same seed, receives the same logical parameters, inputs, upstream gradients and replayed
routes, and is timed over the same warmup and measured iterations. Arms run in a rotated
order per workload, so no arm always runs first. This is a finite MoE-layer benchmark, not a
full-model training or convergence result. Timings exclude evidence copies and reductions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from functools import partialmethod
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import arms as A  # noqa: E402
import rank_load as RL  # noqa: E402
from arm_adapter import ArmAdapter, numerical_error  # noqa: E402
from benchmark import WORKLOADS, min_steps, percentile, routing_for_step  # noqa: E402

TOLERANCE = {"relative_l2": 0.01, "relative_peak": 0.05}


def digest(value):
    """sha256 of a tensor's bytes."""
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def spread(samples):
    """Median and spread of a list of times."""
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median_ms": percentile(ordered, 0.5),
        "p10_ms": percentile(ordered, 0.1),
        "p90_ms": percentile(ordered, 0.9),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def library_versions():
    """Versions that decide which expert and transport kernels run."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for package in ("transformer_engine", "deep_ep", "triton"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    try:
        import transformer_engine_torch as tex

        # TE's native grouped-tensor GEMM, which consumes capacity-padded expert inputs,
        # needs cuBLASLt >= 13.3.
        found["cublaslt"] = tex.get_cublasLt_version()
    except Exception:  # pragma: no cover - depends on the environment
        found["cublaslt"] = None
    found["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    return found


def parse_args(argv=None):
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(A.DEFAULT_ARMS))
    parser.add_argument("--reference-arm", default="hybridep")
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=512, help="per-rank source tokens")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--steps", type=int, default=6, help="routing steps per iteration")
    parser.add_argument("--warmup", type=int, default=2, help="unmeasured step-sequence passes")
    parser.add_argument("--iters", type=int, default=5, help="measured step-sequence passes")
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hot-maxvio", type=float, default=2.0)
    parser.add_argument("--schedule", choices=RL.SCHEDULES, default="static")
    parser.add_argument("--schedule-period", type=int, default=3)
    parser.add_argument("--order", choices=("rotate", "fixed"), default="rotate")
    parser.add_argument(
        "--te-op-fuser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use_transformer_engine_op_fuser for every arm; dev's MoonEP and #8500 need it",
    )
    parser.add_argument(
        "--scheduler-transport", choices=A.SCHEDULER_TRANSPORTS, default="replica_nccl"
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iters < 1 or args.learning_rate < 0:
        parser.error("warmup >= 0, iters >= 1 and a nonnegative learning rate are required")
    try:
        args.arm_list = A.resolve(args.arms)
    except ValueError as exc:
        parser.error(str(exc))
    args.workload_list = args.workloads.split(",")
    for workload in args.workload_list:
        if workload not in WORKLOADS:
            parser.error(f"unknown workload {workload}")
        if args.steps < min_steps(workload, args.schedule, args.schedule_period):
            parser.error(f"--steps {args.steps} does not exercise {workload}")
    return args


def main(argv=None):
    """Run every available arm on every workload and write per-rank JSONL plus a summary."""
    args = parse_args(argv)
    os.environ.setdefault("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    geometry = A.Geometry(
        ep=world,
        experts=args.experts,
        topk=args.topk,
        tokens=args.tokens,
        hidden=args.hidden,
        ffn=args.ffn,
    )
    geometry.validate()
    hosts = [None] * world
    torch.distributed.all_gather_object(hosts, socket.gethostname())
    if len(set(hosts)) != 1 or torch.cuda.device_count() < world:
        raise RuntimeError("requires one host exposing every EP GPU")
    if not all(
        torch.cuda.can_device_access_peer(i, j)
        for i in range(world)
        for j in range(world)
        if i != j
    ):
        raise RuntimeError("all EP GPU pairs must support peer access")
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    parallel_state.initialize_model_parallel(expert_model_parallel_size=world)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    probes = A.probe_requirements(te_op_fuser=args.te_op_fuser)
    table = A.availability_table(args.arm_list, probes)
    runnable = [arm for arm, row in zip(args.arm_list, table) if row["available"]]
    if probes[A.HYBRIDEP][0]:
        from megatron.core.transformer.moe.fused_a2a import HybridEPBuffer

        # Benchmark-only kernel-cache reuse, as in the upstream virtual-expert parity test.
        HybridEPBuffer.__init__ = partialmethod(HybridEPBuffer.__init__, load_cached_kernels=True)
    reference = args.reference_arm
    if reference not in {arm.name for arm in runnable}:
        reference = runnable[0].name if runnable else None
    args.out.mkdir(parents=True, exist_ok=True)
    route_kwargs = dict(
        hot_maxvio=args.hot_maxvio, schedule=args.schedule, period=args.schedule_period
    )
    run_info = {
        "geometry": geometry.__dict__,
        "arms": table,
        "reference_arm": reference,
        "warmup_passes": args.warmup,
        "measured_passes": args.iters,
        "steps": args.steps,
        "order": args.order,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "te_op_fuser": args.te_op_fuser,
        "scheduler_transport": args.scheduler_transport,
        "workloads": args.workload_list,
        "route_kwargs": route_kwargs,
        "world": world,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "versions": library_versions(),
    }
    if rank == 0:
        (args.out / "run.json").write_text(json.dumps(run_info, indent=2))
        print(json.dumps({"availability": table}, indent=2), flush=True)
    samples = {}
    failed = False
    path = args.out / f"rank{rank}.jsonl"
    try:
        with path.open("w") as stream:
            for index, workload in enumerate(args.workload_list):
                values, initial = {}, None
                for arm in A.run_order(runnable, index, args.order):
                    adapter = ArmAdapter(
                        geometry,
                        arm,
                        te_op_fuser=args.te_op_fuser,
                        scheduler_transport=args.scheduler_transport,
                    )
                    error = None
                    try:
                        layer = adapter.build(pg, args.seed)
                        parameters = dict(layer.named_parameters())
                        if initial is None:
                            initial = {k: v.detach().cpu().clone() for k, v in parameters.items()}
                        elif parameters.keys() != initial.keys():
                            raise AssertionError("logical parameter names differ between arms")
                        else:
                            with torch.no_grad():
                                for name, parameter in parameters.items():
                                    parameter.copy_(initial[name])
                        values[arm.name] = []
                        for iteration in range(-args.warmup, args.iters):
                            for step in range(args.steps):
                                ids, _ = routing_for_step(
                                    workload,
                                    geometry.experts,
                                    geometry.topk,
                                    geometry.tokens,
                                    step,
                                    args.seed,
                                    world,
                                    **route_kwargs,
                                )
                                rng = torch.Generator(device="cuda").manual_seed(
                                    args.seed + rank * 10000 + step
                                )
                                x = torch.randn(
                                    geometry.tokens,
                                    1,
                                    geometry.hidden,
                                    device="cuda",
                                    dtype=torch.bfloat16,
                                    generator=rng,
                                )
                                upstream = torch.randn(
                                    x.shape, device="cuda", dtype=x.dtype, generator=rng
                                )
                                collect = iteration == 0
                                result = adapter.run_workload(
                                    x,
                                    ids.cuda(),
                                    upstream,
                                    learning_rate=args.learning_rate if iteration >= 0 else 0.0,
                                    collect=collect,
                                )
                                if iteration < 0:
                                    continue
                                times = [None] * world
                                torch.distributed.all_gather_object(
                                    times, result["elapsed_ms"], group=pg.ep
                                )
                                counts = torch.bincount(
                                    ids.flatten(), minlength=geometry.experts
                                ).cuda()
                                torch.distributed.all_reduce(counts, group=pg.ep)
                                loads = counts.reshape(world, -1).sum(-1).cpu().tolist()
                                row = {
                                    "backend": arm.name,
                                    "family": arm.family,
                                    "workload": workload,
                                    "iteration": iteration,
                                    "step": step,
                                    "rank": rank,
                                    "mode": (
                                        "training_step" if args.learning_rate else "moe_fwd_bwd"
                                    ),
                                    "elapsed_ms_per_rank": times,
                                    "worst_rank_ms": max(times),
                                    "rank_time_max_over_mean": max(times) / (sum(times) / world),
                                    "logical_expert_counts": counts.cpu().tolist(),
                                    "logical_rank_loads": loads,
                                    "logical_rank_maxvio": RL.maxvio(loads),
                                    "seed": args.seed,
                                }
                                samples.setdefault((arm.name, workload), []).append(max(times))
                                if collect:
                                    values[arm.name].append(result["values"])
                                    row["input_sha256"] = digest(x.cpu())
                                    row["routes_sha256"] = digest(ids)
                                    row["value_sha256"] = {
                                        k: digest(v) for k, v in result["values"].items()
                                    }
                                    row["virtual_plan"] = result["plan"]
                                    row["capability"] = adapter.capability()
                                stream.write(json.dumps(row) + "\n")
                                stream.flush()
                    except Exception as exc:  # recorded, and the run exits nonzero
                        error = f"{type(exc).__name__}: {exc}"
                        values.pop(arm.name, None)
                        import traceback

                        traceback.print_exc()
                    finally:
                        adapter.close()
                    flag = torch.tensor(int(error is not None), device="cuda")
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                    if flag.item():
                        failed = True
                        stream.write(
                            json.dumps(
                                {
                                    "backend": arm.name,
                                    "workload": workload,
                                    "rank": rank,
                                    "status": "FAILED",
                                    "error": error or "failed on another rank",
                                }
                            )
                            + "\n"
                        )
                        samples.pop((arm.name, workload), None)
                # Numerics against the reference arm, from the first measured pass.
                if reference in values:
                    for name, per_step in values.items():
                        if name == reference:
                            continue
                        for step, (actual, expected) in enumerate(zip(per_step, values[reference])):
                            if actual.keys() != expected.keys():
                                errors = {"fields": "result fields differ"}
                                passed = False
                            else:
                                errors = {
                                    k: numerical_error(actual[k], expected[k]) for k in actual
                                }
                                passed = all(
                                    e["relative_l2"] <= TOLERANCE["relative_l2"]
                                    and e["relative_peak"] <= TOLERANCE["relative_peak"]
                                    for e in errors.values()
                                )
                            failed |= not passed
                            stream.write(
                                json.dumps(
                                    {
                                        "numerics": True,
                                        "backend": name,
                                        "reference": reference,
                                        "workload": workload,
                                        "step": step,
                                        "rank": rank,
                                        "errors": errors,
                                        "numerical_pass": passed,
                                        "tolerance": TOLERANCE,
                                    }
                                )
                                + "\n"
                            )
                del values, initial
        summary = {
            "run": run_info,
            "results": [
                {"backend": arm, "workload": workload, "worst_rank": spread(times)}
                for (arm, workload), times in sorted(samples.items())
            ],
        }
        if rank == 0:
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            for item in summary["results"]:
                s = item["worst_rank"]
                print(
                    f"{item['workload']:28s} {item['backend']:20s} n={s['n']:3d} "
                    f"median={s['median_ms']:.3f} p10={s['p10_ms']:.3f} p90={s['p90_ms']:.3f}",
                    flush=True,
                )
        flag = torch.tensor(int(failed), device="cuda")
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return int(flag.item())
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
