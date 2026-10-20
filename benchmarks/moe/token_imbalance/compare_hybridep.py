"""Compare existing HybridEP and virtual-expert execution on identical logical routes.

Run with torchrun on one NVLink host. This is a finite MoE layer benchmark, not a
full-model training or convergence result. Timings exclude evidence copies and reductions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark import TRACE_WORKLOADS, min_steps, routing_for_step
from hybridep_adapter import Geometry, HybridEPAdapter, numerical_error


def digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workloads", default=",".join(TRACE_WORKLOADS))
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    workloads = args.workloads.split(",")
    if args.warmup < 0 or args.learning_rate < 0:
        parser.error("warmup and learning rate must be nonnegative")
    for workload in workloads:
        if workload not in TRACE_WORKLOADS or args.steps < min_steps(workload):
            parser.error(f"invalid workload or insufficient steps: {workload}")
    os.environ.setdefault("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    geometry = Geometry(ep=world, tokens=args.tokens)
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
        raise RuntimeError(
            "all EP GPU pairs must support peer access; separately record NVLink topology"
        )
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    parallel_state.initialize_model_parallel(expert_model_parallel_size=world)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"rank{rank}.jsonl"
    failed = False
    try:
        with path.open("w") as stream:
            for workload in workloads:
                baseline = []
                initial = None
                for virtual in (False, True):
                    adapter = HybridEPAdapter(geometry, virtual)
                    layer = None
                    try:
                        layer = adapter.build(pg, args.seed)
                        if initial is None:
                            initial = {
                                name: value.detach().cpu().clone()
                                for name, value in layer.named_parameters()
                            }
                        else:
                            parameters = dict(layer.named_parameters())
                            if parameters.keys() != initial.keys():
                                raise AssertionError("logical parameter names differ")
                            with torch.no_grad():
                                for name, parameter in parameters.items():
                                    parameter.copy_(initial[name])
                        # Warmup has no updates; reset all accumulated gradients each call.
                        for iteration in range(-args.warmup, args.steps):
                            step = max(iteration, 0)
                            ids, _ = routing_for_step(
                                workload,
                                geometry.experts,
                                geometry.topk,
                                geometry.tokens,
                                step,
                                args.seed,
                                world,
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
                            result = adapter.run_workload(
                                x,
                                ids.cuda(),
                                upstream,
                                learning_rate=args.learning_rate if iteration >= 0 else 0.0,
                            )
                            if iteration < 0:
                                continue
                            times = [None] * world
                            torch.distributed.all_gather_object(
                                times, result["elapsed_ms"], group=pg.ep
                            )
                            local_counts = torch.bincount(ids.flatten(), minlength=geometry.experts)
                            counts = local_counts.cuda()
                            torch.distributed.all_reduce(counts, group=pg.ep)
                            logical_loads = counts.reshape(world, -1).sum(-1).cpu().tolist()
                            errors = {}
                            if virtual:
                                reference = baseline[step]
                                if result["values"].keys() != reference.keys():
                                    raise AssertionError("backend result fields differ")
                                errors = {
                                    name: numerical_error(value, reference[name])
                                    for name, value in result["values"].items()
                                }
                            else:
                                baseline.append(result["values"])
                            passed = all(
                                e["relative_l2"] <= 0.01 and e["relative_peak"] <= 0.05
                                for e in errors.values()
                            )
                            failed |= not passed
                            row = {
                                "backend": adapter.name,
                                "workload": workload,
                                "step": step,
                                "rank": rank,
                                "mode": "training_step" if args.learning_rate else "moe_fwd_bwd",
                                "elapsed_ms_per_rank": times,
                                "worst_rank_ms": max(times),
                                "rank_time_max_over_mean": max(times) / (sum(times) / world),
                                "logical_expert_counts": counts.cpu().tolist(),
                                "logical_rank_loads": logical_loads,
                                "logical_rank_max_over_mean": max(logical_loads)
                                / (sum(logical_loads) / world),
                                "input_sha256": digest(x.cpu()),
                                "routes_sha256": digest(ids),
                                "value_sha256": {k: digest(v) for k, v in result["values"].items()},
                                "virtual_plan": result["plan"],
                                "errors_vs_hybridep": errors,
                                "numerical_pass": passed,
                                "tolerance": {"relative_l2": 0.01, "relative_peak": 0.05},
                                "capability": adapter.capability(),
                                "seed": args.seed,
                            }
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                    finally:
                        del layer
                        adapter.close()
            flag = torch.tensor(int(failed), device="cuda")
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            return int(flag.item())
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
