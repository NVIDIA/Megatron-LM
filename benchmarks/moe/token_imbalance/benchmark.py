#!/usr/bin/env python3
"""MoE runtime token-imbalance benchmark (standard all-to-all baseline).

Scope (first slice of the roadmap #6757 item):
    routing workload -> existing standard all-to-all dispatcher -> local expert computation
    -> correctness oracle -> per-rank raw metrics, plus the rank-level load it produces

Two experiments, never mixed:
    trace_replay : a fixed LOGICAL routing (global expert ids, W* workloads) replayed at any
                   EP layout. The resulting rank-level imbalance is a property of the layout
                   and is reported, not controlled.
    rank_target  : a routing CONSTRUCTED for the given layout so that every EP rank receives
                   a chosen load exactly (R* workloads). Severity (maxvio), concentration and
                   temporal persistence are controlled one at a time.

Three measurement modes, never mixed:
    dispatch_only : dispatcher round trip for the workload, with a stand-in local expert.
                    Does NOT represent GEMM cost or a training step.
    moe_fwd_bwd   : real MoELayer forward + backward. The router runs, and its top-k is
                    replaced by the workload through RouterReplay, so the layer routes exactly
                    the workload. Does NOT represent full-model throughput or convergence.
    training_step : moe_fwd_bwd plus a real optimizer update.

Timing boundary:
    INCLUDED  dispatcher preprocess/dispatch/postprocess, local expert compute,
              combine preprocess/combine/postprocess; router, backward and the optimizer
              update in the model modes. Every step is timed on its own as well.
    EXCLUDED  all metric reduction, JSON writing, printing, warmup iterations.

Correctness oracle (exact integer metadata + exact round trip), at every step:
    * every token row has exactly K distinct experts
    * sum_e assignments[e] == T * K, and the global histogram matches T*K*world
    * dispatch -> identity expert -> combine restores every row to its origin
      (values are exact powers of two, so the gated sum is exact in bf16/fp32)
    * dispatcher recv splits agree with the independent group-global count
    * rank_target: the realized per-rank load equals the target exactly
    * model modes: the routing map the layer actually used equals the workload exactly

Usage:
    torchrun --standalone --nproc_per_node=N benchmark.py --mode dispatch_only --ep 2 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules

import rank_load as RL  # noqa: E402

# --------------------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------------------

# trace_replay: defined from GLOBAL expert ids only, identical at every EP layout.
TRACE_WORKLOADS = (
    "W0_balanced",
    "W1_hot_expert_set",
    "W3_spread_hot_experts",
    "W4_random_balanced",
    "W5_zero_expert_rank",
    "W6_rotating_hotspot",
    "W7_burst",
)

WORKLOADS = TRACE_WORKLOADS + RL.RANK_TARGET_WORKLOADS

WORKLOAD_NOTES = {
    "W0_balanced": "round-robin distinct top-k; every expert and destination rank equal",
    "W1_hot_expert_set": "a contiguous hot block of E/4 experts takes 3/4 of the rows",
    "W3_spread_hot_experts": "W1 with its experts relabeled so the hot block is strided across "
    "the expert range: identical per-expert heat, different placement",
    "W4_random_balanced": "uniform random distinct top-k: balanced in expectation, with the "
    "sampling fluctuation W0's exact round-robin does not have",
    "W5_zero_expert_rank": "half the experts never selected; owner ranks receive nothing",
    "W6_rotating_hotspot": "W1's hot block moves by its own width every 3 steps",
    "W7_burst": "3 steady steps, 1 burst step, then recovery (period 4)",
    **RL.RANK_TARGET_NOTES,
}

# Smallest --steps at which a per-step trace workload is actually exercised.
_TRACE_MIN_STEPS = {"W6_rotating_hotspot": 4, "W7_burst": 5}


def is_rank_target(workload: str) -> bool:
    return workload in RL.RANK_TARGET_WORKLOADS


def min_steps(workload: str, schedule: str = "static", period: int = 3) -> int:
    if is_rank_target(workload):
        return RL.min_steps_for_schedule(schedule, period)
    return _TRACE_MIN_STEPS.get(workload, 1)


def _hot_width(num_experts: int, topk: int) -> int:
    return max(topk, num_experts // 4)


def _spread_relabel(num_experts: int, width: int) -> list[int]:
    """Bijection on expert ids sending the block [0, width) to a stride across the range.

    Applied to W1's ids it keeps every per-expert count and moves only the placement.
    """
    stride = max(1, num_experts // width)
    hot_targets = [i * stride for i in range(width)]
    rest = [e for e in range(num_experts) if e not in set(hot_targets)]
    return hot_targets + rest


def _distinct_topk(pool: list[int], k: int, g: torch.Generator) -> list[int]:
    if k > len(pool):
        raise ValueError(f"topk {k} exceeds candidate pool {len(pool)}")
    idx = torch.randperm(len(pool), generator=g)[:k].tolist()
    return sorted(pool[i] for i in idx)


def _generator(*parts: Any) -> torch.Generator:
    key = "|".join(str(p) for p in parts)
    return torch.Generator(device="cpu").manual_seed(
        int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    )


def routing_for_step(
    workload: str, num_experts: int, topk: int, tokens: int, step: int, seed: int, ep_size: int,
    *, hot_maxvio: float = 2.0, schedule: str = "static", period: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(expert_ids [tokens, topk] int64, gates [tokens, topk] float32), deterministic.

    trace_replay (W*) workloads are seeded by (workload, seed, step) alone and do not depend on
    `ep_size`, so one logical workload is comparable across execution methods and layouts;
    `test_workload.py` asserts that. rank_target (R*) workloads are built FOR `ep_size`, so
    they depend on it by design.
    """
    if is_rank_target(workload):
        return _rank_target_routing(workload, num_experts, topk, tokens, step, seed, ep_size,
                                    hot_maxvio, schedule, period)
    if workload not in TRACE_WORKLOADS:
        raise ValueError(f"unknown workload {workload}")

    # W3 reuses W1's random stream: it is W1 relabeled, not an independent draw.
    stream = "W1_hot_expert_set" if workload == "W3_spread_hot_experts" else workload
    g = _generator(stream, seed, step)
    everyone = list(range(num_experts))
    width = _hot_width(num_experts, topk)
    ids = torch.empty((tokens, topk), dtype=torch.int64)

    if workload == "W0_balanced":
        for t in range(tokens):
            start = (t * topk) % num_experts
            ids[t] = torch.tensor([(start + j) % num_experts for j in range(topk)])
    elif workload in ("W1_hot_expert_set", "W3_spread_hot_experts"):
        hot = list(range(width))
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(hot if t % 4 else everyone, topk, g))
        if workload == "W3_spread_hot_experts":
            relabel = torch.tensor(_spread_relabel(num_experts, width), dtype=torch.int64)
            ids = relabel[ids].sort(dim=1).values
    elif workload == "W4_random_balanced":
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(everyone, topk, g))
    elif workload == "W5_zero_expert_rank":
        live = list(range(num_experts // 2, num_experts)) or everyone
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(live, topk, g))
    elif workload == "W6_rotating_hotspot":
        base = ((step // 3) * width) % num_experts
        hot = [(base + i) % num_experts for i in range(width)]
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(hot if t % 4 else everyone, topk, g))
    elif workload == "W7_burst":
        burst = (step % 4) == 3
        hot = list(range(width))
        for t in range(tokens):
            ids[t] = torch.tensor(
                _distinct_topk(hot if (burst and t % 2 == 0) else everyone, topk, g)
            )

    _check_distinct(ids, topk)
    return ids, _gates(tokens, topk, g)


def target_loads_for_step(workload: str, ep_size: int, total: int, step: int, seed: int, *,
                          hot_maxvio: float, schedule: str, period: int) -> list[int]:
    """Per-rank target load at `step` for a rank_target workload (sums to `total`)."""
    base = RL.target_rank_loads(workload, ep_size, total, hot_maxvio)
    perm = RL.schedule_permutation(schedule, period, ep_size, step, seed)
    loads = [0] * ep_size
    for slot, rank in enumerate(perm):
        loads[rank] = base[slot]
    return loads


def _rank_target_routing(workload, num_experts, topk, tokens, step, seed, ep_size, hot_maxvio,
                         schedule, period):
    loads = target_loads_for_step(workload, ep_size, tokens * topk, step, seed,
                                  hot_maxvio=hot_maxvio, schedule=schedule, period=period)
    counts = RL.expert_counts_for_rank_loads(loads, num_experts, tokens, topk)
    g = _generator(workload, seed, step, ep_size, hot_maxvio, schedule, period)
    ids = RL.routing_from_expert_counts(counts, tokens, topk, g)
    _check_distinct(ids, topk)
    return ids, _gates(tokens, topk, g)


def _check_distinct(ids: torch.Tensor, topk: int) -> None:
    if topk > 1:
        for t in range(ids.shape[0]):
            assert len(set(ids[t].tolist())) == topk, "top-k slots must be distinct experts"


def _gates(tokens: int, topk: int, g: torch.Generator) -> torch.Tensor:
    gates = torch.rand((tokens, topk), generator=g, dtype=torch.float32) * 0.5 + 0.5
    return gates / gates.sum(dim=1, keepdim=True)


def probs_matrix_from_gates(ids: torch.Tensor, gates: torch.Tensor, num_experts: int,
                           dtype: torch.dtype) -> torch.Tensor:
    """[T, K] gates -> [T, E] masked probabilities; unrouted entries are exactly 0.

    This is the probability layout the all-to-all dispatcher consumes (`permute` indexes
    a flattened [E, T] view of it), and it matches what Router returns upstream.
    """
    full = torch.zeros((ids.shape[0], num_experts), dtype=dtype)
    full.scatter_(1, ids, gates.to(dtype))
    return full


def routing_map_from_ids(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    m = torch.zeros((ids.shape[0], num_experts), dtype=torch.bool)
    m.scatter_(1, ids, True)
    return m


def workload_fingerprint(
    workload: str, num_experts: int, topk: int, tokens: int, steps: int, seed: int, ep_size: int,
    *, hot_maxvio: float = 2.0, schedule: str = "static", period: int = 3,
) -> str:
    h = hashlib.sha256()
    h.update(f"{workload}|{num_experts}|{topk}|{tokens}|{steps}|{seed}|{ep_size}".encode())
    if is_rank_target(workload):
        h.update(f"|{hot_maxvio}|{schedule}|{period}".encode())
    for step in range(steps):
        ids, gates = routing_for_step(workload, num_experts, topk, tokens, step, seed, ep_size,
                                      hot_maxvio=hot_maxvio, schedule=schedule, period=period)
        h.update(ids.numpy().tobytes())
        h.update(gates.numpy().tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def imbalance_metrics(counts: list[int]) -> dict[str, Any]:
    """max/mean and population CV; explicit null (never a fake epsilon) on zero load."""
    total = sum(counts)
    n = len(counts)
    if n == 0:
        return {"max_over_mean": None, "cv": None, "reason": "empty"}
    if total == 0:
        return {"max_over_mean": None, "cv": None, "reason": "zero_assignments"}
    mean = total / n
    pstd = math.sqrt(sum((c - mean) ** 2 for c in counts) / n)
    return {"max_over_mean": max(counts) / mean, "cv": pstd / mean,
            "std_definition": "population", "reason": None}


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        raise ValueError("empty sample")
    idx = (len(sorted_vals) - 1) * p
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - idx) + sorted_vals[hi] * (idx - lo)


def rank_level_summary(per_step_expert_counts: list[list[int]], ep_size: int) -> dict[str, Any]:
    """Destination rank loads per step, their maxvio/concentration, and hotspot persistence."""
    per_step_loads = [RL.rank_loads(c, ep_size) for c in per_step_expert_counts]
    maxvios = [RL.maxvio(loads) for loads in per_step_loads]
    concs = [RL.concentration(loads) for loads in per_step_loads]
    valid = [i for i, m in enumerate(maxvios) if m is not None]
    worst = max(valid, key=lambda i: maxvios[i]) if valid else None
    return {
        "rank_loads_per_step": per_step_loads,
        "maxvio_per_step": maxvios,
        "concentration_per_step": concs,
        "maxvio_max": maxvios[worst] if worst is not None else None,
        "concentration_at_maxvio_max": concs[worst] if worst is not None else None,
        "hot_rank_persistence": RL.hot_rank_persistence(per_step_loads),
    }


@dataclass
class ArmResult:
    workload_id: str
    workload_checksum: str
    experiment: str
    mode: str
    backend: str
    base_sha: str
    ep: int
    etp: int
    tp: int
    world_size: int
    rank: int
    num_experts: int
    topk: int
    hidden_size: int
    logical_tokens: int
    assignments: int
    steps: int
    measured_iters: int
    warmup_iters: int
    hot_maxvio: float | None = None
    schedule: str | None = None
    schedule_period: int | None = None
    median_ms: float | None = None
    mean_ms: float | None = None
    p95_ms: float | None = None
    min_ms: float | None = None
    per_step_median_ms: list[float] = field(default_factory=list)
    logical_tokens_per_second: float | None = None
    assignments_per_second: float | None = None
    expert_imbalance: float | None = None
    expert_cv: float | None = None
    expert_counts: list[int] = field(default_factory=list)
    expert_counts_per_step: list[list[int]] = field(default_factory=list)
    rank_loads_per_step: list[list[int]] = field(default_factory=list)
    target_rank_loads_per_step: list[list[int]] = field(default_factory=list)
    maxvio_per_step: list[float | None] = field(default_factory=list)
    concentration_per_step: list[float | None] = field(default_factory=list)
    maxvio_max: float | None = None
    concentration_at_maxvio_max: float | None = None
    hot_rank_persistence: float | None = None
    recv_splits_per_source_rank: list[int] = field(default_factory=list)
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    correctness: dict[str, Any] = field(default_factory=dict)
    status: str = "NOT_RUN"
    error: str | None = None


# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dispatch_only", "moe_fwd_bwd", "training_step"),
                    default="dispatch_only")
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--etp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--ffn-hidden-size", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=128, help="per-rank source tokens")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="override the permute geometry precondition (default: topk)")
    ap.add_argument("--steps", type=int, default=1,
                    help="routing steps per measured iteration; per-step workloads and "
                         "non-static schedules refuse a value too small to exercise them")
    ap.add_argument("--hot-maxvio", type=float, default=2.0,
                    help="R1/R2: busiest-rank load as a multiple of the mean rank load")
    ap.add_argument("--schedule", choices=RL.SCHEDULES, default="static",
                    help="R*: which rank carries the hot load at each step; the per-step load "
                         "multiset is the same at every step")
    ap.add_argument("--schedule-period", type=int, default=3,
                    help="R* with --schedule rotate: steps between hotspot moves")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-sha", default="UNKNOWN")
    ap.add_argument("--run-id", default="unknown")
    ap.add_argument("--repo-root", default=None,
                    help="Megatron-LM checkout to import (default: parent of this file)")
    ap.add_argument("--write-trace", default=None,
                    help="directory to persist a trace-v1 record per workload step into")
    ap.add_argument("--instrument", choices=("none", "counters"), default="none",
                    help="extra always-on instrumentation, to measure its overhead")
    ap.add_argument("--capability-json", default=None,
                    help="write the backend capability probe result to this path")
    args = ap.parse_args()

    geom_min = args.min_tokens or args.topk
    if args.tokens < geom_min:
        print(f"ERROR: --tokens {args.tokens} < required geometry minimum {geom_min} "
              f"(the argsort permute needs num_experts*num_tokens >= tokens*topk slots)",
              file=sys.stderr)
        return 2
    workloads = args.workloads.split(",")
    for workload in workloads:
        if workload not in WORKLOADS:
            print(f"ERROR: unknown workload {workload}", file=sys.stderr)
            return 2
        need = min_steps(workload, args.schedule, args.schedule_period)
        if args.steps < need:
            print(f"ERROR: {workload} (schedule {args.schedule}) needs --steps >= {need} to be "
                  f"exercised; with --steps {args.steps} every iteration would replay only "
                  f"the first {args.steps} step(s)", file=sys.stderr)
            return 2

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if args.ep * args.etp * args.tp != world:
        if rank == 0:
            print(f"ERROR: ep {args.ep} * etp {args.etp} * tp {args.tp} != world {world}",
                  file=sys.stderr)
        torch.distributed.destroy_process_group()
        return 2

    repo_root = Path(args.repo_root).resolve() if args.repo_root else \
        Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from backend_adapter import StandardAlltoAllAdapter, report_capability

    adapter = StandardAlltoAllAdapter()
    cap = adapter.capability()
    if rank == 0:
        print("[B1] backend capability probe:\n" + report_capability(cap), flush=True)
        if args.capability_json:
            Path(args.capability_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.capability_json).write_text(
                json.dumps(cap.as_dict(), indent=2), encoding="utf-8"
            )

    from megatron.core import parallel_state
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        context_parallel_size=1,
        expert_tensor_parallel_size=args.etp,
    )

    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    ep_rank = parallel_state.get_expert_model_parallel_rank()
    if args.num_experts % args.ep != 0:
        raise SystemExit(
            f"num_experts {args.num_experts} must be divisible by ep {args.ep} "
            f"(local expert count must be an integer)"
        )
    num_local_experts = args.num_experts // args.ep
    if num_local_experts < 1:
        raise SystemExit(f"ep {args.ep} exceeds num_experts {args.num_experts}")
    local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]
    model_mode = args.mode in ("moe_fwd_bwd", "training_step")

    config = TransformerConfig(
        tensor_model_parallel_size=args.tp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        pipeline_model_parallel_size=1,
        num_layers=1,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=8,
        num_moe_experts=args.num_experts,
        moe_router_topk=args.topk,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,  # SequentialMLP experts: no TE dependency
        moe_router_dtype="fp32",
        # The model modes replace the router's top-k with the workload, so the layer routes
        # exactly what the workload says; the router itself still runs.
        moe_enable_routing_replay=model_mode,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        use_cpu_initialization=True,
        bf16=True,
    )

    dispatcher = MoEAlltoAllTokenDispatcher(
        num_local_experts=num_local_experts,
        local_expert_indices=local_expert_indices,
        config=config,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )

    dev = torch.device("cuda", local_rank)
    # Expert weights are indexed by GLOBAL expert id: draw the full [E, ...] tables with one
    # fixed seed on every rank, then slice out the experts this rank owns. Deriving them from
    # `seed + rank` would silently change the model whenever the EP layout changes, which
    # would break the "same logical workload, different execution method" contract.
    torch.manual_seed(args.seed)
    full_W1 = torch.randn(args.num_experts, args.hidden_size, 2 * args.ffn_hidden_size,
                          dtype=torch.float32) * 0.02
    full_W2 = torch.randn(args.num_experts, args.ffn_hidden_size, args.hidden_size,
                          dtype=torch.float32) * 0.02
    lo, hi = local_expert_indices[0], local_expert_indices[-1] + 1
    assert local_expert_indices == list(range(lo, hi)), "local experts must be contiguous"
    W1 = full_W1[lo:hi].to(device=dev, dtype=torch.bfloat16).contiguous()
    W2 = full_W2[lo:hi].to(device=dev, dtype=torch.bfloat16).contiguous()

    def expert_compute(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """SwiGLU expert slices; per-expert loop (no grouped GEMM in dispatch_only)."""
        out = torch.zeros_like(x)
        offset = 0
        for e in range(num_local_experts):
            n = int(counts[e].item())
            if n == 0:
                continue
            gu = x[offset:offset + n] @ W1[e]
            gate, up = gu.chunk(2, dim=-1)
            out[offset:offset + n] = (torch.nn.functional.silu(gate) * up) @ W2[e]
            offset += n
        return out

    # ---- real MoELayer, built lazily for the model-level modes -------------------------
    _moe_layer = None
    captured: dict[str, torch.Tensor] = {}

    def moe_layer():
        nonlocal _moe_layer
        if _moe_layer is None:
            from megatron.core.models.gpt.gpt_layer_specs import (
                get_gpt_layer_local_submodules,
            )
            from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
            from megatron.core.transformer.spec_utils import get_submodules

            spec = get_gpt_layer_local_submodules(
                num_experts=args.num_experts, moe_grouped_gemm=False
            ).mlp
            subs = get_submodules(spec)
            if not isinstance(subs, MoESubmodules):
                raise RuntimeError(f"unexpected MoE submodule spec {type(subs)}")
            RouterReplay.clear_global_router_replay_instances()
            torch.manual_seed(args.seed)  # identical logical weights on every rank
            _moe_layer = MoELayer(config, subs).cuda().to(dtype=torch.bfloat16)
            _moe_layer.set_layer_number(0)
            _moe_layer.train()
            if len(RouterReplay.global_router_replay_instances) != 1:
                raise RuntimeError("expected exactly one RouterReplay instance for one layer")

            def keep_routing_map(_module, _inputs, output):
                captured["routing_map"] = output[1].detach()

            _moe_layer.router.register_forward_hook(keep_routing_map)
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        return _moe_layer

    def token_permutation(hidden: torch.Tensor, probs: torch.Tensor, rmap: torch.Tensor):
        h, p = dispatcher.dispatch_preprocess(hidden, rmap, probs)
        h, p = dispatcher.token_dispatch(h, p)
        h, counts, p = dispatcher.dispatch_postprocess(h, p)
        return h, counts, p

    def token_unpermutation(hidden: torch.Tensor) -> torch.Tensor:
        h = dispatcher.combine_preprocess(hidden)
        h = dispatcher.token_combine(h)
        return dispatcher.combine_postprocess(h)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"rank{rank}.jsonl"

    def emit(rec: ArmResult) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def sync() -> None:
        torch.cuda.synchronize()
        torch.distributed.barrier()

    # Optional always-on instrumentation. Its overhead is measured by running the same
    # workload with --instrument none and --instrument counters and differencing the result.
    counters = {"permute_calls": 0, "expert_calls": 0, "combine_calls": 0}

    def count(key: str) -> None:
        if args.instrument == "counters":
            counters[key] += 1

    route_kwargs = dict(hot_maxvio=args.hot_maxvio, schedule=args.schedule,
                        period=args.schedule_period)

    def run_arm(rec: ArmResult, workload: str, checksum: str) -> None:
        steps = args.steps
        per_step = [
            routing_for_step(workload, args.num_experts, args.topk, args.tokens, s,
                             args.seed, args.ep, **route_kwargs)
            for s in range(steps)
        ]
        maps_cpu = [routing_map_from_ids(ids, args.num_experts) for ids, _ in per_step]
        # the dispatcher consumes the routing map and the [T, E] probs on the device
        maps = [m.cuda() for m in maps_cpu]
        ids_dev = [ids.cuda() for ids, _ in per_step]

        # ---- optional trace-v1 persistence (rank 0 only: the workload is global) ----
        corr: dict[str, Any] = {}
        if args.write_trace and rank == 0:
            try:
                from trace import save_trace, trace_digest
                digests = []
                for s, (ids_s, gates_s) in enumerate(per_step):
                    tdir = Path(args.write_trace) / workload / f"step{s:04d}"
                    src = torch.zeros(ids_s.shape[0], dtype=torch.int32)
                    save_trace(
                        tdir, ids_s, gates_s, src, num_experts=args.num_experts,
                        topk=args.topk, hidden_size=args.hidden_size, world_size=world,
                        seed=args.seed, workload_id=workload, workload_checksum=checksum,
                        producer_commit=args.base_sha,
                    )
                    digests.append(trace_digest(tdir))
                corr["trace_dir"] = str(Path(args.write_trace) / workload)
                corr["trace_digest_per_step"] = digests
                print(f"[B1] wrote {steps} trace step(s) for {workload}", flush=True)
            except Exception as exc:
                raise RuntimeError(f"trace persistence failed for {workload}: {exc}") from exc
        probs_dev = [
            probs_matrix_from_gates(ids, g, args.num_experts, torch.bfloat16).cuda()
            for ids, g in per_step
        ]

        # ---------------- correctness oracle (outside the timed window) ----------------
        corr["distinct_topk_ok"] = all(
            len(set(ids[t].tolist())) == args.topk
            for ids, _ in per_step for t in range(args.tokens)
        )
        corr["id_range_ok"] = all(
            bool(ids.min().item() >= 0 and ids.max().item() < args.num_experts)
            for ids, _ in per_step
        )
        corr["assignments_match_T_times_k"] = all(
            int(m.sum().item()) == args.tokens * args.topk for m in maps_cpu
        )
        # One all-reduce for every step's histogram.
        local_hist = torch.stack([m.sum(dim=0).to(torch.int64) for m in maps_cpu]).cuda()
        global_hist = local_hist.clone()
        torch.distributed.all_reduce(global_hist)
        global_counts = [[int(x) for x in row] for row in global_hist.cpu().tolist()]
        corr["global_histogram_exact"] = all(
            sum(row) == args.tokens * args.topk * world for row in global_counts
        )
        corr["local_expert_counts"] = [int(x) for x in local_hist[0].cpu().tolist()]
        corr["global_expert_counts"] = global_counts[0]

        level = rank_level_summary(global_counts, args.ep)
        if is_rank_target(workload):
            # Every source rank sends the same constructed block, so the global target is
            # world x the per-source target.
            targets = [
                [world * c for c in target_loads_for_step(
                    workload, args.ep, args.tokens * args.topk, s, args.seed,
                    hot_maxvio=args.hot_maxvio, schedule=args.schedule,
                    period=args.schedule_period)]
                for s in range(steps)
            ]
            rec.target_rank_loads_per_step = targets
            corr["rank_target_exact"] = targets == level["rank_loads_per_step"]

        # dispatch -> identity expert -> combine must restore the input exactly.
        #
        # Construction: every element is an exact multiple of 2**-10 that a bf16 holds
        # exactly, and every row sums to exactly `topk`. With uniform 1/topk gating,
        # combine returns sum_k (row / topk) = row, i.e. an EXACT equality.
        # `rank` shifts the pattern so a rank cannot pass on another rank's data.
        # The dispatcher's combine step SUMS the top-k expert contributions; the routing
        # probabilities are applied inside the expert module upstream, not by the dispatcher.
        # This harness therefore feeds raw expert outputs to combine and the round trip
        # returns a constant multiple of the input. That multiple is measured once here
        # (on the slow path, outside the timed window) and then required to be exact.
        assert (args.topk & (args.topk - 1)) == 0, "oracle prefers a power-of-two topk"
        ones_full = [
            probs_matrix_from_gates(ids, torch.full_like(g, 1.0 / args.topk), args.num_experts,
                                    torch.bfloat16).cuda()
            for ids, g in per_step
        ]
        probe_x = torch.full((args.tokens, args.hidden_size), 0.5, device=dev,
                             dtype=torch.bfloat16)
        ph, _c, _p = token_permutation(probe_x, ones_full[0], maps[0])
        factor = float(token_unpermutation(ph)[0, 0].float().item()) / 0.5
        corr["combine_factor"] = factor
        corr["combine_factor_is_topk"] = abs(factor - args.topk) < 1e-9
        assert corr["combine_factor_is_topk"], (
            f"combine factor {factor} != topk {args.topk}: the dispatcher's scaling contract "
            f"changed, refusing to score this arm"
        )

        # Sentinel row on an integer grid of 2**-10: every |value| <= 255 * 2**-10 is an
        # exact bf16 significand, the row sums to exactly `topk + rank` grid units, and the
        # sign pattern is a rank-dependent zero-sum permutation.
        k = args.topk + rank
        h = args.hidden_size
        assert h % 2 == 0, "round-trip oracle needs an even hidden size"
        unit = 1024
        target = k * unit                                    # integer row sum
        amp = 42                                             # balanced +-amp sums to zero
        signs = torch.empty(h, dtype=torch.int64)
        signs[0::2] = amp
        signs[1::2] = -amp
        assert int(signs.sum().item()) == 0
        signs = signs.roll(rank * 2)                          # rank-dependent permutation
        lift = target // h
        assert lift * h == target, f"row sum {k} must divide evenly over {h} columns"
        ints = signs + lift
        assert int(ints.sum().item()) == target, "integer row sum mismatch"
        assert int(ints.abs().max().item()) < 255, "value leaves the bf16-exact grid"
        hidden = (ints.float() / unit).to(torch.bfloat16)
        hidden = hidden.unsqueeze(0).repeat(args.tokens, 1).contiguous().cuda()
        expected_roundtrip = hidden * torch.tensor(factor, device=dev, dtype=torch.bfloat16)

        roundtrip_ok, recv_ok, mismatch_rows = True, True, 0
        recv: list[int] = []
        for s in range(steps):
            permuted, counts, _pp = token_permutation(hidden, ones_full[s], maps[s])
            restored = token_unpermutation(permuted)
            if not torch.equal(restored, expected_roundtrip):
                roundtrip_ok = False
                mismatch_rows += int((restored != expected_roundtrip).any(dim=1).sum().item())
            if s == 0:
                corr["counts_len_ok"] = len(counts) == num_local_experts
                corr["local_counts_sum_ok"] = (
                    int(counts.sum().item()) <= args.tokens * args.topk * world
                )
            # `output_splits` is per SOURCE rank for this destination only, in assignment rows
            # (the dispatch unit is one token-expert pair). A single rank's row cannot
            # reconcile the group total, so sum over all ranks and require the exact global
            # assignment count.
            osplit = dispatcher.output_splits
            recv_s = [int(x) for x in osplit] if osplit is not None else []
            if s == 0:
                recv = recv_s
            if recv_s:
                total_recv = torch.tensor(sum(recv_s), dtype=torch.int64, device=dev)
                torch.distributed.all_reduce(total_recv)
                recv_ok = recv_ok and int(total_recv.item()) == sum(global_counts[s])
        corr["roundtrip_exact"] = roundtrip_ok
        if not roundtrip_ok:
            corr["roundtrip_mismatch_rows"] = mismatch_rows
        corr["recv_splits_assignments_ok"] = recv_ok if recv else None
        corr["global_tokens"] = args.tokens * world
        corr["global_assignments_per_step"] = [sum(row) for row in global_counts]

        # ---------------- model-mode inputs ----------------
        # Real (non-sentinel) activations. No padding: the workload decides every row's
        # experts, and a padded row would silently drop assignments from it.
        g_in = torch.Generator(device="cpu").manual_seed(args.seed + 991 + rank)
        real_input = torch.randn(args.tokens, args.hidden_size, generator=g_in,
                                 dtype=torch.float32).to(device=dev, dtype=torch.bfloat16)
        step_holder: dict[str, Any] = {}

        def model_step(s: int) -> None:
            layer = moe_layer()
            RouterReplay.set_replay_data([ids_dev[s]])
            if args.mode == "training_step":
                opt = step_holder.get("opt")
                if opt is None:
                    opt = torch.optim.SGD(layer.parameters(), lr=0.01)
                    step_holder["opt"] = opt
                opt.zero_grad(set_to_none=True)
            else:
                layer.zero_grad(set_to_none=True)
            from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
            MoEAuxLossAutoScaler.main_loss_backward_scale = None
            out = layer(real_input.unsqueeze(1))
            hs = out[0] if isinstance(out, tuple) else out
            loss = hs.float().pow(2).mean()
            loss.backward()
            if args.mode == "training_step":
                torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
                step_holder["opt"].step()
            step_holder["last_loss"] = float(loss.detach())

        def dispatch_step(s: int) -> None:
            count("permute_calls")
            ph, cnt, _pp = token_permutation(hidden, probs_dev[s], maps[s])
            count("expert_calls")
            eh = expert_compute(ph, cnt)
            # probs are applied INSIDE the expert module upstream
            # (experts.py: intermediate_parallel * permuted_probs), not by the
            # dispatcher, so combine must receive the raw expert output.
            count("combine_calls")
            _ = token_unpermutation(eh)

        run_step = model_step if model_mode else dispatch_step

        if model_mode:
            # The layer must have routed exactly the workload, at every step.
            exact = True
            for s in range(steps):
                captured.pop("routing_map", None)
                model_step(s)
                got = captured.get("routing_map")
                exact = exact and got is not None and torch.equal(got.bool().cpu(), maps_cpu[s])
            corr["replayed_routing_exact"] = exact

        # ---------------- timing ----------------
        sync()
        for _ in range(args.warmup):
            for s in range(steps):
                run_step(s)
        sync()

        times: list[float] = []
        step_times: list[list[float]] = [[] for _ in range(steps)]
        torch.cuda.reset_peak_memory_stats(dev)
        for _ in range(args.iters):
            events = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)]
            torch.distributed.barrier()
            events[0].record()
            for s in range(steps):
                run_step(s)
                events[s + 1].record()
            torch.cuda.synchronize()
            times.append(events[0].elapsed_time(events[-1]))
            for s in range(steps):
                step_times[s].append(events[s].elapsed_time(events[s + 1]))
        sync()

        rec.median_ms = percentile(sorted(times), 0.5)
        rec.mean_ms = sum(times) / len(times)
        rec.p95_ms = percentile(sorted(times), 0.95)
        rec.min_ms = min(times)
        rec.per_step_median_ms = [percentile(sorted(t), 0.5) for t in step_times]
        secs = rec.median_ms / 1000.0
        rec.logical_tokens_per_second = (args.tokens * steps) / secs
        rec.assignments_per_second = (args.tokens * args.topk * steps) / secs
        rec.peak_allocated_bytes = int(torch.cuda.max_memory_allocated(dev))
        rec.peak_reserved_bytes = int(torch.cuda.max_memory_reserved(dev))
        rec.expert_counts = global_counts[0]
        rec.expert_counts_per_step = global_counts
        rec.recv_splits_per_source_rank = recv
        im = imbalance_metrics(rec.expert_counts)
        rec.expert_imbalance = im["max_over_mean"]
        rec.expert_cv = im["cv"]
        for key, value in level.items():
            setattr(rec, key, value)

        if model_mode:
            last_loss = step_holder.get("last_loss")
            corr["model_step_ran"] = last_loss is not None
            corr["model_step_loss_finite"] = (
                last_loss is not None and math.isfinite(last_loss) and abs(last_loss) < 1e6
            )
            corr["model_step_loss"] = last_loss
            gsum = sum(
                float(p.grad.abs().sum()) for p in _moe_layer.parameters() if p.grad is not None
            ) if _moe_layer is not None else 0.0
            corr["model_step_grad_abs_sum"] = gsum
            corr["model_step_grads_flow"] = gsum > 0
        if args.instrument != "none":
            corr["instrumentation_counters"] = dict(counters)
        rec.correctness = corr
        hard_keys = [
            "distinct_topk_ok", "id_range_ok", "assignments_match_T_times_k",
            "global_histogram_exact", "roundtrip_exact", "counts_len_ok",
            "local_counts_sum_ok", "combine_factor_is_topk",
        ]
        if is_rank_target(workload):
            hard_keys.append("rank_target_exact")
        if model_mode:
            hard_keys += [
                "replayed_routing_exact", "model_step_ran", "model_step_loss_finite",
                "model_step_grads_flow",
            ]
        hard_fail = not all(bool(corr[k]) for k in hard_keys)
        if corr.get("recv_splits_assignments_ok") is False:
            hard_fail = True
        rec.status = "FAILED_CORRECTNESS" if hard_fail else "OK"

    results = []
    for workload in workloads:
        rank_target = is_rank_target(workload)
        checksum = workload_fingerprint(workload, args.num_experts, args.topk, args.tokens,
                                        args.steps, args.seed, args.ep, **route_kwargs)
        rec = ArmResult(
            workload_id=workload, workload_checksum=checksum,
            experiment="rank_target" if rank_target else "trace_replay", mode=args.mode,
            backend="standard_alltoall", base_sha=args.base_sha, ep=args.ep, etp=args.etp,
            tp=args.tp, world_size=world, rank=rank, num_experts=args.num_experts,
            topk=args.topk, hidden_size=args.hidden_size, logical_tokens=args.tokens,
            assignments=args.tokens * args.topk, steps=args.steps,
            measured_iters=args.iters, warmup_iters=args.warmup,
            hot_maxvio=args.hot_maxvio if rank_target else None,
            schedule=args.schedule if rank_target else None,
            schedule_period=args.schedule_period if rank_target else None,
        )
        try:
            run_arm(rec, workload, checksum)
        except Exception as exc:
            rec.status = "FAILED"
            rec.error = f"{type(exc).__name__}: {exc}"
            import traceback
            traceback.print_exc()
        results.append(rec)
        emit(rec)

    sync()
    if rank == 0:
        ok = sum(1 for r in results if r.status == "OK")
        print(f"[B1] run_id={args.run_id} mode={args.mode} ep={args.ep} world={world} "
              f"arms={len(results)} ok={ok} -> {out_dir}")
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
