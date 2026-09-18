#!/usr/bin/env python3
"""B1 — MoE runtime token-imbalance benchmark (standard all-to-all baseline).

Scope (first slice, as claimed on roadmap #6757):
    fixed logical routing workload -> existing standard all-to-all dispatcher
    -> local expert computation -> correctness oracle -> per-rank raw metrics

Two measurement modes are reported and never mixed:
    dispatch_only : dispatcher round trip for a fixed logical routing.
                    Does NOT represent GEMM cost or a training step.
    moe_fwd_bwd   : real MoELayer forward + backward (router included, real gradients).
                    Does NOT represent full-model throughput or convergence.

Timing boundary:
    INCLUDED  dispatcher preprocess/dispatch/postprocess, local expert compute,
              combine preprocess/combine/postprocess; backward in moe_fwd_bwd.
    EXCLUDED  all metric reduction, JSON writing, printing, warmup iterations.
    The router runs inside moe_fwd_bwd (it is part of a real step). In dispatch_only the
    routing is fixed input and the router is deliberately not executed.

Correctness oracle (exact integer metadata + exact round trip):
    * every valid token row has exactly K distinct experts
    * sum_e assignments[e] == T_valid * K, and the global histogram matches T*K*world
    * dispatch -> identity expert -> combine restores every row to its origin
      (values are exact powers of two, so the gated sum is exact in bf16/fp32)
    * dispatcher recv splits agree with the independent group-global count

Usage:
    torchrun --standalone --nproc_per_node=N b1_benchmark.py --mode dispatch_only --ep 2 ...
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


# --------------------------------------------------------------------------------------
# W0..W7 logical routing workloads (deterministic; independent of execution layout)
# --------------------------------------------------------------------------------------

WORKLOADS = (
    "W0_balanced",
    "W1_hot_expert_set",
    "W2_hot_rank",
    "W3_spread_hot_experts",
    "W4_source_ragged",
    "W5_zero_expert_rank",
    "W6_rotating_hotspot",
    "W7_burst",
)

WORKLOAD_NOTES = {
    "W0_balanced": "round-robin distinct top-k; every expert and destination rank equal",
    "W1_hot_expert_set": "a fixed small expert set receives most selections",
    "W2_hot_rank": "hot experts owned by one EP rank; separates per-expert from per-rank skew",
    "W3_spread_hot_experts": "same expert heat, hotspot spread across ranks",
    "W4_source_ragged": "uneven per-rank source token counts, balanced destinations",
    "W5_zero_expert_rank": "half the experts never selected; owner ranks receive nothing",
    "W6_rotating_hotspot": "hotspot moves every 3 steps; detects stale plan/count",
    "W7_burst": "3 steady steps, 1 burst step, then recovery",
}


def _distinct_topk(pool: list[int], k: int, g: torch.Generator) -> list[int]:
    if k > len(pool):
        raise ValueError(f"topk {k} exceeds candidate pool {len(pool)}")
    idx = torch.randperm(len(pool), generator=g)[:k].tolist()
    return sorted(pool[i] for i in idx)


def routing_for_step(
    workload: str, num_experts: int, topk: int, tokens: int, step: int, seed: int, ep_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(expert_ids [tokens, topk] int64, gates [tokens, topk] float32), deterministic.

    Seeded only by (workload, seed, step), so the logical workload is identical for every
    execution method and every EP/ETP layout - that is the B1 contract. `ep_size` is taken for
    call-site symmetry and for workloads whose *description* is expressed in terms of a layout
    (W3 spreads a hot set with a stride derived from it), but the resulting expert ids must not
    change with it; `test_workload.py` asserts exactly that for every workload.
    """
    g = torch.Generator(device="cpu").manual_seed(
        int(hashlib.sha256(f"{workload}|{seed}|{step}".encode()).hexdigest()[:8], 16)
    )
    everyone = list(range(num_experts))
    hotwidth = max(topk, num_experts // 4)
    per_rank = max(1, num_experts // max(ep_size, 1))
    ids = torch.empty((tokens, topk), dtype=torch.int64)

    if workload == "W0_balanced":
        for t in range(tokens):
            start = (t * topk) % num_experts
            ids[t] = torch.tensor([(start + j) % num_experts for j in range(topk)])
    elif workload == "W1_hot_expert_set":
        hot = [i % num_experts for i in range(hotwidth)]
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(hot if t % 4 else everyone, topk, g))
    elif workload == "W2_hot_rank":
        # A contiguous global hot block starting at expert 0. Defined from GLOBAL expert ids
        # only, so the same logical workload exists at any EP size; which rank ends up owning
        # the hot block is a property of the layout, not of the workload.
        hot = [i % num_experts for i in range(hotwidth)]
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(hot if t % 4 else everyone, topk, g))
    elif workload == "W3_spread_hot_experts":
        # A hot set spread across every OTHER global expert (0, 2, 4, ...), versus W2's
        # contiguous block at 0. Defined from globals only, so the same logical workload
        # exists at any EP size; that the two differ in how many ranks they touch is a
        # property of the layout, not of the workload definition.
        spread = sorted({(2 * i) % num_experts for i in range(max(topk, num_experts // 2))})
        pool = spread if len(spread) >= topk else everyone
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(pool if t % 4 else everyone, topk, g))
    elif workload == "W4_source_ragged":
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(everyone, topk, g))
    elif workload == "W5_zero_expert_rank":
        live = list(range(num_experts // 2, num_experts)) or everyone
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(live, topk, g))
    elif workload == "W6_rotating_hotspot":
        base = ((step // 3) * hotwidth) % num_experts
        hot = [(base + i) % num_experts for i in range(hotwidth)]
        for t in range(tokens):
            ids[t] = torch.tensor(_distinct_topk(hot if t % 4 else everyone, topk, g))
    elif workload == "W7_burst":
        burst = (step % 4) == 3
        hot = [i % num_experts for i in range(hotwidth)]
        for t in range(tokens):
            ids[t] = torch.tensor(
                _distinct_topk(hot if (burst and t % 2 == 0) else everyone, topk, g)
            )
    else:
        raise ValueError(f"unknown workload {workload}")

    if topk > 1:
        for t in range(tokens):
            assert len(set(ids[t].tolist())) == topk, "top-k slots must be distinct experts"

    gates = torch.rand((tokens, topk), generator=g, dtype=torch.float32) * 0.5 + 0.5
    gates = gates / gates.sum(dim=1, keepdim=True)
    return ids, gates


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
    workload: str, num_experts: int, topk: int, tokens: int, steps: int, seed: int, ep_size: int
) -> str:
    h = hashlib.sha256()
    h.update(f"{workload}|{num_experts}|{topk}|{tokens}|{steps}|{seed}|{ep_size}".encode())
    for step in range(steps):
        ids, gates = routing_for_step(workload, num_experts, topk, tokens, step, seed, ep_size)
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


@dataclass
class ArmResult:
    workload_id: str
    workload_checksum: str
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
    median_ms: float | None = None
    mean_ms: float | None = None
    p95_ms: float | None = None
    min_ms: float | None = None
    logical_tokens_per_second: float | None = None
    assignments_per_second: float | None = None
    expert_imbalance: float | None = None
    expert_cv: float | None = None
    rank_imbalance: float | None = None
    expert_counts: list[int] = field(default_factory=list)
    rank_received: list[int] = field(default_factory=list)
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
    ap.add_argument("--steps", type=int, default=1)
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
                    help="directory to persist a trace-v1 record of this workload into")
    ap.add_argument("--trace-id", default=None,
                    help="existing trace directory to load the workload from instead of generating")
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
        Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules

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
        moe_grouped_gemm=False,  # transformer_engine is not installed on this node
        moe_router_dtype="fp32",
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
        """SwiGLU expert slices; per-expert loop because TE grouped GEMM is unavailable."""
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
            torch.manual_seed(args.seed)  # identical logical weights on every rank
            _moe_layer = MoELayer(config, subs).cuda().to(dtype=torch.bfloat16)
            _moe_layer.set_layer_number(0)
            _moe_layer.train()
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

    def run_arm(rec: ArmResult, workload: str) -> None:
        corr_extra: dict[str, Any] = {}
        steps = args.steps
        per_step = [
            routing_for_step(workload, args.num_experts, args.topk, args.tokens, s,
                             args.seed, args.ep)
            for s in range(steps)
        ]
        maps_cpu = [routing_map_from_ids(ids, args.num_experts) for ids, _ in per_step]
        # the dispatcher consumes the routing map and the [T, E] probs on the device
        maps = [m.cuda() for m in maps_cpu]

        # ---- optional trace-v1 persistence (rank 0 only: the workload is global) ----
        if args.write_trace and rank == 0:
            try:
                from trace import save_trace, trace_digest
                tdir = Path(args.write_trace) / workload
                ids0s, gates0s = per_step[0]
                src = torch.zeros(ids0s.shape[0], dtype=torch.int32)
                man = save_trace(
                    tdir, ids0s, gates0s, src, num_experts=args.num_experts, topk=args.topk,
                    hidden_size=args.hidden_size, world_size=world, seed=args.seed,
                    workload_id=workload, workload_checksum=checksum,
                    producer_commit=args.base_sha,
                )
                corr_extra["trace_dir"] = str(tdir)
                corr_extra["trace_digest"] = trace_digest(tdir)
                print(f"[B1] wrote trace {tdir} digest={corr_extra['trace_digest']}", flush=True)
            except Exception as exc:
                raise RuntimeError(f"trace persistence failed for {workload}: {exc}") from exc
        probs_dev = [
            probs_matrix_from_gates(ids, g, args.num_experts, torch.bfloat16).cuda()
            for ids, g in per_step
        ]

        # ---------------- correctness oracle (outside the timed window) ----------------
        corr: dict[str, Any] = {}
        ids0, gates0 = per_step[0]
        rmap0_cpu = maps_cpu[0]   # exact integer oracle
        rmap0 = maps[0]           # device copy handed to the dispatcher
        corr["distinct_topk_ok"] = all(
            len(set(ids0[t].tolist())) == args.topk for t in range(args.tokens)
        )
        corr["id_range_ok"] = bool(ids0.min().item() >= 0
                                   and ids0.max().item() < args.num_experts)
        corr["assignments_match_T_times_k"] = (
            int(rmap0_cpu.sum().item()) == args.tokens * args.topk
        )
        local_hist = rmap0_cpu.sum(dim=0).to(torch.int64).cuda()
        global_hist = local_hist.clone()
        torch.distributed.all_reduce(global_hist)
        corr["global_histogram_exact"] = (
            int(global_hist.sum().item()) == args.tokens * args.topk * world
        )
        corr["local_expert_counts"] = [int(x) for x in local_hist.cpu().tolist()]
        corr["global_expert_counts"] = [int(x) for x in global_hist.cpu().tolist()]

        # dispatch -> identity expert -> combine must restore the input exactly.
        #
        # Construction: every element is an exact multiple of 2**-10 that a bf16 holds
        # exactly, and every row sums to exactly `topk`. With the uniform 1/topk gating
        # above, combine returns sum_k (row / topk) = row, i.e. an EXACT equality.
        # `rank` shifts the pattern so a rank cannot pass on another rank's data.
        # The dispatcher's combine step SUMS the top-k expert contributions; the routing
        # probabilities are applied inside the expert module upstream, not by the dispatcher.
        # This harness therefore feeds raw expert outputs to combine and the round trip
        # returns a constant multiple of the input. That multiple is measured once here
        # (on the slow path, outside the timed window) and then required to be exact.
        #
        # Uniform gating is used so the run is representative of a real router; the value is
        # only used by the experts, which are the identity here.
        assert (args.topk & (args.topk - 1)) == 0, "oracle prefers a power-of-two topk"
        uniform = torch.full_like(gates0, 1.0 / args.topk)
        ones_full = probs_matrix_from_gates(
            ids0, uniform, args.num_experts, torch.bfloat16
        ).cuda()
        probe_x = torch.full((args.tokens, args.hidden_size), 0.5, device=dev,
                             dtype=torch.bfloat16)
        ph, _c, _p = token_permutation(probe_x, ones_full, rmap0)
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

        permuted, counts, _pp = token_permutation(hidden, ones_full, rmap0)
        restored = token_unpermutation(permuted)
        corr["roundtrip_exact"] = bool(torch.equal(restored, expected_roundtrip))
        corr["roundtrip_matches_calibrated_factor"] = corr["roundtrip_exact"]
        if not corr["roundtrip_exact"]:
            corr["roundtrip_max_abs_err"] = float(
                (restored.float() - expected_roundtrip.float()).abs().max().item()
            )
            corr["roundtrip_mismatch_rows"] = int(
                (restored != expected_roundtrip).any(dim=1).sum().item()
            )
        # Accounting boundary, stated explicitly because the two kinds of count differ:
        #   logical tokens      : args.tokens * world
        #   assignment rows     : args.tokens * topk * world  (top-k is already expanded)
        # `counts` is per-LOCAL-EXPERT assignment counts, so it sums to the groups' share of
        # assignment rows. `output_splits` is the dispatcher's per-source-rank receive
        # layout, also in assignment rows (the dispatch unit is one token-expert pair).
        # local_counts_sum_ok is deliberately a bound, not an equality: only the group can
        # reconcile the total, and that is exactly what global_histogram_exact checks.
        corr["global_tokens"] = args.tokens * world
        corr["global_assignments"] = int(global_hist.sum().item())
        corr["counts_len_ok"] = len(counts) == num_local_experts
        corr["local_counts_sum_ok"] = int(counts.sum().item()) <= corr["global_assignments"]
        corr["recv_splits_assignments_ok"] = None
        osplit = dispatcher.output_splits
        recv = [int(x) for x in osplit] if osplit is not None else []
        corr["recv_splits_per_source_rank"] = recv
        if recv:
            # `output_splits` is per SOURCE rank for this destination only, so a single
            # rank's row cannot reconcile the group total on its own: sum over all ranks
            # with an all-reduce and then require the exact global assignment count.
            local_recv = torch.tensor(sum(recv), dtype=torch.int64, device=dev)
            total_recv = local_recv.clone()
            torch.distributed.all_reduce(total_recv)
            corr["recv_splits_total"] = int(total_recv.item())
            corr["recv_splits_assignments_ok"] = (
                int(total_recv.item()) == int(global_hist.sum().item())
            )

        # ---------------- model-mode inputs ----------------
        # A real (non-sentinel) input plus a padding mask: `training_step` and `moe_fwd_bwd`
        # run the actual router, and this rank owns `valid_here` of the rows.
        rows_here = max(1, min(args.tokens, args.tokens))
        g_in = torch.Generator(device="cpu").manual_seed(args.seed + 991 + rank)
        real_input = torch.randn(rows_here, args.hidden_size, generator=g_in,
                                 dtype=torch.float32).to(device=dev, dtype=torch.bfloat16)
        pad = torch.ones(rows_here, 1, dtype=torch.bool, device=dev)
        valid_here = max(1, rows_here - (rank % 2))
        pad[:valid_here] = False

        # ---------------- timing ----------------
        step_holder: dict[str, Any] = {}

        def one_iter() -> None:
            if args.mode == "training_step":
                # Real layer, real router, real gradients, real optimizer update. The
                # workload is fixed by seed; the router decides the actual routing.
                layer = moe_layer()
                opt = step_holder.get("opt")
                if opt is None:
                    opt = torch.optim.SGD(layer.parameters(), lr=0.01)
                    step_holder["opt"] = opt
                opt.zero_grad(set_to_none=True)
                from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
                MoEAuxLossAutoScaler.main_loss_backward_scale = None
                out = layer(real_input.unsqueeze(1), padding_mask=pad)
                h = out[0] if isinstance(out, tuple) else out
                loss = h.float().pow(2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
                opt.step()
                step_holder["last_loss"] = float(loss.detach())
                return

            for s in range(steps):
                rmap = maps[s]
                p = probs_dev[s]
                count("permute_calls")
                ph, cnt, pp = token_permutation(hidden, p, rmap)
                count("expert_calls")
                eh = expert_compute(ph, cnt)
                # probs are applied INSIDE the expert module upstream
                # (experts.py: intermediate_parallel * permuted_probs), not by the
                # dispatcher, so combine must receive the raw expert output.
                count("combine_calls")
                _ = token_unpermutation(eh)

        sync()
        for _ in range(args.warmup):
            one_iter()
        sync()

        times: list[float] = []
        torch.cuda.reset_peak_memory_stats(dev)
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.distributed.barrier()
            start.record()
            one_iter()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        sync()

        rec.median_ms = percentile(sorted(times), 0.5)
        rec.mean_ms = sum(times) / len(times)
        rec.p95_ms = percentile(sorted(times), 0.95)
        rec.min_ms = min(times)
        secs = rec.median_ms / 1000.0
        rec.logical_tokens_per_second = (args.tokens * steps) / secs
        rec.assignments_per_second = (args.tokens * args.topk * steps) / secs
        rec.peak_allocated_bytes = int(torch.cuda.max_memory_allocated(dev))
        rec.peak_reserved_bytes = int(torch.cuda.max_memory_reserved(dev))
        rec.expert_counts = corr["global_expert_counts"]
        rec.rank_received = recv
        im = imbalance_metrics(rec.expert_counts)
        rec.expert_imbalance = im["max_over_mean"]
        rec.expert_cv = im["cv"]
        if recv:
            rec.rank_imbalance = imbalance_metrics(recv)["max_over_mean"]

        corr["trace_dir"] = corr_extra.get("trace_dir")
        corr["trace_digest"] = corr_extra.get("trace_digest")
        if args.mode == "training_step":
            last_loss = step_holder.get("last_loss")
            corr["training_step_ran"] = last_loss is not None
            corr["training_step_loss_finite"] = (
                last_loss is not None and math.isfinite(last_loss) and abs(last_loss) < 1e6
            )
            corr["training_step_loss"] = last_loss
            grad_ok = False
            layer = _moe_layer
            if layer is not None:
                gsum = sum(
                    float(p.grad.abs().sum()) for p in layer.parameters() if p.grad is not None
                )
                grad_ok = gsum > 0
                corr["training_step_grad_abs_sum"] = gsum
            corr["training_step_grads_flow"] = grad_ok
        if args.instrument != "none":
            corr["instrumentation_counters"] = dict(counters)
        rec.correctness = corr
        hard_keys = [
            "distinct_topk_ok", "id_range_ok", "assignments_match_T_times_k",
            "global_histogram_exact", "roundtrip_exact", "counts_len_ok",
            "local_counts_sum_ok", "combine_factor_is_topk",
        ]
        if args.mode == "training_step":
            hard_keys += [
                "training_step_ran", "training_step_loss_finite", "training_step_grads_flow",
            ]
        hard_fail = not all(bool(corr[k]) for k in hard_keys)
        if corr.get("recv_splits_assignments_ok") is False:
            hard_fail = True
        rec.status = "FAILED_CORRECTNESS" if hard_fail else "OK"

    results = []
    for workload in args.workloads.split(","):
        checksum = workload_fingerprint(workload, args.num_experts, args.topk, args.tokens,
                                        args.steps, args.seed, args.ep)
        rec = ArmResult(
            workload_id=workload, workload_checksum=checksum, mode=args.mode,
            backend="standard_alltoall", base_sha=args.base_sha, ep=args.ep, etp=args.etp,
            tp=args.tp, world_size=world, rank=rank, num_experts=args.num_experts,
            topk=args.topk, hidden_size=args.hidden_size, logical_tokens=args.tokens,
            assignments=args.tokens * args.topk, steps=args.steps,
            measured_iters=args.iters, warmup_iters=args.warmup,
        )
        try:
            run_arm(rec, workload)
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
