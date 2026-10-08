#!/usr/bin/env python3
"""A2: count device->host synchronisations per ragged dispatch step (deterministic, not timing).

The A2 change removes a redundant `.item()` that re-read a tensor the code had already pulled
to the host with `.tolist()`. Wall-clock on this shared node cannot resolve that (co-tenant
noise exceeds the effect), but the effect itself is *deterministic*: it is one fewer
device->host transfer per ragged step. This probe counts them by wrapping the tensor
methods that force a sync, so base and patch are compared on a number that does not depend
on GPU contention at all.

The exact round-trip oracle still runs, so a lower count cannot come from skipping work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Tensor methods that force a device->host synchronisation.
SYNC_METHODS = ("item", "tolist", "cpu", "numpy")

counts = {m: 0 for m in SYNC_METHODS}
_orig = {}
_depth = {"sync": 0}
call_log = []


def _wrap(name):
    """Wrap a tensor method so it records one sync event (ignoring internal re-entry)."""
    def wrapper(self, *a, **kw):
        inner = _depth["sync"] > 0
        if not inner:
            _depth["sync"] += 1
            counts[name] += 1
            try:
                call_log.append((name, tuple(self.shape), str(self.dtype)))
            finally:
                _depth["sync"] -= 1
        return _orig[name](self, *a, **kw)
    return wrapper


def install():
    for n in SYNC_METHODS:
        _orig[n] = getattr(torch.Tensor, n)
        setattr(torch.Tensor, n, _wrap(n))


def uninstall():
    for n in SYNC_METHODS:
        setattr(torch.Tensor, n, _orig[n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree-label", required=True)
    ap.add_argument("--seq-lens", default="2048,128")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repo-root", default=None)
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if a.repo_root:
        sys.path.insert(0, a.repo_root)
    from tests.unit_tests.inference.test_moe_dispatching_and_routing import (  # noqa: E402
        NANOV3_BASE,
        _make_base_config,
    )
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection  # noqa: E402
    from megatron.core.transformer.moe.token_dispatcher_inference import (  # noqa: E402
        NCCLAllGatherDispatcher,
    )
    from megatron.core.parallel_state import initialize_model_parallel  # noqa: E402

    initialize_model_parallel(1, 1, expert_model_parallel_size=world)

    lens = [int(x) for x in a.seq_lens.split(",")]
    if len(lens) != world:
        lens = [lens[0]] * world

    cfg = _make_base_config(expert_model_parallel_size=world)
    num_local_experts = cfg.num_moe_experts // world
    ep_rank = torch.distributed.get_rank() if world > 1 else 0
    NCCLAllGatherDispatcher.allocate_buffers()

    def make_dispatcher():
        return NCCLAllGatherDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=[ep_rank * num_local_experts + i for i in range(num_local_experts)],
            config=cfg,
            pg_collection=get_default_pg_collection(),
            runs_metadata_sync=True,
        )

    d = make_dispatcher()
    n_local = lens[rank]

    def fresh(n):
        g = torch.Generator(device="cpu").manual_seed(7)
        return (
            (torch.randint(-8, 9, (n, a.hidden), generator=g).float() / 8).to(torch.bfloat16).cuda(),
            torch.rand(n, a.topk, generator=g).cuda(),
            torch.randint(0, cfg.num_moe_experts, (n, a.topk), generator=g).cuda(),
        )

    # correctness first (identical to the profiling probe's oracle)
    NCCLAllGatherDispatcher._use_allgather_v = True
    h, p, rm = fresh(n_local)
    d.routing_map = rm
    hidden, probs = d.token_dispatch(h, p)
    out = d.token_combine(hidden)
    torch.cuda.synchronize()
    exact = bool(torch.equal(out, h * world)) if n_local else bool(out.numel() == 0)
    rows_ok = int(hidden.shape[0]) == sum(lens)

    # The valid-tokens tensor and the host estimate must equal the global token count.
    # This is the value the removed `.item()` used to produce, so asserting it here proves
    # the replacement writes exactly what the original did (same total, same tensor dtype).
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
    )
    expected_total = sum(lens)
    d.routing_map = fresh(n_local)[2]
    d.token_dispatch(h, p)
    torch.cuda.synchronize()
    got_total = int(InferenceAllGatherDispatcherBase._valid_tokens_tensor.item())
    got_estimate = int(InferenceAllGatherDispatcherBase._host_valid_tokens_estimate)
    total_ok = (got_total == expected_total) and (got_estimate == expected_total)
    if not total_ok:
        print(f"[a2sync] TOTAL MISMATCH: tensor={got_total} host={got_estimate} "
              f"expected={expected_total}", flush=True)

    # deterministic sync counting: only the ragged dispatch path, no combine
    for _ in range(a.warmup):
        rm = fresh(n_local)[2]
        d.routing_map = rm
        d.token_dispatch(h, p)
    torch.cuda.synchronize()

    install()
    try:
        counts.clear()
        counts.update({m: 0 for m in SYNC_METHODS})
        call_log.clear()
        for _ in range(a.iters):
            rm = fresh(n_local)[2]
            d.routing_map = rm
            d.token_dispatch(h, p)
        torch.cuda.synchronize()
    finally:
        uninstall()

    total = sum(counts.values())
    rec = dict(tree_label=a.tree_label, rank=rank, world=world, local_tokens=lens,
               iters=a.iters, sync_total=total, sync_per_step=total / a.iters,
               by_method=dict(counts), roundtrip_exact=exact, rows_ok=rows_ok,
               expected_total=expected_total, got_total=got_total,
               got_estimate=got_estimate, total_ok=total_ok,
               call_log=call_log[:24])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with (Path(a.out) / f"rank{rank}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    if rank == 0:
        print(f"[a2sync] {a.tree_label} syncs/step={rec['sync_per_step']:.2f} "
              f"total={total} by={counts} exact={exact} rows_ok={rows_ok} "
              f"total_ok={total_ok}", flush=True)
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
