# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional direct PR #7450 kernel comparison on identical GPUs and inputs.

Stage the unmodified virtual_expert_triton.py from commit 242c8dbc6 in
MOONEP_PR7450_REFERENCE (default /results/pr7450-reference/virtual_expert_triton.py).
The pinned hash prevents silently benchmarking a different PR revision.
Results are diagnostic timings, not a noisy performance assertion in CI.
"""

import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moonep_moe_scheduler import (
    ReplicaPlannerWorkspace,
    plan_replica_routes,
)
from megatron.core.transformer.moe.replica_peer_tma_transport import PeerTmaTransport
from megatron.core.transformer.moe.replica_weight_transport import ReplicaWeightLayout
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_replica_storage as storage

pytestmark = [pytest.mark.launch_on_gb200, pytest.mark.experimental]


@pytest.fixture(scope="module")
def reference():
    source = Path(
        os.environ.get(
            "MOONEP_PR7450_REFERENCE", "/results/pr7450-reference/virtual_expert_triton.py"
        )
    )
    if not source.is_file():
        pytest.skip("Stage the pinned PR source to run this optional comparison.")
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "4add967b56947e9216220b2ed678e5ac9e76bcc29e400246dfea45aaf3f00380"
    )
    spec = importlib.util.spec_from_file_location("pr7450_kernels", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Utils.set_world_size(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
    Utils.initialize_distributed()
    return module


def _time(call, repeats=30):
    for _ in range(5):
        call()
        dist.barrier()
    torch.cuda.synchronize()
    eager = _measure(call, repeats)
    # Eager CUDA event intervals include CPU launch gaps. Replay separately
    # measures device execution of the same operations without Python dispatch.
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        captured_outputs = call()
    replay = _measure(graph.replay, repeats)
    del captured_outputs, graph
    return {"eager": eager, "cuda_graph": replay}


def _measure(call, repeats):
    times = []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        dist.barrier()
        begin.record()
        call()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) * 1000)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, times)
    return {
        "median_max_rank_us": statistics.median(max(row) for row in zip(*gathered)),
        "rank_median_us": [statistics.median(row) for row in gathered],
    }


def _report(name, values):
    if dist.get_rank() == 0:
        result = {
            "case": name,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "pr_commit": "242c8dbc63c17ec804f1e31974d3dec86e5ed56c",
            **values,
        }
        print("MOONEP_BENCHMARK " + json.dumps(result), flush=True)
        output = Path(os.environ.get("MOONEP_BENCHMARK_OUTPUT", "/results/moonep-pr7450-benchmark"))
        output.mkdir(parents=True, exist_ok=True)
        (output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize("tokens,topk", [(256, 2), (4096, 8)])
@torch.no_grad()
def test_planner_against_pr(reference, tokens, topk):
    experts, rank = 32, dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=device).manual_seed(181 + rank)
    scores = torch.rand(tokens, experts, device=device, generator=g)
    scores[:, 0] += 0.75
    ids = scores.topk(topk, dim=-1).indices.sort(dim=-1).values.contiguous()
    probs = torch.full(ids.shape, 1 / topk, device=device)
    ours = ReplicaPlannerWorkspace.allocate(
        num_experts=experts, device=device, group=dist.group.WORLD
    )
    theirs = reference.VirtualExpertPlannerWorkspace(
        num_experts=experts, device=device, group=dist.group.WORLD
    )

    def split():
        return plan_replica_routes(ids, probs, ours, alignment=256)

    def fused():
        return reference.launch_virtual_expert_planner(ids, theirs, 256)

    def fused_dense():
        mapped, copies = fused()
        dense = probs.new_zeros((tokens, 2 * experts)).scatter(1, mapped.long(), probs)
        return mapped, copies, dense

    try:
        plan, dense = split()
        dist.barrier()
        mapped, copies, expected = fused_dense()
        torch.testing.assert_close(plan.virtual_experts, mapped, rtol=0, atol=0)
        torch.testing.assert_close(plan.experts_to_copy, copies, rtol=0, atol=0)
        torch.testing.assert_close(dense, expected, rtol=0, atol=0)
        _report(
            f"planner-t{tokens}-k{topk}",
            {
                "split_dense": _time(split),
                "pr_fused_ids": _time(fused),
                "pr_fused_dense_bridge": _time(fused_dense),
            },
        )
    finally:
        ours.destroy()
        theirs.destroy()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@torch.no_grad()
def test_weight_push_against_pr(reference, precision):
    from dataclasses import replace

    from megatron.core.transformer.moe.replica_weight_triton import launch_replica_weight_prefetch

    cfg = storage.config(dist.group.WORLD, "shared", "reuse", precision)
    shape = (2048, 2048)
    cfg = replace(
        cfg,
        member_shapes=(shape, shape),
        num_sms=32,
        rowwise_scale_shapes=((2048, 64),) * 2 if precision == "mxfp8" else None,
        columnwise_scale_shapes=((64, 2048),) * 2 if precision == "mxfp8" else None,
    )
    transport = PeerTmaTransport(cfg)
    ws = transport.workspace
    layout = ReplicaWeightLayout.ROWWISE if precision == "mxfp8" else ReplicaWeightLayout.PLAIN
    src = storage.sources(cfg, layout)
    placement = storage.placement(cfg)
    sources = tuple(s.data_bases for s in src)
    scales = tuple(s.scale_bases for s in src) if precision == "mxfp8" else None
    adapter = SimpleNamespace(
        weight_arena=ws.weight_arena,
        weight_handle=ws.weight_handle,
        weight_grid_barrier=ws.weight_grid_barrier,
        rank=transport.rank,
        world_size=cfg.world_size,
        num_local_experts=cfg.num_local_home_experts,
        member_numels=ws.member_numels,
        num_sms=ws.num_sms,
    )

    def ours():
        launch_replica_weight_prefetch(
            sources=sources,
            scale_sources=scales,
            arena=ws.weight_arena,
            peer_bases=ws.weight_handle.buffer_ptrs_dev,
            signal_bases=ws.weight_handle.signal_pad_ptrs_dev,
            experts_to_copy=placement.slot_to_expert,
            grid_barrier=ws.weight_grid_barrier,
            rank=transport.rank,
            world_size=cfg.world_size,
            num_local_home_experts=cfg.num_local_home_experts,
            num_local_replica_slots=cfg.num_local_replica_slots,
            member_numels=ws.member_numels,
            num_sms=ws.num_sms,
        )

    def theirs():
        reference.launch_virtual_expert_weight_prefetch(
            adapter, sources=sources, scale_sources=scales, experts_to_copy=placement.slot_to_expert
        )

    try:
        ours()
        torch.cuda.synchronize()
        expected = ws.weight_arena.clone()
        theirs()
        torch.cuda.synchronize()
        torch.testing.assert_close(ws.weight_arena, expected, rtol=0, atol=0)
        _report(f"push-{precision}", {"ours": _time(ours), "pr": _time(theirs)})
    finally:
        transport.destroy()
