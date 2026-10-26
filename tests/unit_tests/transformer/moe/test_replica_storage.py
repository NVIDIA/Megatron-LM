# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed parity and lifecycle tests for both replica storage policies."""

import time
import weakref
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport
from megatron.core.transformer.moe.replica_peer_tma_transport import PeerTmaTransport
from megatron.core.transformer.moe.replica_storage import (
    ReplicaWeightCache,
    get_replica_consumer_fence,
)
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaOwnership,
    ReplicaPlacement,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightSource,
)
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world

pytestmark = pytest.mark.launch_on_gb200


def test_completed_nccl_handle_releases_packed_messages(distributed_world, monkeypatch):
    cfg = config(distributed_world, "shared", "reuse")
    transport = NcclP2PTransport(cfg)
    exchanged = []
    exchange = transport._exchange

    def track_buffers(sends, sizes=None, dtype=None, **kwargs):
        receives, keepalive = exchange(sends, sizes, dtype, **kwargs)
        exchanged.extend(weakref.ref(t) for t in (*sends.values(), *receives.values()))
        return receives, keepalive

    monkeypatch.setattr(transport, "_exchange", track_buffers)
    plan = transport.prepare_plan(placement(cfg))
    src = sources(cfg, ReplicaWeightLayout.ROWWISE)
    try:
        retained_handle = transport.start_weight_sync(sources=src, plan=plan)
        transport.wait_weight_sync(retained_handle)
        torch.cuda.synchronize()
        previous_buffers = tuple(exchanged)
        assert previous_buffers
        next_handle = transport.start_weight_sync(sources=src, plan=plan)
        # A completed handle may outlive the transfer, without retaining the
        # packed sends/receives after workspace completion cleanup.
        assert all(ref() is None for ref in previous_buffers)
        transport.wait_weight_sync(retained_handle)
        transport.wait_weight_sync(next_handle)
        torch.cuda.synchronize()
    finally:
        transport.destroy()


def config(group, storage, layouts, weight_format="mxfp8"):
    shapes = ((128, 128), (128, 128))
    return ReplicaTransportConfig(
        group=group,
        device=torch.device("cuda", torch.cuda.current_device()),
        world_size=dist.get_world_size(group),
        num_local_home_experts=2,
        num_local_replica_slots=2,
        member_shapes=shapes,
        weight_format=weight_format,
        rowwise_scale_shapes=((128, 4),) * 2 if weight_format == "mxfp8" else None,
        columnwise_scale_shapes=((4, 128),) * 2 if weight_format == "mxfp8" else None,
        grad_dtype=torch.float32,
        num_sms=4,
        replica_storage=storage,
        replica_mxfp8_storage=layouts,
        replica_exchange_buffer_bytes=196608,
    )


def placement(cfg, generation=1, flip=False):
    table = torch.tensor(
        [[2 * ((r + 1) % cfg.world_size), 2 * r + 1] for r in range(cfg.world_size)],
        dtype=torch.int32,
        device=cfg.device,
    )
    if flip:
        table = table.flip(1).contiguous()
    return ReplicaPlacement(table, ReplicaOwnership(2), generation)


def sources(cfg, layout, generation=0):
    rank = dist.get_rank(cfg.group)
    result = []
    for p, shape in enumerate(cfg.member_shapes):
        data = tuple(
            torch.full(
                shape,
                2 * rank + e + p * 20 + generation,
                dtype=torch.uint8 if cfg.weight_format == "mxfp8" else torch.bfloat16,
                device=cfg.device,
            )
            for e in range(2)
        )
        scales = None
        if cfg.weight_format == "mxfp8":
            shapes = (
                cfg.columnwise_scale_shapes
                if layout is ReplicaWeightLayout.COLUMNWISE
                else cfg.rowwise_scale_shapes
            )
            scales = tuple(
                torch.full(shapes[p], 120 + e + generation, dtype=torch.uint8, device=cfg.device)
                for e in range(2)
            )
        result.append(
            ReplicaWeightSource(
                data=data,
                scales=scales,
                layout=layout,
                data_bases=torch.tensor(
                    [t.data_ptr() for t in data], dtype=torch.int64, device=cfg.device
                ),
                scale_bases=(
                    torch.tensor(
                        [t.data_ptr() for t in scales], dtype=torch.int64, device=cfg.device
                    )
                    if scales
                    else None
                ),
            )
        )
    return tuple(result)


@pytest.mark.parametrize("backend", [NcclP2PTransport, PeerTmaTransport], ids=["nccl", "peer_tma"])
@pytest.mark.parametrize("storage", ["shared", "layerwise"])
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_weight_push_waits_for_remote_consumers(distributed_world, backend, storage, weight_format):
    """A fast owner must not overwrite a peer's previous generation of weights."""
    cfg = config(distributed_world, storage, "reuse", weight_format)
    transport = backend(cfg)
    rank = dist.get_rank(cfg.group)
    table = placement(cfg)
    layouts = (
        [ReplicaWeightLayout.ROWWISE, ReplicaWeightLayout.COLUMNWISE]
        if weight_format == "mxfp8"
        else [ReplicaWeightLayout.PLAIN] * 2
    )
    # Allocate both generations before introducing skew; allocation and plan
    # construction must not accidentally provide a cross-rank consumer fence.
    inputs = [sources(cfg, layout, 5 * i) for i, layout in enumerate(layouts)]
    plan = transport.prepare_plan(table)
    experts = table.slot_to_expert[rank].tolist()
    failures = []

    def check_generation(generation, layout):
        for projection in range(2):
            for slot, expert in enumerate(experts):
                value = transport.projection_views(projection)[0][slot]
                if weight_format == "mxfp8":
                    row, row_scale, col, col_scale = value
                    value, scale = (
                        (col, col_scale)
                        if layout is ReplicaWeightLayout.COLUMNWISE
                        else (row, row_scale)
                    )
                    failures.append(not torch.all(scale == 120 + expert % 2 + generation).item())
                failures.append(
                    not torch.all(value == expert + projection * 20 + generation).item()
                )

    try:
        handle = transport.start_weight_sync(sources=inputs[0], plan=plan)
        transport.wait_weight_sync(handle)
        torch.cuda.synchronize()
        dist.barrier(group=cfg.group)
        if rank == 0:
            # Other owners can enter the next push while this rank still has
            # readers of the old arena. Do not raise until everyone has joined
            # and drained the second transfer, even when the old code corrupts it.
            time.sleep(0.5)
            check_generation(0, layouts[0])
        handle = transport.start_weight_sync(sources=inputs[1], plan=plan)
        transport.wait_weight_sync(handle)
        check_generation(5, layouts[1])
        torch.cuda.synchronize()
        failed = torch.tensor(int(any(failures)), device=cfg.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=cfg.group)
        assert failed.item() == 0, "Replica weights overwritten before remote consumers completed"
    finally:
        transport.destroy()


@pytest.mark.parametrize("backend", [NcclP2PTransport, PeerTmaTransport], ids=["nccl", "peer_tma"])
@pytest.mark.parametrize("storage", ["shared", "layerwise"])
@pytest.mark.parametrize("layouts", ["reuse", "separate"])
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_storage_matrix_weights_gradients_and_lifetime(
    distributed_world, backend, storage, layouts, weight_format
):
    cfg = config(distributed_world, storage, layouts, weight_format)
    first, second = backend(cfg), backend(cfg)
    rank = dist.get_rank(cfg.group)
    table = placement(cfg)
    try:
        for p in range(2):
            assert (
                first.projection_views(p)[1].data_ptr() == second.projection_views(p)[1].data_ptr()
            ) == (storage == "shared")
        layouts_to_test = (
            [ReplicaWeightLayout.ROWWISE, ReplicaWeightLayout.COLUMNWISE]
            if weight_format == "mxfp8"
            else [ReplicaWeightLayout.PLAIN]
        )
        for i, layout in enumerate(layouts_to_test):
            src = sources(cfg, layout, i * 5)
            plan = first.prepare_plan(table)
            handle = first.start_weight_sync(sources=src, plan=plan)
            # Repeated waits on different consumers must retain the completion.
            for stream in [torch.cuda.current_stream(), torch.cuda.Stream()]:
                with torch.cuda.stream(stream):
                    first.wait_weight_sync(handle)
                    for p in range(2):
                        weights, _ = first.projection_views(p)
                        for slot, expert in enumerate(table.slot_to_expert[rank].tolist()):
                            weight = weights[slot]
                            if weight_format == "mxfp8":
                                row, row_scale, col, col_scale = weight
                                assert (row.data_ptr() == col.data_ptr()) == (layouts == "reuse")
                                value, scale = (col, col_scale) if i else (row, row_scale)
                                assert torch.all(scale == 120 + expert % 2 + i * 5)
                            else:
                                value = weight
                            assert torch.all(value == expert + p * 20 + i * 5)
                            if weight_format == "mxfp8" and layouts == "separate" and i:
                                assert torch.all(row == expert + p * 20)
                stream.synchronize()
        # Add replica gradients to existing native values, local and remote.
        native = tuple(
            torch.full((2, *shape), 3.0, device=cfg.device) for shape in cfg.member_shapes
        )
        destinations = tuple(
            ReplicaGradDestination(
                tuple(tensor),
                torch.tensor([t.data_ptr() for t in tensor], dtype=torch.int64, device=cfg.device),
            )
            for tensor in native
        )
        for p in range(2):
            first.projection_views(p)[1].fill_(rank + 1)
            handle = first.start_grad_reduce(native_grads=destinations, plan=plan, projections=(p,))
            first.wait_grad_reduce(handle)
            expected = torch.full_like(native[p], 3.0)
            for sender, row in enumerate(table.slot_to_expert.tolist()):
                for expert in row:
                    if expert // 2 == rank:
                        expected[expert % 2].add_(sender + 1)
            torch.testing.assert_close(native[p], expected, rtol=0, atol=0)
        first.destroy()
        # Releasing one layer must leave a shared workspace usable by another.
        plan = second.prepare_plan(table)
        handle = second.start_weight_sync(sources=sources(cfg, layouts_to_test[0], 9), plan=plan)
        second.wait_weight_sync(handle)
        torch.cuda.synchronize()
    finally:
        first.destroy()
        second.destroy()


@pytest.mark.parametrize("backend", [NcclP2PTransport, PeerTmaTransport], ids=["nccl", "peer_tma"])
def test_exact_cache_invalidation_is_collective(distributed_world, backend):
    cfg = config(distributed_world, "layerwise", "separate")
    transport = backend(cfg)
    cache = ReplicaWeightCache(cfg.group, cfg.device)
    forward, backward = ReplicaWeightLayout.ROWWISE, ReplicaWeightLayout.COLUMNWISE
    src = sources(cfg, forward)
    table = placement(cfg)

    def ensure(table, src, layout):
        handle = cache.lookup(layout, table.slot_to_expert, src)
        hit = handle is not None
        if not hit:
            handle = transport.start_weight_sync(sources=src, plan=transport.prepare_plan(table))
            cache.remember(layout, table.slot_to_expert, src, handle)
        transport.wait_weight_sync(handle)
        return hit

    try:
        assert not ensure(table, src, forward)
        assert ensure(placement(cfg, generation=99), src, forward)  # New plan, same mapping.
        bwd = sources(cfg, backward, 5)
        assert not ensure(table, bwd, backward)
        assert ensure(table, src, forward)
        assert ensure(table, bwd, backward)
        # Only rank zero changes a source, without incrementing the original
        # tensor's version. Owners and receivers must all take the miss path.
        if dist.get_rank(cfg.group) == 0:
            src[0].data[0].data.add_(1)
        assert not ensure(table, src, forward)
        assert ensure(table, src, forward)
        if dist.get_rank(cfg.group) == 0:
            src[0].scales[0].data.add_(1)
        assert not ensure(table, src, forward)
        other = placement(cfg, flip=True)
        assert not ensure(other, src, forward)
        assert not ensure(table, src, forward)  # Restore an older microbatch plan.
        assert ensure(table, src, forward)
        torch.cuda.synchronize()
    finally:
        cache.clear()
        transport.destroy()


def test_compute_consumer_fence_orders_different_streams(distributed_world):
    device = torch.device("cuda", torch.cuda.current_device())
    fence = get_replica_consumer_fence(distributed_world, device)
    owner, other = object(), object()
    x = torch.zeros(4096, device=device)
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    a.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(a):
        fence.acquire(owner, device)
        with pytest.raises(RuntimeError, match="active compute consumer"):
            fence.acquire(other, device)
        torch.cuda._sleep(1_000_000)
        x.fill_(7)
        saved = x.clone()
        fence.release(owner, device)
    with torch.cuda.stream(b):
        fence.acquire(other, device)
        x.fill_(9)
        fence.release(other, device)
    torch.cuda.synchronize()
    assert torch.all(saved == 7)
    assert torch.all(x == 9)


@pytest.fixture(scope="module")
def single_expert_tp_group(distributed_world):
    group = None
    for rank in range(dist.get_world_size()):
        candidate = dist.new_group([rank], backend="nccl")
        if rank == dist.get_rank():
            group = candidate
    yield group
    dist.destroy_process_group(group)


@pytest.mark.parametrize("backend", [NcclP2PTransport, PeerTmaTransport], ids=["nccl", "peer_tma"])
@pytest.mark.parametrize("storage", ["shared", "layerwise"])
@pytest.mark.parametrize("layouts", ["reuse", "separate"])
def test_real_te_runtime_gemm_and_cached_prefetch(
    distributed_world, single_expert_tp_group, monkeypatch, backend, storage, layouts
):
    from megatron.core.transformer.moe import replica_expert_runtime as runtime_module
    from tests.unit_tests.transformer.moe import test_replica_nccl_transport as reference_tests

    runtime_type = runtime_module.ReplicaExpertRuntime
    created = []

    def make_runtime(**kwargs):
        kwargs.update(
            transport_factory=backend, replica_storage=storage, replica_mxfp8_storage=layouts
        )
        runtime = runtime_type(**kwargs)
        created.append(runtime)
        start, wait = runtime.start_prefetch, runtime.wait_prefetch

        def repeat_prefetch(plan, direction=runtime_module._WeightDirection.FORWARD):
            start(plan, direction)
            wait(plan)
            before = runtime.weight_cache.hits if runtime.weight_cache else 0
            start(plan, direction)
            if runtime.weight_cache:
                assert runtime.weight_cache.hits == before + 1

        runtime.start_prefetch = repeat_prefetch
        return runtime

    monkeypatch.setattr(runtime_module, "ReplicaExpertRuntime", make_runtime)
    # Real TE quantization, grouped runtime parameter binding, and GEMM/dgrad
    # against independent source weights. Both orientations are exercised.
    reference_tests.test_nccl_runtime_forward_and_dgrad(
        distributed_world, single_expert_tp_group, "mxfp8", False, monkeypatch, slots=2
    )
    assert len(created) == 1
