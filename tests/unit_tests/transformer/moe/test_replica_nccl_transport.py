# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NCCL replica transport parity, subgroup routing and completion lifetime tests."""

import gc
import os
import socket
import weakref
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport, _compile_schedule
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaOwnership,
    ReplicaPlacement,
    ReplicaTransferHandle,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightSource,
    create_replica_weight_transport,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


def test_host_schedule_matches_every_sender_and_receiver():
    table = [[2, 0, 2, -1], [0, 3, 4, 0], [-1, 1, 5, 2], [-1, -1, -1, -1]]
    plans = [_compile_schedule(table, 2, rank) for rank in range(4)]
    observed = []
    for owner, plan in enumerate(plans):
        for route in plan.sends:
            receiver = next(r for r in plans[route.peer].receives if r.peer == owner)
            assert receiver.home_indices == route.home_indices
            assert receiver.replica_slots == route.replica_slots
            observed.extend(
                (route.peer, slot, owner * 2 + index)
                for index, slot in zip(route.home_indices, route.replica_slots)
            )
        observed.extend(
            (owner, slot, owner * 2 + index)
            for index, slot in zip(plan.local.home_indices, plan.local.replica_slots)
        )
    expected = [
        (rank, slot, expert)
        for rank, row in enumerate(table)
        for slot, expert in enumerate(row)
        if expert != -1
    ]
    assert sorted(observed) == sorted(expected)
    assert not plans[3].sends and not plans[3].receives and not plans[3].local.replica_slots


@pytest.mark.parametrize("expert", [-2, 8])
def test_host_schedule_rejects_out_of_range_experts(expert):
    with pytest.raises(ValueError, match="expert id"):
        _compile_schedule([[expert], [-1], [-1], [-1]], 2, 0)


@pytest.fixture(scope="module")
def distributed_world():
    if not torch.cuda.is_available():
        pytest.skip("NCCL transport tests require CUDA.")
    # Utils defaults to LOCAL_RANK for its single-node tests. Supply the
    # global torchrun coordinates explicitly when spanning multiple nodes.
    Utils.set_world_size(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
    Utils.initialize_distributed()
    return dist.group.WORLD


def test_distributed_launch_topology(distributed_world):
    rank = dist.get_rank()
    assert rank == int(os.environ["RANK"])
    assert dist.get_world_size() == int(os.environ["WORLD_SIZE"])
    assert torch.cuda.current_device() == int(os.environ["LOCAL_RANK"])
    members = [None] * dist.get_world_size()
    dist.all_gather_object(members, (rank, socket.gethostname(), torch.cuda.current_device()))
    assert len({(host, device) for _, host, device in members}) == len(members)
    if "NUM_NODES" in os.environ:
        assert len({host for _, host, _ in members}) == int(os.environ["NUM_NODES"])
    print(f"NCCL transport topology: {members}", flush=True)


@pytest.fixture(params=[1, 2, 4, 8])
def ep_group(request, distributed_world):
    world_size = dist.get_world_size()
    ep_size = request.param
    if world_size < ep_size or world_size % ep_size:
        pytest.skip(f"Requires a world size divisible by EP={ep_size}.")
    # Strided subgroups exercise global-rank conversion, including groups whose
    # first global rank is not zero. Every world member creates groups in order.
    count = world_size // ep_size
    group = None
    for offset in range(count):
        ranks = list(range(offset, world_size, count))
        candidate = dist.new_group(ranks, backend="nccl", timeout=timedelta(seconds=60))
        if dist.get_rank() in ranks:
            group = candidate
    try:
        yield group
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=group)
        dist.destroy_process_group(group)
        dist.barrier()


@pytest.fixture(scope="module")
def expert_tp_group(distributed_world):
    group = None
    for rank in range(dist.get_world_size()):
        candidate = dist.new_group([rank], backend="nccl")
        if rank == dist.get_rank():
            group = candidate
    yield group
    dist.destroy_process_group(group)


@pytest.mark.parametrize("slots", [1, 2, 3])
@pytest.mark.parametrize(
    "weight_format,swizzled", [("bf16", False), ("mxfp8", False), ("mxfp8", True)]
)
def test_nccl_runtime_forward_and_dgrad(
    ep_group, expert_tp_group, weight_format, swizzled, monkeypatch, slots
):
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
    from megatron.core.transformer.moe.replica_expert_runtime import (
        ReplicaExpertRuntime,
        ReplicaPlan,
        _WeightDirection,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = swizzled
    rank, size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    device = torch.device("cuda", torch.cuda.current_device())
    shapes = ((512, 128), (128, 256))
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2 * size,
        expert_model_parallel_size=size,
        moe_ffn_hidden_size=256,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
    )
    experts = TEGroupedMLP(
        num_local_experts=2,
        config=config,
        submodules=GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        ),
        pg_collection=ProcessGroupCollection(ep=ep_group, expt_tp=expert_tp_group),
    )
    # Quantize deterministic native weights with real TE. The ordinary expert
    # preparation hooks and runtime binding still run through TEGroupedMLP.
    for linear, shape in zip((experts.linear_fc1, experts.linear_fc2), shapes):
        for e in range(2):
            linear.register_parameter(
                f"weight{e}",
                torch.nn.Parameter(
                    quantizer(_mx_dense(shape, 2 * rank + e, device))
                    if weight_format == "mxfp8"
                    else _mx_dense(shape, 2 * rank + e, device)
                ),
            )
    for parameter in experts.parameters():
        parameter.main_grad = torch.zeros(parameter.shape, dtype=torch.float32, device=device)
    runtime = ReplicaExpertRuntime(
        experts=experts,
        group=ep_group,
        num_experts=2 * size,
        num_local_home_experts=2,
        num_local_replica_slots=slots,
        transport_factory=NcclP2PTransport,
    )
    table = [[2 * ((r + 1) % size) + slot % 2 for slot in range(slots)] for r in range(size)]
    placements = torch.tensor(table, dtype=torch.int32, device=device)
    plan = ReplicaPlan(
        torch.cat((torch.arange(2 * size, device=device), placements.flatten())), placements
    )
    try:
        assert runtime.weight_format == weight_format
        assert all(len(p.virtual_weight) == slots for p in runtime.projections)
        if weight_format == "bf16":
            runtime.start_prefetch(plan)
            runtime.wait_prefetch(plan)
            inputs, outputs, expected_grads = [], [], []
            for p, shape in enumerate(shapes):
                weight = runtime.projections[p].runtime_parameters[2]
                inp = _mx_dense((16, shape[1]), 0, device).requires_grad_()
                inputs.append(inp)
                outputs.append(torch.nn.functional.linear(inp, weight))
                reference = _mx_dense(shape, table[rank][0], device)
                expected_grads.append(torch.ones_like(outputs[-1]) @ reference)
            # Reuse storage for another forward, then restore the old plan as
            # the dispatcher's backward hook does. Saved weight views must
            # observe the restored bytes without a spurious version error.
            other = torch.flip(placements, dims=(1,))
            other_plan = ReplicaPlan(None, other)
            runtime.start_prefetch(other_plan)
            runtime.wait_prefetch(other_plan)
            runtime.start_backward_prefetch(plan)
            runtime.wait_prefetch_for_backward(plan)
            sum(output.sum() for output in outputs).backward()
            for inp, expected in zip(inputs, expected_grads):
                torch.testing.assert_close(inp.grad, expected, rtol=0, atol=0)
            return
        for direction in (_WeightDirection.FORWARD, _WeightDirection.BACKWARD):
            if direction is _WeightDirection.BACKWARD:
                # FSDP recreates native transpose caches after forward. The
                # runtime must refresh its wrappers and source pointer tables.
                for parameter in runtime.source_parameters:
                    parameter.__fsdp_param__ = True
                    parameter._columnwise_data = parameter._columnwise_data.clone()
                    parameter._columnwise_scale_inv = parameter._columnwise_scale_inv.clone()
                with monkeypatch.context() as capture_patch:
                    capture_patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
                    with pytest.raises(RuntimeError, match="CUDA-graph source pointers"):
                        runtime.prepare_runtime_parameters()
            runtime.start_prefetch(plan, direction)
            runtime.wait_prefetch(plan)
            for projection in runtime.projections:
                for parameter, source in zip(projection.runtime_parameters, projection.parameters):
                    assert (
                        parameter._columnwise_data.data_ptr() == source._columnwise_data.data_ptr()
                    )
            for p, shape in enumerate(shapes):
                for slot, expert in enumerate(table[rank]):
                    actual_weight = runtime.projections[p].virtual_weight[slot]
                    reference_weight = quantizer(_mx_dense(shape, expert, device))
                    # Fprop uses rowwise weights; dgrad uses columnwise weights.
                    backward = direction is _WeightDirection.BACKWARD
                    width = shape[0] if backward else shape[1]
                    inp = quantizer(_mx_dense((128, width), 0, device))
                    kwargs = dict(
                        out_dtype=torch.bfloat16, layout="NN" if backward else "TN", grad=backward
                    )
                    actual = general_gemm(actual_weight, inp, **kwargs)[0]
                    expected = general_gemm(reference_weight, inp, **kwargs)[0]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        runtime.destroy()


def _config(group, dtype=torch.float32):
    return ReplicaTransportConfig(
        group=group,
        device=torch.device("cuda", torch.cuda.current_device()),
        world_size=dist.get_world_size(group),
        num_local_home_experts=2,
        num_local_replica_slots=4,
        member_shapes=((8, 16), (16, 4)),
        weight_format="bf16",
        rowwise_scale_shapes=None,
        columnwise_scale_shapes=None,
        grad_dtype=dtype,
        num_sms=None,
        replica_storage="layerwise",
        replica_mxfp8_storage="separate",
        replica_exchange_buffer_bytes=3072,  # Force multiple tiles, including tails.
    )


def _table(world_size, pattern):
    if pattern == "mixed":
        return [
            [2 * ((rank + 1) % world_size), 2 * rank, 2 * ((rank + 1) % world_size), -1]
            for rank in range(world_size)
        ]
    if pattern == "local":
        return [[2 * rank, 2 * rank + 1, 2 * rank, -1] for rank in range(world_size)]
    table = [[-1] * 4 for _ in range(world_size)]
    if pattern == "sparse":
        table[min(1, world_size - 1)][0] = 0
    return table


def _placement(config, table, version=1):
    return ReplicaPlacement(
        torch.tensor(table, dtype=torch.int32, device=config.device), ReplicaOwnership(2), version
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_nccl_serialized_layers_share_bounded_scratch_not_native_grads(ep_group, dtype):
    config = _config(ep_group, dtype)
    first, second = NcclP2PTransport(config), NcclP2PTransport(config)
    rank = dist.get_rank(ep_group)
    table = _table(config.world_size, "mixed")
    native = [_native_grads(config), _native_grads(config)]
    try:
        assert first._replica_chunks is second._replica_chunks
        chunks = first._replica_chunks
        assert (
            sum(
                t.numel() * t.element_size()
                for t in (chunks.send, chunks.receive, chunks.accumulate)
            )
            <= config.replica_exchange_buffer_bytes
        )
        assert not hasattr(first, "_native_grads")
        for generation, transport in enumerate((first, second, first)):
            plan = transport.prepare_plan(_placement(config, table, generation))
            producer = torch.cuda.Stream()
            producer.wait_stream(torch.cuda.current_stream())
            base = 16 * (generation + 1)
            with torch.cuda.stream(producer):
                destinations = tuple(
                    ReplicaGradDestination(tuple(v)) for v in native[generation % 2]
                )
                for p in range(2):
                    native[generation % 2][p].fill_(base)
                    transport.projection_views(p)[1].fill_(rank + generation + 1)
                handle = transport.start_grad_reduce(
                    native_grads=destinations, plan=plan, projections=(0, 1)
                )
            transport.wait_grad_reduce(handle)
            for destination in destinations:
                for expert, actual in enumerate(destination.tensors):
                    expected = base + sum(
                        source_rank + generation + 1
                        for source_rank, row in enumerate(table)
                        for logical in row
                        if logical == rank * 2 + expert
                    )
                    torch.testing.assert_close(
                        actual, torch.full_like(actual, expected), rtol=0, atol=0
                    )
        first.destroy()
        assert native[0][0].data_ptr() != native[1][0].data_ptr()
    finally:
        first.destroy()
        second.destroy()


def _native_grads(config):
    # The model/optimizer owns these destinations, not the transport.
    return tuple(
        torch.empty(
            (config.num_local_home_experts, *shape), dtype=config.grad_dtype, device=config.device
        )
        for shape in config.member_shapes
    )


def _weight(config, projection, expert, offset=0):
    shape = config.member_shapes[projection]
    return (
        torch.arange(shape[0] * shape[1], device=config.device).reshape(shape) % 7
        + expert * 8
        + projection * 64
        + offset
    ).to(torch.bfloat16)


def _sources(config, rank, offset=0):
    return tuple(
        ReplicaWeightSource(
            tuple(_weight(config, p, 2 * rank + e, offset) for e in range(2)), scales=None
        )
        for p in range(2)
    )


def _replica_grad(config, projection, rank, slot):
    shape = config.member_shapes[projection]
    return (
        torch.arange(shape[0] * shape[1], device=config.device).reshape(shape) % 4 / 16
        + 1
        + rank / 8
        + slot / 16
        + projection / 4
    ).to(config.grad_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["mixed", "local", "empty", "sparse"])
def test_nccl_weights_and_gradients_match_reference(ep_group, dtype, pattern):
    config = _config(ep_group, dtype)
    rank = dist.get_rank(ep_group)
    table = _table(config.world_size, pattern)
    transport = create_replica_weight_transport("replica_nccl", config)
    try:
        assert isinstance(transport, NcclP2PTransport)
        plan = transport.prepare_plan(_placement(config, table))
        views = [transport.projection_views(p) for p in range(2)]
        pointers = [[w.data_ptr() for w in weights] for weights, _ in views]
        for weights, _ in views:
            for weight in weights:
                weight.fill_(-17)
        # A fresh source allocation on a different producer stream must be
        # visible on every consumer stream, including repeated waits.
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            sources = _sources(config, rank)
            handle = transport.start_weight_sync(sources=sources, plan=plan)
        for _ in range(2):
            consumer = torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                transport.wait_weight_sync(handle)
                copies = [tuple(w.clone() for w in weights) for weights, _ in views]
            consumer.synchronize()
            for p in range(2):
                for slot, expert in enumerate(table[rank]):
                    expected = (
                        _weight(config, p, expert)
                        if expert >= 0
                        else torch.full_like(copies[p][slot], -17)
                    )
                    torch.testing.assert_close(copies[p][slot], expected, rtol=0, atol=0)
        transport.wait_weight_sync(handle)
        # Use distinct gradient destinations to ensure start_grad_reduce honors
        # its argument rather than silently using transport-owned native staging.
        destinations = tuple(
            ReplicaGradDestination(
                tuple(torch.full_like(w, 256 + p * 16, dtype=dtype) for w in sources[p].data)
            )
            for p in range(2)
        )
        for p, (_, grads) in enumerate(views):
            for slot in range(4):
                grads[slot].copy_(_replica_grad(config, p, rank, slot))
                if table[rank][slot] < 0:
                    grads[slot].fill_(float("nan"))
        # Native expected values start nonzero. The reference directly sums the
        # global placement, independently of backend routes or packing offsets.
        expected = [
            [w.float().clone() for w in destination.tensors] for destination in destinations
        ]
        for source_rank, row in enumerate(table):
            for slot, expert in enumerate(row):
                if expert >= 0 and expert // 2 == rank:
                    for p in range(2):
                        expected[p][expert % 2].add_(
                            _replica_grad(config, p, source_rank, slot).float()
                        )
        fc2 = transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(1,))
        transport.wait_grad_reduce(fc2)
        # FC1 must still contain native-only gradients until explicitly started.
        for tensor in destinations[0].tensors:
            torch.testing.assert_close(tensor, torch.full_like(tensor, 256), rtol=0, atol=0)
        fc1 = transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(0,))
        for _ in range(2):
            consumer = torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                transport.wait_grad_reduce(fc1)
                transport.wait_grad_reduce(fc2)
                copies = [[w.clone() for w in destination.tensors] for destination in destinations]
            consumer.synchronize()
            for p in range(2):
                for e in range(2):
                    torch.testing.assert_close(
                        copies[p][e], expected[p][e].to(dtype), rtol=0, atol=0
                    )
        for p in range(2):
            weights, _ = transport.projection_views(p)
            assert [w.data_ptr() for w in weights] == pointers[p]
            assert not hasattr(transport, "_native_grads")
    finally:
        transport.destroy()
        transport.destroy()
    # Teardown must not destroy the caller's process group.
    dist.barrier(group=ep_group)


def test_nccl_old_plan_backward_uses_current_sources(ep_group):
    config = _config(ep_group)
    rank = dist.get_rank(ep_group)
    transport = NcclP2PTransport(config)
    try:
        table = _table(config.world_size, "mixed")
        first = transport.prepare_plan(_placement(config, table))
        second = transport.prepare_plan(_placement(config, _table(config.world_size, "local"), 2))
        for plan, offset in [(first, 0), (second, 8), (first, 16)]:
            handle = transport.start_weight_sync(sources=_sources(config, rank, offset), plan=plan)
            transport.wait_weight_sync(handle)
        for p in range(2):
            for slot, expert in enumerate(table[rank]):
                if expert >= 0:
                    torch.testing.assert_close(
                        transport.projection_views(p)[0][slot],
                        _weight(config, p, expert, 16),
                        rtol=0,
                        atol=0,
                    )
    finally:
        transport.destroy()


def test_nccl_dropped_handles_and_fp32_accumulation(ep_group):
    config = _config(ep_group, torch.bfloat16)
    rank = dist.get_rank(ep_group)
    transport = NcclP2PTransport(config)
    table = [[0] * 4 for _ in range(config.world_size)]
    try:
        plan = transport.prepare_plan(_placement(config, table))
        sources = _sources(config, rank)
        source_ref = weakref.ref(sources[0].data[0])
        weights = [transport.projection_views(p)[0] for p in range(2)]
        transport.start_weight_sync(sources=sources, plan=plan)
        del sources
        gc.collect()
        # Internal keepalive protects inputs even without a caller-owned handle.
        assert source_ref() is not None
        native = _native_grads(config)
        destinations = tuple(ReplicaGradDestination(tuple(native[p])) for p in range(2))
        for p in range(2):
            native[p].fill_(256)
            transport.projection_views(p)[1].fill_(1)
        transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(1, 0))
        # Drain without waiting on either handle. Adding individual ones into
        # BF16 256 would lose every contribution; FP32 accumulation retains them.
        transport.destroy()
        for p in range(2):
            for slot in range(4):
                torch.testing.assert_close(weights[p][slot], _weight(config, p, 0), rtol=0, atol=0)
            for index, tensor in enumerate(destinations[p].tensors):
                expected = 256 + (4 * config.world_size if rank == 0 and index == 0 else 0)
                torch.testing.assert_close(
                    tensor, torch.full_like(tensor, expected), rtol=0, atol=0
                )
    finally:
        transport.destroy()


def test_nccl_validation_and_capture_with_prepared_plan(ep_group, monkeypatch):
    config = _config(ep_group)
    with pytest.raises(ValueError, match="scale shapes"):
        NcclP2PTransport(replace(config, weight_format="mxfp8"))
    with pytest.raises(ValueError, match="indexed CUDA"):
        NcclP2PTransport(replace(config, device=torch.device("cpu")))
    transport = NcclP2PTransport(config)
    try:
        placement = _placement(config, _table(config.world_size, "mixed"))
        plan = transport.prepare_plan(placement)
        sources = _sources(config, dist.get_rank(ep_group))
        with pytest.raises(ValueError, match="different transport"):
            transport.start_weight_sync(sources=sources, plan=replace(plan, transport=object()))
        with pytest.raises(ValueError, match="different transport"):
            transport.wait_weight_sync(ReplicaTransferHandle(object(), None))
        with pytest.raises(ValueError, match="plain BF16"):
            transport.start_weight_sync(
                sources=(replace(sources[0], layout=ReplicaWeightLayout.ROWWISE), sources[1]),
                plan=plan,
            )
        with pytest.raises(ValueError, match="canonical local expert"):
            transport.start_weight_sync(
                sources=(replace(sources[0], data=()), sources[1]), plan=plan
            )
        with pytest.raises(ValueError, match="dtype"):
            transport.start_weight_sync(
                sources=(
                    replace(sources[0], data=tuple(w.float() for w in sources[0].data)),
                    sources[1],
                ),
                plan=plan,
            )
        native = _native_grads(config)
        destinations = tuple(ReplicaGradDestination(tuple(native[p])) for p in range(2))
        with pytest.raises(ValueError, match="distinct"):
            transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(1, 1))
        # Mock the capture predicate so the rejection itself cannot invalidate
        # a real CUDA capture context. All operation entry points must reject,
        # including starts that reuse a schedule prepared outside capture.
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
            with pytest.raises(ValueError, match="CUDA capture"):
                transport.prepare_plan(placement)
            with pytest.raises(ValueError, match="CUDA capture"):
                transport.start_weight_sync(sources=sources, plan=plan)
            with pytest.raises(ValueError, match="CUDA capture"):
                transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(0,))
    finally:
        transport.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        transport.projection_views(0)


def _mx_config(group, dtype=torch.float32):
    # Deliberately padded, unequal directional scale extents. The transport
    # must carry the declared bytes rather than assuming weight_numel / 32.
    return replace(
        _config(group, dtype),
        weight_format="mxfp8",
        member_shapes=((32, 160), (160, 32)),
        rowwise_scale_shapes=((128, 8), (256, 4)),
        columnwise_scale_shapes=((4, 256), (8, 128)),
    )


def _mx_bytes(shape, expert, projection, component, generation, device):
    numel = 1
    for dim in shape:
        numel *= dim
    return (
        (
            (
                torch.arange(numel, device=device) * 17
                + expert * 29
                + projection * 37
                + component * 53
                + generation * 71
            )
            % 256
        )
        .to(torch.uint8)
        .view(shape)
    )


def _mx_sources(config, rank, layout, generation):
    columnwise = layout is ReplicaWeightLayout.COLUMNWISE
    shapes = config.columnwise_scale_shapes if columnwise else config.rowwise_scale_shapes
    return tuple(
        ReplicaWeightSource(
            data=tuple(
                _mx_bytes(
                    config.member_shapes[p],
                    2 * rank + e,
                    p,
                    2 * columnwise,
                    generation,
                    config.device,
                )
                for e in range(2)
            ),
            scales=tuple(
                _mx_bytes(shapes[p], 2 * rank + e, p, 2 * columnwise + 1, generation, config.device)
                for e in range(2)
            ),
            layout=layout,
        )
        for p in range(2)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["mixed", "local", "empty", "sparse"])
def test_nccl_mxfp8_bytes_scales_and_gradients(ep_group, dtype, pattern):
    config = _mx_config(ep_group, dtype)
    rank = dist.get_rank(ep_group)
    table = _table(config.world_size, pattern)
    transport = NcclP2PTransport(config)
    try:
        plan = transport.prepare_plan(_placement(config, table))
        views = [transport.projection_views(p) for p in range(2)]
        pointers = [[tuple(t.data_ptr() for t in slot) for slot in weights] for weights, _ in views]
        for weights, _ in views:
            for slot in weights:
                for tensor in slot:
                    tensor.fill_(19)
        # Reuse the plan with freshly allocated sources; preserve the other
        # direction, including when switching back to forward after backward.
        resident = {}
        for generation, layout in enumerate(
            (
                ReplicaWeightLayout.ROWWISE,
                ReplicaWeightLayout.COLUMNWISE,
                ReplicaWeightLayout.ROWWISE,
            )
        ):
            producer = torch.cuda.Stream()
            producer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(producer):
                sources = _mx_sources(config, rank, layout, generation)
                handle = transport.start_weight_sync(sources=sources, plan=plan)
            direction = int(layout is ReplicaWeightLayout.COLUMNWISE)
            resident[direction] = generation
            for _ in range(2):
                consumer = torch.cuda.Stream()
                with torch.cuda.stream(consumer):
                    transport.wait_weight_sync(handle)
                    copies = [
                        [tuple(t.clone() for t in slot) for slot in weights] for weights, _ in views
                    ]
                consumer.synchronize()
                for p in range(2):
                    for slot, expert in enumerate(table[rank]):
                        for component, actual in enumerate(copies[p][slot]):
                            direction = component // 2
                            expected = (
                                _mx_bytes(
                                    actual.shape,
                                    expert,
                                    p,
                                    component,
                                    resident[direction],
                                    config.device,
                                )
                                if expert >= 0 and direction in resident
                                else torch.full_like(actual, 19)
                            )
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            transport.wait_weight_sync(handle)
            del sources, handle
        native = _native_grads(config)
        destinations = tuple(ReplicaGradDestination(tuple(native[p])) for p in range(2))
        expected = []
        for p in range(2):
            native[p].fill_(256)
            expected.append(
                [torch.full(config.member_shapes[p], 256.0, device=config.device) for _ in range(2)]
            )
            for slot in range(4):
                views[p][1][slot].copy_(_replica_grad(config, p, rank, slot))
                if table[rank][slot] < 0:
                    views[p][1][slot].fill_(float("nan"))
            for source_rank, row in enumerate(table):
                for slot, expert in enumerate(row):
                    if expert >= 0 and expert // 2 == rank:
                        expected[p][expert % 2].add_(
                            _replica_grad(config, p, source_rank, slot).float()
                        )
        for p in (1, 0):
            transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(p,))
        assert pointers == [
            [tuple(t.data_ptr() for t in slot) for slot in transport.projection_views(p)[0]]
            for p in range(2)
        ]
        transport.destroy()  # Also verifies completion with discarded gradient handles.
        for p in range(2):
            for e in range(2):
                torch.testing.assert_close(
                    destinations[p].tensors[e], expected[p][e].to(dtype), rtol=0, atol=0
                )
    finally:
        transport.destroy()


def test_nccl_mxfp8_rejects_invalid_components(ep_group):
    config = _mx_config(ep_group)
    transport = NcclP2PTransport(config)
    try:
        plan = transport.prepare_plan(_placement(config, _table(config.world_size, "mixed")))
        sources = _mx_sources(config, dist.get_rank(ep_group), ReplicaWeightLayout.ROWWISE, 0)
        invalid = [
            replace(sources[0], scales=None),
            replace(sources[0], layout=ReplicaWeightLayout.PLAIN),
            replace(sources[0], layout=ReplicaWeightLayout.COLUMNWISE),
            replace(sources[0], scales=()),
            replace(sources[0], scales=tuple(t.float() for t in sources[0].scales)),
            replace(sources[0], scales=tuple(t.view(-1) for t in sources[0].scales)),
            replace(sources[0], data=tuple(t.float() for t in sources[0].data)),
        ]
        for source in invalid:
            with pytest.raises(ValueError):
                transport.start_weight_sync(sources=(source, sources[1]), plan=plan)
    finally:
        transport.destroy()


def _mx_dense(shape, expert, device):
    rows = torch.arange(shape[0], device=device).view(-1, 1)
    cols = torch.arange(shape[1], device=device).view(1, -1)
    return (((rows * 11 + cols * 7 + expert * 3) % 37 - 18) / 16 * (1 + expert / 8)).to(
        torch.bfloat16
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["replica_nccl", "replica_peer_tma"])
def test_gradients_follow_canonical_source_order(ep_group, dtype, backend):
    """Cancellation makes local-first reduction observably wrong, even in FP32."""
    if dist.get_world_size(ep_group) < 2:
        pytest.skip("Requires both a local and a lower-ranked remote replica.")
    config = replace(_config(ep_group, dtype), member_shapes=((16, 16), (16, 16)), num_sms=4)
    rank = dist.get_rank(ep_group)
    # Rank1 owns expert2. Rank0 has two copies, and rank1 one local copy.
    table = [[-1] * 4 for _ in range(config.world_size)]
    table[0][:2] = [2, 2]
    table[1][0] = 2
    transport = create_replica_weight_transport(backend, config)
    try:
        plan = transport.prepare_plan(_placement(config, table))
        native = _native_grads(config)
        destinations = tuple(ReplicaGradDestination(tuple(tensor)) for tensor in native)
        for projection, tensor in enumerate(native):
            tensor.fill_(7)
            if rank == 1:
                tensor[0].fill_(2**24)
            _, replicas = transport.projection_views(projection)
            replicas.fill_(float("nan"))
            if rank == 0:
                replicas[0].fill_(-(2**24))
                replicas[1].fill_(2)
            elif rank == 1:
                replicas[0].fill_(1)
        handle = transport.start_grad_reduce(
            native_grads=destinations, plan=plan, projections=(0, 1)
        )
        transport.wait_grad_reduce(handle)
        for tensor in native:
            expected = torch.full_like(tensor, 7)
            if rank == 1:
                # (2**24 - 2**24) + 2 + 1 = 3 in canonical rank/slot order.
                # The former local-first order loses the +1 and returns2.
                expected[0].fill_(3)
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    finally:
        transport.destroy()
