# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Correctness and profiling coverage for the replica weight-transfer kernels.

The production-shape benchmark is opt-in because it allocates large symmetric
arenas and compiles two fixed-shape transport kernels on every rank. Run it on one
four-GPU Blackwell node with::

    MCORE_RUN_REPLICA_WEIGHT_PROFILE=1 uv run python -m torch.distributed.run \
      --nproc-per-node 4 -m pytest -q \
      tests/unit_tests/transformer/moe/test_replica_weight_triton.py

The test emits ``replica_weight_owner_push_profile`` and
``replica_grad_reduce_profile`` NVTX ranges, so the same command can be placed
after ``nsys profile -t cuda,nvtx`` for an isolated GPU trace.
"""

import gc
import json
import os
import statistics

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_weight_triton import (
    MAX_REPLICA_WEIGHT_SMS,
    _transport_tile,
    _validate_transport_shape,
    compile_replica_weight_kernels,
    launch_replica_grad_reduce,
    launch_replica_weight_prefetch,
)
from tests.unit_tests.test_utilities import Utils

_PROFILE_ENABLED = os.environ.get("MCORE_RUN_REPLICA_WEIGHT_PROFILE", "0") == "1"


def test_replica_weight_kernel_rejects_more_than_32_sms():
    """Keep replica-weight launches within their reserved SM budget."""
    with pytest.raises(ValueError, match="limited to 32 SMs"):
        _validate_transport_shape(
            world_size=4, num_local_experts=32, num_sms=MAX_REPLICA_WEIGHT_SMS + 1
        )


def test_replica_weight_kernel_rejects_nondivisible_projections():
    """Require a row-aligned transport tile shared by both projection shapes."""
    with pytest.raises(ValueError, match="256-aligned"):
        _transport_tile(32768 // torch.bfloat16.itemsize, 16384, 16385)


def _allocate_symmetric(
    numel: int, dtype: torch.dtype, group: dist.ProcessGroup
) -> tuple[torch.Tensor, object]:
    """Allocate and rendezvous one native NCCL symmetric-memory tensor."""
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", torch.cuda.current_device())
    dist.barrier(group=group, device_ids=[device.index])
    backend = group._get_backend(device)
    if not backend._comm_ptr():
        raise RuntimeError("NCCL communicator is unavailable for symmetric memory.")
    if symm_mem.get_backend(device) != "NCCL":
        symm_mem.set_backend("NCCL")
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    return tensor, symm_mem.rendezvous(tensor, group)


def _pointer_table(members: torch.Tensor) -> torch.Tensor:
    """Return the ``int64`` per-expert base-address table the kernels consume."""
    return torch.tensor(
        [members[index].data_ptr() for index in range(members.shape[0])],
        dtype=torch.int64,
        device=members.device,
    )


def _gather_samples(samples: list[float], group: dist.ProcessGroup) -> list[float]:
    """Gather fixed-size CUDA-event samples from every rank."""
    device_samples = torch.tensor(samples, dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(device_samples) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, device_samples, group=group)
    return torch.cat(gathered).cpu().tolist()


def _summarize(samples: list[float]) -> dict[str, float]:
    """Return stable benchmark summary statistics in milliseconds."""
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
    }


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "grad_dtype", [torch.float32, torch.bfloat16], ids=["fp32-grad", "bf16-grad"]
)
def test_replica_weight_kernels_virtual_only_cases(grad_dtype):
    """Cover owner-push, sparse plans, zero work, and unequal FC shapes."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Replica weight kernel coverage requires a 4-rank torchrun launch")

    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 8
    # Keep the test compact while using the same 8-KiB-aligned E8M0 scale
    # transactions as the production 2048x640 expert projections.
    member_numels = (262144, 524288)
    arena_numel = num_local_experts * sum(member_numels)
    weight_storage, weight_handle = _allocate_symmetric(arena_numel, torch.bfloat16, group)
    grad_storage, grad_handle = _allocate_symmetric(arena_numel, grad_dtype, group)
    weight_arena = weight_storage
    grad_arena = grad_storage
    sources = tuple(
        torch.empty(num_local_experts, member, dtype=torch.bfloat16, device=device)
        for member in member_numels
    )
    for projection, source in enumerate(sources):
        values = (
            torch.arange(num_local_experts, dtype=torch.bfloat16, device=device)
            + rank * num_local_experts
            + projection * 1000
        )
        source.copy_(values[:, None])
    main_grads = tuple(
        torch.empty(num_local_experts, member, dtype=grad_dtype, device=device)
        for member in member_numels
    )
    weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    compile_replica_weight_kernels(
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        num_sms=4,
        device_index=device.index,
        grad_dtype=grad_dtype,
    )

    def make_plan(placement: str, slots: tuple[int, ...]) -> torch.Tensor:
        plan = torch.full((world_size, num_local_experts), -1, dtype=torch.int32, device=device)
        if placement == "asymmetric":
            plan[0, slots[0]] = num_local_experts
            plan[1, slots[0]] = 2 * num_local_experts
            return plan
        for destination in range(world_size):
            peers = [peer for peer in range(world_size) if peer != destination]
            for ordinal, slot in enumerate(slots):
                owner = (
                    (destination + 1) % world_size
                    if placement == "ring"
                    else peers[ordinal % len(peers)]
                )
                plan[destination, slot] = owner * num_local_experts + slot
        return plan

    cases = (
        ("all-peers", tuple(range(num_local_experts))),
        ("ring", tuple(range(num_local_experts))),
        ("all-peers", tuple()),
        ("all-peers", (0, 3, 7)),
        ("asymmetric", (0,)),
    )
    errors = []

    def check_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
        try:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        except AssertionError as exc:
            errors.append(f"{label}: {exc}")

    try:
        for placement, slots in cases:
            plan = make_plan(placement, slots)
            local_slots = tuple(
                slot for slot in range(num_local_experts) if int(plan[rank, slot]) >= 0
            )
            weight_storage.fill_(-123)
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            launch_replica_weight_prefetch(
                sources=tuple(_pointer_table(source) for source in sources),
                arena=weight_arena,
                peer_bases=weight_handle.buffer_ptrs_dev,
                signal_bases=weight_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan,
                grid_barrier=weight_grid_barrier,
                rank=rank,
                world_size=world_size,
                num_local_experts=num_local_experts,
                member_numels=member_numels,
                num_sms=4,
            )
            torch.cuda.synchronize(device)

            grad_storage.fill_(-77)
            for projection, member in enumerate(member_numels):
                offset = num_local_experts * sum(member_numels[:projection])
                view = grad_arena.narrow(0, offset, num_local_experts * member).view(
                    num_local_experts, member
                )
                for slot in local_slots:
                    view[slot].fill_(projection * 1000 + rank * 100 + slot + 1)
                main_grads[projection].fill_(projection + 5)
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            launch_replica_grad_reduce(
                arena=grad_arena,
                native_grads=tuple(_pointer_table(grad) for grad in main_grads),
                peer_bases=grad_handle.buffer_ptrs_dev,
                signal_bases=grad_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan,
                grid_barrier=grad_grid_barrier,
                rank=rank,
                world_size=world_size,
                num_local_experts=num_local_experts,
                member_numels=member_numels,
                num_sms=4,
            )
            torch.cuda.synchronize(device)
            plan_rows = plan.tolist()
            for projection, member in enumerate(member_numels):
                expected_fp32 = torch.full(
                    (num_local_experts,),
                    torch.tensor(projection + 5, dtype=grad_dtype).item(),
                    dtype=torch.float32,
                    device=device,
                )
                for local_expert in range(num_local_experts):
                    semantic_expert = rank * num_local_experts + local_expert
                    for destination in range(world_size):
                        for slot in range(num_local_experts):
                            if plan_rows[destination][slot] == semantic_expert:
                                expected_fp32[local_expert] += torch.tensor(
                                    projection * 1000 + destination * 100 + slot + 1,
                                    dtype=grad_dtype,
                                    device=device,
                                )
                expected = expected_fp32.to(grad_dtype)
                check_close(
                    main_grads[projection][:, 0],
                    expected,
                    f"{placement}/{slots} projection {projection} main_grad",
                )
                offset = num_local_experts * sum(member_numels[:projection])
                view = grad_arena.narrow(0, offset, num_local_experts * member).view(
                    num_local_experts, member
                )
                for slot in range(num_local_experts):
                    # The reduction reads the slots and leaves them as they were;
                    # TE's overwriting wgrad GEMM refreshes them on the next backward.
                    expected_slot = torch.tensor(
                        projection * 1000 + rank * 100 + slot + 1 if slot in local_slots else -77,
                        dtype=grad_dtype,
                    ).item()
                    if view[slot, 0].item() != expected_slot:
                        errors.append(
                            f"{placement}/{slots} projection {projection} slot {slot} "
                            f"gradient start={view[slot, 0].item()} expected={expected_slot}"
                        )
                    if view[slot, -1].item() != expected_slot:
                        errors.append(
                            f"{placement}/{slots} projection {projection} slot {slot} "
                            f"gradient end={view[slot, -1].item()} expected={expected_slot}"
                        )
            for projection, member in enumerate(member_numels):
                offset = num_local_experts * sum(member_numels[:projection])
                view = weight_arena.narrow(0, offset, num_local_experts * member).view(
                    num_local_experts, member
                )
                for slot in range(num_local_experts):
                    semantic_expert = int(plan[rank, slot])
                    expected = -123 if semantic_expert < 0 else projection * 1000 + semantic_expert
                    expected = torch.tensor(expected, dtype=torch.bfloat16).item()
                    if view[slot, 0].item() != expected:
                        errors.append(
                            f"{placement}/{slots} projection {projection} slot {slot} "
                            f"weight start={view[slot, 0].item()} expected={expected}"
                        )
                    if view[slot, -1].item() != expected:
                        errors.append(
                            f"{placement}/{slots} projection {projection} slot {slot} "
                            f"weight end={view[slot, -1].item()} expected={expected}"
                        )
        gathered_errors = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_errors, errors, group=group)
        all_errors = [error for rank_errors in gathered_errors for error in rank_errors]
        assert not all_errors, "\n".join(all_errors)
    finally:
        dist.barrier(group=group, device_ids=[device.index])
        del weight_arena, grad_arena, weight_handle, grad_handle
        del weight_storage, grad_storage
        gc.collect()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_replica_mxfp8_weight_kernel_copies_data_and_scales_by_orientation():
    """Copy MXFP8 bytes/scales exactly without touching the other GEMM orientation."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Replica MXFP8 kernel coverage requires a 4-rank torchrun launch")

    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 4
    member_numels = (16384, 32768)
    rowwise_scale_numels = tuple(member // 32 for member in member_numels)
    columnwise_scale_numels = rowwise_scale_numels

    def allocate_orientation(scale_numels):
        numel = num_local_experts * sum(
            member + scale for member, scale in zip(member_numels, scale_numels)
        )
        return _allocate_symmetric(numel, torch.uint8, group)

    rowwise_arena, rowwise_handle = allocate_orientation(rowwise_scale_numels)
    columnwise_arena, columnwise_handle = allocate_orientation(columnwise_scale_numels)
    rowwise_data = tuple(
        torch.empty(num_local_experts, member, dtype=torch.uint8, device=device)
        for member in member_numels
    )
    columnwise_data = tuple(torch.empty_like(source) for source in rowwise_data)
    rowwise_scales = tuple(
        torch.empty(num_local_experts, scale, dtype=torch.uint8, device=device)
        for scale in rowwise_scale_numels
    )
    columnwise_scales = tuple(torch.empty_like(source) for source in rowwise_scales)
    for projection in range(2):
        for expert in range(num_local_experts):
            semantic_expert = rank * num_local_experts + expert
            rowwise_data[projection][expert].fill_(semantic_expert + 1 + 20 * projection)
            rowwise_scales[projection][expert].fill_(semantic_expert + 65 + 20 * projection)
            columnwise_data[projection][expert].fill_(semantic_expert + 129 + 20 * projection)
            columnwise_scales[projection][expert].fill_(semantic_expert + 193 + 20 * projection)

    experts_to_copy = torch.full(
        (world_size, num_local_experts), -1, dtype=torch.int32, device=device
    )
    for destination in range(world_size):
        owner = (destination + 1) % world_size
        experts_to_copy[destination] = torch.arange(
            owner * num_local_experts,
            (owner + 1) * num_local_experts,
            dtype=torch.int32,
            device=device,
        )
    rowwise_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    columnwise_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    compile_replica_weight_kernels(
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        mxfp8=True,
        num_sms=4,
        device_index=device.index,
    )

    def launch(orientation):
        dist.barrier(group=group, device_ids=[device.index])
        rowwise = orientation == "rowwise"
        arena = rowwise_arena if rowwise else columnwise_arena
        handle = rowwise_handle if rowwise else columnwise_handle
        launch_replica_weight_prefetch(
            sources=tuple(
                _pointer_table(source) for source in (rowwise_data if rowwise else columnwise_data)
            ),
            scale_sources=tuple(
                _pointer_table(source)
                for source in (rowwise_scales if rowwise else columnwise_scales)
            ),
            arena=arena,
            peer_bases=handle.buffer_ptrs_dev,
            signal_bases=handle.signal_pad_ptrs_dev,
            experts_to_copy=experts_to_copy,
            grid_barrier=rowwise_barrier if rowwise else columnwise_barrier,
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_numels=member_numels,
            num_sms=4,
        )

    def projection_views(arena, scale_numels, projection):
        projection_offset = num_local_experts * sum(
            member + scale
            for member, scale in zip(member_numels[:projection], scale_numels[:projection])
        )
        data = arena.narrow(
            0, projection_offset, num_local_experts * member_numels[projection]
        ).view(num_local_experts, member_numels[projection])
        scale = arena.narrow(
            0,
            projection_offset + num_local_experts * member_numels[projection],
            num_local_experts * scale_numels[projection],
        ).view(num_local_experts, scale_numels[projection])
        return data, scale

    try:
        rowwise_arena.fill_(17)
        columnwise_arena.fill_(23)
        launch("rowwise")
        torch.cuda.synchronize(device)
        owner = (rank + 1) % world_size
        for projection in range(2):
            data, scale = projection_views(rowwise_arena, rowwise_scale_numels, projection)
            expected_data = torch.arange(
                owner * num_local_experts + 1 + 20 * projection,
                (owner + 1) * num_local_experts + 1 + 20 * projection,
                dtype=torch.uint8,
                device=device,
            )
            expected_scale = expected_data + 64
            torch.testing.assert_close(data[:, 0], expected_data, rtol=0, atol=0)
            torch.testing.assert_close(data[:, -1], expected_data, rtol=0, atol=0)
            torch.testing.assert_close(scale[:, 0], expected_scale, rtol=0, atol=0)
            torch.testing.assert_close(scale[:, -1], expected_scale, rtol=0, atol=0)
        torch.testing.assert_close(
            columnwise_arena, torch.full_like(columnwise_arena, 23), rtol=0, atol=0
        )

        rowwise_snapshot = rowwise_arena.clone()
        launch("columnwise")
        torch.cuda.synchronize(device)
        torch.testing.assert_close(rowwise_arena, rowwise_snapshot, rtol=0, atol=0)
        for projection in range(2):
            data, scale = projection_views(columnwise_arena, columnwise_scale_numels, projection)
            expected_data = torch.arange(
                owner * num_local_experts + 129 + 20 * projection,
                (owner + 1) * num_local_experts + 129 + 20 * projection,
                dtype=torch.uint8,
                device=device,
            )
            expected_scale = expected_data + 64
            torch.testing.assert_close(data[:, 0], expected_data, rtol=0, atol=0)
            torch.testing.assert_close(data[:, -1], expected_data, rtol=0, atol=0)
            torch.testing.assert_close(scale[:, 0], expected_scale, rtol=0, atol=0)
            torch.testing.assert_close(scale[:, -1], expected_scale, rtol=0, atol=0)
    finally:
        dist.barrier(group=group, device_ids=[device.index])
        del rowwise_handle, columnwise_handle, rowwise_arena, columnwise_arena
        gc.collect()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not _PROFILE_ENABLED, reason="set MCORE_RUN_REPLICA_WEIGHT_PROFILE=1")
def test_replica_mxfp8_weight_kernel_production_bandwidth():
    """Require production-shape MXFP8 owner-push to approach NVLink bandwidth."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Replica MXFP8 profiling requires a 4-rank torchrun launch")

    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = int(os.environ.get("MCORE_REPLICA_WEIGHT_LOCAL_EXPERTS", "32"))
    member_numels = (
        int(os.environ.get("MCORE_REPLICA_WEIGHT_FC1_NUMEL", str(2048 * 640))),
        int(os.environ.get("MCORE_REPLICA_WEIGHT_FC2_NUMEL", str(2048 * 640))),
    )
    scale_numels = tuple(member // 32 for member in member_numels)
    num_sms = int(os.environ.get("MCORE_REPLICA_WEIGHT_NUM_SMS", "32"))
    active_slots = int(os.environ.get("MCORE_REPLICA_WEIGHT_ACTIVE_SLOTS", str(num_local_experts)))
    warmups = int(os.environ.get("MCORE_REPLICA_WEIGHT_WARMUPS", "3"))
    iterations = int(os.environ.get("MCORE_REPLICA_WEIGHT_ITERATIONS", "10"))
    batches = int(os.environ.get("MCORE_REPLICA_WEIGHT_BATCHES", "5"))
    minimum_gbps = float(os.environ.get("MCORE_REPLICA_WEIGHT_MIN_GBPS", "800"))

    arena_numel = num_local_experts * sum(
        member + scale for member, scale in zip(member_numels, scale_numels)
    )
    arena, handle = _allocate_symmetric(arena_numel, torch.uint8, group)
    data_sources = tuple(
        torch.empty(num_local_experts, member, dtype=torch.uint8, device=device)
        for member in member_numels
    )
    scale_sources = tuple(
        torch.empty(num_local_experts, scale, dtype=torch.uint8, device=device)
        for scale in scale_numels
    )
    for projection in range(2):
        for expert in range(num_local_experts):
            value = rank * num_local_experts + expert + projection * 97
            data_sources[projection][expert].fill_(value % 256)
            scale_sources[projection][expert].fill_((value + 41) % 256)
    # The production bridge binds these device tables once. Keep the benchmark
    # on that steady-state path instead of timing four tiny host-to-device table
    # constructions on every owner-push launch.
    data_source_bases = tuple(_pointer_table(source) for source in data_sources)
    scale_source_bases = tuple(_pointer_table(source) for source in scale_sources)

    experts_to_copy = torch.full(
        (world_size, num_local_experts), -1, dtype=torch.int32, device=device
    )
    for destination in range(world_size):
        owner = (destination + 1) % world_size
        experts_to_copy[destination, :active_slots] = torch.arange(
            owner * num_local_experts,
            owner * num_local_experts + active_slots,
            dtype=torch.int32,
            device=device,
        )
    grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    compile_replica_weight_kernels(
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        mxfp8=True,
        num_sms=num_sms,
        device_index=device.index,
    )

    def prefetch() -> None:
        torch.cuda.nvtx.range_push("replica_mxfp8_weight_owner_push_profile")
        launch_replica_weight_prefetch(
            sources=data_source_bases,
            scale_sources=scale_source_bases,
            arena=arena,
            peer_bases=handle.buffer_ptrs_dev,
            signal_bases=handle.signal_pad_ptrs_dev,
            experts_to_copy=experts_to_copy,
            grid_barrier=grid_barrier,
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_numels=member_numels,
            num_sms=num_sms,
        )
        torch.cuda.nvtx.range_pop()

    try:
        for _ in range(warmups):
            prefetch()
        torch.cuda.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        samples = []
        for _ in range(batches):
            # Enqueue a batch so rank-local Python launch skew is paid once,
            # while every kernel's device-side completion barrier keeps the
            # steady-state transport sequence aligned across ranks.
            dist.barrier(group=group, device_ids=[device.index])
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                prefetch()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iterations)

        owner = (rank + 1) % world_size
        for projection, (member, scale) in enumerate(zip(member_numels, scale_numels)):
            projection_offset = num_local_experts * sum(
                data_bytes + scale_bytes
                for data_bytes, scale_bytes in zip(
                    member_numels[:projection], scale_numels[:projection]
                )
            )
            data_view = arena.narrow(0, projection_offset, num_local_experts * member).view(
                num_local_experts, member
            )
            scale_view = arena.narrow(
                0, projection_offset + num_local_experts * member, num_local_experts * scale
            ).view(num_local_experts, scale)
            expected = torch.arange(
                owner * num_local_experts + projection * 97,
                (owner + 1) * num_local_experts + projection * 97,
                dtype=torch.int64,
                device=device,
            ).to(torch.uint8)
            if active_slots:
                torch.testing.assert_close(
                    data_view[:active_slots, -1], expected[:active_slots], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    scale_view[:active_slots, -1], expected[:active_slots] + 41, rtol=0, atol=0
                )

        gathered_samples = _gather_samples(samples, group)
        median_ms = statistics.median(gathered_samples)
        payload_bytes = active_slots * sum(
            member + scale for member, scale in zip(member_numels, scale_numels)
        )
        # Every payload byte is read from owner-local HBM and written across
        # NVLink to the peer arena. Report the same aggregate wire traffic
        # convention as the existing BF16 benchmark (read + remote write).
        effective_gbps = 2 * payload_bytes / (median_ms * 1.0e6)
        if rank == 0:
            result = {
                "shape": {
                    "world_size": world_size,
                    "num_local_experts": num_local_experts,
                    "active_slots": active_slots,
                    "member_numels": member_numels,
                    "scale_numels": scale_numels,
                    "num_sms": num_sms,
                    "payload_bytes_per_rank": payload_bytes,
                },
                "prefetch_ms": _summarize(gathered_samples),
                "effective_gbps": effective_gbps,
                "minimum_gbps": minimum_gbps,
            }
            print("REPLICA_MXFP8_WEIGHT_PROFILE=" + json.dumps(result, sort_keys=True), flush=True)
        assert effective_gbps >= minimum_gbps, (
            f"MXFP8 replica prefetch achieved {effective_gbps:.1f} GB/s, "
            f"below the {minimum_gbps:.1f} GB/s target"
        )
    finally:
        dist.barrier(group=group, device_ids=[device.index])
        del handle, arena
        gc.collect()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not _PROFILE_ENABLED, reason="set MCORE_RUN_REPLICA_WEIGHT_PROFILE=1")
def test_replica_weight_kernels_production_profile():
    """Profile correct prefetch and grad-reduce results at the production shape."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Replica weight profiling requires a 4-rank torchrun launch")

    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = int(os.environ.get("MCORE_REPLICA_WEIGHT_LOCAL_EXPERTS", "32"))
    member_numels = (
        int(os.environ.get("MCORE_REPLICA_WEIGHT_FC1_NUMEL", str(2048 * 640))),
        int(os.environ.get("MCORE_REPLICA_WEIGHT_FC2_NUMEL", str(2048 * 640))),
    )
    num_sms = int(os.environ.get("MCORE_REPLICA_WEIGHT_NUM_SMS", "32"))
    active_slots = int(os.environ.get("MCORE_REPLICA_WEIGHT_ACTIVE_SLOTS", str(num_local_experts)))
    warmups = int(os.environ.get("MCORE_REPLICA_WEIGHT_WARMUPS", "3"))
    iterations = int(os.environ.get("MCORE_REPLICA_WEIGHT_ITERATIONS", "10"))
    grad_dtype_name = os.environ.get("MCORE_REPLICA_WEIGHT_GRAD_DTYPE", "fp32")
    grad_dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16}
    if grad_dtype_name not in grad_dtypes:
        raise ValueError(
            "MCORE_REPLICA_WEIGHT_GRAD_DTYPE must be 'fp32' or 'bf16', " f"got {grad_dtype_name!r}."
        )
    grad_dtype = grad_dtypes[grad_dtype_name]
    if not 0 <= active_slots <= num_local_experts:
        raise ValueError(f"active slots must be in [0, {num_local_experts}], got {active_slots}.")

    arena_numel = num_local_experts * sum(member_numels)
    weight_arena, weight_handle = _allocate_symmetric(arena_numel, torch.bfloat16, group)
    grad_arena, grad_handle = _allocate_symmetric(arena_numel, grad_dtype, group)
    weight_arena.fill_(-123)
    sources = tuple(
        torch.empty(num_local_experts, member, dtype=torch.bfloat16, device=device)
        for member in member_numels
    )
    for projection, source in enumerate(sources):
        for expert in range(num_local_experts):
            source[expert].fill_(projection * 1000 + rank * num_local_experts + expert)
    source_bases = tuple(_pointer_table(source) for source in sources)

    experts_to_copy = torch.full(
        (world_size, num_local_experts), -1, dtype=torch.int32, device=device
    )
    for destination in range(world_size):
        owner = (destination + 1) % world_size
        experts_to_copy[destination, :active_slots] = torch.arange(
            owner * num_local_experts,
            owner * num_local_experts + active_slots,
            dtype=torch.int32,
            device=device,
        )
    weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    main_grads = tuple(
        torch.zeros(num_local_experts, member, dtype=grad_dtype, device=device)
        for member in member_numels
    )
    main_grad_bases = tuple(_pointer_table(grad) for grad in main_grads)
    compile_replica_weight_kernels(
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        num_sms=num_sms,
        device_index=device.index,
        grad_dtype=grad_dtype,
    )

    def prefetch() -> None:
        torch.cuda.nvtx.range_push("replica_weight_owner_push_profile")
        launch_replica_weight_prefetch(
            sources=source_bases,
            arena=weight_arena,
            peer_bases=weight_handle.buffer_ptrs_dev,
            signal_bases=weight_handle.signal_pad_ptrs_dev,
            experts_to_copy=experts_to_copy,
            grid_barrier=weight_grid_barrier,
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_numels=member_numels,
            num_sms=num_sms,
        )
        torch.cuda.nvtx.range_pop()

    for _ in range(warmups):
        prefetch()
    torch.cuda.synchronize(device)
    dist.barrier(group=group, device_ids=[device.index])
    prefetch_samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prefetch()
        end.record()
        end.synchronize()
        prefetch_samples.append(start.elapsed_time(end))

    for projection, member in enumerate(member_numels):
        offset = num_local_experts * sum(member_numels[:projection])
        view = weight_arena.narrow(0, offset, num_local_experts * member).view(
            num_local_experts, member
        )
        if active_slots:
            owner = (rank + 1) % world_size
            replica_expected = torch.arange(
                owner * num_local_experts,
                owner * num_local_experts + active_slots,
                dtype=torch.bfloat16,
                device=device,
            ) + (projection * 1000)
            torch.testing.assert_close(view[:active_slots, -1], replica_expected)
        if active_slots < num_local_experts:
            torch.testing.assert_close(
                view[active_slots:, 0], torch.full_like(view[active_slots:, 0], -123)
            )

    def prepare_grads() -> None:
        for projection, member in enumerate(member_numels):
            offset = num_local_experts * sum(member_numels[:projection])
            view = grad_arena.narrow(0, offset, num_local_experts * member).view(
                num_local_experts, member
            )
            view.zero_()
            view[:active_slots].fill_(2 + projection)
            main_grads[projection].fill_(1 + projection)

    def grad_reduce() -> None:
        torch.cuda.nvtx.range_push("replica_grad_reduce_profile")
        launch_replica_grad_reduce(
            arena=grad_arena,
            native_grads=main_grad_bases,
            peer_bases=grad_handle.buffer_ptrs_dev,
            signal_bases=grad_handle.signal_pad_ptrs_dev,
            experts_to_copy=experts_to_copy,
            grid_barrier=grad_grid_barrier,
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_numels=member_numels,
            num_sms=num_sms,
        )
        torch.cuda.nvtx.range_pop()

    grad_samples = []
    for iteration in range(warmups + iterations):
        prepare_grads()
        torch.cuda.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        grad_reduce()
        end.record()
        end.synchronize()
        if iteration >= warmups:
            grad_samples.append(start.elapsed_time(end))

    for projection, main_grad in enumerate(main_grads):
        expected = torch.full((num_local_experts,), 1 + projection, dtype=grad_dtype, device=device)
        expected[:active_slots].add_(2 + projection)
        torch.testing.assert_close(main_grad[:, 0], expected)

    prefetch_samples = _gather_samples(prefetch_samples, group)
    grad_samples = _gather_samples(grad_samples, group)
    if rank == 0:
        result = {
            "shape": {
                "world_size": world_size,
                "num_local_experts": num_local_experts,
                "member_numels": member_numels,
                "active_slots": active_slots,
                "num_sms": num_sms,
                "grad_dtype": grad_dtype_name,
            },
            "prefetch_ms": _summarize(prefetch_samples),
            "grad_reduce_ms": _summarize(grad_samples),
        }
        print("REPLICA_WEIGHT_PROFILE=" + json.dumps(result, sort_keys=True), flush=True)

    dist.barrier(group=group, device_ids=[device.index])
    del weight_handle, grad_handle, weight_arena, grad_arena
    gc.collect()
    Utils.destroy_model_parallel()
