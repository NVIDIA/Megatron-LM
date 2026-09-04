# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-rank correctness for the virtual-expert weight-transfer kernels.

Run on one four-GPU NVLink node::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
      tests/unit_tests/transformer/moe/test_virtual_expert_triton.py

These tests cover what the kernels put on the wire. Wire *bandwidth* is not measured here;
``bench_virtual_expert_weight_sol.py`` is the benchmark of record and sweeps plan occupancy,
which is what actually moves the number.
"""

import gc
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.virtual_expert_triton import (
    MAX_VIRTUAL_EXPERT_WEIGHT_SMS,
    _transport_tile,
    _validate_transport_shape,
    compile_virtual_expert_weight_kernels,
    launch_virtual_expert_grad_reduce,
    launch_virtual_expert_weight_prefetch,
)
from tests.unit_tests.test_utilities import Utils

NUM_SMS = 4
requires_four_ranks = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available(),
    reason="Virtual-expert transport coverage requires a 4-rank torchrun launch on CUDA",
)


def test_virtual_expert_transport_shape_guards():
    """Reject launches the transport kernels cannot serve."""
    with pytest.raises(ValueError, match="limited to 32 SMs"):
        _validate_transport_shape(
            world_size=4, num_local_experts=32, num_sms=MAX_VIRTUAL_EXPERT_WEIGHT_SMS + 1
        )
    # Both projections share one row-aligned tile, so an odd member breaks it.
    with pytest.raises(ValueError, match="256-aligned"):
        _transport_tile(32768 // torch.bfloat16.itemsize, 16384, 16385)


def _allocate_symmetric(numel, dtype, group):
    """Allocate and rendezvous one native NCCL symmetric-memory tensor."""
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", torch.cuda.current_device())
    dist.barrier(group=group, device_ids=[device.index])
    if not group._get_backend(device)._comm_ptr():
        raise RuntimeError("NCCL communicator is unavailable for symmetric memory.")
    if symm_mem.get_backend(device) != "NCCL":
        symm_mem.set_backend("NCCL")
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    return tensor, symm_mem.rendezvous(tensor, group)


def _pointer_table(members):
    """Return the ``int64`` per-expert base-address table the kernels consume."""
    return torch.tensor(
        [members[index].data_ptr() for index in range(members.shape[0])],
        dtype=torch.int64,
        device=members.device,
    )


def _arena_view(arena, member_numels, scale_numels, projection, num_local_experts):
    """Return the ``[num_local_experts, numel]`` data and scale views of one projection."""
    stride = (
        member_numels
        if scale_numels is None
        else tuple(member + scale for member, scale in zip(member_numels, scale_numels))
    )
    offset = num_local_experts * sum(stride[:projection])
    member = member_numels[projection]
    data = arena.narrow(0, offset, num_local_experts * member).view(num_local_experts, member)
    if scale_numels is None:
        return data, None
    scale = arena.narrow(
        0, offset + num_local_experts * member, num_local_experts * scale_numels[projection]
    ).view(num_local_experts, scale_numels[projection])
    return data, scale


def _check_ends(view, expected_per_slot, label, errors):
    """Compare the first and last element of every slot row against its expectation."""
    for slot, expected in enumerate(expected_per_slot):
        for column, end in ((0, "head"), (-1, "tail")):
            actual = view[slot, column].item()
            if actual != expected:
                errors.append(f"{label} slot={slot} {end}: got {actual}, expected {expected}")


def _report(errors, group):
    """Fail on every rank if any rank saw a mismatch."""
    gathered = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered, errors, group=group)
    combined = [error for rank_errors in gathered for error in rank_errors]
    assert not combined, "\n".join(combined)


def _make_plan(placement, slots, world_size, num_local_experts, device):
    """Build one ``[world_size, num_local_experts]`` virtual-expert assignment."""
    plan = torch.full((world_size, num_local_experts), -1, dtype=torch.int32, device=device)
    if placement == "asymmetric":
        # Only two ranks receive anything, and only into slot 0.
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


@pytest.mark.internal
@requires_four_ranks
@pytest.mark.parametrize(
    "grad_dtype", [torch.float32, torch.bfloat16], ids=["fp32-grad", "bf16-grad"]
)
def test_virtual_expert_weight_transport(grad_dtype):
    """Push BF16 weights and reduce virtual-expert gradients over full, sparse and empty plans."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 8
    # Keep the test compact while using the same 8-KiB-aligned transactions as
    # the production 2048x640 expert projections.
    member_numels = (262144, 524288)
    arena_numel = num_local_experts * sum(member_numels)
    weight_arena, weight_handle = _allocate_symmetric(arena_numel, torch.bfloat16, group)
    grad_arena, grad_handle = _allocate_symmetric(arena_numel, grad_dtype, group)
    sources = tuple(
        torch.empty(num_local_experts, member, dtype=torch.bfloat16, device=device)
        for member in member_numels
    )
    for projection, source in enumerate(sources):
        source.copy_(
            (
                torch.arange(num_local_experts, dtype=torch.bfloat16, device=device)
                + rank * num_local_experts
                + projection * 1000
            )[:, None]
        )
    main_grads = tuple(
        torch.empty(num_local_experts, member, dtype=grad_dtype, device=device)
        for member in member_numels
    )
    weight_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    grad_barrier = torch.zeros(1, dtype=torch.int32, device=device)
    compile_virtual_expert_weight_kernels(
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        num_sms=NUM_SMS,
        device_index=device.index,
        grad_dtype=grad_dtype,
    )

    cases = (
        ("all-peers", tuple(range(num_local_experts))),
        ("ring", tuple(range(num_local_experts))),
        ("all-peers", tuple()),
        ("all-peers", (0, 3, 7)),
        ("asymmetric", (0,)),
    )
    errors = []

    def scalar(value):
        """Round a reference value the same way the kernel's store does."""
        return torch.tensor(value, dtype=grad_dtype).item()

    try:
        for placement, slots in cases:
            case = f"{placement}/{slots}"
            plan = _make_plan(placement, slots, world_size, num_local_experts, device)
            rows = plan.tolist()
            local_slots = tuple(slot for slot in range(num_local_experts) if rows[rank][slot] >= 0)

            weight_arena.fill_(-123)
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            launch_virtual_expert_weight_prefetch(
                sources=tuple(_pointer_table(source) for source in sources),
                arena=weight_arena,
                peer_bases=weight_handle.buffer_ptrs_dev,
                signal_bases=weight_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan,
                grid_barrier=weight_barrier,
                rank=rank,
                world_size=world_size,
                num_local_experts=num_local_experts,
                member_numels=member_numels,
                num_sms=NUM_SMS,
            )
            torch.cuda.synchronize(device)

            for projection in range(len(member_numels)):
                view, _ = _arena_view(
                    weight_arena, member_numels, None, projection, num_local_experts
                )
                _check_ends(
                    view,
                    [
                        torch.tensor(
                            -123 if rows[rank][slot] < 0 else projection * 1000 + rows[rank][slot],
                            dtype=torch.bfloat16,
                        ).item()
                        for slot in range(num_local_experts)
                    ],
                    f"{case} p{projection} weight",
                    errors,
                )

            grad_arena.fill_(-77)
            for projection in range(len(member_numels)):
                view, _ = _arena_view(
                    grad_arena, member_numels, None, projection, num_local_experts
                )
                for slot in local_slots:
                    view[slot].fill_(projection * 1000 + rank * 100 + slot + 1)
                main_grads[projection].fill_(projection + 5)
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            launch_virtual_expert_grad_reduce(
                arena=grad_arena,
                native_grads=tuple(_pointer_table(grad) for grad in main_grads),
                peer_bases=grad_handle.buffer_ptrs_dev,
                signal_bases=grad_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan,
                grid_barrier=grad_barrier,
                rank=rank,
                world_size=world_size,
                num_local_experts=num_local_experts,
                member_numels=member_numels,
                num_sms=NUM_SMS,
            )
            torch.cuda.synchronize(device)

            for projection in range(len(member_numels)):
                # BF16 partials accumulate in FP32 registers and round once on
                # the final store, so build the reference the same way.
                expected = torch.full(
                    (num_local_experts,), scalar(projection + 5), dtype=torch.float32, device=device
                )
                for destination in range(world_size):
                    for slot in range(num_local_experts):
                        expert = rows[destination][slot]
                        if expert // num_local_experts == rank and expert >= 0:
                            expected[expert % num_local_experts] += scalar(
                                projection * 1000 + destination * 100 + slot + 1
                            )
                try:
                    torch.testing.assert_close(
                        main_grads[projection][:, 0], expected.to(grad_dtype), rtol=0, atol=0
                    )
                except AssertionError as exc:
                    errors.append(f"{case} p{projection} main_grad: {exc}")

                # The reduction reads the slots and leaves them as they were;
                # TE's overwriting wgrad GEMM refreshes them next backward.
                view, _ = _arena_view(
                    grad_arena, member_numels, None, projection, num_local_experts
                )
                _check_ends(
                    view,
                    [
                        scalar(
                            projection * 1000 + rank * 100 + slot + 1
                            if slot in local_slots
                            else -77
                        )
                        for slot in range(num_local_experts)
                    ],
                    f"{case} p{projection} grad",
                    errors,
                )
        _report(errors, group)
    finally:
        dist.barrier(group=group, device_ids=[device.index])
        del weight_arena, grad_arena, weight_handle, grad_handle
        gc.collect()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@requires_four_ranks
def test_virtual_expert_mxfp8_transport_moves_one_orientation_at_a_time():
    """Copy MXFP8 bytes and scales exactly without touching the other GEMM orientation."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 4
    member_numels = (16384, 32768)
    scale_numels = tuple(member // 32 for member in member_numels)
    arena_numel = num_local_experts * sum(
        member + scale for member, scale in zip(member_numels, scale_numels)
    )

    arenas = {}
    handles = {}
    for orientation in ("rowwise", "columnwise"):
        arenas[orientation], handles[orientation] = _allocate_symmetric(
            arena_numel, torch.uint8, group
        )
    # Distinct byte ranges per orientation and component: any crossed wire shows
    # up as a wrong value rather than a coincidental match.
    bases = {("rowwise", "data"): 1, ("rowwise", "scale"): 65}
    bases.update({("columnwise", "data"): 129, ("columnwise", "scale"): 193})
    sources = {}
    for (orientation, kind), base in bases.items():
        numels = member_numels if kind == "data" else scale_numels
        tensors = tuple(
            torch.empty(num_local_experts, numel, dtype=torch.uint8, device=device)
            for numel in numels
        )
        for projection, tensor in enumerate(tensors):
            for expert in range(num_local_experts):
                tensor[expert].fill_(base + rank * num_local_experts + expert + 20 * projection)
        sources[(orientation, kind)] = tensors

    # Every rank materializes its right-hand neighbour's whole expert set.
    plan = torch.empty((world_size, num_local_experts), dtype=torch.int32, device=device)
    for destination in range(world_size):
        owner = (destination + 1) % world_size
        plan[destination] = torch.arange(
            owner * num_local_experts,
            (owner + 1) * num_local_experts,
            dtype=torch.int32,
            device=device,
        )
    barriers = {
        orientation: torch.zeros(1, dtype=torch.int32, device=device) for orientation in arenas
    }
    compile_virtual_expert_weight_kernels(
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        mxfp8=True,
        num_sms=NUM_SMS,
        device_index=device.index,
    )

    def launch(orientation):
        dist.barrier(group=group, device_ids=[device.index])
        launch_virtual_expert_weight_prefetch(
            sources=tuple(_pointer_table(s) for s in sources[(orientation, "data")]),
            scale_sources=tuple(_pointer_table(s) for s in sources[(orientation, "scale")]),
            arena=arenas[orientation],
            peer_bases=handles[orientation].buffer_ptrs_dev,
            signal_bases=handles[orientation].signal_pad_ptrs_dev,
            experts_to_copy=plan,
            grid_barrier=barriers[orientation],
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_numels=member_numels,
            num_sms=NUM_SMS,
        )
        torch.cuda.synchronize(device)

    def verify(orientation):
        owner = (rank + 1) % world_size
        for projection in range(len(member_numels)):
            data, scale = _arena_view(
                arenas[orientation], member_numels, scale_numels, projection, num_local_experts
            )
            experts = torch.arange(
                owner * num_local_experts,
                (owner + 1) * num_local_experts,
                dtype=torch.int64,
                device=device,
            )
            for view, kind in ((data, "data"), (scale, "scale")):
                expected = (experts + bases[(orientation, kind)] + 20 * projection).to(torch.uint8)
                for column, label in ((0, "head"), (-1, "tail")):
                    torch.testing.assert_close(
                        view[:, column],
                        expected,
                        rtol=0,
                        atol=0,
                        msg=lambda msg: f"{orientation} p{projection} {kind} {label}: {msg}",
                    )

    try:
        arenas["rowwise"].fill_(17)
        arenas["columnwise"].fill_(23)
        launch("rowwise")
        verify("rowwise")
        # Forward pushes the rowwise orientation only; the backward arena must
        # still hold its fill.
        torch.testing.assert_close(
            arenas["columnwise"], torch.full_like(arenas["columnwise"], 23), rtol=0, atol=0
        )

        rowwise_snapshot = arenas["rowwise"].clone()
        launch("columnwise")
        verify("columnwise")
        torch.testing.assert_close(arenas["rowwise"], rowwise_snapshot, rtol=0, atol=0)
    finally:
        dist.barrier(group=group, device_ids=[device.index])
        arenas.clear()
        handles.clear()
        gc.collect()
        Utils.destroy_model_parallel()
