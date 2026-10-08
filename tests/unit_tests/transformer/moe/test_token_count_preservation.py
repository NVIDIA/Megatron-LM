# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Focused regression tests for the masked-routing local token count (A3).

What is being protected
-----------------------
``get_tokens_per_expert_and_token_count(..., with_padding_mask=True)`` returns

    (global_tokens_per_expert, local_num_tokens, total_num_tokens)

where ``local_num_tokens`` must be THIS RANK's valid (non-padding) token count and
``total_num_tokens`` the group-global valid token count.

``local_tokens_per_expert = routing_map.sum(dim=0)`` is contiguous, and
``reduce_from_tensor_model_parallel_region`` -> ``_reduce()`` all-reduces a contiguous
input IN PLACE (see megatron/core/tensor_parallel/mappings.py). Deriving
``local_num_tokens`` from that same buffer after the collective therefore yields the
group-global count, so the two return values silently collapse into one and the
per-token-loss scaling in ``Router.attach_and_log_load_balancing_loss``
(``aux_loss * num_local_tokens * tp_cp_group.size()``) is wrong by the group size.

These tests use a real NCCL process group. Run them with torchrun, e.g.

    torchrun --standalone --nproc_per_node=2 -m pytest -q \
        tests/unit_tests/transformer/moe/test_token_count_preservation.py
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_utils import get_tokens_per_expert_and_token_count

pytestmark = pytest.mark.internal

NUM_EXPERTS = 8
PAD_ROWS = 2


def _build_routing_map(valid_rows: int, topk: int, seed: int) -> torch.Tensor:
    """[valid_rows + PAD_ROWS, NUM_EXPERTS] bool with exactly `topk` distinct experts/row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for _ in range(valid_rows):
        picked = torch.randperm(NUM_EXPERTS, generator=g)[:topk]
        row = torch.zeros(NUM_EXPERTS, dtype=torch.bool)
        row[picked] = True
        rows.append(row)
    valid = torch.stack(rows) if rows else torch.zeros((0, NUM_EXPERTS), dtype=torch.bool)
    return torch.cat([valid, torch.zeros((PAD_ROWS, NUM_EXPERTS), dtype=torch.bool)], dim=0)


@pytest.fixture(scope="module")
def _nccl_group(nccl_session):
    """Reuse the session-wide NCCL group.

    This fixture must not destroy the group: the module conftest's session-scoped cleanup
    owns that, and a module-scoped destroy races the next module's init.
    """
    if nccl_session < 2:
        pytest.skip("needs at least 2 ranks to observe the aliasing")
    return nccl_session


def _valid_rows_per_rank(world: int) -> list[int]:
    """A distribution where no two ranks share the same valid token count."""
    plans = {
        2: [3, 1],
        3: [7, 2, 4],
        4: [6, 0, 3, 5],
        8: [6, 0, 3, 5, 9, 1, 4, 2],
    }
    if world not in plans:
        pytest.skip(f"no rank plan for world_size={world}")
    return plans[world]


def _check_case(topk: int, valid_rows: list[int], with_padding_mask: bool, seed: int) -> None:
    """Compare the helper against an independent oracle on every rank."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_valid = valid_rows[rank]
    # rank-specific routing so no rank can pass by accident from a shared fixture
    seed = valid_rows[rank] * 1000 + topk + (0 if with_padding_mask else 7)

    routing_map = _build_routing_map(local_valid, topk, seed).cuda()

    global_hist, local_num, total_num = get_tokens_per_expert_and_token_count(
        routing_map=routing_map,
        reduce_group=dist.group.WORLD,
        topk=topk,
        with_padding_mask=with_padding_mask,
    )

    # independent oracle: local histogram, then an explicit all-reduce of our own buffer
    expected_local_hist = routing_map.sum(dim=0).to(torch.int64)
    expected_global_hist = expected_local_hist.clone()
    dist.all_reduce(expected_global_hist)

    assert torch.equal(global_hist.to(torch.int64), expected_global_hist), (
        f"global histogram mismatch on rank {rank}: "
        f"{global_hist.tolist()} != {expected_global_hist.tolist()}"
    )

    # current main floors the masked division (`// topk`, #7183 int64 for the fused aux
    # loss), so the counts are compared as integers; the unmasked branch keeps its previous
    # definition exactly.
    if with_padding_mask:
        assert int(local_num) == local_valid, (
            f"local token count is not rank-local on rank {rank}: "
            f"got {int(local_num)}, expected {local_valid} "
            f"(group-global would be {int(expected_global_hist.sum()) // topk})"
        )
        assert int(total_num) == int(expected_global_hist.sum()) // topk
    else:
        physical_rows = routing_map.shape[0]
        assert float(local_num) == physical_rows, (
            f"unmasked local count changed on rank {rank}: {float(local_num)} != {physical_rows}"
        )
        assert float(total_num) == physical_rows * world


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_masked_local_count_uneven(_nccl_group, topk):
    """Uneven valid rows per rank: local must stay rank-local, global must stay global."""
    _check_case(topk, _valid_rows_per_rank(_nccl_group), with_padding_mask=True, seed=1)


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_masked_local_count_zero_rank(_nccl_group, topk):
    """A rank with zero valid rows must report 0 locally, not the group total."""
    world = _nccl_group
    rows = _valid_rows_per_rank(world)
    if 0 not in rows:
        pytest.skip(f"rank plan for world_size={world} has no zero-load rank")
    _check_case(topk, rows, with_padding_mask=True, seed=2)


@pytest.mark.parametrize("topk", [1, 2])
def test_unmasked_count_unchanged(_nccl_group, topk):
    """with_padding_mask=False keeps its original definition: physical rows."""
    _check_case(topk, _valid_rows_per_rank(_nccl_group), with_padding_mask=False, seed=3)


@pytest.mark.parametrize("topk", [2])
def test_helper_all_zero_counts(_nccl_group, topk):
    """All ranks with only padding rows: helper returns zero counts, no NaN/fake epsilon."""
    rank = dist.get_rank()
    routing_map = _build_routing_map(0, topk, seed=99 + rank).cuda()
    global_hist, local_num, total_num = get_tokens_per_expert_and_token_count(
        routing_map=routing_map,
        reduce_group=dist.group.WORLD,
        topk=topk,
        with_padding_mask=True,
    )
    assert float(local_num) == 0.0
    assert float(total_num) == 0.0
    assert int(global_hist.sum().item()) == 0


def test_local_count_storage_is_independent():
    """The local scalar must not be a view of the histogram that the collective rewrites.

    Pure bookkeeping check with no collective: reading the local count before the reduce
    is only meaningful if it materialises its own storage.
    """
    routing_map = _build_routing_map(5, 2, seed=7)
    local_hist = routing_map.sum(dim=0)
    local_scalar = local_hist.sum() / 2
    expected = float(local_scalar)
    local_hist.add_(1000)  # simulate an in-place all-reduce
    assert float(local_scalar) == expected, "local scalar aliases the histogram buffer"
