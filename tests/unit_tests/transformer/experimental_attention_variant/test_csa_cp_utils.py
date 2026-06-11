# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_TWO_CHUNK,
    build_global_compressed_cu_seqlens,
    exchange_cp_boundary_hidden,
    local_kv_cp_chunk_ranges,
    local_q_cp_chunk_ranges,
)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 CP CUDA utility tests require CUDA.")


def test_csa_cp_partition_mode_contract():
    """Validate the public CSA CP partition-mode names.

    Expected: omitted mode defaults to the original contiguous partition,
    supported strings round-trip exactly, and unknown values fail before any CP
    layout helper can silently choose the wrong row order.
    """
    assert local_q_cp_chunk_ranges(None, 4, 1, 0) == ((0, 4),)
    assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_CONTIGUOUS, 4, 1, 0) == ((0, 4),)
    assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 4, 1, 0) == ((0, 4),)

    with pytest.raises(RuntimeError, match="Unsupported CSA CP partition mode"):
        local_q_cp_chunk_ranges("invalid_mode", 4, 1, 0)


def test_two_chunk_cp_ranges_match_expected_order():
    """Validate the two-chunk CP partition contract.

    Expected: rank r owns global chunk r followed by chunk 2*cp_size-1-r.
    A failure here means DSv4 chunk-aware CP helpers would disagree with the
    two-chunk local row order.
    """
    assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 16, 1, 0) == ((0, 16),)
    expected_by_rank = {
        0: ((0, 2), (14, 16)),
        1: ((2, 4), (12, 14)),
        2: ((4, 6), (10, 12)),
        3: ((6, 8), (8, 10)),
    }
    for rank, expected in expected_by_rank.items():
        assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 4, 4, rank) == expected

    with pytest.raises(RuntimeError, match="even local_rows"):
        local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 5, 4, 0)


def test_cp_chunk_ranges_match_partition_mode():
    """Validate local and left-boundary ranges for each CP partition mode.

    Expected: each mode returns the current rank's local rows and matching
    left-boundary rows in global packed-token coordinates.
    """
    assert local_q_cp_chunk_ranges(None, local_rows=4, cp_size=4, cp_rank=2) == ((8, 12),)
    assert local_kv_cp_chunk_ranges(
        DSV4_CP_PARTITION_CONTIGUOUS,
        local_rows=4,
        boundary_rows=2,
        cp_size=4,
        cp_rank=2,
    ) == ((6, 8), (8, 12))

    assert local_q_cp_chunk_ranges(
        DSV4_CP_PARTITION_TWO_CHUNK, local_rows=4, cp_size=4, cp_rank=0
    ) == ((0, 2), (14, 16))
    assert local_kv_cp_chunk_ranges(
        DSV4_CP_PARTITION_TWO_CHUNK,
        local_rows=4,
        boundary_rows=4,
        cp_size=4,
        cp_rank=0,
    ) == ((-2, 0), (12, 14), (0, 2), (14, 16))

    with pytest.raises(RuntimeError, match="boundary rows must be divisible"):
        local_kv_cp_chunk_ranges(
            DSV4_CP_PARTITION_TWO_CHUNK,
            local_rows=4,
            boundary_rows=3,
            cp_size=4,
            cp_rank=0,
        )


def test_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP boundary exchange contract.

    Expected: with cp_group=None, the fixed left boundary is zero-filled and its
    backward path contributes no gradient to the local tensor. A failure here
    means the CP path could invent boundary tokens or bogus local gradients.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_cp_boundary_hidden(local, [], 2, DSV4_CP_PARTITION_CONTIGUOUS, None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_two_chunk_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP two-chunk boundary exchange contract.

    Expected: with cp_group=None, the helper degenerates to one zero-filled
    boundary and contributes no local gradient. A failure here means chunk-aware
    CP boundary plumbing could perturb disabled-CP paths.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_cp_boundary_hidden(local, [], 2, DSV4_CP_PARTITION_TWO_CHUNK, None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_global_compressed_cu_seqlens_matches_reference():
    """Validate compressed-prefix metadata for ragged padded sequences.

    Expected: each sequence contributes ``padded_seq_len // ratio`` compressed
    rows to the global seq-major prefix. A failure here means rank-major
    compressed rows could be repacked with the wrong sequence offsets.
    """
    _require_cuda()
    cu_seqlens_cpu = torch.tensor([0, 7, 128, 381, 512], dtype=torch.int32)
    ratio = 4

    actual = build_global_compressed_cu_seqlens(cu_seqlens_cpu.cuda(), ratio)

    assert torch.equal(actual.cpu(), torch.tensor([0, 1, 31, 94, 126], dtype=torch.int32))
