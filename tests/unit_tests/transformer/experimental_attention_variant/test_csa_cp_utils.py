# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_MODES,
    DSV4_CP_PARTITION_TWO_CHUNK,
    build_global_compressed_cu_seqlens_fused,
    two_chunk_cp_partition,
    exchange_two_chunk_left_boundary_tensor,
    exchange_left_boundary_tensor,
    normalize_dsv4_cp_partition_mode,
)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 CP CUDA utility tests require CUDA.")


def test_dsv4_cp_partition_mode_contract():
    """Validate the public DSv4 CP partition-mode names.

    Expected: omitted mode defaults to the original contiguous partition,
    supported strings round-trip exactly, and unknown values fail before any CP
    layout helper can silently choose the wrong row order.
    """
    assert DSV4_CP_PARTITION_MODES == (
        DSV4_CP_PARTITION_CONTIGUOUS,
        DSV4_CP_PARTITION_TWO_CHUNK,
    )
    assert normalize_dsv4_cp_partition_mode(None) == DSV4_CP_PARTITION_CONTIGUOUS
    assert (
        normalize_dsv4_cp_partition_mode(DSV4_CP_PARTITION_CONTIGUOUS)
        == DSV4_CP_PARTITION_CONTIGUOUS
    )
    assert (
        normalize_dsv4_cp_partition_mode(DSV4_CP_PARTITION_TWO_CHUNK)
        == DSV4_CP_PARTITION_TWO_CHUNK
    )

    with pytest.raises(RuntimeError, match="Unsupported DSv4 CP partition mode"):
        normalize_dsv4_cp_partition_mode("invalid_mode")


def test_two_chunk_cp_partition_matches_expected_order():
    """Validate the two-chunk CP partition contract.

    Expected: rank r owns global chunk r followed by chunk 2*cp_size-1-r.
    A failure here means DSv4 chunk-aware CP helpers would disagree with the
    two-chunk local row order.
    """
    assert two_chunk_cp_partition(16, cp_size=1, cp_rank=0) == ((0, 16),)
    assert two_chunk_cp_partition(16, cp_size=4, cp_rank=0) == ((0, 2), (14, 16))
    assert two_chunk_cp_partition(16, cp_size=4, cp_rank=1) == ((2, 4), (12, 14))
    assert two_chunk_cp_partition(16, cp_size=4, cp_rank=2) == ((4, 6), (10, 12))
    assert two_chunk_cp_partition(16, cp_size=4, cp_rank=3) == ((6, 8), (8, 10))

    with pytest.raises(RuntimeError, match=r"padded_total_tokens % \(2 \* cp_size\)"):
        two_chunk_cp_partition(18, cp_size=4, cp_rank=0)


def test_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP boundary exchange contract.

    Expected: with cp_group=None, the fixed left boundary is zero-filled and its
    backward path contributes no gradient to the local tensor. A failure here
    means the CP path could invent boundary tokens or bogus local gradients.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_left_boundary_tensor(local, d_window=2, cp_group=None)

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

    boundary = exchange_two_chunk_left_boundary_tensor(local, d_window=2, cp_group=None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_cute_global_compressed_cu_seqlens_matches_reference():
    """Validate fused compressed-prefix metadata for ragged padded sequences.

    Expected: each sequence contributes ``padded_seq_len // ratio`` compressed
    rows to the global seq-major prefix. A failure here means rank-major
    compressed rows could be repacked with the wrong sequence offsets.
    """
    _require_cuda()
    cu_seqlens_cpu = torch.tensor([0, 7, 128, 381, 512], dtype=torch.int32)
    ratio = 4

    fused = build_global_compressed_cu_seqlens_fused(cu_seqlens_cpu.cuda(), ratio)

    assert torch.equal(fused.cpu(), torch.tensor([0, 1, 31, 94, 126], dtype=torch.int32))
