# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams


def test_from_cu_seqlens_normalizes_and_infers_defaults():
    params = PackedSeqParams.from_cu_seqlens(
        cu_seqlens_q=torch.tensor([[0, 3, 7]], dtype=torch.int64),
        max_seqlen_q=torch.tensor([4], dtype=torch.int64),
    )

    expected_cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    assert params.qkv_format == "thd"
    assert torch.equal(params.cu_seqlens_q, expected_cu_seqlens)
    assert torch.equal(params.cu_seqlens_kv, expected_cu_seqlens)
    assert params.max_seqlen_q == 4
    assert params.max_seqlen_kv == 4
    assert params.total_tokens == 7
    assert torch.equal(
        params.seq_idx,
        torch.tensor([[0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32),
    )


def test_from_cu_seqlens_uses_padded_metadata_for_seq_idx():
    params = PackedSeqParams.from_cu_seqlens(
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int64),
        max_seqlen_q=torch.tensor(4, dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([[0, 4, 8]], dtype=torch.int64),
        total_tokens=torch.tensor([7], dtype=torch.int64),
    )

    assert torch.equal(
        params.cu_seqlens_q_padded,
        torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    assert torch.equal(params.cu_seqlens_kv_padded, params.cu_seqlens_q_padded)
    assert params.total_tokens == 7
    assert torch.equal(
        params.seq_idx,
        torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32),
    )


def test_single_sequence_returns_canonical_thd_params():
    params = PackedSeqParams.single_sequence(
        seq_len=6,
        device=torch.device("cpu"),
        dtype=torch.int64,
    )

    expected_cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
    assert params.qkv_format == "thd"
    assert torch.equal(params.cu_seqlens_q, expected_cu_seqlens)
    assert torch.equal(params.cu_seqlens_kv, expected_cu_seqlens)
    assert params.max_seqlen_q == 6
    assert params.max_seqlen_kv == 6
    assert params.total_tokens == 6
    assert torch.equal(
        params.seq_idx,
        torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int32),
    )


def test_from_cu_seqlens_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="qkv_format='thd'"):
        PackedSeqParams.from_cu_seqlens(
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            max_seqlen_q=2,
            qkv_format="sbhd",
        )

    with pytest.raises(ValueError, match=r"shape \[1, N\]"):
        PackedSeqParams.from_cu_seqlens(
            cu_seqlens_q=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
            max_seqlen_q=1,
        )

    with pytest.raises(ValueError, match="nondecreasing"):
        PackedSeqParams.from_cu_seqlens(
            cu_seqlens_q=torch.tensor([0, 3, 2], dtype=torch.int32),
            max_seqlen_q=3,
        )

    with pytest.raises(TypeError, match="integer dtype"):
        PackedSeqParams.from_cu_seqlens(
            cu_seqlens_q=torch.tensor([0.0, 2.0, 5.0], dtype=torch.float32),
            max_seqlen_q=3,
        )

    with pytest.raises(ValueError, match="same shape"):
        PackedSeqParams.from_cu_seqlens(
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_seqlen_q=3,
            cu_seqlens_q_padded=torch.tensor([0, 6], dtype=torch.int32),
        )

    with pytest.raises(ValueError, match="total_tokens must be >="):
        PackedSeqParams.from_cu_seqlens(
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_seqlen_q=3,
            total_tokens=4,
        )
