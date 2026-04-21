# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for THD (packed sequence) support in multimodal_dev."""

import pytest
import torch

import sys
import os

# Ensure the repo root is on the path so that the examples package is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.forward_step import (
    _build_packed_seq_params,
    _pack_batch,
)


# ===================================================================
# Unit tests — CPU, no distributed / GPU required
# ===================================================================


class TestBuildPackedSeqParams:
    """Tests for ``_build_packed_seq_params``."""

    def test_basic(self):
        params = _build_packed_seq_params([5, 3, 7], device="cpu")
        assert params.qkv_format == "thd"
        assert params.cu_seqlens_q.tolist() == [0, 5, 8, 15]
        assert params.cu_seqlens_kv.tolist() == [0, 5, 8, 15]
        assert params.max_seqlen_q == 7
        assert params.max_seqlen_kv == 7
        assert params.total_tokens == 15
        # Padded cu_seqlens mirror actual cu_seqlens (no per-subseq padding).
        assert params.cu_seqlens_q_padded is not None
        assert params.cu_seqlens_q_padded.tolist() == [0, 5, 8, 15]
        assert params.cu_seqlens_kv_padded is not None
        assert params.cu_seqlens_kv_padded.tolist() == [0, 5, 8, 15]

    def test_equal_lengths(self):
        params = _build_packed_seq_params([4, 4, 4], device="cpu")
        assert params.cu_seqlens_q.tolist() == [0, 4, 8, 12]
        assert params.max_seqlen_q == 4
        assert params.total_tokens == 12

    def test_single_sample(self):
        params = _build_packed_seq_params([10], device="cpu")
        assert params.cu_seqlens_q.tolist() == [0, 10]
        assert params.max_seqlen_q == 10
        assert params.total_tokens == 10

    def test_dtype_is_int32(self):
        params = _build_packed_seq_params([3, 5], device="cpu")
        assert params.cu_seqlens_q.dtype == torch.int32

    def test_seq_idx_computed(self):
        """Verify __post_init__ computes seq_idx for Mamba compatibility."""
        params = _build_packed_seq_params([3, 2], device="cpu")
        # seq_idx should be [0,0,0,1,1] (shape [1, 5])
        assert params.seq_idx is not None
        assert params.seq_idx.shape == (1, 5)
        assert params.seq_idx[0].tolist() == [0, 0, 0, 1, 1]


class TestPackBatch:
    """Tests for ``_pack_batch``."""

    def test_no_padding(self):
        """All tokens valid — T == B*S."""
        B, S = 2, 8
        batch = {
            "input_ids": torch.arange(B * S).reshape(B, S),
            "labels": torch.arange(B * S).reshape(B, S) + 100,
            "loss_mask": torch.ones(B, S),
            "position_ids": torch.arange(S).unsqueeze(0).unsqueeze(0).expand(
                3, B, S,
            ).clone(),
        }
        packed = _pack_batch(batch)

        T = B * S
        assert packed["input_ids"].shape == (1, T)
        assert packed["labels"].shape == (1, T)
        assert packed["loss_mask"].shape == (1, T)
        assert packed["position_ids"].shape == (3, 1, T)
        assert packed["attention_mask"] is None
        assert packed["packed_seq_params"].total_tokens == T

    def test_with_padding(self):
        """attention_mask strips padding — T < B*S."""
        B, S = 2, 8
        batch = {
            "input_ids": torch.arange(B * S).reshape(B, S),
            "labels": torch.arange(B * S).reshape(B, S),
            "loss_mask": torch.ones(B, S),
            "position_ids": torch.zeros(3, B, S, dtype=torch.long),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 0, 0, 0],  # 5 valid
                [1, 1, 1, 0, 0, 0, 0, 0],  # 3 valid
            ]),
        }
        packed = _pack_batch(batch)

        T = 5 + 3
        assert packed["input_ids"].shape == (1, T)
        assert packed["labels"].shape == (1, T)
        assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [0, 5, 8]
        assert packed["packed_seq_params"].max_seqlen_q == 5
        assert packed["packed_seq_params"].total_tokens == T

    def test_token_order_preserved(self):
        """Packed tokens appear in sample-0 then sample-1 order."""
        batch = {
            "input_ids": torch.tensor([[10, 20, 30], [40, 50, 60]]),
            "position_ids": torch.zeros(3, 2, 3, dtype=torch.long),
        }
        packed = _pack_batch(batch)
        assert packed["input_ids"].tolist() == [[10, 20, 30, 40, 50, 60]]

    def test_position_ids_mrope(self):
        """MRoPE [3, B, S] → [3, 1, T] with correct concatenation."""
        B, S = 2, 4
        pos = torch.zeros(3, B, S, dtype=torch.long)
        # Sample 0: positions [0,1,2,3] on all 3 dims
        # Sample 1: positions [10,11,12,13] on all 3 dims
        for d in range(3):
            pos[d, 0] = torch.tensor([0, 1, 2, 3])
            pos[d, 1] = torch.tensor([10, 11, 12, 13])

        batch = {
            "input_ids": torch.zeros(B, S, dtype=torch.long),
            "position_ids": pos,
        }
        packed = _pack_batch(batch)

        assert packed["position_ids"].shape == (3, 1, 8)
        # Each dim: [0,1,2,3,10,11,12,13]
        for d in range(3):
            assert packed["position_ids"][d, 0].tolist() == [
                0, 1, 2, 3, 10, 11, 12, 13,
            ]

    def test_standard_position_ids(self):
        """Standard [B, S] position_ids → [1, T]."""
        B, S = 2, 3
        batch = {
            "input_ids": torch.zeros(B, S, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        }
        packed = _pack_batch(batch)
        assert packed["position_ids"].shape == (1, 6)
        assert packed["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]

    def test_no_labels_no_loss_mask(self):
        """Gracefully handle missing labels and loss_mask."""
        batch = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "position_ids": torch.zeros(3, 2, 2, dtype=torch.long),
        }
        packed = _pack_batch(batch)
        assert packed["input_ids"].shape == (1, 4)
        assert packed.get("labels") is None
        assert packed.get("loss_mask") is None

    def test_variable_length_with_attention_mask(self):
        """Variable-length sequences: attention_mask strips padding."""
        B, S = 3, 10
        seq_lengths = [8, 5, 10]  # valid tokens per sample
        batch = {
            "input_ids": torch.arange(B * S).reshape(B, S),
            "labels": torch.arange(B * S).reshape(B, S) + 100,
            "loss_mask": torch.ones(B, S),
            "position_ids": torch.zeros(3, B, S, dtype=torch.long),
            "attention_mask": torch.zeros(B, S),
        }
        for i, sl in enumerate(seq_lengths):
            batch["attention_mask"][i, :sl] = 1.0

        packed = _pack_batch(batch)

        T = sum(seq_lengths)  # 23
        assert packed["input_ids"].shape == (1, T)
        assert packed["labels"].shape == (1, T)
        assert packed["loss_mask"].shape == (1, T)
        assert packed["position_ids"].shape == (3, 1, T)
        assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [
            0, 8, 13, 23,
        ]
        assert packed["packed_seq_params"].total_tokens == T

        # Verify correct tokens were kept (first sl tokens per sample).
        ids = packed["input_ids"][0].tolist()
        expected = list(range(0, 8)) + list(range(10, 15)) + list(range(20, 30))
        assert ids == expected

    def test_packed_seq_params_cumsum_matches_loop(self):
        """Verify torch.cumsum produces the same cu_seqlens as a Python loop."""
        lengths = [17, 31, 11, 42, 1]
        params = _build_packed_seq_params(lengths, device="cpu")
        # Manual cumulative sum
        expected = [0]
        for sl in lengths:
            expected.append(expected[-1] + sl)
        assert params.cu_seqlens_q.tolist() == expected
