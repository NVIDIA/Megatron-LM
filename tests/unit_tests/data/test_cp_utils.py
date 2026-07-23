# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for context-parallel data utilities in megatron.core.utils."""

import pytest
import torch

from megatron.core.utils import pad_thd_sequences_for_cp


def _reference_pad(tensors_with_pad_values, cu_seqlens, divisibility_factor):
    """Slow but obviously-correct reference implementation."""
    out_tensors = [[] for _ in tensors_with_pad_values]
    cu_padded = [0]
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        seg_len = end - start
        padded_len = (
            (seg_len + divisibility_factor - 1) // divisibility_factor
        ) * divisibility_factor
        pad = padded_len - seg_len
        for i, (tensor, pad_value) in enumerate(tensors_with_pad_values):
            out_tensors[i].append(tensor[start:end])
            if pad > 0:
                out_tensors[i].append(torch.full((pad,), pad_value, dtype=tensor.dtype))
        cu_padded.append(cu_padded[-1] + padded_len)
    return (
        [torch.cat(pieces) for pieces in out_tensors],
        torch.tensor(cu_padded, dtype=cu_seqlens.dtype),
    )


class TestPadThdSequencesForCp:
    """Unit tests for ``pad_thd_sequences_for_cp``."""

    def test_shorter_than_divisibility_factor(self):
        """All segments shorter than the divisibility factor (matches TE unit test)."""
        input_ids = torch.tensor([1, 1, 1, 2, 2, 3, 3, 3, 3])
        labels = torch.tensor([-100, -100, -100, -100, -100, -100, -100, 13, -100])
        cu_seqlens = torch.tensor([0, 3, 5, 9], dtype=torch.int32)

        (ids_p, lab_p), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 777), (labels, -200)], cu_seqlens, divisibility_factor=8
        )
        (ref_ids, ref_lab), ref_cu = _reference_pad(
            [(input_ids, 777), (labels, -200)], cu_seqlens, 8
        )
        assert torch.equal(ids_p, ref_ids)
        assert torch.equal(lab_p, ref_lab)
        assert torch.equal(cu_p, ref_cu)
        assert cu_p[-1].item() == 24

    def test_mixed_sequence_lengths(self):
        """Segments mixing lengths shorter and longer than the divisibility factor."""
        input_ids = torch.tensor(
            [1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        )
        labels = torch.arange(input_ids.numel(), dtype=torch.int64)
        cu_seqlens = torch.tensor([0, 2, 9, 13, 23], dtype=torch.int32)

        (ids_p, lab_p), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 999), (labels, -300)], cu_seqlens, divisibility_factor=6
        )
        (ref_ids, ref_lab), ref_cu = _reference_pad(
            [(input_ids, 999), (labels, -300)], cu_seqlens, 6
        )
        assert torch.equal(ids_p, ref_ids)
        assert torch.equal(lab_p, ref_lab)
        assert torch.equal(cu_p, ref_cu)
        # Per-segment padded lengths: 6, 12, 6, 12.
        assert cu_p.tolist() == [0, 6, 18, 24, 36]

    def test_longer_than_divisibility_factor(self):
        """Segments longer than the divisibility factor are rounded up to the next multiple."""
        # Seg 1: 7 -> 8 (pad 1); Seg 2: 11 -> 12 (pad 1); Seg 3: 5 -> 8 (pad 3).
        input_ids = torch.cat(
            [
                torch.ones(7, dtype=torch.int64),
                torch.full((11,), 2, dtype=torch.int64),
                torch.full((5,), 3, dtype=torch.int64),
            ]
        )
        labels = torch.arange(input_ids.numel(), dtype=torch.int64) + 100
        cu_seqlens = torch.tensor([0, 7, 18, 23], dtype=torch.int32)

        (ids_p, lab_p), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 888), (labels, -400)], cu_seqlens, divisibility_factor=4
        )
        (ref_ids, ref_lab), ref_cu = _reference_pad(
            [(input_ids, 888), (labels, -400)], cu_seqlens, 4
        )
        assert torch.equal(ids_p, ref_ids)
        assert torch.equal(lab_p, ref_lab)
        assert torch.equal(cu_p, ref_cu)
        assert cu_p.tolist() == [0, 8, 20, 28]

    def test_already_divisible_is_noop(self):
        """When every segment already satisfies divisibility, output equals input."""
        input_ids = torch.arange(16, dtype=torch.int64)
        labels = torch.arange(16, dtype=torch.int64) + 100
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

        (ids_p, lab_p), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 0), (labels, -100)], cu_seqlens, divisibility_factor=4
        )
        assert torch.equal(ids_p, input_ids)
        assert torch.equal(lab_p, labels)
        assert torch.equal(cu_p, cu_seqlens)

    def test_accepts_2d_input(self):
        """A leading batch dim of 1 (from DataLoader collation) is squeezed away."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[10, 20, 30, 40, 50]])
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

        (ids_p, lab_p), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 0), (labels, -100)], cu_seqlens, divisibility_factor=4
        )
        # Seg 1: [1, 2] + 2 pads = 4; Seg 2: [3, 4, 5] + 1 pad = 4.
        assert ids_p.dim() == 1
        assert ids_p.tolist() == [1, 2, 0, 0, 3, 4, 5, 0]
        assert lab_p.tolist() == [10, 20, -100, -100, 30, 40, 50, -100]
        assert cu_p.tolist() == [0, 4, 8]

    def test_preserves_cu_seqlens_int32_dtype(self):
        """``torch.cumsum`` promotes int32 -> int64 by default; verify we preserve int32."""
        input_ids = torch.tensor([1, 2, 3])
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

        _, cu_p = pad_thd_sequences_for_cp([(input_ids, 0)], cu_seqlens, divisibility_factor=8)
        assert cu_p.dtype == torch.int32

    def test_preserves_cu_seqlens_int64_dtype(self):
        """Dtype preservation also holds when cu_seqlens is int64."""
        input_ids = torch.tensor([1, 2, 3])
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int64)

        _, cu_p = pad_thd_sequences_for_cp([(input_ids, 0)], cu_seqlens, divisibility_factor=8)
        assert cu_p.dtype == torch.int64

    def test_preserves_tensor_dtypes(self):
        """Per-tensor dtypes propagate to the padded outputs."""
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        labels = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        loss_mask = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

        (ids_p, lab_p, lm_p), _ = pad_thd_sequences_for_cp(
            [(input_ids, 0), (labels, 0), (loss_mask, 0)], cu_seqlens, divisibility_factor=4
        )
        assert ids_p.dtype == torch.int32
        assert lab_p.dtype == torch.float32
        assert lm_p.dtype == torch.float32

    def test_pads_multiple_tensors_in_one_call(self):
        """Passing N tensors yields N padded outputs sharing one cu_seqlens_padded."""
        input_ids = torch.tensor([10, 11, 20, 21, 22])
        labels = torch.tensor([100, 101, 200, 201, 202])
        loss_mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

        (ids_p, lab_p, lm_p), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 0), (labels, -100), (loss_mask, 0.0)], cu_seqlens, divisibility_factor=4
        )
        # Seg 1: 2 -> 4 (2 pads); Seg 2: 3 -> 4 (1 pad).
        assert ids_p.tolist() == [10, 11, 0, 0, 20, 21, 22, 0]
        assert lab_p.tolist() == [100, 101, -100, -100, 200, 201, 202, -100]
        assert lm_p.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        assert cu_p.tolist() == [0, 4, 8]

    def test_single_tensor(self):
        """A single (tensor, pad_value) pair is also valid input."""
        input_ids = torch.tensor([7, 7, 7])
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

        (ids_p,), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 99)], cu_seqlens, divisibility_factor=4
        )
        assert ids_p.tolist() == [7, 7, 7, 99]
        assert cu_p.tolist() == [0, 4]

    def test_divisibility_factor_one_is_noop(self):
        """divisibility_factor=1 means every length is already valid -> no padding."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        labels = torch.tensor([6, 7, 8, 9, 10])
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

        (ids_p, lab_p), cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 0), (labels, -100)], cu_seqlens, divisibility_factor=1
        )
        assert torch.equal(ids_p, input_ids)
        assert torch.equal(lab_p, labels)
        assert torch.equal(cu_p, cu_seqlens)

    @pytest.mark.parametrize("divisibility_factor", [2, 4, 8, 16])
    def test_post_condition_segments_divisible(self, divisibility_factor):
        """The padded cu_seqlens differences are all multiples of divisibility_factor."""
        torch.manual_seed(0)
        seg_lens = [int(x) for x in torch.randint(1, 50, (8,))]
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(seg_lens).cumsum(0).tolist(), dtype=torch.int32
        )
        total = cu_seqlens[-1].item()
        input_ids = torch.arange(total, dtype=torch.int64)
        labels = torch.arange(total, dtype=torch.int64)

        _, cu_p = pad_thd_sequences_for_cp(
            [(input_ids, 0), (labels, -100)], cu_seqlens, divisibility_factor=divisibility_factor
        )
        diffs = (cu_p[1:] - cu_p[:-1]).tolist()
        assert all(d % divisibility_factor == 0 for d in diffs), diffs

    @pytest.mark.parametrize("divisibility_factor", [2, 4, 8])
    def test_against_reference(self, divisibility_factor):
        """Random shapes against the slow reference implementation."""
        torch.manual_seed(42)
        seg_lens = [int(x) for x in torch.randint(1, 30, (6,))]
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(seg_lens).cumsum(0).tolist(), dtype=torch.int32
        )
        total = cu_seqlens[-1].item()
        input_ids = torch.randint(0, 1000, (total,), dtype=torch.int64)
        labels = torch.randint(-100, 1000, (total,), dtype=torch.int64)
        loss_mask = torch.ones(total, dtype=torch.float32)
        spec = [(input_ids, 42), (labels, -100), (loss_mask, 0.0)]

        padded_tensors, cu_p = pad_thd_sequences_for_cp(
            spec, cu_seqlens, divisibility_factor=divisibility_factor
        )
        ref_tensors, ref_cu = _reference_pad(spec, cu_seqlens, divisibility_factor)
        for got, want in zip(padded_tensors, ref_tensors):
            assert torch.equal(got, want)
        assert torch.equal(cu_p, ref_cu)
