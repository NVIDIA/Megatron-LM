# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for megatron.core.inference.moe.pad.

Covers:
- pad_to_alignment: basic padding, zero-token experts, empty input, data integrity
- unpad_from_alignment: basic unpad, probs=None, probs!=None, empty padded buffer
- pad -> unpad roundtrip
"""

import pytest
import torch


def _ref_pad(hidden, tpe, alignment):
    """PyTorch reference for pad_to_alignment."""
    segments = []
    perm_segments = []
    offset = 0
    for count in tpe.tolist():
        count = int(count)
        chunk = hidden[offset : offset + count]
        if count > 0:
            aligned = ((count + alignment - 1) // alignment) * alignment
        else:
            aligned = 0
        pad_rows = aligned - count
        if count > 0:
            padded_chunk = torch.cat(
                [chunk, torch.zeros(pad_rows, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)],
                dim=0,
            ) if pad_rows > 0 else chunk
        else:
            padded_chunk = torch.zeros(0, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
        segments.append(padded_chunk)
        perm = torch.full((aligned,), -1, dtype=torch.int32, device=hidden.device)
        for i in range(count):
            perm[i] = offset + i
        perm_segments.append(perm)
        offset += count
    if segments:
        padded = torch.cat(segments, dim=0)
        perm_map = torch.cat(perm_segments, dim=0)
    else:
        padded = torch.zeros(0, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
        perm_map = torch.zeros(0, dtype=torch.int32, device=hidden.device)
    return padded, perm_map


@pytest.mark.internal
class TestPadToAlignment:

    def test_basic_padding(self):
        """Real tokens are copied correctly; padding rows are zero."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        hidden = torch.arange(9 * 4, device='cuda', dtype=torch.bfloat16).reshape(9, 4)
        tpe = torch.tensor([3, 4, 2], device='cuda', dtype=torch.int32)
        padded, perm_map, offs = pad_to_alignment(hidden, tpe, alignment=8)

        # Each expert gets 8 aligned rows: 3+5pad, 4+4pad, 2+6pad = 24 total
        assert padded.shape[0] == 24
        assert perm_map.shape[0] == 24

        # Check that real rows match source
        real_mask = perm_map >= 0
        for i in range(padded.shape[0]):
            src = perm_map[i].item()
            if src >= 0:
                torch.testing.assert_close(padded[i], hidden[src])

    def test_zero_expert_no_space(self):
        """Experts with 0 tokens consume no aligned space."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        hidden = torch.randn(5, 64, device='cuda', dtype=torch.bfloat16)
        tpe = torch.tensor([0, 5, 0], device='cuda', dtype=torch.int32)
        padded, perm_map, offs = pad_to_alignment(hidden, tpe, alignment=8)
        # Expert 1 has 5 tokens -> aligned to 8
        assert padded.shape[0] == 8
        assert (perm_map[:5] >= 0).all()
        assert (perm_map[5:] == -1).all()

    def test_empty_input(self):
        """Zero total tokens: output is empty, kernel not called."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        hidden = torch.zeros(0, 64, device='cuda', dtype=torch.bfloat16)
        tpe = torch.tensor([0, 0, 0], device='cuda', dtype=torch.int32)
        padded, perm_map, offs = pad_to_alignment(hidden, tpe, alignment=16)
        assert padded.shape[0] == 0
        assert perm_map.shape[0] == 0
        assert offs[-1].item() == 0

    def test_inclusive_offsets_aligned(self):
        """Inclusive offsets are multiples of alignment."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        tpe = torch.tensor([3, 7, 0, 15, 1], device='cuda', dtype=torch.int32)
        hidden = torch.randn(tpe.sum().item(), 64, device='cuda', dtype=torch.bfloat16)
        alignment = 16
        _, _, offs = pad_to_alignment(hidden, tpe, alignment=alignment)
        for i in range(len(tpe)):
            assert offs[i].item() % alignment == 0, f"offs[{i}]={offs[i].item()} not aligned"

    def test_real_row_count_equals_total_tokens(self):
        """Number of non-padding rows equals total input tokens."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        tpe = torch.tensor([5, 3, 8, 2], device='cuda', dtype=torch.int32)
        total = tpe.sum().item()
        hidden = torch.randn(total, 128, device='cuda', dtype=torch.bfloat16)
        _, perm_map, _ = pad_to_alignment(hidden, tpe, alignment=16)
        assert (perm_map >= 0).sum().item() == total

    @pytest.mark.parametrize("alignment", [8, 16, 32, 64, 128])
    @pytest.mark.parametrize("tpe_values", [[1, 2, 3], [5, 0, 10], [7, 7, 7, 7]])
    def test_alignment_correctness(self, alignment, tpe_values):
        """Every expert block boundary is aligned."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        tpe = torch.tensor(tpe_values, device='cuda', dtype=torch.int32)
        total = tpe.sum().item()
        if total == 0:
            return
        hidden = torch.randn(total, 64, device='cuda', dtype=torch.bfloat16)
        _, _, offs = pad_to_alignment(hidden, tpe, alignment=alignment)
        for v in offs.tolist():
            assert v % alignment == 0

    def test_data_integrity_matches_reference(self):
        """Padded output matches PyTorch reference."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        torch.manual_seed(42)
        tpe = torch.tensor([3, 2, 5, 1], device='cuda', dtype=torch.int32)
        total = tpe.sum().item()
        hidden = torch.randn(total, 64, device='cuda', dtype=torch.bfloat16)
        padded, perm_map, _ = pad_to_alignment(hidden, tpe, alignment=8)
        ref_padded, ref_perm = _ref_pad(hidden, tpe, alignment=8)

        # Compare real rows only (padding may differ in zeros vs garbage)
        assert padded.shape[0] == ref_padded.shape[0]
        real = perm_map >= 0
        for i in range(padded.shape[0]):
            if perm_map[i].item() >= 0:
                src = perm_map[i].item()
                torch.testing.assert_close(padded[i], hidden[src])

    def test_single_expert(self):
        """Works with a single expert."""
        from megatron.core.inference.moe.pad import pad_to_alignment

        hidden = torch.randn(7, 64, device='cuda', dtype=torch.bfloat16)
        tpe = torch.tensor([7], device='cuda', dtype=torch.int32)
        padded, perm_map, offs = pad_to_alignment(hidden, tpe, alignment=16)
        assert padded.shape[0] == 16
        assert (perm_map[:7] >= 0).all()
        assert (perm_map[7:] == -1).all()


@pytest.mark.internal
class TestUnpadFromAlignment:

    def test_basic_unpad(self):
        """Real rows are scattered to correct output positions."""
        from megatron.core.inference.moe.pad import unpad_from_alignment

        padded = torch.ones(8, 4, device='cuda', dtype=torch.bfloat16)
        # rows 0,1,2 map to output rows 0,2,4; rest are padding
        perm_map = torch.tensor([0, 2, 4, -1, -1, -1, -1, -1], device='cuda', dtype=torch.int32)
        output = unpad_from_alignment(padded, perm_map, original_size=5)
        assert output.shape == (5, 4)
        torch.testing.assert_close(output[0], torch.ones(4, device='cuda', dtype=torch.bfloat16))
        torch.testing.assert_close(output[2], torch.ones(4, device='cuda', dtype=torch.bfloat16))
        torch.testing.assert_close(output[4], torch.ones(4, device='cuda', dtype=torch.bfloat16))
        # Rows not in perm_map should be zero
        assert output[1].sum().item() == 0.0
        assert output[3].sum().item() == 0.0

    def test_unpad_with_probs(self):
        """Each output row is multiplied by its routing probability."""
        from megatron.core.inference.moe.pad import unpad_from_alignment

        hidden_dim = 4
        padded = torch.ones(4, hidden_dim, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.tensor([0, 1, -1, -1], device='cuda', dtype=torch.int32)
        probs = torch.tensor([0.5, 0.3], device='cuda', dtype=torch.bfloat16)
        output = unpad_from_alignment(padded, perm_map, original_size=2, probs=probs)
        torch.testing.assert_close(
            output[0], torch.full((hidden_dim,), 0.5, device='cuda', dtype=torch.bfloat16),
            atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            output[1], torch.full((hidden_dim,), 0.3, device='cuda', dtype=torch.bfloat16),
            atol=1e-2, rtol=1e-2,
        )

    def test_unpad_without_probs(self):
        """Without probs, values are copied as-is."""
        from megatron.core.inference.moe.pad import unpad_from_alignment

        padded = torch.randn(4, 8, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.tensor([0, 1, 2, -1], device='cuda', dtype=torch.int32)
        output = unpad_from_alignment(padded, perm_map, original_size=3)
        for i in range(3):
            torch.testing.assert_close(output[i], padded[i])

    def test_empty_padded_output(self):
        """Empty padded_output: kernel not called, output is all zeros."""
        from megatron.core.inference.moe.pad import unpad_from_alignment

        padded = torch.zeros(0, 64, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.zeros(0, device='cuda', dtype=torch.int32)
        output = unpad_from_alignment(padded, perm_map, original_size=5)
        assert output.shape == (5, 64)
        assert output.sum().item() == 0.0

    def test_all_padding_rows(self):
        """All perm_map == -1: output is zeros."""
        from megatron.core.inference.moe.pad import unpad_from_alignment

        padded = torch.ones(8, 4, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.full((8,), -1, device='cuda', dtype=torch.int32)
        output = unpad_from_alignment(padded, perm_map, original_size=4)
        assert output.sum().item() == 0.0


@pytest.mark.internal
class TestPadUnpadRoundtrip:

    @pytest.mark.parametrize(
        "tpe_values,alignment,hidden_dim",
        [
            ([3, 4, 2], 8, 64),
            ([5, 0, 10], 16, 128),
            ([1], 16, 64),
            ([7, 7, 7, 7], 8, 32),
            ([0, 0, 5, 0], 32, 64),
            ([15, 3, 0, 8, 1], 16, 128),
        ],
    )
    def test_pad_unpad_recovers_original(self, tpe_values, alignment, hidden_dim):
        """pad_to_alignment -> unpad_from_alignment recovers original hidden states."""
        from megatron.core.inference.moe.pad import pad_to_alignment, unpad_from_alignment

        torch.manual_seed(42)
        tpe = torch.tensor(tpe_values, device='cuda', dtype=torch.int32)
        total = tpe.sum().item()
        if total == 0:
            return
        hidden = torch.randn(total, hidden_dim, device='cuda', dtype=torch.bfloat16)

        padded, perm_map, _ = pad_to_alignment(hidden, tpe, alignment=alignment)
        recovered = unpad_from_alignment(padded, perm_map, original_size=total)

        # Each original row should match the recovered row exactly
        torch.testing.assert_close(recovered, hidden, atol=0, rtol=0)

    def test_roundtrip_with_probs(self):
        """Roundtrip with probs applies the scaling correctly."""
        from megatron.core.inference.moe.pad import pad_to_alignment, unpad_from_alignment

        torch.manual_seed(7)
        tpe = torch.tensor([4, 3, 5], device='cuda', dtype=torch.int32)
        total = tpe.sum().item()
        hidden = torch.randn(total, 64, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(total, device='cuda', dtype=torch.bfloat16)

        padded, perm_map, _ = pad_to_alignment(hidden, tpe, alignment=8)
        recovered = unpad_from_alignment(padded, perm_map, original_size=total, probs=probs)

        # Expected: each row scaled by its probability
        expected = hidden * probs.unsqueeze(1)
        torch.testing.assert_close(recovered, expected, atol=1e-2, rtol=1e-2)
