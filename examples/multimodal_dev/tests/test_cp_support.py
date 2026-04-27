# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Context Parallelism (CP) support in multimodal_dev.

Tests cover:
  1. _cp_split_tensor — zigzag split correctness, reconstruction, and edge cases
  2. _pad_batch_for_cp — padding shapes, values, and position_ids continuity
  3. _NoCPGroup — dummy process group behaviour
  4. Cross-validation against megatron.core.utils.get_batch_on_this_cp_rank

Run with:  pytest examples/multimodal_dev/tests/test_cp_support.py -v
"""

import pytest
import torch

from examples.multimodal_dev.models.base import _cp_split_tensor

# ---------------------------------------------------------------------------
# Test _cp_split_tensor
# ---------------------------------------------------------------------------



class TestCpSplitTensor:
    """Tests for zigzag CP splitting."""

    def test_basic_2d_cp2(self):
        """[B, S] tensor with CP=2 splits and reconstructs correctly."""
        B, S = 2, 16
        t = torch.arange(B * S).reshape(B, S)
        cp_size = 2

        chunks = []
        for rank in range(cp_size):
            chunks.append(_cp_split_tensor(t, seq_dim=1, cp_size=cp_size, cp_rank=rank))

        # Each rank gets S / CP = 8 tokens
        for c in chunks:
            assert c.shape == (B, S // cp_size)

        # Reconstruct: rank 0 gets chunks [0, 3], rank 1 gets chunks [1, 2]
        # Original split into 4 chunks of size 4:
        # chunk0=[0..3], chunk1=[4..7], chunk2=[8..11], chunk3=[12..15]
        # rank0 = [chunk0, chunk3] = [0..3, 12..15]
        # rank1 = [chunk1, chunk2] = [4..7, 8..11]
        assert torch.equal(chunks[0][0], torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]))
        assert torch.equal(chunks[1][0], torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))

    def test_3d_mrope_cp2(self):
        """[3, B, S] MRoPE tensor with CP=2."""
        B, S = 1, 8
        cp_size = 2
        t = torch.arange(3 * B * S).reshape(3, B, S)

        chunk = _cp_split_tensor(t, seq_dim=2, cp_size=cp_size, cp_rank=0)
        assert chunk.shape == (3, B, S // cp_size)

        # All 3 MRoPE components should be split consistently
        for d in range(3):
            original_row = t[d, 0]  # [S]
            # With S=8, CP=2: 4 chunks of size 2
            # rank0 gets chunks [0, 3] = positions [0,1, 6,7]
            expected = torch.cat([original_row[0:2], original_row[6:8]])
            assert torch.equal(chunk[d, 0], expected)

    def test_sbh_decoder_input(self):
        """[S, B, H] decoder input split along dim=0."""
        S, B, H = 16, 2, 4
        cp_size = 2
        t = torch.randn(S, B, H)

        chunk = _cp_split_tensor(t, seq_dim=0, cp_size=cp_size, cp_rank=0)
        assert chunk.shape == (S // cp_size, B, H)

    def test_cp4(self):
        """CP=4 zigzag pattern."""
        S = 32
        cp_size = 4
        t = torch.arange(S).unsqueeze(0)  # [1, 32]

        all_chunks = []
        for rank in range(cp_size):
            c = _cp_split_tensor(t, seq_dim=1, cp_size=cp_size, cp_rank=rank)
            all_chunks.append(c)
            assert c.shape == (1, S // cp_size)

        # All tokens should appear exactly once across ranks
        combined = torch.cat(all_chunks, dim=1)
        assert torch.equal(combined.sort(dim=1).values, t.sort(dim=1).values)

    def test_cp8(self):
        """CP=8 zigzag pattern — all tokens appear exactly once."""
        S = 64
        cp_size = 8
        t = torch.arange(S).unsqueeze(0)  # [1, 64]

        all_chunks = []
        for rank in range(cp_size):
            c = _cp_split_tensor(t, seq_dim=1, cp_size=cp_size, cp_rank=rank)
            all_chunks.append(c)
            assert c.shape == (1, S // cp_size)

        combined = torch.cat(all_chunks, dim=1)
        assert torch.equal(combined.sort(dim=1).values, t.sort(dim=1).values)

    def test_not_divisible_raises(self):
        """Should raise when seq_len not divisible by 2*cp_size."""
        t = torch.randn(2, 10)  # S=10, not divisible by 4
        with pytest.raises(AssertionError):
            _cp_split_tensor(t, seq_dim=1, cp_size=2, cp_rank=0)

    def test_zigzag_symmetry(self):
        """rank 0 and rank (cp_size-1) should get mirror chunks."""
        S = 16
        cp_size = 2
        t = torch.arange(S).unsqueeze(0)  # [1, 16]

        c0 = _cp_split_tensor(t, seq_dim=1, cp_size=cp_size, cp_rank=0)
        c1 = _cp_split_tensor(t, seq_dim=1, cp_size=cp_size, cp_rank=1)

        # rank0 gets chunks [0, 3], rank1 gets chunks [1, 2]
        # chunk0=[0..3], chunk3=[12..15] -> rank0 gets [0..3, 12..15]
        # chunk1=[4..7], chunk2=[8..11] -> rank1 gets [4..7, 8..11]
        # rank0's first half is earliest, rank1's first half is next
        assert c0[0, 0].item() < c1[0, 0].item()  # rank0 starts earlier

    def test_matches_megatron_core(self):
        """Cross-validate against megatron.core.utils.get_batch_on_this_cp_rank logic.

        We simulate the core function's logic (seq_dim=1, attention_mask seq_dim=2)
        and compare.
        """
        B, S = 2, 32
        cp_size = 4

        input_ids = torch.arange(B * S).reshape(B, S)
        labels = torch.arange(B * S).reshape(B, S) + 1000

        for cp_rank in range(cp_size):
            # Our implementation
            our_ids = _cp_split_tensor(input_ids, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)
            our_labels = _cp_split_tensor(labels, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)

            # Simulate megatron core logic inline
            def core_split(val, seq_dim):
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1):],
                )
                index = torch.zeros(2, dtype=torch.int64, device=val.device)
                index[0].fill_(cp_rank)
                index[1].fill_(2 * cp_size - cp_rank - 1)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
                return val

            ref_ids = core_split(input_ids.clone(), seq_dim=1)
            ref_labels = core_split(labels.clone(), seq_dim=1)

            assert torch.equal(our_ids, ref_ids), f"input_ids mismatch at rank {cp_rank}"
            assert torch.equal(our_labels, ref_labels), f"labels mismatch at rank {cp_rank}"

    def test_batch_dim_preserved(self):
        """Batch dimension must be unchanged after split."""
        B, S = 4, 32
        cp_size = 4
        t = torch.randn(B, S)

        for rank in range(cp_size):
            c = _cp_split_tensor(t, seq_dim=1, cp_size=cp_size, cp_rank=rank)
            assert c.shape[0] == B


# ---------------------------------------------------------------------------
# Test _pad_batch_for_cp (requires mocking parallel_state)
# ---------------------------------------------------------------------------

class TestPadBatchForCp:
    """Tests for CP padding logic.

    These tests mock the parallel_state functions to avoid needing
    a real distributed environment.
    """

    def _make_batch(self, B=2, S=10):
        """Create a minimal batch dict."""
        return {
            "input_ids": torch.randint(0, 1000, (B, S)),
            "labels": torch.randint(0, 1000, (B, S)),
            "loss_mask": torch.ones(B, S),
            "position_ids": torch.arange(S).unsqueeze(0).unsqueeze(0).expand(3, B, S).clone(),
            "attention_mask": torch.ones(B, S, dtype=torch.bool),
        }

    def test_padding_shape(self, monkeypatch):
        """Padded shapes should be divisible by cp_size * 2."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        batch = self._make_batch(B=2, S=10)
        padded = fs._pad_batch_for_cp(batch)

        # S=10 should be padded to 12 (next multiple of 2*2=4)
        assert padded["input_ids"].shape[1] == 12
        assert padded["labels"].shape[1] == 12
        assert padded["loss_mask"].shape[1] == 12
        assert padded["position_ids"].shape[2] == 12
        assert padded["attention_mask"].shape[1] == 12

    def test_padding_values(self, monkeypatch):
        """Check pad values: input_ids=0, labels=-100, loss_mask=0."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        batch = self._make_batch(B=1, S=6)
        padded = fs._pad_batch_for_cp(batch)

        # S=6 -> padded to 8 (2 padding tokens)
        assert padded["input_ids"][0, -1].item() == 0
        assert padded["labels"][0, -1].item() == -100
        assert padded["loss_mask"][0, -1].item() == 0
        assert padded["attention_mask"][0, -1].item() == 0

    def test_position_ids_padding_continuity(self, monkeypatch):
        """MRoPE position_ids padding should continue incrementing."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        B, S = 1, 6
        pos = torch.arange(S).unsqueeze(0).unsqueeze(0).expand(3, B, -1).clone()
        # pos[:, 0, :] = [0, 1, 2, 3, 4, 5] for each of the 3 components
        batch = self._make_batch(B=B, S=S)
        batch["position_ids"] = pos

        padded = fs._pad_batch_for_cp(batch)
        # Should pad to 8: last original value is 5, padded values should be 6, 7
        for d in range(3):
            assert padded["position_ids"][d, 0, 6].item() == 6
            assert padded["position_ids"][d, 0, 7].item() == 7

    def test_no_padding_when_divisible(self, monkeypatch):
        """No padding when seq_len already divisible."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        batch = self._make_batch(B=2, S=8)  # 8 % 4 == 0
        padded = fs._pad_batch_for_cp(batch)
        assert padded["input_ids"].shape[1] == 8

    def test_cp1_noop(self, monkeypatch):
        """CP=1 should return batch unchanged."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 1)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        batch = self._make_batch(B=2, S=10)
        original_shape = batch["input_ids"].shape
        padded = fs._pad_batch_for_cp(batch)
        assert padded["input_ids"].shape == original_shape

    def test_padding_with_tp_sp(self, monkeypatch):
        """When SP is enabled, alignment includes tp_size."""
        import examples.multimodal_dev.forward_step as fs
        import megatron.training as mt

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 4)
        monkeypatch.setattr(
            mt,
            "get_args",
            lambda: type("Args", (), {"sequence_parallel": True})(),
        )

        # divisible_by = tp_size * cp_size * 2 = 4 * 2 * 2 = 16
        batch = self._make_batch(B=1, S=10)
        padded = fs._pad_batch_for_cp(batch)
        assert padded["input_ids"].shape[1] == 16  # next multiple of 16

    def test_padding_with_cp4(self, monkeypatch):
        """CP=4 should pad to multiple of 8."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 4)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        # divisible_by = 4 * 2 = 8
        batch = self._make_batch(B=1, S=5)
        padded = fs._pad_batch_for_cp(batch)
        assert padded["input_ids"].shape[1] == 8

    def test_padding_with_cp8(self, monkeypatch):
        """CP=8 should pad to multiple of 16."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 8)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        # divisible_by = 8 * 2 = 16
        batch = self._make_batch(B=1, S=10)
        padded = fs._pad_batch_for_cp(batch)
        assert padded["input_ids"].shape[1] == 16

    def test_standard_position_ids_padding(self, monkeypatch):
        """Standard [B, S] position_ids (non-MRoPE) should also be padded."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        B, S = 1, 6
        batch = self._make_batch(B=B, S=S)
        batch["position_ids"] = torch.arange(S).unsqueeze(0)  # [1, 6] standard

        padded = fs._pad_batch_for_cp(batch)
        # S=6 -> padded to 8
        assert padded["position_ids"].shape == (1, 8)
        assert padded["position_ids"][0, 6].item() == 6
        assert padded["position_ids"][0, 7].item() == 7

    def test_loss_mask_padding_zeroes_out(self, monkeypatch):
        """Padded loss_mask positions should be 0 so they don't contribute to loss."""
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        B, S = 2, 6
        batch = self._make_batch(B=B, S=S)
        batch["loss_mask"] = torch.ones(B, S)  # all 1s

        padded = fs._pad_batch_for_cp(batch)
        # Original 6 positions should still be 1
        assert padded["loss_mask"][:, :6].sum().item() == B * S
        # Padded 2 positions should be 0
        assert padded["loss_mask"][:, 6:].sum().item() == 0


# ---------------------------------------------------------------------------
# Test _NoCPGroup
# ---------------------------------------------------------------------------

from examples.multimodal_dev.models.base import _NoCPGroup


class TestNoCPGroup:
    """Tests for the dummy CP group used by the vision encoder."""

    def test_size_is_one(self):
        g = _NoCPGroup()
        assert g.size() == 1

    def test_rank_is_zero(self):
        g = _NoCPGroup()
        assert g.rank() == 0


# ---------------------------------------------------------------------------
# Test pad + split round-trip
# ---------------------------------------------------------------------------

class TestPadAndSplitRoundTrip:
    """Verify that padding + splitting gives consistent results across ranks."""

    def test_padded_batch_splits_cleanly(self, monkeypatch):
        """After padding, the batch should split evenly across all CP ranks."""
        import examples.multimodal_dev.forward_step as fs

        for cp_size in [2, 4, 8]:
            monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda cs=cp_size: cs)
            monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

            B, S = 2, 17  # deliberately not aligned
            batch = {
                "input_ids": torch.arange(B * S).reshape(B, S),
                "labels": torch.arange(B * S).reshape(B, S) + 1000,
                "loss_mask": torch.ones(B, S),
            }

            padded = fs._pad_batch_for_cp(batch)
            padded_S = padded["input_ids"].shape[1]

            # Should be divisible by 2 * cp_size
            assert padded_S % (2 * cp_size) == 0, (
                f"CP={cp_size}: padded S={padded_S} not divisible by {2 * cp_size}"
            )

            # Split should work without assertion errors
            all_ids = []
            for rank in range(cp_size):
                chunk = _cp_split_tensor(
                    padded["input_ids"], seq_dim=1, cp_size=cp_size, cp_rank=rank,
                )
                assert chunk.shape == (B, padded_S // cp_size)
                all_ids.append(chunk)

            # All tokens appear exactly once across ranks
            combined = torch.cat(all_ids, dim=1)
            for b in range(B):
                assert torch.equal(
                    combined[b].sort().values,
                    padded["input_ids"][b].sort().values,
                ), f"Token mismatch at batch {b} for CP={cp_size}"

    def test_loss_mask_zero_for_padded_on_all_ranks(self, monkeypatch):
        """After pad + split, padded positions have loss_mask=0 on whichever rank they land."""
        import examples.multimodal_dev.forward_step as fs

        cp_size = 4
        monkeypatch.setattr(fs, "get_context_parallel_world_size", lambda: cp_size)
        monkeypatch.setattr(fs, "get_tensor_model_parallel_world_size", lambda: 1)

        B, S = 1, 10  # 10 -> padded to 16 for CP=4
        batch = {
            "input_ids": torch.randint(1, 1000, (B, S)),  # non-zero tokens
            "loss_mask": torch.ones(B, S),
        }

        padded = fs._pad_batch_for_cp(batch)
        padded_S = padded["input_ids"].shape[1]
        num_padded = padded_S - S  # 6 padded positions

        # Count total loss_mask=0 across all ranks
        total_zero = 0
        for rank in range(cp_size):
            lm_chunk = _cp_split_tensor(
                padded["loss_mask"], seq_dim=1, cp_size=cp_size, cp_rank=rank,
            )
            total_zero += (lm_chunk == 0).sum().item()

        assert total_zero == B * num_padded, (
            f"Expected {B * num_padded} zero loss_mask positions, got {total_zero}"
        )


# ---------------------------------------------------------------------------
# Test _thd_cp_partition_index (THD + CP per-sample partitioning)
# ---------------------------------------------------------------------------

try:
    from transformer_engine.pytorch import cpp_extensions as _tex  # noqa: F401

    _HAS_TE = True
except Exception:
    _HAS_TE = False


@pytest.mark.skipif(not _HAS_TE, reason="TransformerEngine not installed")
class TestThdCpPartition:
    """Verify TE-based per-sample THD + CP partition matches THD semantics.

    Each packed sub-sample of length ``s_i`` (where ``s_i % (2*cp_size) == 0``)
    is split into ``2*cp_size`` zigzag chunks per sample; rank ``r`` gets
    chunks ``[r, 2*cp_size - r - 1]`` of every sample.  The union across
    ranks must cover every token position exactly once.
    """

    @staticmethod
    def _make_padded_packed(seqlens, divisor):
        """Concatenate per-sample dummy tokens after padding each sample to a
        multiple of *divisor*.  Returns ``(input_ids[1, T], cu_seqlens_padded)``.
        """
        import math
        padded = [math.ceil(s / divisor) * divisor for s in seqlens]
        chunks = []
        next_id = 1
        for s, p in zip(seqlens, padded):
            chunks.append(torch.arange(next_id, next_id + s, dtype=torch.int64))
            chunks.append(torch.zeros(p - s, dtype=torch.int64))  # padding
            next_id += s
        input_ids = torch.cat(chunks, dim=0).unsqueeze(0)  # [1, T]
        cu_seqlens_padded = torch.tensor(
            [0] + list(torch.tensor(padded).cumsum(0).tolist()),
            dtype=torch.int32,
        )
        return input_ids, cu_seqlens_padded

    def _ensure_cuda(self, x):
        return x.cuda() if torch.cuda.is_available() else x

    def test_partition_covers_all_positions_cp2(self):
        from examples.multimodal_dev.models.base import _thd_cp_partition_index

        cp_size = 2
        seqlens = [5, 7, 3]  # valid lengths
        input_ids, cu_seqlens_padded = self._make_padded_packed(
            seqlens, divisor=2 * cp_size,
        )
        input_ids = self._ensure_cuda(input_ids)
        cu_seqlens_padded = self._ensure_cuda(cu_seqlens_padded)
        T = input_ids.shape[1]

        # Union of per-rank indices must be all positions exactly once.
        seen = torch.zeros(T, dtype=torch.long, device=input_ids.device)
        for cp_rank in range(cp_size):
            idx = _thd_cp_partition_index(
                cu_seqlens_padded, T, cp_size, cp_rank,
            )
            assert idx.numel() == T // cp_size, (
                f"rank {cp_rank}: expected {T // cp_size} tokens, got {idx.numel()}"
            )
            seen.scatter_add_(
                0, idx.long(), torch.ones_like(idx, dtype=seen.dtype),
            )
        assert torch.all(seen == 1), (
            f"Position coverage broken: counts={seen.tolist()}"
        )

    def test_index_select_aligns_inputs_and_position_ids_cp2(self):
        """input_ids, loss_mask, and (3, 1, T) position_ids index_select with
        the same partition index produce shape-consistent per-rank tensors."""
        from examples.multimodal_dev.models.base import _thd_cp_partition_index

        cp_size = 2
        seqlens = [8, 4]
        input_ids, cu_seqlens_padded = self._make_padded_packed(
            seqlens, divisor=2 * cp_size,
        )
        input_ids = self._ensure_cuda(input_ids)
        cu_seqlens_padded = self._ensure_cuda(cu_seqlens_padded)
        T = input_ids.shape[1]
        labels = input_ids + 1000
        loss_mask = (input_ids != 0).float()
        position_ids = (
            torch.arange(T, device=input_ids.device)
            .unsqueeze(0).unsqueeze(0).expand(3, 1, T).contiguous()
        )
        H = 4
        decoder_input = (
            torch.arange(T * H, dtype=torch.float32, device=input_ids.device)
            .view(T, 1, H)
        )

        for cp_rank in range(cp_size):
            idx = _thd_cp_partition_index(
                cu_seqlens_padded, T, cp_size, cp_rank,
            )
            ii = input_ids.index_select(1, idx)
            ll = labels.index_select(1, idx)
            lm = loss_mask.index_select(1, idx)
            pi = position_ids.index_select(2, idx)
            di = decoder_input.index_select(0, idx)

            assert ii.shape == (1, T // cp_size)
            assert ll.shape == (1, T // cp_size)
            assert lm.shape == (1, T // cp_size)
            assert pi.shape == (3, 1, T // cp_size)
            assert di.shape == (T // cp_size, 1, H)
            # Sliced position_ids is just the partition index itself
            # (since position_ids was arange(T) over all positions).
            assert torch.equal(pi[0, 0], idx.to(pi.dtype))
            # All MRoPE rows agree.
            assert torch.equal(pi[1, 0], pi[0, 0])
            assert torch.equal(pi[2, 0], pi[0, 0])

    def test_partition_cp4_three_samples(self):
        from examples.multimodal_dev.models.base import _thd_cp_partition_index

        cp_size = 4
        seqlens = [12, 4, 8]
        input_ids, cu_seqlens_padded = self._make_padded_packed(
            seqlens, divisor=2 * cp_size,
        )
        input_ids = self._ensure_cuda(input_ids)
        cu_seqlens_padded = self._ensure_cuda(cu_seqlens_padded)
        T = input_ids.shape[1]

        seen = torch.zeros(T, dtype=torch.long, device=input_ids.device)
        for cp_rank in range(cp_size):
            idx = _thd_cp_partition_index(
                cu_seqlens_padded, T, cp_size, cp_rank,
            )
            assert idx.numel() == T // cp_size
            seen.scatter_add_(
                0, idx.long(), torch.ones_like(idx, dtype=seen.dtype),
            )
        assert torch.all(seen == 1)

    def test_loss_mask_zero_kept_per_rank(self):
        """Pad-token positions (loss_mask=0) survive as 0 on whichever rank
        they land — sanity check that we don't accidentally discard them."""
        from examples.multimodal_dev.models.base import _thd_cp_partition_index

        cp_size = 2
        seqlens = [5, 3]
        input_ids, cu_seqlens_padded = self._make_padded_packed(
            seqlens, divisor=2 * cp_size,
        )
        input_ids = self._ensure_cuda(input_ids)
        cu_seqlens_padded = self._ensure_cuda(cu_seqlens_padded)
        T = input_ids.shape[1]
        loss_mask = (input_ids != 0).float()
        total_zeros = (loss_mask == 0).sum().item()

        zeros_seen = 0
        for cp_rank in range(cp_size):
            idx = _thd_cp_partition_index(
                cu_seqlens_padded, T, cp_size, cp_rank,
            )
            zeros_seen += (
                loss_mask.index_select(1, idx) == 0
            ).sum().item()
        assert zeros_seen == total_zeros
