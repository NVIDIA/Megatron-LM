# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Context Parallelism (CP) support in multimodal_dev.

Tests cover:
  1. _cp_split_tensor — zigzag split correctness, reconstruction, and edge cases
  2. _NoCPGroup — dummy process group behaviour
  3. _thd_cp_partition_index — TE-based per-sample THD CP partitioning
  4. Cross-validation against megatron.core.utils.get_batch_on_this_cp_rank

Run with:  pytest examples/multimodal_dev/tests/test_cp_support.py -v
"""

import pytest
import torch

from examples.multimodal_dev.models.base import _cp_split_tensor, _NoCPGroup


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


class TestNoCPGroup:
    """Tests for the dummy CP group used by the vision encoder."""

    def test_size_is_one(self):
        g = _NoCPGroup()
        assert g.size() == 1

    def test_rank_is_zero(self):
        g = _NoCPGroup()
        assert g.rank() == 0


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
