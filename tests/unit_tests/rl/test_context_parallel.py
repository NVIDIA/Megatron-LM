# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for context-parallel helpers in megatron/rl/rl_utils.py.

These tests exercise _scatter_for_context_parallel and
_gather_logprobs_context_parallel without requiring a real distributed
environment (megatron.core.mpu is patched) or a Transformer Engine install
(rl_utils.tex is patched with a pure-torch reference partitioner).

The token-to-rank partition is owned by TE's ``thd_get_partitioned_indices``:
every padded sequence in the bin is split into ``2*cp_size`` chunks and rank
``r`` receives chunks ``(r, 2*cp_size-r-1)`` of each sequence (load-balanced
zigzag, applied per sequence). The reference implementation below mirrors that
layout; the production code is deliberately agnostic to the exact index order
(it only requires a disjoint and complete partition), so these tests validate
the plumbing around the indices, not TE's kernel itself.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_fake_pg():
    pg = MagicMock()
    pg.__class__ = torch.distributed.ProcessGroup
    return pg


def _reference_thd_partition_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
    """Pure-torch reference for tex.thd_get_partitioned_indices.

    For every padded slot [s, e) in cu_seqlens_padded, rank r owns chunks
    (r, 2*cp_size-r-1) of a 2*cp_size partition of that slot. Zero-length
    (ghost) slots contribute nothing.
    """
    idxs = []
    cu = [int(x) for x in cu_seqlens_padded]
    for s, e in zip(cu[:-1], cu[1:]):
        slot_len = e - s
        if slot_len == 0:
            continue
        assert slot_len % (2 * cp_size) == 0, (s, e, cp_size)
        chunk = slot_len // (2 * cp_size)
        idxs += list(range(s + cp_rank * chunk, s + (cp_rank + 1) * chunk))
        idxs += list(
            range(s + (2 * cp_size - cp_rank - 1) * chunk, s + (2 * cp_size - cp_rank) * chunk)
        )
    return torch.tensor(idxs, dtype=torch.int64)


def _fake_tex():
    return SimpleNamespace(thd_get_partitioned_indices=_reference_thd_partition_indices)


def _single_seq_psp(seq_len):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = torch.tensor([0, seq_len], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        total_tokens=seq_len,
    )


def _packed_psp(cu_padded, cu_actual, seq_len):
    from megatron.core.packed_seq_params import PackedSeqParams

    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=torch.tensor(cu_actual, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(cu_actual, dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor(cu_padded, dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor(cu_padded, dtype=torch.int32),
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        total_tokens=seq_len,
    )


def _zigzag_chunks(t: torch.Tensor, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Whole-tensor zigzag reference (the single-sequence degenerate case)."""
    seq_len = t.shape[1]
    chunk_size = seq_len // (2 * cp_size)
    chunks = t.view(t.shape[0], 2 * cp_size, chunk_size, *t.shape[2:])
    a = chunks[:, cp_rank]
    b = chunks[:, 2 * cp_size - cp_rank - 1]
    return torch.cat([a, b], dim=1)


# ---------------------------------------------------------------------------
# Tests for _scatter_for_context_parallel
# ---------------------------------------------------------------------------

class TestScatterForContextParallel:
    """Test _scatter_for_context_parallel in isolation."""

    def _run(self, cp_size, cp_rank, psp, seq_len, batch=1):
        from megatron.rl.rl_utils import _scatter_for_context_parallel

        tokens = torch.arange(batch * seq_len).reshape(batch, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
        fake_pg = _make_fake_pg()

        with (
            patch('megatron.rl.rl_utils.mpu') as mock_mpu,
            patch('megatron.rl.rl_utils.tex', _fake_tex()),
        ):
            mock_mpu.get_context_parallel_rank.return_value = cp_rank
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            result = _scatter_for_context_parallel(tokens, position_ids, psp, cp_size)

        return result, tokens, psp

    # --- shape tests ----------------------------------------------------------

    def test_local_shapes_single_sequence(self):
        (lt, lp, _, ll, idx), _, _ = self._run(
            cp_size=2, cp_rank=0, psp=_single_seq_psp(8), seq_len=8
        )
        assert lt.shape == (1, 4)
        assert lp.shape == (1, 4)
        assert ll.shape == (1, 4)
        assert idx.shape == (4,)

    def test_cp4_shapes(self):
        (lt, _, _, ll, idx), _, _ = self._run(
            cp_size=4, cp_rank=2, psp=_single_seq_psp(16), seq_len=16
        )
        assert lt.shape == (1, 4)
        assert ll.shape == (1, 4)
        assert idx.shape == (4,)

    # --- value tests: layout --------------------------------------------------

    def test_single_sequence_degenerates_to_zigzag(self):
        """With one full-length sequence, the TE partition equals the whole-tensor zigzag."""
        for cp_rank in range(2):
            (lt, _, _, _, _), tokens, _ = self._run(
                cp_size=2, cp_rank=cp_rank, psp=_single_seq_psp(8), seq_len=8
            )
            torch.testing.assert_close(lt, _zigzag_chunks(tokens, 2, cp_rank))

    def test_multi_sequence_bin_respects_boundaries(self):
        """Indices must partition each padded slot independently (per-sequence zigzag)."""
        # Bin of 24 tokens: slots [0, 8), [8, 16) (real sequences) and [16, 24) (trailing ghost).
        psp = _packed_psp(cu_padded=[0, 8, 16, 24], cu_actual=[0, 5, 12, 20], seq_len=24)
        (_, _, _, _, idx), _, _ = self._run(cp_size=2, cp_rank=0, psp=psp, seq_len=24)
        # Rank 0 owns chunks 0 and 3 of every slot: [0,1] and [6,7] within each slot.
        expected = torch.tensor([0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23])
        torch.testing.assert_close(idx, expected)

    def test_partition_is_disjoint_and_complete(self):
        psp = _packed_psp(cu_padded=[0, 8, 16, 24], cu_actual=[0, 5, 12, 20], seq_len=24)
        all_idx = []
        for cp_rank in range(2):
            (_, _, _, _, idx), _, _ = self._run(cp_size=2, cp_rank=cp_rank, psp=psp, seq_len=24)
            all_idx.append(idx)
        cat = torch.cat(all_idx)
        assert cat.shape[0] == 24
        torch.testing.assert_close(torch.sort(cat).values, torch.arange(24))

    def test_labels_are_shifted_then_partitioned(self):
        (_, _, _, ll, idx), tokens, _ = self._run(
            cp_size=2, cp_rank=1, psp=_single_seq_psp(8), seq_len=8
        )
        tokens_shifted = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
        torch.testing.assert_close(ll, tokens_shifted.index_select(1, idx))

    # --- PackedSeqParams mutation test ----------------------------------------

    def test_cp_fields_set_on_copy(self):
        """cp_group, local_cp_size and cu_seqlens_*_padded must be set;
        original must be unchanged."""
        (_, _, cp_params, _, _), _, orig = self._run(
            cp_size=2, cp_rank=0, psp=_single_seq_psp(8), seq_len=8
        )
        assert cp_params.local_cp_size == 2
        assert cp_params.cp_group is not None
        # THD CP path requires cu_seqlens_*_padded — set from cu_seqlens_* fallback.
        assert cp_params.cu_seqlens_q_padded is not None
        assert cp_params.cu_seqlens_kv_padded is not None
        # Original must not have been mutated.
        assert orig.local_cp_size is None
        assert orig.cp_group is None
        assert orig.cu_seqlens_q_padded is None
        assert orig.cu_seqlens_kv_padded is None

    def test_explicit_padded_boundaries_are_used(self):
        """When the packer supplied cu_seqlens_*_padded, the partition must use them."""
        psp = _packed_psp(cu_padded=[0, 8, 16, 24], cu_actual=[0, 5, 12, 20], seq_len=24)
        (_, _, cp_params, _, _), _, _ = self._run(cp_size=2, cp_rank=0, psp=psp, seq_len=24)
        torch.testing.assert_close(cp_params.cu_seqlens_q_padded, psp.cu_seqlens_q_padded)

    def test_assertion_on_indivisible_seq_len(self):
        """The total length must be divisible by 2*cp_size."""
        from megatron.rl.rl_utils import _scatter_for_context_parallel

        tokens = torch.zeros(1, 9, dtype=torch.long)
        pos = torch.zeros(1, 9, dtype=torch.long)
        psp = _single_seq_psp(9)
        fake_pg = _make_fake_pg()
        with (
            patch('megatron.rl.rl_utils.mpu') as mock_mpu,
            patch('megatron.rl.rl_utils.tex', _fake_tex()),
        ):
            mock_mpu.get_context_parallel_rank.return_value = 0
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            with pytest.raises(AssertionError, match="divisible"):
                _scatter_for_context_parallel(tokens, pos, psp, cp_size=2)


# ---------------------------------------------------------------------------
# Tests for _gather_logprobs_context_parallel
# ---------------------------------------------------------------------------

class TestGatherLogprobsContextParallel:
    """Test _gather_logprobs_context_parallel without a real dist backend.

    The gather scatters each rank's local logprobs into a zero tensor at its
    partition indices and all-reduces (SUM) across the CP group; the partition
    being disjoint and complete makes the sum a reconstruction.
    """

    def _gather_no_grad(self, cp_size, seq_len, locals_and_indices, calling_rank=0):
        """Simulate the no_grad gather by patching all_reduce with the global sum."""
        from megatron.rl.rl_utils import _gather_logprobs_context_parallel

        fake_pg = _make_fake_pg()
        batch = locals_and_indices[0][0].shape[0]
        total = torch.zeros(batch, seq_len)
        for local_lp, idx in locals_and_indices:
            total += torch.zeros(batch, seq_len).index_copy(1, idx, local_lp)

        def fake_all_reduce(tensor, *args, **kwargs):
            tensor.copy_(total)

        local_lp, idx = locals_and_indices[calling_rank]
        with (
            patch('megatron.rl.rl_utils.mpu') as mock_mpu,
            patch('torch.distributed.all_reduce', side_effect=fake_all_reduce),
        ):
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            result = _gather_logprobs_context_parallel(local_lp, idx, seq_len, no_grad=True)
        return result

    def test_shape_after_gather(self):
        """Output shape must be [batch, seq_len - 1]."""
        seq_len, cp_size = 8, 2
        locals_and_indices = [
            (torch.zeros(1, 4), _reference_thd_partition_indices([0, 8], 8, cp_size, r))
            for r in range(cp_size)
        ]
        out = self._gather_no_grad(cp_size, seq_len, locals_and_indices)
        assert out.shape == (1, seq_len - 1)

    def test_values_invert_partition(self):
        """The gathered tensor must place every rank's values at its global positions."""
        seq_len, cp_size = 8, 2
        full = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0) + 100.0
        locals_and_indices = []
        for r in range(cp_size):
            idx = _reference_thd_partition_indices([0, 8], 8, cp_size, r)
            locals_and_indices.append((full.index_select(1, idx), idx))
        out = self._gather_no_grad(cp_size, seq_len, locals_and_indices)
        torch.testing.assert_close(out, full[:, :-1])

    def test_cp4_multi_sequence(self):
        """CP=4 with two padded slots of 8 tokens each."""
        seq_len, cp_size = 16, 4
        cu_padded = [0, 8, 16]
        full = torch.randn(2, seq_len)
        locals_and_indices = []
        for r in range(cp_size):
            idx = _reference_thd_partition_indices(cu_padded, seq_len, cp_size, r)
            locals_and_indices.append((full.index_select(1, idx), idx))
        out = self._gather_no_grad(cp_size, seq_len, locals_and_indices)
        torch.testing.assert_close(out, full[:, :-1])


# ---------------------------------------------------------------------------
# Tests that verify scatter + per-rank logprobs + gather == reference logprobs
# ---------------------------------------------------------------------------

class TestScatterGatherEquivalence:
    """Verify that partitioning tokens, computing logprobs per rank, then
    gathering gives the same result as the single-rank reference computation,
    for both single-sequence and packed multi-sequence (padded-slot) inputs."""

    @staticmethod
    def _reference_logprobs(logits, tokens):
        """selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])."""
        from megatron.rl.rl_utils import selective_log_softmax

        return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])

    @staticmethod
    def _cp_logprobs(logits, tokens, cu_padded, cp_size):
        """Simulate the CP path: per-sequence partition, compute per rank,
        scatter + sum (no actual distributed runtime)."""
        from megatron.rl.rl_utils import selective_log_softmax

        seq_len = tokens.shape[1]
        tokens_shifted = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
        full = torch.zeros(tokens.shape[0], seq_len)
        for cp_rank in range(cp_size):
            idx = _reference_thd_partition_indices(cu_padded, seq_len, cp_size, cp_rank)
            local_logits = logits.index_select(1, idx)
            local_labels = tokens_shifted.index_select(1, idx)
            local_lp = selective_log_softmax(local_logits, local_labels)
            full += torch.zeros(tokens.shape[0], seq_len).index_copy(1, idx, local_lp)
        return full[:, :-1]

    def test_cp2_single_sequence_matches_reference(self):
        torch.manual_seed(0)
        batch, seq_len, vocab = 1, 8, 16
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp = self._cp_logprobs(logits, tokens, [0, seq_len], cp_size=2)
        torch.testing.assert_close(ref, cp)

    def test_cp4_single_sequence_matches_reference(self):
        torch.manual_seed(42)
        batch, seq_len, vocab = 2, 16, 32
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp = self._cp_logprobs(logits, tokens, [0, seq_len], cp_size=4)
        torch.testing.assert_close(ref, cp)

    def test_cp2_packed_bin_matches_reference(self):
        """A packed bin with two aligned slots and a trailing ghost slot."""
        torch.manual_seed(7)
        batch, seq_len, vocab = 1, 24, 8
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp = self._cp_logprobs(logits, tokens, [0, 8, 16, 24], cp_size=2)
        torch.testing.assert_close(ref, cp)

    def test_cp2_packed_bin_with_ghost_padding_matches_reference(self):
        """Fixed-size cu_seqlens with repeated trailing boundaries (zero-length ghosts)."""
        torch.manual_seed(11)
        batch, seq_len, vocab = 1, 16, 8
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp = self._cp_logprobs(logits, tokens, [0, 8, 16, 16, 16], cp_size=2)
        torch.testing.assert_close(ref, cp)
