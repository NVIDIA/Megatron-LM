# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for context-parallel helpers in megatron/rl/rl_utils.py.

These tests exercise _scatter_for_context_parallel,
_gather_logprobs_context_parallel and create_packed_seq_params_for_bin
without requiring a real distributed environment (or a GPU) by patching
megatron.core.mpu and torch.distributed.all_gather.

The helpers implement the TE THD per-sequence zigzag CP layout: EACH
sequence in a packed bin, padded to a multiple of 2*CP, is split into 2*CP
equal chunks and rank r owns that sequence's chunks (r, 2*CP-r-1), in that
order, with sequences kept in bin order. This is the convention assumed by
tex.thd_get_partitioned_indices and its consumers (TE ring attention,
RoPE-CP, and _undo/_redo_attention_load_balancing in
megatron/core/ssm/mamba_context_parallel.py). Zigzag-splitting the whole
bin as if it were one sequence silently reassembles multi-sequence bins in
the wrong token order.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

try:  # pragma: no cover
    import triton  # noqa: F401
except ImportError:
    # CPU-only environments (e.g. macOS) have no triton wheel; stub the one
    # module in the rl_utils import chain that imports it unconditionally.
    sys.modules.setdefault('megatron.core.transformer.moe.paged_stash', MagicMock())

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.rl.rl_utils import (
    _gather_logprobs_context_parallel,
    _scatter_for_context_parallel,
    _thd_partitioned_indices,
    selective_log_softmax,
)
from megatron.rl.sequence_packing_utils import (
    PackingInfo,
    SequencePacker,
    create_packed_seq_params_for_bin,
)

PAD = 0
MAX_SEQUENCES_PER_BIN = 8


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------

def _make_fake_pg():
    pg = MagicMock()
    pg.__class__ = torch.distributed.ProcessGroup
    return pg


def _round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def _make_bin(seq_lengths, bin_size, cp_size):
    """Build a synthetic packed bin plus its PackedSeqParams.

    Tokens are globally unique (arange + 1) so any reordering is detectable;
    the trailing bin capacity is filled with PAD like SequencePacker does.
    """
    total = sum(seq_lengths)
    assert total <= bin_size
    tokens = torch.full((1, bin_size), PAD, dtype=torch.long)
    tokens[0, :total] = torch.arange(1, total + 1)
    position_ids = torch.zeros(1, bin_size, dtype=torch.long)
    start = 0
    for length in seq_lengths:
        position_ids[0, start : start + length] = torch.arange(length)
        start += length

    packing_info = PackingInfo(
        bin_seq_indices=[list(range(len(seq_lengths)))],
        seq_starts={0: list(torch.tensor([0] + list(seq_lengths)).cumsum(0).tolist())},
        seq_lengths=list(seq_lengths),
        seq_to_bin_idx=[0] * len(seq_lengths),
        packing_algo='fifo',
    )
    params = create_packed_seq_params_for_bin(
        packing_info=packing_info,
        bin_idx=0,
        bin_size=bin_size,
        max_sequences_per_bin=MAX_SEQUENCES_PER_BIN,
        device=torch.device('cpu'),
        cp_size=cp_size,
    )
    return tokens, position_ids, params


def _run_scatter(tokens, position_ids, params, cp_size, cp_rank):
    fake_pg = _make_fake_pg()
    with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
        mock_mpu.get_context_parallel_rank.return_value = cp_rank
        mock_mpu.get_context_parallel_group.return_value = fake_pg
        return _scatter_for_context_parallel(
            tokens, position_ids, params, cp_size, pad_token=PAD
        )


def _run_gather(local_logprobs_per_rank, cp_gather_index, cp_size, calling_rank=0):
    """Simulate the no_grad gather by patching all_gather."""
    fake_pg = _make_fake_pg()

    def fake_all_gather(out_list, tensor, group):
        for i, t in enumerate(local_logprobs_per_rank):
            out_list[i].copy_(t)

    with (
        patch('megatron.rl.rl_utils.mpu') as mock_mpu,
        patch('torch.distributed.all_gather', side_effect=fake_all_gather),
    ):
        mock_mpu.get_context_parallel_group.return_value = fake_pg
        mock_mpu.get_context_parallel_world_size.return_value = cp_size
        return _gather_logprobs_context_parallel(
            local_logprobs_per_rank[calling_rank], cp_gather_index, no_grad=True
        )


# ---------------------------------------------------------------------------
# Brute-force reference, written independently of the implementation
# ---------------------------------------------------------------------------

def _ref_padded_frame(tokens, seq_lengths, bin_size, cp_size):
    """Reference padded layout: each sequence padded to a 2*CP multiple with
    PAD, then the remaining bin capacity appended as one trailing pad block."""
    multiple = 2 * cp_size
    seqs = []
    start = 0
    for length in seq_lengths:
        seq = tokens[0, start : start + length]
        pad = torch.full((_round_up(length, multiple) - length,), PAD, dtype=tokens.dtype)
        seqs.append(torch.cat([seq, pad]))
        start += length
    tail = bin_size - sum(len(s) for s in seqs)
    assert tail >= 0
    if tail > 0:
        seqs.append(torch.full((tail,), PAD, dtype=tokens.dtype))
    return seqs


def _ref_rank_slice(padded_seqs, cp_size, cp_rank):
    """Per-sequence zigzag: rank r takes chunks (r, 2*CP-1-r) of each padded
    sequence, in that order, sequences kept in order."""
    out = []
    for seq in padded_seqs:
        n = len(seq) // (2 * cp_size)
        chunks = [seq[i * n : (i + 1) * n] for i in range(2 * cp_size)]
        out.append(torch.cat([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]]))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# (a) Scatter matches the per-sequence zigzag reference for all ranks
# ---------------------------------------------------------------------------

class TestScatterPerSequenceZigzag:

    @pytest.mark.parametrize('cp_size', [2, 4])
    @pytest.mark.parametrize(
        'seq_lengths',
        [
            [13],                    # 1 sequence, odd length
            [7, 5],                  # 2 sequences
            [9, 3, 5],               # 3 sequences
            [11, 1, 7, 3],           # 4 sequences incl. length-1
            [5, 9, 3, 7, 1],         # 5 sequences
        ],
    )
    def test_local_tokens_match_reference(self, cp_size, seq_lengths):
        bin_size = 32 * cp_size
        tokens, position_ids, params = _make_bin(seq_lengths, bin_size, cp_size)
        padded_seqs = _ref_padded_frame(tokens, seq_lengths, bin_size, cp_size)
        for cp_rank in range(cp_size):
            local_tokens, _, _, _, _ = _run_scatter(
                tokens, position_ids, params, cp_size, cp_rank
            )
            expected = _ref_rank_slice(padded_seqs, cp_size, cp_rank)
            assert local_tokens.shape == (1, bin_size // cp_size)
            torch.testing.assert_close(local_tokens[0], expected)

    @pytest.mark.parametrize('cp_size', [2, 4])
    def test_union_of_ranks_reconstructs_padded_frame(self, cp_size):
        seq_lengths = [9, 3, 5]
        bin_size = 16 * cp_size
        tokens, position_ids, params = _make_bin(seq_lengths, bin_size, cp_size)
        padded_frame = torch.cat(_ref_padded_frame(tokens, seq_lengths, bin_size, cp_size))
        reconstructed = torch.empty_like(padded_frame)
        for cp_rank in range(cp_size):
            local_tokens, _, _, _, _ = _run_scatter(
                tokens, position_ids, params, cp_size, cp_rank
            )
            index = _thd_partitioned_indices(
                params.cu_seqlens_q_padded, bin_size, cp_size, cp_rank
            )
            reconstructed[index] = local_tokens[0]
        torch.testing.assert_close(reconstructed, padded_frame)

    def test_partitioned_indices_are_a_permutation(self):
        cp_size, bin_size = 4, 64
        _, _, params = _make_bin([9, 3, 5], bin_size, cp_size)
        all_index = torch.cat(
            [
                _thd_partitioned_indices(params.cu_seqlens_q_padded, bin_size, cp_size, r)
                for r in range(cp_size)
            ]
        )
        assert torch.equal(torch.sort(all_index).values, torch.arange(bin_size))

    def test_labels_are_next_padded_token(self):
        """Local labels must equal the next token of the padded frame so that
        within-sequence positions predict the true next token."""
        cp_size, seq_lengths = 2, [7, 5]
        bin_size = 32
        tokens, position_ids, params = _make_bin(seq_lengths, bin_size, cp_size)
        padded_frame = torch.cat(_ref_padded_frame(tokens, seq_lengths, bin_size, cp_size))
        shifted = torch.cat([padded_frame[1:], padded_frame[-1:]])
        for cp_rank in range(cp_size):
            _, _, _, local_labels, _ = _run_scatter(
                tokens, position_ids, params, cp_size, cp_rank
            )
            index = _thd_partitioned_indices(
                params.cu_seqlens_q_padded, bin_size, cp_size, cp_rank
            )
            torch.testing.assert_close(local_labels[0], shifted[index])
            assert local_labels.is_contiguous()

    def test_cp_fields_set_on_copy(self):
        """cp_group, local_cp_size and cu_seqlens_*_padded must be set on the
        returned copy; the original must be unchanged."""
        cp_size, bin_size = 2, 32
        tokens, position_ids, params = _make_bin([7, 5], bin_size, cp_size)
        _, _, cp_params, _, _ = _run_scatter(tokens, position_ids, params, cp_size, 0)
        assert cp_params.local_cp_size == cp_size
        assert cp_params.cp_group is not None
        assert cp_params.cu_seqlens_q_padded is not None
        assert cp_params.cu_seqlens_kv_padded is not None
        # Original must not have been mutated.
        assert params.local_cp_size is None
        assert params.cp_group is None

    def test_assertion_on_indivisible_seq_len(self):
        """The bin length itself must be divisible by 2*cp_size."""
        tokens = torch.zeros(1, 9, dtype=torch.long)
        pos = torch.zeros(1, 9, dtype=torch.long)
        cu = torch.tensor([0, 9], dtype=torch.int32)
        psp = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                              max_seqlen_q=9, max_seqlen_kv=9, total_tokens=9)
        with pytest.raises(AssertionError, match="divisible"):
            _run_scatter(tokens, pos, psp, cp_size=2, cp_rank=0)

    def test_assertion_on_seq_len_divisible_by_cp_but_not_2cp(self):
        """seq_len=6, cp_size=2: 6 % 2 == 0 but 6 % 4 != 0 — must fail."""
        tokens = torch.zeros(1, 6, dtype=torch.long)
        pos = torch.zeros(1, 6, dtype=torch.long)
        cu = torch.tensor([0, 6], dtype=torch.int32)
        psp = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                              max_seqlen_q=6, max_seqlen_kv=6, total_tokens=6)
        with pytest.raises(AssertionError, match="divisible"):
            _run_scatter(tokens, pos, psp, cp_size=2, cp_rank=0)


# ---------------------------------------------------------------------------
# (b) Round-trip: gather(scatter(x)) == x for every rank composition
# ---------------------------------------------------------------------------

class TestScatterGatherRoundTrip:

    @pytest.mark.parametrize('cp_size', [1, 2, 4])
    @pytest.mark.parametrize(
        'seq_lengths',
        [[13], [7, 5], [9, 3, 5], [11, 1, 7, 3], [5, 9, 3, 7, 1]],
    )
    def test_round_trip_restores_original_layout(self, cp_size, seq_lengths):
        bin_size = 32 * max(cp_size, 1)
        tokens, position_ids, params = _make_bin(seq_lengths, bin_size, cp_size)

        results = [
            _run_scatter(tokens, position_ids, params, cp_size, cp_rank)
            for cp_rank in range(cp_size)
        ]
        cp_gather_index = results[0][4]
        # All ranks must compute the identical gather index.
        for r in range(1, cp_size):
            torch.testing.assert_close(results[r][4], cp_gather_index)

        # Feed each rank's local tokens back as if they were its local
        # logprob outputs; the gather must restore the original (unpadded)
        # bin layout at every real-token position.
        locals_as_vals = [res[0].float() for res in results]
        gathered = _run_gather(locals_as_vals, cp_gather_index, cp_size)
        assert gathered.shape == (1, bin_size - 1)

        expected = tokens[0].float()
        # Real tokens (and the residual pad tail retained as a pseudo-
        # sequence) must round-trip exactly. Original positions displaced by
        # the inserted per-sequence padding are junk in both layouts (always
        # loss-masked), so they are excluded.
        multiple = 2 * cp_size
        mapped = sum(seq_lengths) + bin_size - sum(
            _round_up(length, multiple) for length in seq_lengths
        )
        mapped = min(mapped, bin_size - 1)  # the final position is dropped
        torch.testing.assert_close(gathered[0, :mapped], expected[:mapped])

    def test_round_trip_is_exact_identity_without_padding(self):
        """When every sequence length is already a 2*CP multiple no padding is
        inserted, and the full round trip (including the pad tail) is exact."""
        cp_size, seq_lengths, bin_size = 2, [8, 4, 12], 32
        tokens, position_ids, params = _make_bin(seq_lengths, bin_size, cp_size)
        results = [
            _run_scatter(tokens, position_ids, params, cp_size, cp_rank)
            for cp_rank in range(cp_size)
        ]
        gathered = _run_gather(
            [res[0].float() for res in results], results[0][4], cp_size
        )
        torch.testing.assert_close(gathered[0], tokens[0, :-1].float())

    def test_logprobs_match_single_rank_reference(self):
        """Slice per-position logits with the scatter, compute local logprobs
        with the scatter's labels, gather, and compare against the plain
        selective_log_softmax(logits[:, :-1], tokens[:, 1:]) reference at all
        within-sequence positions."""
        torch.manual_seed(0)
        cp_size, seq_lengths, bin_size = 2, [9, 3, 5], 32
        # Encode each token as its own position so logits can be looked up in
        # any frame: vocab == bin_size, token id == original position.
        tokens = torch.full((1, bin_size), PAD, dtype=torch.long)
        tokens[0, : sum(seq_lengths)] = torch.arange(1, sum(seq_lengths) + 1)
        position_ids = torch.zeros(1, bin_size, dtype=torch.long)
        _, _, params = _make_bin(seq_lengths, bin_size, cp_size)
        vocab = bin_size + 1
        logits = torch.randn(1, vocab, vocab)  # logits keyed by token id

        ref = selective_log_softmax(logits[:, tokens[0, :-1], :], tokens[:, 1:])

        per_rank = []
        gather_index = None
        for cp_rank in range(cp_size):
            local_tokens, _, _, local_labels, gather_index = _run_scatter(
                tokens, position_ids, params, cp_size, cp_rank
            )
            local_logits = logits[:, local_tokens[0], :]
            per_rank.append(selective_log_softmax(local_logits, local_labels))
        out = _run_gather(per_rank, gather_index, cp_size)

        # Valid positions: token j and j+1 belong to the same sequence.
        valid = torch.zeros(bin_size - 1, dtype=torch.bool)
        start = 0
        for length in seq_lengths:
            valid[start : start + length - 1] = True
            start += length
        torch.testing.assert_close(out[0, valid], ref[0, valid])


# ---------------------------------------------------------------------------
# (c) cu_seqlens_padded invariants
# ---------------------------------------------------------------------------

class TestPackedSeqParamsForCP:

    @pytest.mark.parametrize('cp_size', [2, 4, 8])
    @pytest.mark.parametrize(
        'seq_lengths', [[13], [7, 5], [9, 3, 5], [11, 1, 7, 3], [5, 9, 3, 7, 1]]
    )
    def test_cu_seqlens_padded_invariants(self, cp_size, seq_lengths):
        bin_size = 32 * cp_size
        _, _, params = _make_bin(seq_lengths, bin_size, cp_size)
        cu = params.cu_seqlens_q.long()
        cu_padded = params.cu_seqlens_q_padded.long()

        assert cu_padded is not None
        assert cu.shape == cu_padded.shape == (MAX_SEQUENCES_PER_BIN + 2,)
        # Monotonic (non-decreasing), starting at 0, padded frame tiles the bin.
        assert cu[0] == 0 and cu_padded[0] == 0
        assert (cu[1:] >= cu[:-1]).all()
        assert (cu_padded[1:] >= cu_padded[:-1]).all()
        assert cu_padded[-1] == bin_size
        # Every padded sequence length is a multiple of 2*cp_size.
        padded_lens = cu_padded[1:] - cu_padded[:-1]
        assert (padded_lens % (2 * cp_size) == 0).all()
        # Unpadded lengths match the input sequences and never exceed the
        # padded ones.
        actual_lens = cu[1:] - cu[:-1]
        assert actual_lens[: len(seq_lengths)].tolist() == seq_lengths
        assert (actual_lens <= padded_lens).all()

    def test_padding_overflow_asserts(self):
        """A bin packed beyond its CP-padded capacity must fail loudly."""
        cp_size = 4  # multiple = 8: [7, 5, 9] pads to 8 + 8 + 16 = 32 > 24
        packing_info = PackingInfo(
            bin_seq_indices=[[0, 1, 2]],
            seq_starts={0: [0, 7, 12, 21]},
            seq_lengths=[7, 5, 9],
            seq_to_bin_idx=[0, 0, 0],
            packing_algo='fifo',
        )
        with pytest.raises(AssertionError, match="overflows"):
            create_packed_seq_params_for_bin(
                packing_info=packing_info,
                bin_idx=0,
                bin_size=24,
                max_sequences_per_bin=MAX_SEQUENCES_PER_BIN,
                device=torch.device('cpu'),
                cp_size=cp_size,
            )


# ---------------------------------------------------------------------------
# (d) cp_size == 1 keeps the old behavior exactly
# ---------------------------------------------------------------------------

class TestCpSize1Unchanged:

    @pytest.mark.parametrize(
        'seq_lengths', [[13], [7, 5], [9, 3, 5], [11, 1, 7, 3], [5, 9, 3, 7, 1]]
    )
    def test_builder_matches_legacy_layout(self, seq_lengths):
        """cp_size=1 params must be byte-identical to the pre-CP-fix builder:
        cu_seqlens = [0, cumsums..., bin_size, ghosts=bin_size...], padded=None."""
        bin_size = 64
        _, _, params = _make_bin(seq_lengths, bin_size, cp_size=1)
        legacy = torch.full((MAX_SEQUENCES_PER_BIN + 2,), bin_size, dtype=torch.int32)
        legacy[0] = 0
        cumsum = 0
        for i, length in enumerate(seq_lengths):
            cumsum += length
            legacy[i + 1] = cumsum
        assert params.cu_seqlens_q_padded is None
        assert params.cu_seqlens_kv_padded is None
        torch.testing.assert_close(params.cu_seqlens_q, legacy)

    def test_scatter_is_identity_at_cp1(self):
        """With cp_size=1 the (never-taken in production) scatter degenerates
        to an exact identity: local == global, gather index == arange."""
        seq_lengths, bin_size = [8, 4, 12], 32
        tokens, position_ids, params = _make_bin(seq_lengths, bin_size, cp_size=1)
        local_tokens, local_pos, _, local_labels, gather_index = _run_scatter(
            tokens, position_ids, params, cp_size=1, cp_rank=0
        )
        torch.testing.assert_close(local_tokens, tokens)
        torch.testing.assert_close(local_pos, position_ids)
        shifted = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
        torch.testing.assert_close(local_labels, shifted)
        torch.testing.assert_close(gather_index, torch.arange(bin_size - 1))

    def test_packer_capacity_unchanged_with_multiple_1(self):
        """seq_length_multiple=1 (the cp_size==1 configuration) must produce
        the same packing as the legacy capacity check."""
        torch.manual_seed(0)
        bin_size, pad = 16, PAD
        lengths = [9, 7, 5, 3, 11, 2, 6]
        trajs = torch.full((len(lengths), bin_size), pad, dtype=torch.long)
        for i, length in enumerate(lengths):
            trajs[i, :length] = torch.randint(1, 50, (length,))
        packer = SequencePacker(bin_size=bin_size, pad_token=pad, seq_length_multiple=1)
        _, _, _, packing_info = packer.pack_sequences(trajs)

        # Legacy first-fit over lengths sorted descending.
        expected_bins, current, current_len = [], [], 0
        for idx in sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True):
            if current_len + lengths[idx] <= bin_size and len(current) < 16:
                current.append(idx)
                current_len += lengths[idx]
            else:
                expected_bins.append(current)
                current, current_len = [idx], lengths[idx]
        expected_bins.append(current)
        assert packing_info.bin_seq_indices == expected_bins


# ---------------------------------------------------------------------------
# Packer capacity accounting under CP padding
# ---------------------------------------------------------------------------

class TestPackerCpCapacity:

    @pytest.mark.parametrize('cp_size', [2, 4])
    def test_padded_lengths_never_overflow_bins(self, cp_size):
        torch.manual_seed(1)
        multiple = 2 * cp_size
        bin_size = 8 * multiple
        lengths = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        trajs = torch.full((len(lengths), bin_size), PAD, dtype=torch.long)
        for i, length in enumerate(lengths):
            trajs[i, :length] = torch.randint(1, 50, (length,))
        packer = SequencePacker(
            bin_size=bin_size, pad_token=PAD, seq_length_multiple=multiple
        )
        _, _, _, packing_info = packer.pack_sequences(trajs)
        for seq_indices in packing_info.bin_seq_indices:
            padded_total = sum(
                _round_up(packing_info.seq_lengths[idx], multiple) for idx in seq_indices
            )
            assert padded_total <= bin_size
            # And therefore the params builder accepts every bin.
            params = create_packed_seq_params_for_bin(
                packing_info=packing_info,
                bin_idx=packing_info.bin_seq_indices.index(seq_indices),
                bin_size=bin_size,
                max_sequences_per_bin=16,
                device=torch.device('cpu'),
                cp_size=cp_size,
            )
            assert params is not None
