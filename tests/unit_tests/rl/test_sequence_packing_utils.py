# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
from unittest.mock import patch

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.rl import rl_utils, sequence_packing_utils
from megatron.training import arguments, global_vars


class MockTokenizer:
    def __init__(self):
        self.pad = 42
        self.eod = 43
        self.vocab_size = 754
        self.bos = None

    def detokenize(self, tokens):
        return [str(tok) for tok in tokens]


def test_get_actual_sequence_lengths():
    pad_token = 42

    sequences = torch.tensor(
        [
            [1, 2, 3, pad_token, pad_token],
            [4, 5, 6, 7, 8],
            [9, pad_token, pad_token, pad_token, pad_token],
            [pad_token, pad_token, pad_token, pad_token, pad_token],
        ]
    )

    lengths = sequence_packing_utils.get_actual_sequence_lengths(sequences, pad_token)

    assert lengths == [3, 5, 1, 0]


def test_get_actual_sequence_lengths_with_interior_padding():
    pad_token = 42

    sequences = torch.tensor(
        [[1, pad_token, 3, pad_token, pad_token], [pad_token, 2, 3, 4, pad_token]]
    )

    lengths = sequence_packing_utils.get_actual_sequence_lengths(sequences, pad_token)

    assert lengths == [3, 4]


def test_get_actual_sequence_lengths_invalid_shape():
    pad_token = 42
    sequences_1d = torch.tensor([1, 2, 3])

    try:
        sequence_packing_utils.get_actual_sequence_lengths(sequences_1d, pad_token)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected 2D tensor" in str(e)


def test_sequence_packing_basic():
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 16)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 16
    packer = sequence_packing_utils.SequencePacker(bin_size=bin_size, pad_token=tokenizer.pad)

    max_len = 5
    sequences = [
        torch.cat(
            [
                torch.tensor([1, 2, 3, tokenizer.eod]),
                torch.full((1,), tokenizer.pad, dtype=torch.long),
            ]
        ),
        torch.cat(
            [torch.tensor([4, 5, tokenizer.eod]), torch.full((2,), tokenizer.pad, dtype=torch.long)]
        ),
        torch.tensor([6, 7, 8, 9, tokenizer.eod]),
        torch.cat(
            [torch.tensor([10, tokenizer.eod]), torch.full((3,), tokenizer.pad, dtype=torch.long)]
        ),
    ]

    generation_masks = torch.tensor(
        [
            [False, True, True, True, False],
            [False, True, True, False, False],
            [False, True, True, True, True],
            [False, True, False, False, False],
        ]
    )

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

    sequences_tensor = torch.stack(sequences)
    packed_trajs, packed_position_ids, packed_loss_mask, packing_info = packer.pack_sequences(
        sequences_tensor, generation_masks
    )

    assert packed_trajs is not None
    assert packed_position_ids is not None
    assert packed_loss_mask is not None
    assert packing_info is not None

    assert packed_trajs.shape[0] >= 1
    assert packed_trajs.shape[1] == bin_size

    for bin_idx in range(packed_trajs.shape[0]):
        for i in range(packed_trajs.shape[1]):
            if i == 0 or packed_trajs[bin_idx, i - 1] == tokenizer.eod:
                if packed_trajs[bin_idx, i] != tokenizer.pad:
                    assert packed_position_ids[bin_idx, i] == 0


def test_sequence_packing_with_generation_masks():
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 20)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 20
    packer = sequence_packing_utils.SequencePacker(bin_size=bin_size, pad_token=tokenizer.pad)

    sequences = [torch.tensor([1, 2, 3, tokenizer.eod]), torch.tensor([4, 5, 6, 7, tokenizer.eod])]

    max_len = max(len(s) for s in sequences)
    padded_sequences = []
    for seq in sequences:
        padded = torch.cat([seq, torch.full((max_len - len(seq),), tokenizer.pad, dtype=seq.dtype)])
        padded_sequences.append(padded)

    generation_masks = torch.tensor(
        [[False, True, True, True, False], [False, True, True, True, True]]
    )

    padded_sequences_tensor = torch.stack(padded_sequences)
    packed_trajs, packed_position_ids, packed_loss_mask, packing_info = packer.pack_sequences(
        padded_sequences_tensor, generation_masks
    )

    assert packed_trajs.shape[0] == 1
    assert packed_trajs.shape[1] == bin_size


def test_sequence_packing_empty_bins():
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 8)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 8
    num_empty_bins = 3

    packed_trajs = torch.tensor(
        [[1, 2, 3, tokenizer.eod, tokenizer.pad, tokenizer.pad, tokenizer.pad, tokenizer.pad]]
    )
    packed_position_ids = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0]])
    packed_loss_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float)

    empty_trajs, empty_position_ids, empty_loss_mask, empty_packing_info = (
        sequence_packing_utils.create_empty_bins(
            num_empty_bins=num_empty_bins,
            bin_size=bin_size,
            packed_trajs=packed_trajs,
            packed_position_ids=packed_position_ids,
            packed_loss_mask=packed_loss_mask,
            tokenizer=tokenizer,
        )
    )

    assert empty_trajs.shape[0] == num_empty_bins
    assert empty_trajs.shape[1] == bin_size

    for i in range(num_empty_bins):
        assert torch.all(empty_trajs[i] == tokenizer.pad)
        assert torch.all(empty_position_ids[i] == 0)
        assert torch.all(empty_loss_mask[i] == 0)

    assert len(empty_packing_info) == num_empty_bins
    for info in empty_packing_info:
        assert len(info['bin_seq_indices']) == 0
        assert len(info['seq_starts']) == 0


def test_sequence_packing_integration():
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 16)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 16

    packer = sequence_packing_utils.SequencePacker(bin_size=bin_size, pad_token=tokenizer.pad)

    max_len = 5
    sequences = [
        torch.cat(
            [
                torch.tensor([1, 2, 3, tokenizer.eod]),
                torch.full((1,), tokenizer.pad, dtype=torch.long),
            ]
        ),
        torch.cat(
            [torch.tensor([4, 5, tokenizer.eod]), torch.full((2,), tokenizer.pad, dtype=torch.long)]
        ),
        torch.tensor([6, 7, 8, 9, tokenizer.eod]),
    ]
    generation_masks = [
        torch.tensor([False, True, True, True, False]),
        torch.tensor([False, True, True, False, False]),
        torch.tensor([False, True, True, True, True]),
    ]

    sequences_tensor = torch.stack(sequences)
    packed_trajs, packed_position_ids, packed_loss_mask, packing_info = packer.pack_sequences(
        sequences_tensor, generation_masks
    )

    assert packed_trajs is not None
    assert packed_trajs.shape[1] == bin_size
    assert packed_position_ids.shape == packed_trajs.shape
    assert packed_loss_mask.shape == packed_trajs.shape

    assert packed_trajs.shape[0] == 1

    expected_start = torch.tensor(
        [6, 7, 8, 9, tokenizer.eod, 1, 2, 3, tokenizer.eod, 4, 5, tokenizer.eod]
    )
    assert torch.all(packed_trajs[0, :12] == expected_start)

    assert torch.all(packed_trajs[0, 12:] == tokenizer.pad)


class MockGroupStats:
    """Mock group stats object for testing."""

    def __init__(self):
        self.min_piold_to_inf_prob = None
        self.max_piold_to_inf_prob = None
        self.mean_piold_to_inf_prob = None
        self.min_inf_train_prob_abs_diff = None
        self.max_inf_train_prob_abs_diff = None
        self.mean_inf_train_prob_abs_diff = None
        self.min_inf_prob = None
        self.max_inf_prob = None
        self.mean_inf_prob = None


def test_update_inference_logprobs_group_stats():
    """Test the common statistics computation helper function."""
    # Create matching logprobs (should give ratio ~1.0)
    old_logprobs = torch.tensor([[-0.5, -0.3, -0.2, 0.0]])
    inference_logprobs = torch.tensor([[-0.5, -0.3, -0.2, 0.0]])
    mask = torch.tensor([[True, True, True, False]])

    group_stats = MockGroupStats()

    rl_utils.update_inference_logprobs_group_stats(
        old_logprobs=old_logprobs,
        inference_logprobs=inference_logprobs,
        mask=mask,
        group_stats=group_stats,
    )

    # When logprobs match exactly, ratio should be 1.0 and diff should be 0.0
    assert abs(group_stats.mean_piold_to_inf_prob - 1.0) < 1e-6
    assert abs(group_stats.mean_inf_train_prob_abs_diff) < 1e-6


def test_update_inference_logprobs_group_stats_empty_mask():
    """Test statistics computation with empty mask."""
    old_logprobs = torch.tensor([[-0.5, -0.3]])
    inference_logprobs = torch.tensor([[-0.5, -0.3]])
    mask = torch.tensor([[False, False]])  # Empty mask

    group_stats = MockGroupStats()

    rl_utils.update_inference_logprobs_group_stats(
        old_logprobs=old_logprobs,
        inference_logprobs=inference_logprobs,
        mask=mask,
        group_stats=group_stats,
    )

    # With empty mask, stats should remain None
    assert group_stats.mean_piold_to_inf_prob is None


def test_update_inference_logprobs_group_stats_with_mismatch():
    """Test statistics when inference and old logprobs differ."""
    # Old logprobs
    old_logprobs = torch.tensor([[-0.5, -0.5, -0.5]])
    # Inference logprobs with different values
    inference_logprobs = torch.tensor([[-1.0, -1.0, -1.0]])
    mask = torch.tensor([[True, True, True]])

    group_stats = MockGroupStats()

    rl_utils.update_inference_logprobs_group_stats(
        old_logprobs=old_logprobs,
        inference_logprobs=inference_logprobs,
        mask=mask,
        group_stats=group_stats,
    )

    # With different logprobs, ratio should not be 1.0
    # exp(-0.5) / exp(-1.0) = exp(0.5) ≈ 1.65
    assert group_stats.mean_piold_to_inf_prob > 1.0

    # Abs diff should be non-zero
    assert group_stats.mean_inf_train_prob_abs_diff > 0.0


def test_compute_packed_inference_logprobs_stats():
    """Test compute_packed_inference_logprobs_stats with packed data."""
    # Create packed data (simulating 2 bins)
    # old_logprobs shape: [num_bins, seq_len-1]
    old_logprobs = torch.tensor(
        [
            [-0.5, -0.3, -0.2, 0.0, 0.0, 0.0, 0.0],  # bin 0
            [-0.4, -0.6, -0.1, 0.0, 0.0, 0.0, 0.0],  # bin 1
        ]
    )

    # packed_inference_logprobs with same values (should give ratio ~1.0)
    packed_inference_logprobs = torch.tensor(
        [
            [-0.5, -0.3, -0.2, 0.0, 0.0, 0.0, 0.0],  # bin 0
            [-0.4, -0.6, -0.1, 0.0, 0.0, 0.0, 0.0],  # bin 1
        ]
    )

    # packed_loss_mask: [num_bins, seq_len] - indicates valid positions
    # Note: function shifts by 1, so packed_loss_mask[:, 1:] is used
    packed_loss_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # bin 0: 3 valid tokens
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # bin 1: 3 valid tokens
        ]
    )

    group_stats = MockGroupStats()

    sequence_packing_utils.compute_packed_inference_logprobs_stats(
        old_logprobs=old_logprobs,
        packed_inference_logprobs=packed_inference_logprobs,
        packed_loss_mask=packed_loss_mask,
        group_stats=group_stats,
    )

    # Verify statistics were computed
    assert group_stats.min_piold_to_inf_prob is not None
    assert group_stats.max_piold_to_inf_prob is not None
    assert group_stats.mean_piold_to_inf_prob is not None

    # When logprobs match exactly, ratio should be 1.0
    assert abs(group_stats.mean_piold_to_inf_prob - 1.0) < 1e-6
    assert abs(group_stats.mean_inf_train_prob_abs_diff) < 1e-6


def test_compute_packed_inference_logprobs_stats_with_mismatch():
    """Test compute_packed_inference_logprobs_stats when values differ."""
    # old_logprobs
    old_logprobs = torch.tensor([[-0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])

    # Different inference logprobs
    packed_inference_logprobs = torch.tensor([[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]])

    # packed_loss_mask
    packed_loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    group_stats = MockGroupStats()

    sequence_packing_utils.compute_packed_inference_logprobs_stats(
        old_logprobs=old_logprobs,
        packed_inference_logprobs=packed_inference_logprobs,
        packed_loss_mask=packed_loss_mask,
        group_stats=group_stats,
    )

    # With different logprobs, ratio should not be 1.0
    assert group_stats.mean_piold_to_inf_prob > 1.0
    assert group_stats.mean_inf_train_prob_abs_diff > 0.0


def test_compute_packed_inference_logprobs_stats_shape_mismatch():
    """Test that function handles shape mismatch gracefully."""
    # Mismatched shapes
    old_logprobs = torch.tensor([[-0.5, -0.3, -0.2]])  # 3 elements
    packed_inference_logprobs = torch.tensor([[-0.5, -0.3, -0.2]])
    packed_loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])  # 6 elements -> 5 after shift

    group_stats = MockGroupStats()

    # Should not raise, but stats should remain None due to shape mismatch
    sequence_packing_utils.compute_packed_inference_logprobs_stats(
        old_logprobs=old_logprobs,
        packed_inference_logprobs=packed_inference_logprobs,
        packed_loss_mask=packed_loss_mask,
        group_stats=group_stats,
    )

    # Stats should remain None due to shape mismatch
    assert group_stats.mean_piold_to_inf_prob is None


def test_packing_observability_metrics():
    """Test various observability metrics related to sequence packing."""

    # 4 sequences with known lengths packed into 2 bins of size 16.
    # Bin 0 holds seqs 0 (len 5) and 1 (len 3) → 8 actual tokens
    # Bin 1 holds seqs 2 (len 10) and 3 (len 4) → 14 actual tokens
    seq_lengths = [5, 3, 10, 4]
    packing_info = sequence_packing_utils.PackingInfo(
        bin_seq_indices=[[0, 1], [2, 3]],
        seq_starts={0: [0, 5], 1: [0, 10]},
        seq_lengths=seq_lengths,
        seq_to_bin_idx=[0, 0, 1, 1],
        packing_algo='fifo',
    )

    num_bins, bin_size = 2, 16
    packed_trajs = torch.zeros(num_bins, bin_size, dtype=torch.long)
    ctx = sequence_packing_utils.PackingContext(
        bin_size=bin_size,
        packer=None,
        packing_info=packing_info,
        original_generation_masks=None,
        original_trajs=None,
        packed_trajs=packed_trajs,
        packed_position_ids=None,
        packed_loss_mask=None,
    )

    # actual tokens = sum of all seq_lengths referenced by bin_seq_indices
    assert sequence_packing_utils.get_packing_actual_tokens(ctx) == 5 + 3 + 10 + 4

    # compute tokens = num_bins * bin_size
    assert sequence_packing_utils.get_packing_compute_tokens(ctx) == 2 * 16

    # avg seq length = mean of seq_lengths
    assert sequence_packing_utils.get_packing_avg_seq_length(ctx) == pytest.approx(22 / 4)

    # efficiency = total_actual / (bins_per_rank * bin_size * num_ranks)
    with patch('megatron.core.mpu.get_data_parallel_world_size', return_value=4):
        eff = sequence_packing_utils.get_packing_efficiency(ctx)
        # total_actual = sum(seq_lengths) = 22, capacity = 2 * 16 * 4 = 128
        assert eff == pytest.approx(22 / 128)


@pytest.mark.parametrize("num_sequences", [1, 10, 48, 49, 50])
def test_cu_seqlens_size(num_sequences):
    """Test that cu_seqlens always has a fixed size regardless of how many sequences are packed."""
    max_sequences_per_bin = 50
    bin_size = 1024

    seq_len = bin_size // max_sequences_per_bin
    seq_lengths = [seq_len] * num_sequences

    packing_info = sequence_packing_utils.PackingInfo(
        bin_seq_indices=[list(range(num_sequences))],
        # Back-to-back starts plus the padded end sentinel, as pack_sequences records them.
        seq_starts={0: [i * seq_len for i in range(num_sequences + 1)]},
        seq_lengths=seq_lengths,
        seq_to_bin_idx=[0] * num_sequences,
        packing_algo='fifo',
    )

    params = sequence_packing_utils.create_packed_seq_params_for_bin(
        packing_info=packing_info,
        bin_idx=0,
        bin_size=bin_size,
        max_sequences_per_bin=max_sequences_per_bin,
        device=torch.device('cpu'),
    )

    expected_size = max_sequences_per_bin + 2
    assert params.cu_seqlens_q.shape[0] == expected_size, (
        f"cu_seqlens_q has size {params.cu_seqlens_q.shape[0]} but expected {expected_size} "
        f"for {num_sequences} sequences"
    )
    assert params.cu_seqlens_kv.shape[0] == expected_size
    assert params.cu_seqlens_q[0] == 0
    assert params.cu_seqlens_q[-1] == bin_size


@pytest.mark.parametrize(
    "ratio,local_bins,world,expected_bs",
    [
        (1.0, 1, 8, 8),  # no stale data (ratio 1.), everything divides perfectly.
        (1.0, 42, 8, 42 * 8),  # no stale data (ratio 1.), everything divides perfectly, more bins
        (
            0.5,
            1,
            8,
            8,
        ),  # 0.5 means we use half of all seqs per step, they all fit 1 bin -> we should reuse
        (1 / 3, 4, 8, 16),  # third of the data per step, nonint division
    ],
)
def test_get_bins_bs_and_steps(ratio, local_bins, world, expected_bs):
    # Make a dummy struct to check only the required fields.
    # Divide by ratio to make sure the samples are divisible by global_bs in the test.
    n_seqs = int(world * 7 / ratio)
    global_bs_in_seq = int(n_seqs * ratio)

    def side_eff(
        rank, global_batch_size, micro_batch_size, data_parallel_size, decrease_batch_size_if_needed
    ):
        # Inside of the get_microbatch_dataloader, we compute the batch size in bins.
        # We want to test this variable.
        global actual_bs
        actual_bs = global_batch_size

    with patch('megatron.rl.sequence_packing_utils.get_num_microbatches', return_value=1):
        with patch(
            'megatron.rl.sequence_packing_utils.reconfigure_num_microbatches_calculator',
            side_effect=side_eff,
        ):
            with patch('megatron.core.mpu.get_data_parallel_world_size', return_value=world):
                sequence_packing_utils.update_microbatch_calculator(
                    samples_ratio_per_step=ratio,
                    num_bins_this_rank=local_bins,
                    bin_seq_indices=[],
                    global_batch_size=global_bs_in_seq,
                    micro_batch_size=1,
                    decrease_batch_size_if_needed=False,
                )

    # Iterator is local, batch size is global
    assert expected_bs == actual_bs


@pytest.mark.parametrize(
    "multiple, bin_size, expected_starts, expected_cu_padded, expected_cu",
    [
        # Back-to-back placement: the layout equals the actual boundaries, so the
        # padded fields stay None (keeps TE off its pad_between_seqs GPU sync,
        # which would break CUDA graph capture).
        pytest.param(1, 16, [0, 7, 12], None, [0, 7, 12, 16, 16, 16], id="legacy_back_to_back"),
        # 4-aligned placement (cp=2): lengths 7 and 5 reserve footprints of 8;
        # cu_seqlens_*_padded is the slot grid (with the trailing ghost slot),
        # cu_seqlens_* the real token counts (the ghost keeps its full size).
        pytest.param(
            4, 24, [0, 8, 16], [0, 8, 16, 24, 24, 24], [0, 7, 12, 20, 20, 20], id="aligned_for_cp"
        ),
    ],
)
def test_sequence_packing_alignment_pipeline(
    multiple, bin_size, expected_starts, expected_cu_padded, expected_cu
):
    """pack_sequences places each sequence at a seq_length_multiple-aligned offset
    (gaps stay pad tokens with a zero loss mask) and create_packed_seq_params_for_bin
    turns that placement into the cu_seqlens / cu_seqlens_padded pair, materializing
    the padded boundaries only when alignment makes them differ."""
    tokenizer = MockTokenizer()
    seq_a = torch.cat([torch.arange(1, 6), torch.full((2,), tokenizer.pad)])  # length 5
    seq_b = torch.arange(10, 17)  # length 7, placed first (packer sorts by length)
    packer = sequence_packing_utils.SequencePacker(
        bin_size=bin_size,
        pad_token=tokenizer.pad,
        max_sequences_per_bin=4,
        seq_length_multiple=multiple,
    )

    packed_trajs, packed_position_ids, packed_loss_mask, packing_info = packer.pack_sequences(
        torch.stack([seq_a, seq_b])
    )

    assert packed_trajs.shape == (1, bin_size)
    assert packing_info.bin_seq_indices[0] == [1, 0]
    # Aligned starts plus the padded end sentinel.
    assert packing_info.seq_starts[0] == expected_starts

    # Data at the aligned starts; everything outside the two sequences
    # (alignment gaps + bin tail) stays pad tokens with a zero loss mask.
    start_a = expected_starts[1]
    assert torch.equal(packed_trajs[0, 0:7], seq_b)
    assert torch.equal(packed_trajs[0, start_a : start_a + 5], seq_a[:5])
    is_data = torch.zeros(bin_size, dtype=torch.bool)
    is_data[0:7] = True
    is_data[start_a : start_a + 5] = True
    assert torch.all(packed_trajs[0, ~is_data] == tokenizer.pad)
    assert torch.all(packed_loss_mask[0, ~is_data] == 0)
    # Position ids restart at each start.
    assert packed_position_ids[0, start_a] == 0

    params = sequence_packing_utils.create_packed_seq_params_for_bin(
        packing_info=packing_info,
        bin_idx=0,
        bin_size=bin_size,
        max_sequences_per_bin=4,
        device=torch.device('cpu'),
        seq_length_multiple=multiple,
    )
    if expected_cu_padded is None:
        assert params.cu_seqlens_q_padded is None
        assert params.cu_seqlens_kv_padded is None
        assert params.pad_between_seqs is False
    else:
        assert params.cu_seqlens_q_padded.tolist() == expected_cu_padded
        assert params.cu_seqlens_kv_padded.tolist() == expected_cu_padded
        assert params.pad_between_seqs is True
    assert params.cu_seqlens_q.tolist() == expected_cu


def test_sequence_packing_aligned_fit_accounting():
    """Fit decisions must use the reserved (rounded) footprint, not the raw length."""
    tokenizer = MockTokenizer()
    packer = sequence_packing_utils.SequencePacker(
        bin_size=8, pad_token=tokenizer.pad, seq_length_multiple=4
    )
    # Three sequences of length 3 (footprint 4): only two fit per 8-token bin.
    sequences = torch.arange(1, 10).reshape(3, 3)
    packed_trajs, _, _, packing_info = packer.pack_sequences(sequences)
    assert packed_trajs.shape[0] == 2
    assert sorted(len(idxs) for idxs in packing_info.bin_seq_indices) == [1, 2]


def test_sequence_packer_rejects_indivisible_bin():
    with pytest.raises(AssertionError, match="divisible"):
        sequence_packing_utils.SequencePacker(bin_size=10, pad_token=0, seq_length_multiple=4)


def test_default_packed_seq_params_signature_matches_bin_params():
    """The empty-padding-bin fallback (get_default_packed_seq_params) must be structurally
    interchangeable with real bins' params — same None-ness, shape, and dtype per field —
    or CUDA graph replay rejects the microbatches that fall back to it."""
    tokenizer = MockTokenizer()
    packer = sequence_packing_utils.SequencePacker(
        bin_size=16, pad_token=tokenizer.pad, max_sequences_per_bin=4
    )
    _, _, _, packing_info = packer.pack_sequences(torch.arange(1, 11).reshape(2, 5))
    bin_params = sequence_packing_utils.create_packed_seq_params_for_bin(
        packing_info=packing_info,
        bin_idx=0,
        bin_size=16,
        max_sequences_per_bin=4,
        device=torch.device('cpu'),
    )
    default_params = sequence_packing_utils.get_default_packed_seq_params(
        seq_length=16, max_sequences_per_bin=4, device=torch.device('cpu')
    )

    for field in dataclasses.fields(PackedSeqParams):
        bin_value = getattr(bin_params, field.name)
        default_value = getattr(default_params, field.name)
        assert type(default_value) is type(bin_value), field.name
        if isinstance(bin_value, torch.Tensor):
            assert default_value.shape == bin_value.shape, field.name
            assert default_value.dtype == bin_value.dtype, field.name


@pytest.mark.parametrize(
    "override, match",
    [
        pytest.param({'max_sequences_per_bin': 1}, "max_sequences_per_bin", id="overfull_bin"),
        pytest.param({'seq_length_multiple': 4}, "divisible", id="misaligned_slots"),
        pytest.param({'seq_starts': []}, "sentinel", id="missing_seq_starts"),
    ],
)
def test_create_packed_seq_params_rejects_invalid_bin(override, match):
    """Bins violating construction contracts must fail loudly: more sequences than the
    fixed-size cu_seqlens can hold, padded slots misaligned to seq_length_multiple
    (TE's THD context-parallel partitioning contract), or missing seq_starts."""
    tokenizer = MockTokenizer()
    packer = sequence_packing_utils.SequencePacker(
        bin_size=16, pad_token=tokenizer.pad, max_sequences_per_bin=4
    )
    _, _, _, packing_info = packer.pack_sequences(torch.arange(1, 11).reshape(2, 5))
    kwargs = dict(
        packing_info=packing_info,
        bin_idx=0,
        bin_size=16,
        max_sequences_per_bin=4,
        device=torch.device('cpu'),
    )
    if 'seq_starts' in override:
        packing_info.seq_starts[0] = override.pop('seq_starts')
    kwargs.update(override)
    with pytest.raises(AssertionError, match=match):
        sequence_packing_utils.create_packed_seq_params_for_bin(**kwargs)
