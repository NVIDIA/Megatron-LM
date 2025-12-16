# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

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
    packed_trajs, packed_position_ids, packed_attention_mask, packed_loss_mask, packing_info = (
        packer.pack_sequences(sequences_tensor, generation_masks)
    )

    assert packed_trajs is not None
    assert packed_position_ids is not None
    assert packed_attention_mask is not None
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
    packed_trajs, packed_position_ids, packed_attention_mask, packed_loss_mask, packing_info = (
        packer.pack_sequences(padded_sequences_tensor, generation_masks)
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
    packed_attention_mask = torch.ones(1, bin_size, bin_size)

    empty_trajs, empty_position_ids, empty_loss_mask, empty_attention_mask, empty_packing_info = (
        sequence_packing_utils.create_empty_bins(
            num_empty_bins=num_empty_bins,
            bin_size=bin_size,
            packed_trajs=packed_trajs,
            packed_position_ids=packed_position_ids,
            packed_loss_mask=packed_loss_mask,
            packed_attention_mask=packed_attention_mask,
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
    packed_trajs, packed_position_ids, packed_attention_mask, packed_loss_mask, packing_info = (
        packer.pack_sequences(sequences_tensor, generation_masks)
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
