# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for packed SFT statistics and the legacy training import path."""

import pytest
import torch

import megatron.training.training as training_module
from megatron.training.logging import packed_sequence_stats
from megatron.training.logging.packed_sequence_stats import (
    consume_packed_sequence_stats_in_iteration,
    update_packed_sequence_stats,
)


class TestPackedSequenceStatsAccumulator:
    def setup_method(self):
        packed_sequence_stats._reset_packed_sequence_stats_in_iteration()

    def teardown_method(self):
        packed_sequence_stats._reset_packed_sequence_stats_in_iteration()

    def test_legacy_imports_share_implementation(self):
        # Both import paths must update and consume the same module-owned state.
        assert training_module.update_packed_sequence_stats is update_packed_sequence_stats
        assert (
            training_module.consume_packed_sequence_stats_in_iteration
            is consume_packed_sequence_stats_in_iteration
        )

    def test_no_updates_returns_none(self):
        assert consume_packed_sequence_stats_in_iteration() is None

    def test_update_accumulates_batch_stats(self):
        sample_lengths_1 = torch.tensor([[100, 150, 0], [25, 0, 0]], dtype=torch.int32)
        loss_mask_1 = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
        sample_lengths_2 = torch.tensor([[200, 0, 0]], dtype=torch.int32)
        loss_mask_2 = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)

        update_packed_sequence_stats(sample_lengths_1, loss_mask_1)
        update_packed_sequence_stats(sample_lengths_2, loss_mask_2)
        stats = consume_packed_sequence_stats_in_iteration()

        lengths = torch.tensor([100, 150, 25, 200], dtype=torch.float64)
        # Each rank contributes these samples; the consumer reports global totals.
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        assert stats["packed_sequence/total_tokens"] == lengths.sum().item() * world_size
        assert stats["packed_sequence/trained_tokens"] == 6.0 * world_size
        assert stats["packed_sequence/original_samples"] == 4.0 * world_size
        assert stats["packed_sequence/original_sample_length_min"] == 25.0
        assert stats["packed_sequence/original_sample_length_mean"] == lengths.mean().item()
        assert stats["packed_sequence/original_sample_length_max"] == 200.0
        assert (
            stats["packed_sequence/original_sample_length_median"]
            == torch.quantile(lengths, 0.5).item()
        )
        assert stats["packed_sequence/original_sample_length_stdv"] == pytest.approx(
            lengths.std(unbiased=False).item()
        )
        assert consume_packed_sequence_stats_in_iteration() is None
