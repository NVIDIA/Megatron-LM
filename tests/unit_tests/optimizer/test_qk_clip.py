# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``clip_qk`` reduces the maximum attention logits over the group its caller passes."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer.qk_clip import clip_qk
from tests.unit_tests.test_utilities import Utils


class _SelfAttention:
    """The parts of an attention module that ``clip_qk`` touches."""

    def __init__(self, max_attn_logits):
        self.core_attention = SimpleNamespace(current_max_attn_logits=max_attn_logits)
        self.clipped_with = None

    def clip_qk(self):
        self.clipped_with = self.core_attention.current_max_attn_logits.clone()


def _model_with_logits(max_attn_logits):
    """One model chunk, wrapped like the training loop's, with one attention layer."""
    self_attention = _SelfAttention(max_attn_logits)
    layer = SimpleNamespace(self_attention=self_attention)
    decoder = SimpleNamespace(layers=[layer])
    chunk = SimpleNamespace(module=SimpleNamespace(module=SimpleNamespace(decoder=decoder)))
    return [chunk], self_attention


def _rank_logits():
    """Per-head maxima that differ on every rank: [rank, -rank]."""
    rank = torch.distributed.get_rank()
    return torch.tensor([float(rank), -float(rank)], device='cuda')


class TestClipQKGroup:

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_reduces_over_the_given_group_without_global_reads(self):
        world_size = Utils.world_size
        if world_size % 2 != 0:
            pytest.skip(f"world size {world_size} must be even to form pairs")

        # The global DP x CP group spans every rank; the model's group pairs neighbours.
        pair_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [[r, r + 1] for r in range(0, world_size, 2)]
        )
        model, self_attention = _model_with_logits(_rank_logits())

        with mock.patch.object(
            parallel_state,
            'get_data_parallel_group',
            side_effect=AssertionError("clip_qk read the global data-parallel group"),
        ):
            log_max = clip_qk(model, dp_cp_group=pair_group)

        rank = torch.distributed.get_rank()
        low, high = rank - rank % 2, rank - rank % 2 + 1
        expected = torch.tensor([float(high), -float(low)], device='cuda')
        assert torch.equal(self_attention.clipped_with, expected)
        assert log_max == float(high)

    def test_without_a_group_reduces_over_the_global_dp_cp_group(self):
        model, self_attention = _model_with_logits(_rank_logits())

        log_max = clip_qk(model)

        world_size = Utils.world_size
        expected = torch.tensor([float(world_size - 1), 0.0], device='cuda')
        assert torch.equal(self_attention.clipped_with, expected)
        assert log_max == float(world_size - 1)
