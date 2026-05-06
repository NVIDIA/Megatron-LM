# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.static_context import StaticInferenceContext


class TestStaticInferenceContext:

    def test_init_sets_dimensions_and_defaults(self):
        """__init__ records the max sizes and zeroes the offsets."""
        ctx = StaticInferenceContext(max_batch_size=4, max_sequence_length=128)
        assert ctx.max_batch_size == 4
        assert ctx.max_sequence_length == 128
        assert ctx.sequence_len_offset == 0
        assert ctx.batch_size_offset == 0
        assert ctx.key_value_memory_dict == {}
        assert ctx.decode_mode is False

    def test_is_static_batching_returns_true(self):
        """StaticInferenceContext.is_static_batching is always True."""
        ctx = StaticInferenceContext(2, 32)
        assert ctx.is_static_batching() is True

    def test_is_dynamic_batching_returns_false(self):
        """is_dynamic_batching (from base class) returns False for static contexts."""
        ctx = StaticInferenceContext(2, 32)
        assert ctx.is_dynamic_batching() is False

    def test_enable_prefill_mode(self):
        """enable_prefill_mode flips decode_mode to False."""
        ctx = StaticInferenceContext(2, 32)
        ctx.decode_mode = True
        ctx.enable_prefill_mode()
        assert ctx.decode_mode is False

    def test_enable_decode_mode(self):
        """enable_decode_mode flips decode_mode to True."""
        ctx = StaticInferenceContext(2, 32)
        ctx.enable_decode_mode()
        assert ctx.decode_mode is True

    def test_is_decode_only_reflects_decode_mode(self):
        """is_decode_only mirrors the decode_mode attribute."""
        ctx = StaticInferenceContext(2, 32)
        assert ctx.is_decode_only() is False
        ctx.enable_decode_mode()
        assert ctx.is_decode_only() is True

    def test_reset_zeros_offsets_and_returns_to_prefill(self):
        """reset zeroes both offsets and re-enters prefill mode."""
        ctx = StaticInferenceContext(2, 32)
        ctx.sequence_len_offset = 9
        ctx.batch_size_offset = 3
        ctx.enable_decode_mode()
        ctx.reset()
        assert ctx.sequence_len_offset == 0
        assert ctx.batch_size_offset == 0
        assert ctx.decode_mode is False

    def test_increment_offsets_through_base_class(self):
        """increment_sequence_len_offset / increment_batch_size_offset adjust the static offsets."""
        ctx = StaticInferenceContext(2, 32)
        ctx.increment_sequence_len_offset(7)
        ctx.increment_batch_size_offset(1)
        assert ctx.sequence_len_offset == 7
        assert ctx.batch_size_offset == 1
        ctx.reset_batch_size_offset()
        assert ctx.batch_size_offset == 0

    def test_swap_key_value_dict_raises_when_empty(self):
        """swap_key_value_dict raises ValueError when the dict is empty."""
        ctx = StaticInferenceContext(2, 32)
        with pytest.raises(ValueError, match="empty"):
            ctx.swap_key_value_dict([0, 1])

    def test_swap_key_value_dict_reorders_along_batch_dim(self):
        """swap_key_value_dict permutes the batch dimension of cached k/v tensors."""
        ctx = StaticInferenceContext(2, 32)
        # Shape [seq_len, batch_size, ...]
        k = torch.tensor([[[10], [20], [30]], [[40], [50], [60]]])  # [2, 3, 1]
        v = torch.tensor([[[1], [2], [3]], [[4], [5], [6]]])
        ctx.key_value_memory_dict[0] = (k, v)
        ctx.swap_key_value_dict(batch_idx=[2, 0, 1])
        new_k, new_v = ctx.key_value_memory_dict[0]
        assert new_k[0, 0, 0].item() == 30
        assert new_k[0, 1, 0].item() == 10
        assert new_k[0, 2, 0].item() == 20
        assert new_v[0, 0, 0].item() == 3

    def test_swap_key_value_dict_size_mismatch_asserts(self):
        """swap_key_value_dict asserts when len(batch_idx) != cached batch size."""
        ctx = StaticInferenceContext(2, 32)
        ctx.key_value_memory_dict[0] = (torch.zeros(1, 4, 1), torch.zeros(1, 4, 1))
        with pytest.raises(AssertionError):
            ctx.swap_key_value_dict([0, 1])  # len 2 != 4
