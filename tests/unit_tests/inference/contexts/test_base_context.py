# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.base_context import BaseInferenceContext


class _StaticContext(BaseInferenceContext):
    """Minimal concrete context that reports static batching."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sequence_len_offset = 0
        self.batch_size_offset = 0

    def is_static_batching(self) -> bool:
        return True


class _DynamicContext(BaseInferenceContext):
    """Minimal concrete context that reports dynamic batching."""

    def is_static_batching(self) -> bool:
        return False


class TestBaseInferenceContext:

    def test_cannot_instantiate_abstract_directly(self):
        """BaseInferenceContext is abstract; instantiation must fail."""
        with pytest.raises(TypeError):
            BaseInferenceContext(InferenceConfig())

    def test_init_stores_config(self):
        """The provided InferenceConfig is stored as self.config."""
        cfg = InferenceConfig()
        ctx = _StaticContext(cfg)
        assert ctx.config is cfg

    def test_is_dynamic_batching_is_complement_of_static(self):
        """is_dynamic_batching returns the negation of is_static_batching."""
        static = _StaticContext(InferenceConfig())
        dynamic = _DynamicContext(InferenceConfig())
        assert static.is_dynamic_batching() is False
        assert dynamic.is_dynamic_batching() is True

    def test_increment_sequence_len_offset_static(self):
        """For static batching, increment_sequence_len_offset adds to the offset."""
        ctx = _StaticContext(InferenceConfig())
        ctx.sequence_len_offset = 5
        ctx.increment_sequence_len_offset(3)
        assert ctx.sequence_len_offset == 8

    def test_increment_sequence_len_offset_dynamic_is_noop(self):
        """For dynamic batching, increment_sequence_len_offset is a no-op."""
        ctx = _DynamicContext(InferenceConfig())
        # Should not raise even though sequence_len_offset is unset.
        ctx.increment_sequence_len_offset(7)
        assert not hasattr(ctx, "sequence_len_offset")

    def test_increment_batch_size_offset_static(self):
        """For static batching, increment_batch_size_offset adds to the offset."""
        ctx = _StaticContext(InferenceConfig())
        ctx.batch_size_offset = 1
        ctx.increment_batch_size_offset(2)
        assert ctx.batch_size_offset == 3

    def test_increment_batch_size_offset_dynamic_is_noop(self):
        """For dynamic batching, increment_batch_size_offset is a no-op."""
        ctx = _DynamicContext(InferenceConfig())
        ctx.increment_batch_size_offset(4)
        assert not hasattr(ctx, "batch_size_offset")

    def test_reset_batch_size_offset_static(self):
        """For static batching, reset_batch_size_offset zeros the offset."""
        ctx = _StaticContext(InferenceConfig())
        ctx.batch_size_offset = 9
        ctx.reset_batch_size_offset()
        assert ctx.batch_size_offset == 0

    def test_reset_batch_size_offset_dynamic_is_noop(self):
        """For dynamic batching, reset_batch_size_offset does nothing."""
        ctx = _DynamicContext(InferenceConfig())
        ctx.reset_batch_size_offset()
        assert not hasattr(ctx, "batch_size_offset")
