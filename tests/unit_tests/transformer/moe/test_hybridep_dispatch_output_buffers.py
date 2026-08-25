# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**kwargs):
    return TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4, **kwargs)


def test_dispatch_output_buffer_reuse_is_opt_in_with_four_slots_by_default():
    config = _config()

    assert config.moe_hybridep_reuse_dispatch_output_buffers is False
    assert config.moe_hybridep_num_dispatch_output_buffers == 4


def test_disabled_dispatch_output_buffer_reuse_accepts_custom_slot_count():
    config = _config(moe_hybridep_num_dispatch_output_buffers=2)

    assert config.moe_hybridep_reuse_dispatch_output_buffers is False
    assert config.moe_hybridep_num_dispatch_output_buffers == 2


def test_enabled_dispatch_output_buffer_reuse_requires_a_positive_slot_count():
    with pytest.raises(ValueError, match="must be positive"):
        _config(
            moe_hybridep_reuse_dispatch_output_buffers=True,
            moe_hybridep_num_dispatch_output_buffers=0,
        )
