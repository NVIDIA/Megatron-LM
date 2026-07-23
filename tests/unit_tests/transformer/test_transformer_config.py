# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


class TestTransformerConfig:
    def test_num_query_groups_divides_num_attention_heads(self):
        config = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=32, num_query_groups=8
        )
        assert config.num_query_groups == 8

    def test_num_query_groups_defaults_to_num_attention_heads(self):
        config = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=32)
        assert config.num_query_groups == 32

    def test_num_query_groups_not_dividing_num_attention_heads_raises(self):
        with pytest.raises(ValueError, match="must be a positive divisor of num_attention_heads"):
            TransformerConfig(
                num_layers=2, hidden_size=128, num_attention_heads=32, num_query_groups=5
            )

    def test_num_query_groups_larger_than_num_attention_heads_raises(self):
        with pytest.raises(ValueError, match="must be a positive divisor of num_attention_heads"):
            TransformerConfig(
                num_layers=2, hidden_size=128, num_attention_heads=4, num_query_groups=8
            )

    def test_negative_num_query_groups_raises(self):
        with pytest.raises(ValueError, match="must be a positive divisor of num_attention_heads"):
            TransformerConfig(
                num_layers=2, hidden_size=128, num_attention_heads=4, num_query_groups=-1
            )

    def test_zero_num_query_groups_normalized_to_num_attention_heads(self):
        # num_query_groups == 0 is treated like None: normalized to num_attention_heads, so a
        # real attention config never reaches attention init with a zero query-group count.
        config = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=2, num_query_groups=0
        )
        assert config.num_query_groups == 2

    def test_minimal_config_without_attention_heads_is_allowed(self):
        # num_attention_heads defaults to 0 in minimal configs used by many non-attention tests;
        # num_query_groups then normalizes to 0 and validation is skipped.
        config = TransformerConfig(num_layers=1, kv_channels=1)
        assert config.num_query_groups == 0
