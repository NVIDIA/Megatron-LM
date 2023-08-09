# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


class TestTransformerConfig:
    def test_transformer_config(self, transformer_config):

        assert transformer_config.hidden_size == 12
        assert transformer_config.ffn_hidden_size == 48
        assert transformer_config.num_attention_heads == 4
        assert transformer_config.kv_channels == 3
