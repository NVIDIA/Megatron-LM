# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


class TestTransformerConfig:
    def test_transformer_config(self, transformer_config):

        assert transformer_config.hidden_size == 2
        assert transformer_config.ffn_hidden_size == 8
        assert transformer_config.padded_vocab_size == 10
