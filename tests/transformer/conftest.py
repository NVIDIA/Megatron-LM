# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture
def transformer_config():
    return TransformerConfig(hidden_size=12, num_attention_heads=4, padded_vocab_size=10, use_cpu_initialization=True)
