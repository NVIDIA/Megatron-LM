# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture
def transformer_config():
    return TransformerConfig(hidden_size=2, num_attention_heads=2, padded_vocab_size=10, use_cpu_initialization=True)
