# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture
def transformer_config():
    return TransformerConfig(hidden_size=2, ffn_hidden_size=8, padded_vocab_size=10, use_cpu_initialization=True)
