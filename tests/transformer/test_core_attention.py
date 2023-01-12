# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest

import torch

from megatron.core import parallel_state
from megatron.core.transformer.core_attention import CoreAttention

parallel_state.set_tensor_model_parallel_world_size(1)
parallel_state.set_tensor_model_parallel_rank(0)


@pytest.fixture
def core_attention(transformer_config):
    return CoreAttention(transformer_config)


class TestCoreAttention:
    def test_constructor(self, core_attention):
        assert isinstance(core_attention, CoreAttention)
        assert core_attention.layer_number == 1
        assert core_attention.norm_factor == 1.0

        num_weights = sum([p.numel() for p in core_attention.parameters()])
        assert num_weights == 0

