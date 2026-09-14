# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""LocalSpecProvider must expose a non-TE backend.linear() for DSA/MLA."""

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.post_training.modelopt.layers import Linear
from megatron.core.tensor_parallel.layers import ColumnParallelLinear


def test_local_spec_provider_linear_is_replicated_local_linear():
    backend = LocalSpecProvider()
    assert backend.linear() is Linear
    assert backend.linear() is not TELinear
    assert backend.column_parallel_linear() is ColumnParallelLinear
    assert backend.linear() is not backend.column_parallel_linear()
