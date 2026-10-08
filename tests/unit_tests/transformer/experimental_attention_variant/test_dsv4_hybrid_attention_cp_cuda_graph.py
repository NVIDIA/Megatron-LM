# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep the CP/graph selector while checking the unsupported dynamic-CP boundary.

Original transport coverage is recoverable from
aa0d61f457063dd1638ca339af9539d4b9c4cba3. This test does not claim graph coverage.
"""

from types import SimpleNamespace

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def test_dynamic_cp_is_rejected_before_execution():
    with pytest.raises(ValueError, match="dynamic context parallelism"):
        TransformerConfig._validate_dsv4_execution_scope(
            SimpleNamespace(
                experimental_attention_variant="dsv4_hybrid", dynamic_context_parallel=True
            )
        )
