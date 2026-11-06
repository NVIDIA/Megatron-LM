# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only checks for the examples in docs/user-guide/troubleshooting-configuration.md.

Each case asserts that the invalid configuration raises the documented error and that the
corrected configuration in the same section constructs without raising.
"""

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig

BASE = dict(num_layers=4, hidden_size=512, num_attention_heads=8)


def test_attention_heads_must_divide_tensor_parallel_size():
    with pytest.raises(ValueError, match=r"num_attention_heads \(8\) must be a multiple of"):
        TransformerConfig(**BASE, tensor_model_parallel_size=3)

    TransformerConfig(**BASE, tensor_model_parallel_size=4)


def test_query_groups_must_divide_attention_heads():
    with pytest.raises(ValueError, match=r"num_query_groups \(3\) must be a divisor of"):
        TransformerConfig(**BASE, num_query_groups=3)

    TransformerConfig(**BASE, num_query_groups=2)


def test_sequence_parallel_requires_tensor_parallelism():
    with pytest.raises(ValueError, match="Cannot use sequence parallelism without tensor"):
        TransformerConfig(**BASE, sequence_parallel=True)

    TransformerConfig(
        **BASE, tensor_model_parallel_size=2, sequence_parallel=True
    )


def test_pipeline_parallel_requires_pipeline_dtype():
    with pytest.raises(ValueError, match="pipeline_dtype must be specified"):
        TransformerConfig(**BASE, pipeline_model_parallel_size=2)

    TransformerConfig(**BASE, pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16)


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_softcapping_must_be_positive_and_finite(cap):
    with pytest.raises(ValueError, match="must be a positive finite value"):
        TransformerConfig(**BASE, attn_logit_softcapping=cap)

    TransformerConfig(**BASE, attn_logit_softcapping=None)


def test_pipeline_layout_is_exclusive():
    layout = "E|tt|tt|L"
    with pytest.raises(ValueError, match="cannot be set with other pipeline layout arguments"):
        TransformerConfig(
            **BASE,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float,
            pipeline_model_parallel_layout=layout,
            num_layers_in_first_pipeline_stage=2,
        )

    TransformerConfig(
        **BASE,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float,
        pipeline_model_parallel_layout=layout,
    )


def test_virtual_pipeline_size_must_match_layout():
    layout = "E|tt|tt|L"
    with pytest.raises(AssertionError, match="conflicts with pipeline_model_parallel_layout"):
        TransformerConfig(
            **BASE,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float,
            pipeline_model_parallel_layout=layout,
            virtual_pipeline_model_parallel_size=5,
        )

    TransformerConfig(
        **BASE,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float,
        pipeline_model_parallel_layout=layout,
    )
