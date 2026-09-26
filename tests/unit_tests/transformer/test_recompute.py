# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from megatron.core import recompute
from megatron.core.transformer import TransformerConfig


class _TraceLayer(torch.nn.Module):
    def __init__(self, layer_idx, calls):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_number = layer_idx + 1
        self.calls = calls

    def forward(self, hidden_states, **_kwargs):
        self.calls[self.layer_idx] += 1
        return hidden_states * 1.01 + float(self.layer_number)


class _RecomputeBlock:
    def __init__(self, num_layers, recompute_method, recompute_num_layers, skip_final_recompute):
        self.calls = Counter()
        self.config = SimpleNamespace(
            fp8=False,
            fp4=False,
            distribute_saved_activations=False,
            recompute_method=recompute_method,
            recompute_num_layers=recompute_num_layers,
            skip_final_recompute=skip_final_recompute,
        )
        self.layers = [_TraceLayer(layer_idx, self.calls) for layer_idx in range(num_layers)]
        self.num_layers_per_pipeline_rank = num_layers
        self.pg_collection = SimpleNamespace(tp=None)


def _run_forward_backward(block):
    hidden_states = torch.zeros((2, 1, 4), requires_grad=True)
    output = recompute.checkpointed_forward(
        block,
        hidden_states=hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        attention_bias=None,
        packed_seq_params=None,
        use_inner_quantization_context=False,
    )
    output.sum().backward()
    return dict(block.calls)


def test_skip_final_recompute_requires_full_recompute():
    with pytest.raises(ValueError, match="full activation recomputation"):
        TransformerConfig(num_layers=2, num_attention_heads=1, skip_final_recompute=True)
    with pytest.raises(ValueError, match="full activation recomputation"):
        TransformerConfig(
            num_layers=2,
            num_attention_heads=1,
            recompute_granularity='selective',
            skip_final_recompute=True,
        )

    config = TransformerConfig(
        num_layers=2,
        num_attention_heads=1,
        recompute_granularity='full',
        recompute_method='uniform',
        recompute_num_layers=1,
        skip_final_recompute=True,
    )

    assert config.skip_final_recompute


@pytest.mark.parametrize(
    ("recompute_method", "recompute_num_layers", "skip_final_recompute", "expected_calls"),
    [
        ("uniform", 2, False, {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}),
        ("uniform", 2, True, {0: 2, 1: 2, 2: 2, 3: 2, 4: 1}),
        ("block", 3, False, {0: 2, 1: 2, 2: 2, 3: 1, 4: 1}),
        ("block", 3, True, {0: 2, 1: 2, 2: 1, 3: 1, 4: 1}),
    ],
)
def test_skip_final_recompute_avoids_final_backward_recompute(
    recompute_method, recompute_num_layers, skip_final_recompute, expected_calls
):
    block = _RecomputeBlock(
        num_layers=5,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        skip_final_recompute=skip_final_recompute,
    )

    calls = _run_forward_backward(block)

    assert calls == expected_calls
