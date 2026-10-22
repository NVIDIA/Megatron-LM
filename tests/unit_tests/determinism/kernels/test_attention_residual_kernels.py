# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay coverage for the Attention Residuals depth aggregation."""

import pytest
import torch

from megatron.core.transformer.attention_residual import AttentionResidual
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_module_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


class _Aggregate(torch.nn.Module):
    def __init__(self, attn_res):
        super().__init__()
        self.attn_res = attn_res

    def forward(self, *values):
        return self.attn_res(list(values))


@pytest.mark.parametrize("attn_res_impl", ["eager", "compile"])
@pytest.mark.parametrize("num_values", [2, 5])
def test_attention_residual_aggregation_replays(attn_res_impl, num_values):
    seeded()
    hidden_size = 256
    config = TransformerConfig(
        num_layers=4,
        hidden_size=hidden_size,
        num_attention_heads=4,
        enable_attention_residuals=True,
        attn_res_block_layers=2,
        attn_res_impl=attn_res_impl,
    )
    attn_res = AttentionResidual(config).cuda()
    with torch.no_grad():
        # Non-trivial depth weights; the zero-initialized query would make them uniform.
        attn_res.pseudo_query.normal_(std=0.05)
        attn_res.key_norm_weight.normal_(mean=1.0, std=0.1)
    values = tuple(
        torch.randn((64, 2, hidden_size), device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(num_values)
    )
    assert_module_replays_bit_exact(
        _Aggregate(attn_res), values, replays=3, what=f"AttnRes {attn_res_impl} aggregation"
    )
