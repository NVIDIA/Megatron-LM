# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Muon's QKV layout metadata and the opt-in per-head and MLA up-projection splits."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, emerging_optimizers
from megatron.core.transformer.qkv_layout import QKVLayout

pytestmark = pytest.mark.skipif(
    not emerging_optimizers.HAVE_EMERGING_OPTIMIZERS, reason="requires emerging-optimizers"
)


def test_qkv_layouts():
    config = SimpleNamespace(
        num_attention_heads=8, num_query_groups=2, kv_channels=16, attention_output_gate=True
    )
    layout = QKVLayout.from_transformer_config(config)
    assert layout.num_groups == 2 and not layout.is_mla
    assert layout.projection_split_shapes == (64, 64, 16, 16)
    assert layout.per_head_split_shapes == (16,) * 10
    # The default projection split matches the shapes Muon always used.
    assert emerging_optimizers._get_qkv_split_shapes(config) == [64, 64, 16, 16]
    assert emerging_optimizers._get_qkv_split_shapes(config, split_qkv_per_head=True) == [16] * 10

    mla = QKVLayout.for_mla_up_projection(4, (24, 8))
    assert mla.is_mla and mla.num_groups == 4
    assert mla.projection_split_shapes == mla.per_head_split_shapes == (24, 8)
    with pytest.raises(ValueError, match="same rows"):
        QKVLayout(num_groups=1, projection_split_shapes=(4, 4), per_head_split_shapes=(4,))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(muon_split_qkv=False, muon_split_qkv_per_head=True), "muon_split_qkv_per_head"),
        (dict(muon_split_qkv=False, muon_split_mla_qkv=True), "muon_split_mla_qkv"),
    ],
)
def test_split_options_require_qkv_split(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OptimizerConfig(optimizer='muon', **kwargs).validate()


@pytest.mark.parametrize(
    "head_split_shapes",
    [
        pytest.param((8, 8, 8, 8, 8, 8), id="uniform-heads"),
        pytest.param((12, 4), id="mla-unequal-components"),
        pytest.param((8, 8, 8, 8, 4, 4), id="unequal-qkv-heads"),
    ],
)
def test_per_head_orthogonalization_matches_each_head(head_split_shapes):
    torch.manual_seed(0)
    optimizer = emerging_optimizers.TensorParallelMuon(
        [torch.nn.Parameter(torch.zeros(1, 1))],
        split_qkv=True,
        split_qkv_per_head=True,
        fp32_matmul_prec="highest",
    )
    num_groups, cols = 3, 20
    grad = torch.randn(num_groups * sum(head_split_shapes), cols)
    param = SimpleNamespace(is_qkv=True, qkv_split_shapes=list(head_split_shapes))
    result = optimizer._orthogonalize_qkv_per_head(param, grad, list(head_split_shapes))
    # Reference: the optimizer's own 2D Newton-Schulz on every head, one at a time.
    expected = torch.cat(
        [
            optimizer.scaled_orthogonalize_fn(head, None, None, "duplicated")
            for group in grad.split(sum(head_split_shapes), dim=0)
            for head in group.split(list(head_split_shapes), dim=0)
        ]
    )
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
