# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Muon's QKV layout metadata and the opt-in per-head and MLA up-projection splits."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, emerging_optimizers, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
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


@pytest.mark.parametrize("per_head", [False, True])
def test_scalar_head_gate_keeps_the_whole_muon_matrix(per_head):
    """A divisible total row count does not make appended scalar gates part of a QKV group."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks = []
    # 96 QKV rows, plus either 24 scalar gates (one per head) or 48 elementwise gates.
    for rows, scalar_gate, output_gate in [
        (120, True, False),
        (96, False, False),
        (144, False, True),
    ]:
        chunk = torch.nn.Module()
        chunk.config = SimpleNamespace(
            num_attention_heads=24,
            num_query_groups=12,
            kv_channels=2,
            head_wise_attn_gate=scalar_gate,
            attention_output_gate=output_gate,
        )
        chunk.linear_qkv = torch.nn.Linear(48, rows, bias=False, device=device)
        chunks.append(chunk)
    chain = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="muon", lr=0.01, muon_split_qkv_per_head=per_head, muon_tp_mode="duplicated"
        ),
        chunks,
        pg_collection=ProcessGroupCollection(),
        use_gloo_process_groups=False,
    )
    weight, ordinary, output_gated = (chunk.linear_qkv.weight for chunk in chunks)
    assert ordinary.qkv_split_shapes == ([2, 2, 2, 2] if per_head else [4, 2, 2])
    assert output_gated.qkv_split_shapes == ([2] * 6 if per_head else [4, 4, 2, 2])
    assert not getattr(weight, "is_qkv", False)
    muon = chain.chained_optimizers[0].optimizer
    grad = torch.randn_like(weight)
    expected = muon.orthogonalize(torch.nn.Parameter(weight.detach().clone()), grad.clone())
    torch.testing.assert_close(muon.orthogonalize(weight, grad), expected, rtol=0, atol=0)


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
