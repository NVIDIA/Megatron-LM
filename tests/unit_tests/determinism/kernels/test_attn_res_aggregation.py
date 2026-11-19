# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the Attention Residuals (AttnRes) depth aggregation.

``AttentionResidual`` (``megatron/core/transformer/attention_residual.py``) mixes the depth
sources of the residual stream with a per-token softmax over depth: fp32 statistics, RMSNorm
keys and a learnable pseudo-query. ``attn_res_impl`` selects one of three backends:

* ``eager``: the memory-lean ``_AttnResAggregation`` custom Function. Its backward recomputes
  the fp32 intermediates from saved per-token statistics and reduces the pseudo-query /
  key-norm-weight gradients over all tokens with a gemv.
* ``compile``: the plain-autograd math under ``torch.compile`` (Inductor row reductions over
  the hidden size, the AOTAutograd-partitioned backward; the source arity is a dynamo guard).
* ``fla``: flash-linear-attention's ``fused_attnres`` Triton kernels, optionally with the
  consuming RMSNorm fused in (``attn_res_norm_fusion`` hands the norm weight to the kernel as
  ``output_norm_weight``).

Every cell replays forward and backward on 2048 tokens, so many CTAs contend on the token
reductions behind the parameter gradients; outputs, source gradients and parameter gradients
must be byte-identical across replays under side-stream contention.
"""

import pytest
import torch

from megatron.core.transformer import attention_residual
from megatron.core.transformer.attention_residual import AttentionResidual
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_module_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN = 256
SEQ, BATCH = 1024, 2
NUM_SOURCES = 3


def _make_config(impl, **overrides):
    return TransformerConfig(
        num_layers=4,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        enable_attention_residuals=True,
        attn_res_block_layers=2,
        attn_res_impl=impl,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        **overrides,
    )


def _make_module(impl, **overrides):
    """Build the module on the requested backend with a non-trivial depth attention."""
    if impl == "fla" and attention_residual._get_fla_fused_attnres() is None:
        pytest.skip("flash-linear-attention with fused AttnRes is unavailable")
    seeded()
    module = AttentionResidual(_make_config(impl, **overrides)).cuda()
    with torch.no_grad():
        # Zero init makes the depth softmax uniform; perturb the parameters so the
        # softmax jacobian and the key-RMSNorm path carry signal into every gradient.
        module.pseudo_query.normal_(std=0.05)
        module.key_norm_weight.uniform_(0.8, 1.2)
    return module


def _make_values(num_sources=NUM_SOURCES):
    return [
        (torch.randn(SEQ, BATCH, HIDDEN, device="cuda", dtype=torch.bfloat16) * (0.5 + index))
        .detach()
        .requires_grad_(True)
        for index in range(num_sources)
    ]


@pytest.mark.parametrize("impl", ["eager", "compile", "fla"])
def test_attn_res_aggregation_replays(impl):
    """Replay the depth aggregation forward and backward on each backend."""
    module = _make_module(impl)
    values = _make_values()
    if impl == "compile":
        # A failed torch.compile wrap silently falls back to the eager Function.
        assert attention_residual._get_compiled_attn_res() is not None
    outputs, grads = assert_module_replays_bit_exact(
        module, (values,), contention=True, what=f"AttnRes aggregation ({impl})"
    )
    assert len(outputs) == 1
    assert set(grads) == {
        "chunk0.pseudo_query",
        "chunk0.key_norm_weight",
        *(f"in[0][{index}]" for index in range(NUM_SOURCES)),
    }


def test_attn_res_fla_fused_output_norm_replays():
    """Replay FLA's aggregation with the consuming RMSNorm fused in (attn_res_norm_fusion)."""
    module = _make_module("fla", attn_res_norm_fusion=True)
    values = _make_values()
    norm_weight = (
        (torch.ones(HIDDEN, device="cuda") + 0.1 * torch.randn(HIDDEN, device="cuda"))
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    outputs, grads = assert_module_replays_bit_exact(
        module,
        {"values": values, "output_norm_weight": norm_weight},
        contention=True,
        what="AttnRes FLA aggregation with fused output RMSNorm",
    )
    assert len(outputs) == 1
    assert set(grads) == {
        "chunk0.pseudo_query",
        "chunk0.key_norm_weight",
        "in.output_norm_weight",
        *(f"in.values[{index}]" for index in range(NUM_SOURCES)),
    }
