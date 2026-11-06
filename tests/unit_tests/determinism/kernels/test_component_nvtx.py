# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component NVTX ranges must not change the numerics of the modules they decorate.

``megatron/core/component_nvtx.py`` decorates entry points of the CSA2, DSv4 attention,
mHC and MoE modules. With ``MCORE_COMPONENT_NVTX`` unset (the default, and how this suite
runs) the decorators return the original functions, so those modules execute exactly their
undecorated code. With the switch on, the ranges add host-side NVTX calls, CPU-only
sequence-number probes and autograd hooks that return ``None``; a decorated graph must
replay bit-exactly and match the undecorated graph bit for bit, forward and backward.
"""

import inspect

import pytest
import torch
import torch.nn.functional as F

from megatron.core import component_nvtx
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.deepseek_v41.hybrid_adapter import DeepSeekV41Adapter
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.transformer.experimental_attention_variant import csa2
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridAttention,
    DSv4HybridSelfAttention,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule, SinglePassMHCState
from megatron.core.transformer.moe.moe_layer import MoELayer
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    run_once,
    seeded,
)

# Every callable the ranges decorate (the CSA2 ``forward_range`` blocks run inside them).
INSTRUMENTED = (
    (LanguageModelEmbedding, "forward"),
    (DeepSeekV41Adapter, "run_layer"),
    (HyperConnectionHybridLayer, "forward"),
    (CSA2Compressor, "forward"),
    (CSA2Indexer, "project_keys"),
    (CSA2Indexer, "forward_with_scores"),
    (CompressedSparseAttention2, "_forward_fused_no_indexer"),
    (CompressedSparseAttention2, "_forward_fused_indexer_training"),
    (CompressedSparseAttention2, "_forward_fused_indexer_no_loss"),
    (CompressedSparseAttention2, "_forward_fused_reuse"),
    (csa2, "_zero_padding_rows"),
    (csa2, "_zero_padding_rows_inplace"),
    (DSv4HybridAttention, "forward"),
    (DSv4HybridAttention, "_output_projection"),
    (DSv4HybridSelfAttention, "get_query_key_value_tensors"),
    (SinglePassMHCState, "contract"),
    (HyperConnectionModule, "forward"),
    (HyperConnectionModule, "input_expand"),
    (HyperConnectionModule, "output_contract"),
    (HyperConnectionModule, "fused_h_res_h_post_bda"),
    (MoELayer, "forward"),
)


def _identity(fn):
    return fn


def _wrapper_chain(attribute):
    """``attribute`` and every function it wraps (through ``functools.wraps``)."""
    function = (
        attribute.__func__ if isinstance(attribute, (staticmethod, classmethod)) else attribute
    )
    while function is not None:
        yield function
        function = getattr(function, "__wrapped__", None)


def _range_wrapper_codes(monkeypatch):
    """Code objects of the wrappers that ``component`` and ``layer_scope`` install when on."""
    monkeypatch.setattr(component_nvtx, "ENABLED", True)
    codes = {
        component_nvtx.component("probe")(_identity).__code__,
        component_nvtx.layer_scope(_identity).__code__,
        component_nvtx.layer_scope(layer_arg=True)(_identity).__code__,
    }
    monkeypatch.setattr(component_nvtx, "ENABLED", False)
    return codes


@pytest.mark.skipif(
    component_nvtx.ENABLED, reason="checks the default, with MCORE_COMPONENT_NVTX unset"
)
def test_disabled_ranges_leave_instrumented_code_untouched(monkeypatch):
    """Off, the decorators return their argument, so the kernels' callers are unchanged."""
    assert component_nvtx.component("probe")(_identity) is _identity
    assert component_nvtx.layer_scope(_identity) is _identity
    assert component_nvtx.layer_scope(layer_arg=True)(_identity) is _identity
    with component_nvtx.forward_range("probe"):
        pass

    wrapper_codes = _range_wrapper_codes(monkeypatch)
    assert len(wrapper_codes) == 2
    for owner, name in INSTRUMENTED:
        chain = list(_wrapper_chain(inspect.getattr_static(owner, name)))
        assert chain and all(callable(function) for function in chain), (owner, name)
        assert not any(
            getattr(function, "__code__", None) in wrapper_codes for function in chain
        ), f"{owner.__name__}.{name} carries an NVTX range wrapper with the switch off"


class _ScaledSoftmax(torch.autograd.Function):
    """Custom Function with a forward-only range inside, as in the CSA2 sparse attention."""

    @staticmethod
    def forward(ctx, x, scale):
        with component_nvtx.forward_range("softmax"):
            y = torch.softmax(x * scale, dim=-1)
        ctx.save_for_backward(y)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return y * (grad - (grad * y).sum(dim=-1, keepdim=True)) * ctx.scale, None


def _build(component, layer_scope):
    """A layer of nested components; ``component``/``layer_scope`` decorate (or not) its parts."""

    @component("project")
    def project(x, weight):
        return F.layer_norm(x @ weight, (weight.shape[-1],))

    @component("block")
    def block(x, w1, w2):
        hidden = project(x, w1)
        # A tuple output, and an intermediate that feeds two components.
        return project(_ScaledSoftmax.apply(hidden, 0.5), w2), hidden

    class Layer:
        layer_number = 3

        @layer_scope
        def forward(self, x, w1, w2):
            out, hidden = block(x, w1, w2)
            return out * hidden.sum(dim=-1, keepdim=True), hidden

    layer = Layer()
    return layer.forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_enabled_ranges_are_bit_exact(monkeypatch):
    """On, outputs and gradients match the undecorated graph bit for bit and replay exactly."""
    seeded()
    x = torch.randn(4096, 256, device="cuda", requires_grad=True)
    w1 = (torch.randn(256, 256, device="cuda") * 0.05).requires_grad_(True)
    w2 = (torch.randn(256, 256, device="cuda") * 0.05).requires_grad_(True)
    reference_out, reference_grad = run_once(_build(lambda name: _identity, _identity), (x, w1, w2))

    monkeypatch.setattr(component_nvtx, "ENABLED", True)
    ranged = _build(component_nvtx.component, component_nvtx.layer_scope)
    outputs, grads = assert_replays_bit_exact(
        ranged, (x, w1, w2), replays=3, contention=True, what="component NVTX ranges"
    )

    assert outputs.keys() == reference_out.keys() and grads.keys() == reference_grad.keys()
    assert len(grads) == 3
    for name in reference_out:
        assert bytes_equal(outputs[name], reference_out[name]), name
    for name in reference_grad:
        assert bytes_equal(grads[name], reference_grad[name]), name
    # Every backward range closed and no forward call is left open.
    assert not component_nvtx._open
    assert not component_nvtx._claimed_stack
