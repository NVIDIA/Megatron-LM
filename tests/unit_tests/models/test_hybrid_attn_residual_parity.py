# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BF16 parity for Kimi residual updates and fused KDA output normalization.

Independent formulas: moonshotai/Kimi-K3 at
 a590ce090cb049c93a33dfe8c208ec652aa20503, modeling_kimi_linear.py:
KimiDecoderLayer._forward_attn_residual and KimiDeltaAttention.o_norm.
Shapes retain the published 29-layer slim proxy's hidden/FFN/head dimensions.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situlu
from megatron.core.models.hybrid.hybrid_block import AttnResHybridLayer
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.kda import KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def distributed_state():
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    model_parallel_cuda_manual_seed(20260821)
    yield
    Utils.destroy_model_parallel()


def _config(**kwargs):
    return TransformerConfig(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        transformer_impl="transformer_engine",
        **kwargs,
    )


def _assert_similarity(actual, expected, *, relative_l2=0.005):
    a, b = actual.double().flatten(), expected.double().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    denom = a.square().sum() + b.square().sum()
    if denom == 0:
        return
    assert F.cosine_similarity(a[None], b[None]).item() >= 1 - 1e-3
    assert (2 * (a * b).sum() / denom).item() >= 1 - 1e-3
    assert ((a - b).norm() / b.norm().clamp_min(1e-30)).item() <= relative_l2


def _native_aggregate(values, query, norm_weight, eps):
    values = torch.stack(values, dim=-2)
    vf = values.float()
    keys = vf * torch.rsqrt(vf.square().mean(-1, keepdim=True) + eps)
    scores = (keys * (query.float() * norm_weight.float())).sum(-1)
    return (scores.softmax(-1).unsqueeze(-2) @ vf).squeeze(-2).to(values.dtype)


@pytest.mark.parametrize("layer_number", [1, 2])
@pytest.mark.parametrize("fused_bda", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_hybrid_partial_matches_native(layer_number, fused_bda, training):
    config = _config(
        enable_attention_residuals=True,
        attn_res_block_layers=24,
        activation_func=situlu,
        gated_linear_unit=True,
        bias_activation_fusion=False,
        bias_dropout_fusion=fused_bda,
    )
    inner = build_module(
        hybrid_stack_spec.submodules.mlp_layer,
        config=config,
        layer_number=layer_number,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    layer = AttnResHybridLayer(config, inner).cuda().train(training)
    with torch.no_grad():
        layer.attn_res.pseudo_query.normal_(std=0.01)
    params = dict(layer.named_parameters())
    names = [
        "attn_res.pseudo_query",
        "attn_res.key_norm_weight",
        "inner_layer.mlp.linear_fc1.layer_norm_weight",
        "inner_layer.mlp.linear_fc1.weight",
        "inner_layer.mlp.linear_fc2.weight",
    ]
    assert set(params) == set(names)
    ref = {name: params[name].detach().clone().requires_grad_(True) for name in names}
    source = torch.randn(128, 1, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    partial = source if layer_number == 1 else torch.randn_like(source, requires_grad=True)
    rsource = source.detach().clone().requires_grad_(True)
    rpartial = rsource if layer_number == 1 else partial.detach().clone().requires_grad_(True)
    values = [rsource] if layer_number == 1 else [rsource, rpartial]
    aggregated = _native_aggregate(values, ref[names[0]], ref[names[1]], 1e-5)
    normalized = aggregated.float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + 1e-5)
    normalized = normalized.to(aggregated.dtype) * ref[names[2]]
    gate, up = F.linear(normalized, ref[names[3]]).float().chunk(2, -1)
    activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))).to(
        torch.bfloat16
    )
    branch = F.linear(activated, ref[names[4]])
    expected = branch if layer_number == 1 else rpartial + branch
    raw = []
    hooks = []
    hook = inner.mlp.register_forward_hook(lambda m, a, out: raw.append(out[0].clone()))
    pre_hook = inner.register_forward_pre_hook(lambda m, a: hooks.append(True))
    try:
        actual = layer(partial, attn_res_sources=(source,))
    finally:
        hook.remove()
        pre_hook.remove()
    assert hooks == [True]
    # The old residual-add/subtract implementation fails this exact update contract.
    direct = raw[0] if layer_number == 1 else partial + raw[0]
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    _assert_similarity(actual, expected, relative_l2=0.001)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    _assert_similarity(source.grad, rsource.grad)
    if layer_number != 1:
        _assert_similarity(partial.grad, rpartial.grad)
    for name in names:
        assert params[name].grad is not None and ref[name].grad is not None, name
        _assert_similarity(params[name].grad, ref[name].grad)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_kda_gated_norm_matches_native(deterministic, zero_centered):
    config = _config(
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=32,
        linear_num_value_heads=32,
        kda_f_lora_rank=64,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        experimental_attention_variant="kda",
        deterministic_mode=deterministic,
        layernorm_zero_centered_gamma=zero_centered,
    )
    spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    layer = KimiDeltaAttention(
        config,
        spec.submodules,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    with torch.no_grad():
        layer.out_norm.weight.normal_(mean=0.0 if zero_centered else 1.0, std=0.1)
    x = torch.randn(1, 128, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    rx = x.detach().clone().requires_grad_(True)
    rg = gate.detach().clone().requires_grad_(True)
    rw = layer.out_norm.weight.detach().clone().requires_grad_(True)
    effective_weight = rw + 1 if zero_centered else rw
    rf = rx.float()
    normalized = rf * torch.rsqrt(rf.square().mean(-1, keepdim=True) + config.layernorm_epsilon)
    expected = (normalized * effective_weight.float() * torch.sigmoid(rg.float())).to(x.dtype)
    actual = layer._apply_gated_norm(x, gate).view_as(expected)
    _assert_similarity(actual, expected, relative_l2=0.0005)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for a, b in [(x.grad, rx.grad), (gate.grad, rg.grad), (layer.out_norm.weight.grad, rw.grad)]:
        _assert_similarity(a, b)
