# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK runs both spellings of SiTU-GLU: ``situlu`` and gated silu with both tanh clamps.

``activation_func=F.silu`` with ``activation_func_tanh_clamp_scale=s`` and
``activation_func_tanh_clamp_scale_linear=l`` computes ``s*tanh(g/s)*sigmoid(g) * l*tanh(u/l)``,
which is ``situlu(x, s, l)``. MOK takes either spelling as ``situ_beta=(s, l)``. MOK always
clamps the up half, so the spelling without ``activation_func_tanh_clamp_scale_linear`` stays
rejected, as do the other clamp combinations.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situ_glu, situlu
from megatron.core.transformer.moe.megakernel.mok import backend as mok_backend
from megatron.core.transformer.transformer_config import TransformerConfig


def _mok_config(**overrides):
    values = dict(
        num_layers=2,
        bf16=True,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_latent_size=64,
        moe_shared_expert_intermediate_size=256,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_tanh_clamp_scale=4.0,
        activation_func_tanh_clamp_scale_linear=25.0,
        bias_activation_fusion=False,
        add_bias_linear=False,
        gradient_accumulation_fusion=True,
        moe_megakernel_backend="mok",
    )
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.parametrize("betas", [(4.0, 25.0), (2.0, 8.0), (1.0, 64.0)])
@pytest.mark.parametrize("sigma", [1.0, 8.0, 64.0])
def test_tanh_clamped_silu_glu_is_situlu(betas, sigma):
    """The two spellings agree bitwise in FP32, forward and input gradient."""
    generator = torch.Generator().manual_seed(1234)
    x = (torch.randn(257, 2 * 96, generator=generator) * sigma).requires_grad_()
    y = x.detach().clone().requires_grad_()
    upstream = torch.randn(257, 96, generator=generator)
    tanh_spelling = situ_glu(x, betas[0], betas[1], 0.0)
    situ_spelling = situlu(y, betas[0], betas[1])
    torch.testing.assert_close(tanh_spelling, situ_spelling, rtol=0, atol=0)
    tanh_spelling.backward(upstream)
    situ_spelling.backward(upstream)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)


def test_mok_accepts_tanh_clamped_situ_glu():
    config = _mok_config()
    assert config.moe_megakernel_backend == "mok"
    assert config.activation_func is F.silu


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"activation_func_tanh_clamp_scale_linear": None}, id="no-up-clamp"),
        pytest.param(
            {"gated_linear_unit": False, "activation_func_tanh_clamp_scale_linear": None},
            id="not-gated",
        ),
        pytest.param({"activation_func_clamp_value": 7.0}, id="hard-clamp"),
        pytest.param({"activation_func": situlu}, id="situlu-with-tanh-clamp"),
    ],
)
def test_mok_rejects_other_tanh_clamp_spellings(overrides):
    with pytest.raises(ValueError, match="MOK"):
        _mok_config(**overrides)


def _backend(monkeypatch, **activation):
    """Build the real adapter around a stand-in ``mok`` package (no MOK build needed)."""
    package = ModuleType("mok")
    functional = ModuleType("mok.functional")
    functional.MoKConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    package.functional = functional
    monkeypatch.setitem(sys.modules, "mok", package)
    monkeypatch.setitem(sys.modules, "mok.functional", functional)
    experts = torch.nn.Module()
    experts.linear_fc1 = torch.nn.Module()
    experts.linear_fc2 = torch.nn.Module()
    for linear, shape in ((experts.linear_fc1, (8, 8)), (experts.linear_fc2, (8, 4))):
        linear.single_grouped_weight = False
        for index in range(2):
            linear.register_parameter(
                f"weight{index}", torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
            )
    config = SimpleNamespace(
        gradient_accumulation_fusion=True,
        moe_mlp_glu_interleave_size=None,
        moe_shared_expert_glu_interleave_size=None,
        moe_pad_expert_input_to_capacity=False,
        moe_shared_expert_gate=False,
        hidden_size=16,
        moe_latent_size=8,
        moe_ffn_hidden_size=4,
        moe_shared_expert_intermediate_size=12,
        moe_router_topk=2,
        activation_func_clamp_value=None,
        situ_glu_beta1=2.0,
        situ_glu_beta2=8.0,
        fp8=None,
        fp8_recipe=None,
        fp8_param=False,
        moe_single_grouped_weight=False,
        moe_megakernel_backend_config={"shared_expert_mode": "native"},
        **activation,
    )
    return mok_backend.MoKMegakernel(config, object(), experts, None, 2)


@pytest.mark.parametrize(
    "activation, situ_beta",
    [
        pytest.param(
            dict(
                activation_func=F.silu,
                activation_func_tanh_clamp_scale=4.0,
                activation_func_tanh_clamp_scale_linear=25.0,
            ),
            (4.0, 25.0),
            id="tanh-clamped-silu",
        ),
        pytest.param(dict(activation_func=situlu), (2.0, 8.0), id="situlu"),
        pytest.param(dict(activation_func=F.silu), None, id="swiglu"),
    ],
)
def test_backend_passes_situ_beta_for_either_spelling(monkeypatch, activation, situ_beta):
    assert _backend(monkeypatch, **activation).situ_beta == situ_beta
