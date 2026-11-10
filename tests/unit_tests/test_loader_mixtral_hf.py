# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from tools.checkpoint import loader_mixtral_hf


@pytest.mark.parametrize(
    'version', ['4.36.0', '4.49.0', '4.57.0rc1', '5.0.0.dev0', '5.8.1', '6.0.0']
)
def test_supported_transformers_versions(monkeypatch, version):
    monkeypatch.setattr(loader_mixtral_hf.transformers, '__version__', version)

    loader_mixtral_hf.verify_transformers_version()


@pytest.mark.parametrize('version', ['3.99.0', '4.35.2', '4.36.0rc1'])
def test_unsupported_transformers_versions(monkeypatch, version):
    monkeypatch.setattr(loader_mixtral_hf.transformers, '__version__', version)

    with pytest.raises(AssertionError, match=r'requires transformers>=4\.36\.0, found '):
        loader_mixtral_hf.verify_transformers_version()


def _make_mlp_layers(layout):
    num_experts, hidden_size, intermediate_size = 3, 4, 6
    args = SimpleNamespace(num_experts=num_experts)
    gate = torch.arange(num_experts * hidden_size, dtype=torch.float32).reshape(
        num_experts, hidden_size
    )
    w1 = torch.arange(num_experts * intermediate_size * hidden_size, dtype=torch.float32).reshape(
        num_experts, intermediate_size, hidden_size
    )
    w3 = w1 + 1000
    w2 = (
        torch.arange(num_experts * hidden_size * intermediate_size, dtype=torch.float32).reshape(
            num_experts, hidden_size, intermediate_size
        )
        + 2000
    )

    if layout == 'legacy':
        hf_experts = torch.nn.ModuleList()
        for expert_idx in range(num_experts):
            expert = torch.nn.Module()
            expert.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            expert.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
            expert.w3 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            with torch.no_grad():
                expert.w1.weight.copy_(w1[expert_idx])
                expert.w2.weight.copy_(w2[expert_idx])
                expert.w3.weight.copy_(w3[expert_idx])
            hf_experts.append(expert)
        moe_attribute = 'block_sparse_moe'
    else:
        hf_experts = torch.nn.Module()
        hf_experts.gate_up_proj = torch.nn.Parameter(torch.cat([w1, w3], dim=1))
        hf_experts.down_proj = torch.nn.Parameter(w2)
        hf_experts.is_transposed = False
        moe_attribute = 'mlp'

    hf_moe = SimpleNamespace(gate=SimpleNamespace(weight=gate), experts=hf_experts)
    hf_layer = SimpleNamespace(**{moe_attribute: hf_moe})
    local_experts = [
        SimpleNamespace(
            linear_fc1=torch.nn.Linear(hidden_size, 2 * intermediate_size, bias=False),
            linear_fc2=torch.nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        for _ in range(num_experts)
    ]
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            router=torch.nn.Linear(hidden_size, num_experts, bias=False),
            experts=SimpleNamespace(local_experts=local_experts),
        )
    )
    return args, layer, hf_layer, gate, w1, w2, w3


@pytest.mark.parametrize('layout', ['legacy', 'fused'])
def test_set_mlp_state_preserves_router_and_expert_weights(layout):
    args, layer, hf_layer, gate, w1, w2, w3 = _make_mlp_layers(layout)

    loader_mixtral_hf.set_mlp_state(args, layer, hf_layer)

    torch.testing.assert_close(layer.mlp.router.weight, gate, rtol=0, atol=0)
    for expert_idx, expert in enumerate(layer.mlp.experts.local_experts):
        gate_weight, up_weight = expert.linear_fc1.weight.chunk(2, dim=0)
        torch.testing.assert_close(gate_weight, w1[expert_idx], rtol=0, atol=0)
        torch.testing.assert_close(up_weight, w3[expert_idx], rtol=0, atol=0)
        torch.testing.assert_close(expert.linear_fc2.weight, w2[expert_idx], rtol=0, atol=0)


def test_transposed_fused_experts_are_rejected():
    args, layer, hf_layer, *_ = _make_mlp_layers('fused')
    hf_layer.mlp.experts.is_transposed = True

    with pytest.raises(AssertionError, match='Transposed HF expert weight layout is not supported'):
        loader_mixtral_hf.set_mlp_state(args, layer, hf_layer)
