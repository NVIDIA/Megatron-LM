# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU operation-topology guards, NOT production BF16 numerical qualification.

Oracle: huggingface/transformers@93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py, SHA256
2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b:
TextMLP:86-105, TextExperts:108-142, TextMoE:194-206.

Small expert widths/counts bound these graph-contract tests; hidden width stays
4096. The dispatch test supplies indices directly ONLY to test slot traversal,
not router correctness. Real Torch Linear/autograd and real reference FFNs run;
no target kernels or mocked tensor arithmetic are used. These tests do not
replace the production-geometry 288-expert composite forward/all-gradient gate.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.models import glm53_native_reference as reference


def _initialize(module):
    # Local deterministic CPU fixture; does not touch composite seeds/weights.
    generator = torch.Generator(device='cpu').manual_seed(731)
    module.float()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.01, 0.01, generator=generator)
    return module


def _linear_shapes(module, hidden, monkeypatch):
    shapes = []
    original = F.linear

    def observe(input, weight, bias=None):
        shapes.append(tuple(weight.shape))
        return original(input, weight, bias)

    with monkeypatch.context() as context:
        context.setattr(F, 'linear', observe)
        output = module(hidden)
    return output, shapes


@pytest.mark.parametrize('combined', [False, True])
def test_ffn_projection_topology_and_leaf_inventory(combined, monkeypatch):
    width = 8
    module = _initialize(reference.NativeFFN(width, torch.device('cpu'), combined_gate_up=combined))
    assert set(dict(module.named_parameters())) == {
        'gate_proj.weight',
        'up_proj.weight',
        'down_proj.weight',
    }
    assert not dict(module.named_buffers())
    hidden = torch.ones(3, reference.HIDDEN, requires_grad=True)
    output, shapes = _linear_shapes(module, hidden, monkeypatch)
    expected = (
        [(2 * width, reference.HIDDEN)]
        if combined
        else [(width, reference.HIDDEN), (width, reference.HIDDEN)]
    )
    assert shapes == expected + [(reference.HIDDEN, width)]
    output.sum().backward()
    assert hidden.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_moe_selects_combined_routed_experts_but_split_shared(monkeypatch):
    monkeypatch.setattr(reference, 'EXPERTS', 2)
    monkeypatch.setattr(reference, 'EXPERT_WIDTH', 8)
    module = _initialize(reference.NativeMoE(torch.device('cpu')))
    assert all(expert.combined_gate_up is True for expert in module.experts)
    assert module.shared_experts.combined_gate_up is False
    expected = {'gate.weight'}
    expected.update(
        f'{prefix}.{name}.weight'
        for prefix in ('experts.0', 'experts.1', 'shared_experts')
        for name in ('gate_proj', 'up_proj', 'down_proj')
    )
    assert set(dict(module.named_parameters())) == expected
    assert set(dict(module.named_buffers())) == {'e_score_correction_bias'}
    hidden = torch.ones(3, reference.HIDDEN)
    _, expert_shapes = _linear_shapes(module.experts[0], hidden, monkeypatch)
    _, shared_shapes = _linear_shapes(module.shared_experts, hidden, monkeypatch)
    assert expert_shapes == [(16, reference.HIDDEN), (reference.HIDDEN, 8)]
    assert shared_shapes == [(8, reference.HIDDEN), (8, reference.HIDDEN), (reference.HIDDEN, 8)]


def test_expert_dispatch_uses_hf_slot_major_token_order(monkeypatch):
    monkeypatch.setattr(reference, 'EXPERTS', 2)
    monkeypatch.setattr(reference, 'EXPERT_WIDTH', 8)
    module = _initialize(reference.NativeMoE(torch.device('cpu')))
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64)
    probabilities = torch.full((3, 2), 0.5, dtype=torch.float32)
    # A declared dispatch control: no assertion about the top-k router is made.
    monkeypatch.setattr(module, 'routing', lambda hidden: (indices, probabilities))
    hidden = torch.arange(1, 4, dtype=torch.float32)[:, None].expand(-1, reference.HIDDEN)
    observed = {}
    handles = []
    for expert, child in enumerate(module.experts):

        def observe(_module, args, expert=expert):
            observed[expert] = args[0][:, 0].detach().clone()
            return None

        handles.append(child.register_forward_pre_hook(observe))
    try:
        module(hidden)
    finally:
        for handle in handles:
            handle.remove()
    # HF one_hot(...).permute(2,1,0): slot first, then token.
    torch.testing.assert_close(observed[0], torch.tensor([1.0, 3.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(observed[1], torch.tensor([2.0, 1.0, 3.0]), rtol=0, atol=0)
    assert observed[0].tolist() != [1.0, 2.0, 3.0]  # old token-major implementation


def test_combined_expert_matches_literal_hf_forward_and_autograd():
    module = _initialize(reference.NativeFFN(8, torch.device('cpu'), combined_gate_up=True))
    hidden = torch.linspace(-0.5, 0.5, 3 * reference.HIDDEN).reshape(3, reference.HIDDEN)
    actual_input = hidden.clone().requires_grad_()
    native_input = hidden.clone().requires_grad_()
    gate_weight = module.gate_proj.weight.detach().clone().requires_grad_()
    up_weight = module.up_proj.weight.detach().clone().requires_grad_()
    down_weight = module.down_proj.weight.detach().clone().requires_grad_()
    actual = module(actual_input)
    # Literal HF eager expert formula, independently written here.
    gate_up = F.linear(native_input, torch.cat((gate_weight, up_weight), dim=0))
    gate, up = gate_up.chunk(2, dim=-1)
    expected = F.linear(F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0), down_weight)
    gradient = torch.linspace(-1.0, 1.0, actual.numel()).reshape_as(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    pairs = [
        (actual, expected),
        (actual_input.grad, native_input.grad),
        (module.gate_proj.weight.grad, gate_weight.grad),
        (module.up_proj.weight.grad, up_weight.grad),
        (module.down_proj.weight.grad, down_weight.grad),
    ]
    for left, right in pairs:
        assert left is not None and right is not None
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
