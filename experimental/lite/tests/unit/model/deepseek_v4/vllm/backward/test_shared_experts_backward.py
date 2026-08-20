from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.moe import DeepseekV4MoE
from megatron.lite.primitive.modules.mlp import SwiGLUMLP


class _VisibleLinear(nn.Module):
    def forward(self, value, weight):
        return F.linear(value, weight)


def test_shared_experts_block_fp8_bridges_cover_bf16_master_gradients() -> None:
    torch.manual_seed(31)
    hidden_size = 8
    shared_intermediate = 8
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(moe)
    moe.shared_experts = SwiGLUMLP(hidden_size, shared_intermediate).to(
        dtype=torch.bfloat16
    )
    moe.shared_gate_up_fp8 = _VisibleLinear()
    moe.shared_down_fp8 = _VisibleLinear()
    hidden = (torch.randn(5, hidden_size, dtype=torch.bfloat16) * 8).requires_grad_(
        True
    )
    grad_output = torch.randn_like(hidden)

    output = moe._visible_shared_experts(hidden)
    output.backward(grad_output)
    actual_grads = (
        hidden.grad,
        moe.shared_experts.gate_up.weight.grad,
        moe.shared_experts.down.weight.grad,
    )

    ref_hidden = hidden.detach().float().requires_grad_(True)
    ref_gate_up = (
        moe.shared_experts.gate_up.weight.detach().float().requires_grad_(True)
    )
    ref_down = moe.shared_experts.down.weight.detach().float().requires_grad_(True)
    gate_up = F.linear(ref_hidden, ref_gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    reference = F.linear(activated, ref_down)
    expected_grads = torch.autograd.grad(
        reference,
        (ref_hidden, ref_gate_up, ref_down),
        grad_output.float(),
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected, rtol=5e-2, atol=5e-2
        )
