from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.moe import _SharedExpertsState
from megatron.lite.primitive.modules.experts import swiglu_with_probs


class _VisibleLinear(nn.Module):
    def forward(self, value, weight):
        return F.linear(value, weight)


def test_shared_experts_block_fp8_bridges_cover_bf16_master_gradients() -> None:
    torch.manual_seed(31)
    config = SimpleNamespace(
        hidden_size=8,
        n_shared_experts=2,
        moe_intermediate_size=4,
        swiglu_limit=10.0,
    )
    shared = _SharedExpertsState(config)
    shared.gate_up_fp8 = _VisibleLinear()
    shared.down_fp8 = _VisibleLinear()
    hidden = (
        torch.randn(5, config.hidden_size, dtype=torch.bfloat16) * 8
    ).requires_grad_(True)
    grad_output = torch.randn_like(hidden)

    output = shared(hidden)
    output.backward(grad_output)
    actual_grads = (
        hidden.grad,
        shared.gate_up.weight.grad,
        shared.down.weight.grad,
    )

    ref_hidden = hidden.detach().float().requires_grad_(True)
    ref_gate_up = shared.gate_up.weight.detach().float().requires_grad_(True)
    ref_down = shared.down.weight.detach().float().requires_grad_(True)
    gate_up = F.linear(ref_hidden, ref_gate_up)
    activated = swiglu_with_probs(gate_up, None, config.swiglu_limit)
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
