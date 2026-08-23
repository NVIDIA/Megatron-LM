from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import DeepseekV4MoE
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import visible_clamped_swiglu
from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.primitive.modules.mlp import SwiGLUMLP


class _VisibleLinear(nn.Module):
    def forward(self, value, weight):
        return F.linear(value, weight)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None,
    reason="requires CUDA and vLLM activation kernels",
)
def test_shared_swiglu_matches_vllm_clamp_bitwise() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    torch.manual_seed(17)
    swiglu_limit = 10.0
    gate_up = torch.randn(
        257,
        512,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 20
    with set_current_vllm_config(VllmConfig()):
        expected = SiluAndMulWithClamp(swiglu_limit)(gate_up)
    with set_current_vllm_config(VllmConfig()):
        actual = visible_clamped_swiglu(gate_up, swiglu_limit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_experts_preserve_clamp_and_bf16_master_gradients() -> None:
    torch.manual_seed(31)
    hidden_size = 8
    shared_intermediate = 8
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(moe)
    swiglu_limit = 10.0
    moe.shared_experts = SwiGLUMLP(
        hidden_size,
        shared_intermediate,
        swiglu_limit=swiglu_limit,
    ).to(
        dtype=torch.bfloat16
    )
    moe.shared_gate_up_fp8 = _VisibleLinear()
    moe.shared_down_fp8 = _VisibleLinear()
    hidden = (torch.randn(5, hidden_size, dtype=torch.bfloat16) * 8).requires_grad_(
        True
    )
    grad_output = torch.randn_like(hidden)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    with set_current_vllm_config(VllmConfig()):
        output = moe._shared_expert_forward(hidden)
        gate_up_visible = F.linear(hidden, moe.shared_experts.gate_up.weight)
        expected_visible = F.linear(
            SiluAndMulWithClamp(swiglu_limit)(gate_up_visible),
            moe.shared_experts.down.weight,
        )
    unclamped_gate, unclamped_up = gate_up_visible.chunk(2, dim=-1)
    unclamped_visible = F.linear(
        F.silu(unclamped_gate) * unclamped_up,
        moe.shared_experts.down.weight,
    )
    torch.testing.assert_close(output, expected_visible, rtol=0, atol=0)
    assert not torch.equal(output, unclamped_visible)
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
    activated = swiglu_with_probs(gate_up, None, swiglu_limit)
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
