# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""#260 MegaMoE backward math (the A/B alternative to the grouped VJP) on CPU."""
import torch

from megatron.lite.model.deepseek_v41.vllm import moe


def test_mlp_grads_match_autograd_through_rounded_operands():
    torch.manual_seed(81)
    a = torch.randn(5, 64).bfloat16().float()
    w1, w2 = torch.randn(64, 64).bfloat16(), torch.randn(64, 32).bfloat16()
    score, grad, limit = torch.rand(5), torch.randn(5, 64), 3.0
    da, ds, dw1, dw2 = moe._mlp_grads(a, score, grad, w1, w2, limit, (True,) * 4)

    leaves = [t.clone().requires_grad_() for t in (a, score, w1.float(), w2.float())]
    x, s, u, v = leaves
    h = (x.bfloat16().float() @ u.T).bfloat16().float()
    h = h + (x @ u.T - (x @ u.T).detach())  # BF16 rounding of the GEMM1 output: identity STE
    gate, up = h.chunk(2, -1)
    act = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit) * s[:, None]
    act = act + (moe._requant_l2(act) - act).detach()
    expected = torch.autograd.grad(act @ v.T, leaves, grad.bfloat16().float())
    for actual, reference in zip((da, ds, dw1, dw2), expected):
        cosine = torch.nn.functional.cosine_similarity(
            actual.double().flatten(), reference.double().flatten(), dim=0
        )
        assert cosine > 0.999


def test_requant_l2_is_a_power_of_two_e4m3_grid():
    torch.manual_seed(82)
    h = torch.randn(3, 64) * 10
    q = moe._requant_l2(h)
    groups = h.reshape(3, 2, 32)
    scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True) / 448)))
    expected = ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape(h.shape)
    torch.testing.assert_close(q, expected, rtol=0, atol=0)
