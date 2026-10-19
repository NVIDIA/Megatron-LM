# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU derivative/row controls for the reused B1 one-buffer head; not CUDA parity."""
import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.head import _HeadLogProbs, head_log_probs


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("entropy", [False, True])
def test_chunked_head_owned_vjp_matches_dense_fp32_reference(rows, entropy):
    torch.manual_seed(42)
    hidden = torch.randn(7, 16, requires_grad=True)
    weight = torch.randn(32, 16, requires_grad=True)
    labels = torch.tensor([0, 31, 2, 4, 11, 8, 3])
    picked, spread = _HeadLogProbs.apply(hidden, weight, labels, 0.7, entropy, rows)
    distribution = (hidden @ weight.T / 0.7).log_softmax(-1)
    target = distribution.gather(1, labels[:, None]).squeeze(1)
    loss = picked.sum()
    reference = target.sum()
    if entropy:
        loss = loss + spread.sum() * 0.3
        reference = reference - (distribution.exp() * distribution).sum() * 0.3
    expected = torch.autograd.grad(reference, (hidden, weight))
    actual = torch.autograd.grad(loss, (hidden, weight))
    torch.testing.assert_close(picked, target)
    for value, gradient in zip(actual, expected):
        torch.testing.assert_close(value, gradient, atol=2e-5, rtol=2e-5)


def test_public_head_has_no_deployment_cpu_fallback():
    with pytest.raises(RuntimeError, match="CUDA"):
        head_log_probs(
            torch.zeros(1, 16),
            torch.zeros(32, 16),
            torch.zeros(1, dtype=torch.long),
            1,
            entropy=False,
        )
