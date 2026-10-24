# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The head hands Core CE an owned buffer, without altering its raw values."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.model import Head


class ViewLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return (x @ weight.T).view(x.shape[0], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        return grad @ weight, grad.T @ x


class InplaceCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        logits.sub_(logits.max(-1, keepdim=True).values)
        selected = logits.gather(-1, labels[:, None]).squeeze(-1).clone()
        logits.exp_()
        total = logits.sum(-1, keepdim=True)
        logits.div_(total)
        ctx.save_for_backward(logits, labels)
        return (total.log().squeeze(-1) - selected).mean()

    @staticmethod
    def backward(ctx, grad):
        probability, labels = ctx.saved_tensors
        result = probability.clone()
        result[torch.arange(labels.numel()), labels] -= 1
        return result * (grad / labels.numel()), None


@pytest.mark.parametrize('owned', [False, True])
def test_core_ce_buffer_ownership_and_vjp(owned):
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4).div(13).requires_grad_()
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4).div(23).requires_grad_()
    labels = torch.tensor([0, 2, 4])
    expected = torch.autograd.grad(
        torch.nn.functional.cross_entropy(x @ weight.T, labels), (x, weight)
    )
    with patch(
        'megatron.lite.model.deepseek_v41.vllm.model.head_logits',
        lambda x, w, *, dtype: ViewLinear.apply(x, w),
    ):
        logits = (
            Head.forward(SimpleNamespace(weight=weight, logits_dtype="float32"), x)[0]
            if owned
            else ViewLinear.apply(x, weight)
        )
    assert torch.equal(logits.detach().view(torch.int32), (x @ weight.T).detach().view(torch.int32))
    loss = InplaceCE.apply(logits, labels)
    if not owned:
        with pytest.raises(RuntimeError, match='view.*modified inplace'):
            loss.backward()
    else:
        loss.backward()
        for actual, reference in zip((x.grad, weight.grad), expected):
            torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)
