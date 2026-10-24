# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

from megatron.core.tensor_parallel import random as checkpoint_module


@pytest.fixture(autouse=True)
def restore_checkpointing_state():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    previous = checkpoint_module.IS_CHECKPOINTING
    checkpoint_module.IS_CHECKPOINTING = False
    try:
        yield
    finally:
        checkpoint_module.IS_CHECKPOINTING = previous


@pytest.mark.parametrize("fail_during_recompute", [False, True])
def test_checkpointing_state_restored_after_run_function_error(fail_during_recompute):
    value = torch.ones(4, device="cuda", requires_grad=True)
    error = RuntimeError("checkpoint run function failed")
    calls = 0

    def run_function(x):
        nonlocal calls
        calls += 1
        assert checkpoint_module.is_checkpointing()
        if torch.is_grad_enabled() == fail_during_recompute:
            raise error
        return x.square()

    with pytest.raises(RuntimeError) as caught:
        output = checkpoint_module.checkpoint(run_function, False, value)
        output.sum().backward()

    assert caught.value is error
    assert calls == (2 if fail_during_recompute else 1)
    assert not checkpoint_module.is_checkpointing()

    recovered = checkpoint_module.checkpoint(lambda x: x * 3, False, value)
    recovered.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 3))
    assert not checkpoint_module.is_checkpointing()


def test_checkpointing_state_restored_after_autograd_error():
    error = RuntimeError("checkpoint autograd failed")

    class FailingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            raise error

    value = torch.ones(4, device="cuda", requires_grad=True)
    output = checkpoint_module.checkpoint(FailingBackward.apply, False, value)
    assert not checkpoint_module.is_checkpointing()
    with pytest.raises(RuntimeError) as caught:
        output.sum().backward()
    assert caught.value is error
    assert not checkpoint_module.is_checkpointing()


def test_nested_checkpoint_preserves_outer_state_and_gradient():
    value = torch.arange(1, 5, device="cuda", dtype=torch.float32, requires_grad=True)

    def outer(x):
        assert checkpoint_module.is_checkpointing()
        inner = checkpoint_module.checkpoint(lambda y: y.square(), False, x)
        assert checkpoint_module.is_checkpointing()
        return inner * 3

    output = checkpoint_module.checkpoint(outer, False, value)
    assert not checkpoint_module.is_checkpointing()
    torch.testing.assert_close(output, value.square() * 3)
    output.sum().backward()
    torch.testing.assert_close(value.grad, value * 6)
    assert not checkpoint_module.is_checkpointing()


def test_checkpoint_preserves_output_gradient_and_rng_state():
    value = torch.ones(4, device="cuda", requires_grad=True)

    def run_function(x):
        assert checkpoint_module.is_checkpointing()
        return x * torch.rand_like(x)

    output = checkpoint_module.checkpoint(run_function, False, value)
    cpu_rng_after_forward = torch.get_rng_state()
    cuda_rng_after_forward = torch.cuda.get_rng_state()
    assert not checkpoint_module.is_checkpointing()
    output.sum().backward()
    torch.testing.assert_close(value.grad, output.detach())
    assert torch.equal(torch.get_rng_state(), cpu_rng_after_forward)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_after_forward)
    assert not checkpoint_module.is_checkpointing()
