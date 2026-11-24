# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MFSDP gradient reduction when fused weight-gradient accumulation returns None."""

import importlib

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_model
from tests.unit_tests.test_utilities import Utils


class _FusedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return torch.nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        weight.get_main_grad().add_(grad_output.t() @ x)
        weight.grad_added_to_main_grad = True
        return grad_output @ weight, None


class _FusedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 16, device="cuda"))

    def forward(self, x):
        return _FusedLinearFunction.apply(x, self.weight)


class TestMFSDPNoneGradHook:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("use_tensor_hook", [False, True])
    def test_fused_wgrad_returns_none(self, monkeypatch, use_tensor_hook):
        fsdp_module = importlib.import_module(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp"
        )
        is_torch_min_version = fsdp_module.is_torch_min_version
        monkeypatch.setattr(
            fsdp_module,
            "is_torch_min_version",
            lambda version: (
                use_tensor_hook if version == "2.8.0" else is_torch_min_version(version)
            ),
        )

        module = _FusedLinear()
        reference_weight = module.weight.detach().clone().requires_grad_()
        model = fully_shard_model(
            module, fsdp_unit_modules=[_FusedLinear], zero_dp_strategy="optim_grads_params"
        )
        weight = next(p for p, name in model.param_to_name.items() if name.endswith("weight"))
        rank = dist.get_rank()
        x = torch.arange(32, device="cuda", dtype=torch.float32).reshape(2, 16) / 16 + rank
        reference_loss = torch.nn.functional.linear(x, reference_weight).sum()
        expected = torch.autograd.grad(reference_loss, reference_weight)[0]
        dist.all_reduce(expected, op=dist.ReduceOp.AVG)

        seen = []
        grad_acc = torch.autograd.graph.get_gradient_edge(weight).node
        hook = grad_acc.register_hook(
            lambda *args: seen.append(weight not in model._params_require_handle_grad)
        )
        try:
            model(x).sum().backward()
            model.finish_grad_sync()
        finally:
            hook.remove()

        assert weight.grad is None
        assert seen == [True], "the weight must be processed by its hook before the root fallback"
        assert not weight.grad_added_to_main_grad
        assert weight not in model._params_require_handle_grad
        actual = next(p.grad for name, p in model.named_parameters() if name.endswith("weight"))
        torch.testing.assert_close(actual.full_tensor(), expected)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 model parameters require Blackwell",
    )
    @pytest.mark.parametrize("recipe", [DelayedScaling(), MXFP8BlockScaling()])
    def test_quantized_weight_legacy_hook_registration(self, monkeypatch, recipe):
        fsdp_module = importlib.import_module(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp"
        )
        is_torch_min_version = fsdp_module.is_torch_min_version
        monkeypatch.setattr(
            fsdp_module,
            "is_torch_min_version",
            lambda version: False if version == "2.8.0" else is_torch_min_version(version),
        )
        with te.quantized_model_init(
            enabled=True, recipe=recipe, preserve_high_precision_init_val=True
        ):
            module = te.Linear(64, 64, params_dtype=torch.bfloat16, device="cuda")
        assert fsdp_module.is_float8tensor(module.weight)

        model = fully_shard_model(
            module, fsdp_unit_modules=[te.Linear], zero_dp_strategy="optim_grads_params"
        )
        weight = next(p for p, name in model.param_to_name.items() if name.endswith("weight"))
        assert fsdp_module.is_float8tensor(weight)
        assert torch.autograd.graph.get_gradient_edge(weight).node in model.grad_acc_nodes
