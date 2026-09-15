# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import megatron.core.optimizer as optimizer_module
from megatron.core.optimizer import _multi_tensor_adam_batched
from megatron.core.optimizer.optimizer_config import OptimizerConfig


@pytest.mark.parametrize("num_params", [0, 1, 1024, 1025, 2049])
def test_adam_dispatch_preserves_aligned_lists_and_arguments(num_params):
    tensor_lists = [[object() for _ in range(num_params)] for _ in range(5)]
    original_lists = [list(tensors) for tensors in tensor_lists]
    noop = object()
    args = (0.01, 0.9, 0.95, 1.0e-8, 7, 1, 1, 0.1)
    kernel = Mock(return_value=None)

    assert _multi_tensor_adam_batched(kernel, 65536, noop, tensor_lists, *args) is None

    if num_params <= 1024:
        assert kernel.call_count == 1
        assert kernel.call_args.args[2] is tensor_lists
    else:
        assert [len(call.args[2][0]) for call in kernel.call_args_list] == (
            [1024] * (num_params // 1024) + ([num_params % 1024] if num_params % 1024 else [])
        )
    for call in kernel.call_args_list:
        assert call.args[0] == 65536
        assert call.args[1] is noop
        assert call.args[3:] == args
        assert all(len(tensors) == len(call.args[2][0]) for tensors in call.args[2])
    for list_index, original in enumerate(original_lists):
        dispatched = [
            tensor for call in kernel.call_args_list for tensor in call.args[2][list_index]
        ]
        assert dispatched == original
        assert tensor_lists[list_index] == original


def _make_adam(params, *, batched):
    fused_adam = pytest.importorskip("transformer_engine.pytorch.optimizers").FusedAdam
    optimizer = fused_adam(
        [
            {"params": params[:2049], "lr": 0.001, "weight_decay": 0.1},
            {"params": params[2049:], "lr": 0.003, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        master_weights=True,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        use_decoupled_grad=True,
        store_param_remainders=True,
    )
    if batched:
        optimizer.multi_tensor_adam_param_remainder = partial(
            _multi_tensor_adam_batched, optimizer.multi_tensor_adam_param_remainder
        )
    return optimizer


def _set_grads(params, step, dtype):
    for index, param in enumerate(params):
        param.decoupled_grad = (
            None if index == 17 else torch.full_like(param, (index % 7 + step) / 32, dtype=dtype)
        )


def _assert_same_adam(params, optimizer, reference_params, reference_optimizer):
    torch.testing.assert_close(torch.cat(params), torch.cat(reference_params), rtol=0, atol=0)
    for key, dtype in (
        ("exp_avg", torch.float32),
        ("exp_avg_sq", torch.float32),
        ("master_param", torch.int16),
    ):
        values = [optimizer.state[param][key] for param in params]
        reference_values = [reference_optimizer.state[param][key] for param in reference_params]
        assert all(value.dtype == dtype for value in values)
        torch.testing.assert_close(torch.cat(values), torch.cat(reference_values), rtol=0, atol=0)
    assert [group["step"] for group in optimizer.param_groups] == [
        group["step"] for group in reference_optimizer.param_groups
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_batched_adam_matches_unbatched_and_checkpoint_resume(grad_dtype):
    params = [
        torch.nn.Parameter(
            torch.full((4,), (index % 13 + 1) / 16, device="cuda", dtype=torch.bfloat16)
        )
        for index in range(2056)
    ]
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    optimizer = _make_adam(params, batched=True)
    reference_optimizer = _make_adam(reference_params, batched=False)
    original_groups = list(optimizer.param_groups)

    for step in (1, 2):
        for current_params, current_optimizer in (
            (params, optimizer),
            (reference_params, reference_optimizer),
        ):
            _set_grads(current_params, step, grad_dtype)
            closure = Mock(return_value=4.5)
            assert current_optimizer.step(closure) == 4.5
            closure.assert_called_once_with()
        _assert_same_adam(params, optimizer, reference_params, reference_optimizer)
        assert all(group["step"] == step for group in optimizer.param_groups)
        assert all(
            group is original for group, original in zip(optimizer.param_groups, original_groups)
        )

    checkpoint_buffer = io.BytesIO()
    torch.save({"params": params, "optimizer": optimizer.state_dict()}, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    checkpoint = torch.load(checkpoint_buffer, weights_only=True)
    resumed_params = [torch.nn.Parameter(param.detach().clone()) for param in checkpoint["params"]]
    resumed_optimizer = _make_adam(resumed_params, batched=True)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    assert len(resumed_optimizer.param_groups) == 2

    for step in (3, 4):
        for current_params, current_optimizer in (
            (params, optimizer),
            (reference_params, reference_optimizer),
            (resumed_params, resumed_optimizer),
        ):
            _set_grads(current_params, step, grad_dtype)
            current_optimizer.step()
        _assert_same_adam(params, optimizer, reference_params, reference_optimizer)
        _assert_same_adam(resumed_params, resumed_optimizer, reference_params, reference_optimizer)
        assert all(group["step"] == step for group in resumed_optimizer.param_groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("store_param_remainders", [False, True])
def test_optimizer_factory_batches_only_remainder_kernel(store_param_remainders):
    fused_adam = pytest.importorskip("transformer_engine.pytorch.optimizers").FusedAdam
    config = OptimizerConfig(
        optimizer="adam",
        lr=0.001,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        store_param_remainders=store_param_remainders,
    )
    param = torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.bfloat16))
    with patch.object(
        optimizer_module,
        "DistributedOptimizer",
        side_effect=lambda optimizer, *args, **kwargs: SimpleNamespace(optimizer=optimizer),
    ):
        wrapped = optimizer_module._get_megatron_optimizer_based_on_param_groups(
            config,
            model_chunks=[torch.nn.Module()],
            param_groups=[{"params": [param]}],
            pg_collection=SimpleNamespace(tp=None, expt_tp=None),
        )
    optimizer = wrapped.optimizer
    assert isinstance(optimizer, fused_adam)
    kernel = optimizer.multi_tensor_adam_param_remainder
    assert isinstance(kernel, partial) is store_param_remainders
    if store_param_remainders:
        assert kernel.func is _multi_tensor_adam_batched
    assert not isinstance(optimizer.multi_tensor_adam, partial)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_adam_handles_more_than_te_tensor_capacity():
    # Five lists for 6000 parameters exceed TE's 20321 temporary-tensor handle limit.
    # Do not run the unbatched failing call: allocator exhaustion can poison the process.
    params = [
        torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.bfloat16)) for _ in range(6000)
    ]
    optimizer = _make_adam(params, batched=True)
    optimizer.param_groups[0]["params"] = params
    optimizer.param_groups[1]["params"] = []
    for step in (1, 2):
        _set_grads(params, step, torch.float32)
        optimizer.step()
        torch.cuda.synchronize()
        assert optimizer.param_groups[0]["step"] == step
    assert torch.all(torch.cat([param for index, param in enumerate(params) if index != 17]) < 1)
    assert all(optimizer.state[param]["master_param"].dtype == torch.int16 for param in params)
