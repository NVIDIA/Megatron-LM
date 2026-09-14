# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import MethodType, SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src import megatron_fsdp


@pytest.mark.parametrize(
    ("nccl_ub", "fsdp_double_buffer", "expected_grad_comm_dtype"),
    [(False, False, torch.float16), (True, True, torch.bfloat16), (False, True, torch.bfloat16)],
)
def test_mixed_precision_context_preserves_fixed_buffer_dtype_and_restores_policy(
    nccl_ub, fsdp_double_buffer, expected_grad_comm_dtype
):
    original_policy = megatron_fsdp.MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.bfloat16,
    )
    requested_policy = megatron_fsdp.MixedPrecisionPolicy(grad_comm_dtype=torch.float16)
    param_and_grad_buffer = SimpleNamespace(mp_policy=original_policy)
    fsdp = SimpleNamespace(
        mp_policy=original_policy,
        ddp_config=SimpleNamespace(nccl_ub=nccl_ub, fsdp_double_buffer=fsdp_double_buffer),
        param_and_grad_buffer=param_and_grad_buffer,
    )
    fsdp.reset_mixed_precision_policy = MethodType(
        megatron_fsdp.MegatronFSDP.reset_mixed_precision_policy, fsdp
    )

    with megatron_fsdp.MegatronFSDP.mixed_precision_context(fsdp, requested_policy):
        assert fsdp.mp_policy.main_params_dtype == original_policy.main_params_dtype
        assert fsdp.mp_policy.main_grads_dtype == original_policy.main_grads_dtype
        assert fsdp.mp_policy.grad_comm_dtype == expected_grad_comm_dtype
        assert param_and_grad_buffer.mp_policy is fsdp.mp_policy

    assert fsdp.mp_policy.main_params_dtype == original_policy.main_params_dtype
    assert fsdp.mp_policy.main_grads_dtype == original_policy.main_grads_dtype
    assert fsdp.mp_policy.grad_comm_dtype == original_policy.grad_comm_dtype
    assert param_and_grad_buffer.mp_policy is fsdp.mp_policy
