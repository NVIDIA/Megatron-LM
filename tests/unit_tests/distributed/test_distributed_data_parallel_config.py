# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validation of DDP and MFSDP gradient precision settings."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    ("main_dtype", "comm_dtype"),
    [
        (None, None),
        (None, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
    ],
)
def test_mfsdp_rejects_conflicting_fp32_flag(version, main_dtype, comm_dtype):
    """Reject stale recipes and policies that only promote one gradient dtype."""
    with pytest.raises(ValueError, match="grad_reduce_in_fp32=True with Megatron-FSDP"):
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=version,
            grad_reduce_in_fp32=True,
            megatron_fsdp_main_grads_dtype=main_dtype,
            megatron_fsdp_grad_comm_dtype=comm_dtype,
        )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("comm_dtype", [None, torch.float32])
def test_mfsdp_accepts_fp32_policy_without_overriding(version, comm_dtype):
    """An unset communication dtype inherits FP32 main gradients."""
    config = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=version,
        grad_reduce_in_fp32=True,
        megatron_fsdp_main_grads_dtype=torch.float32,
        megatron_fsdp_grad_comm_dtype=comm_dtype,
    )
    assert config.megatron_fsdp_main_grads_dtype == torch.float32
    assert config.megatron_fsdp_grad_comm_dtype == comm_dtype


@pytest.mark.parametrize("use_mfsdp, fp32_flag", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("main_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("comm_dtype", [None, torch.bfloat16, torch.float32])
def test_fp32_validation_leaves_other_policies_unchanged(
    use_mfsdp, fp32_flag, main_dtype, comm_dtype
):
    """DDP ignores MFSDP dtypes; flag-off MFSDP retains independent dtype choices."""
    config = DistributedDataParallelConfig(
        use_megatron_fsdp=use_mfsdp,
        grad_reduce_in_fp32=fp32_flag,
        megatron_fsdp_main_grads_dtype=main_dtype,
        megatron_fsdp_grad_comm_dtype=comm_dtype,
    )
    assert config.megatron_fsdp_main_grads_dtype == main_dtype
    assert config.megatron_fsdp_grad_comm_dtype == comm_dtype
