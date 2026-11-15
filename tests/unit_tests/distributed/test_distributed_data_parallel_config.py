# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolution and validation of data-parallel gradient precision."""

from dataclasses import asdict

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig


@pytest.mark.parametrize("fsdp_version", [1, 2])
@pytest.mark.parametrize(
    "reduce_in_fp32,main_dtype,comm_dtype,expected_main_dtype",
    [
        (True, None, None, torch.float32),
        (True, None, torch.float32, torch.float32),
        (True, torch.float32, None, torch.float32),
        (False, None, None, None),
        (False, torch.float32, torch.bfloat16, torch.float32),
    ],
)
def test_mfsdp_gradient_precision_resolves_after_recipe_edits(
    fsdp_version, reduce_in_fp32, main_dtype, comm_dtype, expected_main_dtype
):
    """The reduction flag resolves MFSDP's unset dtype when wrappers finalize config."""
    config = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=fsdp_version,
        megatron_fsdp_main_grads_dtype=main_dtype,
        megatron_fsdp_grad_comm_dtype=comm_dtype,
    )
    config.grad_reduce_in_fp32 = reduce_in_fp32
    before = asdict(config)
    config.validate()
    assert asdict(config) == before

    for _ in range(2):
        config.finalize()
        assert config.megatron_fsdp_main_grads_dtype == expected_main_dtype
        assert config.megatron_fsdp_grad_comm_dtype == comm_dtype


@pytest.mark.parametrize("fsdp_version", [1, 2])
@pytest.mark.parametrize(
    "main_dtype,comm_dtype,field",
    [
        (torch.bfloat16, None, "megatron_fsdp_main_grads_dtype"),
        (None, torch.bfloat16, "megatron_fsdp_grad_comm_dtype"),
    ],
)
def test_mfsdp_fp32_reduction_rejects_conflicting_explicit_dtypes(
    fsdp_version, main_dtype, comm_dtype, field
):
    """An explicit lower-precision dtype must not silently defeat FP32 reduction."""
    config = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=fsdp_version,
        grad_reduce_in_fp32=True,
        megatron_fsdp_main_grads_dtype=main_dtype,
        megatron_fsdp_grad_comm_dtype=comm_dtype,
    )
    with pytest.raises(ValueError, match=field):
        config.validate()
    with pytest.raises(ValueError, match=field):
        config.finalize()
