# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construction-time selection for the validated BF16 residual norm contract."""

import os
from importlib.metadata import PackageNotFoundError, version

import torch

from megatron.core.ops._backends import require


def select_bf16_add_rmsnorm(hidden_size, dtype, fp32_residual_connection):
    """Return the tested kernel, or None for a different numerical contract.

    The initial default is deliberately limited to the measured TE/Triton/GPU
    combination. Widening this selection requires byte comparisons on the new
    backend; a shared RMSNorm formula alone does not imply equal arithmetic.
    """
    if (
        not 8 <= hidden_size <= 8192
        or hidden_size % 8
        or dtype != torch.bfloat16
        or fp32_residual_connection
    ):
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return None
    try:
        te_version = version("transformer_engine")
        triton_version = version("triton")
    except PackageNotFoundError:
        return None
    if te_version != "2.15.0+42b84005" or not triton_version.startswith("3.6."):
        return None
    if not torch.__version__.startswith("2.11."):
        return None
    if any(
        os.environ.get(name, "0") != "0"
        for name in ("NVTE_NORM_FWD_USE_CUDNN", "NVTE_NORM_USE_CUDNN", "NVTE_USE_CUDNN_RMSNORM")
    ):
        return None
    module = require(
        "megatron.core.ops.normalization.bf16_add_rmsnorm",
        "fused_bf16_add_rmsnorm",
        needed_by="BF16 residual addition and RMSNorm",
    )
    return module.fused_bf16_add_rmsnorm
