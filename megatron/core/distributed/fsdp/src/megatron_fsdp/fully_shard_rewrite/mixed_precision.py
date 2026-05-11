# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Mixed precision policy helpers for the Megatron-FSDP fully_shard rewrite path.

The adapter builds the v2 policy here and passes it to ``fully_shard``. Keep
FP8-specific policy decisions in this module instead of spreading them through
the adapter or ``ParameterGroup``.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class FullyShardMixedPrecisionPolicy:
    """Mixed precision dtype policy owned by the v2 ``fully_shard`` path."""

    main_params_dtype: Optional[torch.dtype] = torch.float32
    main_grads_dtype: Optional[torch.dtype] = None
    grad_comm_dtype: Optional[torch.dtype] = None


def build_fully_shard_mixed_precision_policy(ddp_config) -> FullyShardMixedPrecisionPolicy:
    """Build the v2 mixed precision policy from Megatron's DDP/FSDP config."""

    if ddp_config.grad_reduce_in_fp32:
        main_grads_dtype = torch.float32
        grad_comm_dtype = torch.float32
    else:
        main_grads_dtype = ddp_config.megatron_fsdp_main_grads_dtype
        grad_comm_dtype = ddp_config.megatron_fsdp_grad_comm_dtype

    return FullyShardMixedPrecisionPolicy(
        main_params_dtype=ddp_config.megatron_fsdp_main_params_dtype,
        main_grads_dtype=main_grads_dtype,
        grad_comm_dtype=grad_comm_dtype,
    )
