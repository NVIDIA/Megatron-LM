# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixed precision policy helpers for Megatron-FSDP2.

This module owns the v2 policy data model. Translation from Megatron/MCore
config objects belongs in the adapter layer.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class FullyShardMixedPrecisionPolicy:
    """Mixed precision dtype policy owned by the v2 ``fully_shard`` path."""

    main_params_dtype: Optional[torch.dtype] = None
    main_grads_dtype: Optional[torch.dtype] = None
    grad_comm_dtype: Optional[torch.dtype] = None
