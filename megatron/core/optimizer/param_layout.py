# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter layout dataclasses for optimizer-driven buffer layout.

These dataclasses describe how parameters are laid out in contiguous buffers.
Each distributed optimizer implementation (e.g., DistributedOptimizer) is
responsible for computing these layouts via a _compute_per_buffer_param_layout method,
applying its own padding, alignment, and bucket splitting rules. DDP and
buffers consume the resulting layouts without any optimizer-specific knowledge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class BufferKey:
    """Identifies a distinct parameter buffer.

    Each unique combination of these fields corresponds to a separate contiguous
    buffer in DDP. Parameters are grouped into buffers by these dimensions.

    Attributes:
        param_dtype: Storage dtype (torch.uint8 for FP8 parameters, else param.dtype).
        grad_dtype: Gradient reduction dtype.
        is_expert_parallel: Whether the buffer holds expert-parallel parameters,
            which use a separate data-parallel group.
    """

    param_dtype: torch.dtype
    grad_dtype: torch.dtype
    is_expert_parallel: bool


@dataclass
class PerBufferParamLayout:
    """Layout for parameters within a single contiguous buffer.

    Describes how parameters should be laid out in the contiguous buffer.

    Attributes:
        param_index_map: Mapping from parameter to (start_index, end_index, bucket_id) in buffer.
        bucket_indices: List of (start_index, end_index) for each bucket.
        per_bucket_numel_unpadded: Number of unpadded elements per bucket.
        param_indices: The index of each param among same-dtype params (using the "fake"
            high-precision dtype for FP8 params). Needed for loading non-native-fp8
            checkpoints in native-fp8 mode. Order matches param_index_map iteration order.
    """

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(default_factory=dict)
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)
    param_indices: List[int] = field(default_factory=list)


@dataclass
class FullParamLayout:
    """Layout for all parameters across all buffer groups in a model chunk.

    Maps BufferKey to per-buffer PerBufferParamLayout objects. Each PerBufferParamLayout has its
    own independent index space since different buffer groups are physically
    separate buffers.

    Attributes:
        layouts: Mapping from BufferKey to PerBufferParamLayout.
    """

    layouts: Dict[BufferKey, PerBufferParamLayout] = field(default_factory=dict)
