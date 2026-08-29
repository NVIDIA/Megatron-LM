# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter layout dataclasses for optimizer-driven buffer layout.

These dataclasses describe how parameters are laid out in contiguous buffers.
Each distributed optimizer implementation (e.g., DistributedOptimizer) is
responsible for computing these layouts via a _compute_per_buffer_param_layout method,
applying its own padding, alignment, and bucket splitting rules. DDP and
buffers consume the resulting layouts without any optimizer-specific knowledge.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


def pad_to_divisor(value: int, divisor: int) -> int:
    """Round up ``value`` to the nearest multiple of ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)


def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary."""
    return pad_to_divisor(param_start_index, 64)


def bucket_end_divisor(data_parallel_world_size: int, pad_for_high_nccl_busbw: bool) -> int:
    """Divisor used to pad bucket ends for DP-divisibility (and optional NCCL busbw)."""
    if pad_for_high_nccl_busbw:
        return math.lcm(data_parallel_world_size, 128, 2**16)
    return math.lcm(data_parallel_world_size, 128)


def pad_bucket_end(
    bucket_end_index: int, data_parallel_world_size: int, pad_for_high_nccl_busbw: bool
) -> int:
    """Pad bucket end for DP-divisibility (and optionally high NCCL bus bandwidth)."""
    return pad_to_divisor(
        bucket_end_index, bucket_end_divisor(data_parallel_world_size, pad_for_high_nccl_busbw)
    )


def shared_storage_group_key(
    param: torch.nn.Parameter,
) -> Optional[Tuple[torch.device, torch.dtype, int]]:
    """Identify a plain parameter that is a view into a larger shared storage.

    Some modules expose several independently named parameters to training and checkpointing,
    while deliberately allocating their data as adjacent views of one tensor. The parameter
    buffer must preserve that structural relationship so consumers can recover a fused view
    after DDP remaps the parameters.

    This intentionally uses storage identity rather than an attribute set out-of-band by the
    module. Standalone parameters and non-strided or meta tensors are not grouped.
    """
    data = param.data
    if data.device.type == "meta" or data.layout != torch.strided or not data.is_contiguous():
        return None
    storage = data.untyped_storage()
    if storage.nbytes() <= data.numel() * data.element_size():
        return None
    return (data.device, data.dtype, storage.data_ptr())


def params_share_gapless_storage(first: torch.nn.Parameter, second: torch.nn.Parameter) -> bool:
    """Return whether two parameters are consecutive views of one storage."""
    first_key = shared_storage_group_key(first)
    return (
        first_key is not None
        and first_key == shared_storage_group_key(second)
        and first.data.storage_offset() + first.numel() == second.data.storage_offset()
    )


def order_params_for_layout(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    """Return reverse-registration order while preserving adjacent shared-storage groups.

    DDP normally packs parameters in reverse registration order. Reversing each member of a
    shared allocation would destroy its layout, so a gapless run of views into one storage is
    kept in ascending storage-offset order. The groups themselves remain in normal reverse
    registration order.
    """
    reversed_params = list(params)[::-1]
    ordered: List[torch.nn.Parameter] = []
    index = 0
    while index < len(reversed_params):
        group_key = shared_storage_group_key(reversed_params[index])
        if group_key is None:
            ordered.append(reversed_params[index])
            index += 1
            continue

        run_end = index + 1
        while (
            run_end < len(reversed_params)
            and shared_storage_group_key(reversed_params[run_end]) == group_key
        ):
            run_end += 1

        run = reversed_params[index:run_end]
        ascending = sorted(run, key=lambda param: param.data.storage_offset())
        is_gapless_group = len(ascending) > 1 and all(
            params_share_gapless_storage(first, second)
            for first, second in zip(ascending, ascending[1:])
        )

        ordered.extend(ascending if is_gapless_group else run)
        index = run_end
    return ordered


@dataclass(frozen=True)
class BufferKey:
    """Identifies a distinct parameter buffer.

    Each unique combination of these fields corresponds to a separate contiguous
    buffer in DDP. Parameters are grouped into buffers by these dimensions.

    Attributes:
        param_dtype: Storage dtype (torch.uint8 for FP8/NVFP4 parameters, else param.dtype).
        grad_dtype: Gradient reduction dtype.
        is_expert_parallel: Whether the buffer holds expert-parallel parameters,
            which use a separate data-parallel group.
        is_managed_by_layer_wise_optimizer: Whether parameters in this buffer are
            managed by :class:`LayerWiseDistributedOptimizer` (shard-aligned layout
            so each whole param lives in one shard). Non-LayerWise params get
            :class:`DistributedOptimizer`'s byte-level layout in a separate buffer.
    """

    param_dtype: torch.dtype
    grad_dtype: torch.dtype
    is_expert_parallel: bool
    is_managed_by_layer_wise_optimizer: bool = False


@dataclass
class PerBufferParamLayout:
    """Layout for parameters within a single contiguous buffer.

    Describes how parameters should be laid out in the contiguous buffer.

    Attributes:
        param_index_map: Mapping from parameter to (start_index, end_index, bucket_id) in buffer.
        bucket_indices: List of (start_index, end_index) for each bucket.
        per_bucket_numel_unpadded: Number of unpadded elements per bucket.
        param_indices: The index of each param among same-dtype params (using the "fake"
            high-precision dtype for FP8/NVFP4 params). Needed for loading non-native-fp8
            checkpoints in native-fp8 mode. Order matches param_index_map iteration order.
        num_optimizer_shards: Number of optimizer shards. Set by the distributed optimizer
            that computes the layout so that shard assignment at runtime uses the same
            value. ``None`` for non-distributed-optimizer layouts.
    """

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(default_factory=dict)
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)
    param_indices: List[int] = field(default_factory=list)
    num_optimizer_shards: Optional[int] = None


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
