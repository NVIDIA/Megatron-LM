# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from NVIDIA-NeMo/Megatron-Bridge (Apache-2.0).
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/utils.py

"""Distributed checkpoint layouts for grouped expert adapters."""

import logging
from typing import List, Optional, Tuple

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.process_groups_config import ProcessGroupCollection

from .process_groups import (
    _get_process_group,
    _get_tensor_parallel_group,
    _process_group_rank,
    _process_group_size,
)

logger = logging.getLogger(__name__)


def _divide_exact(value: int, divisor: int, name: str) -> int:
    """Divide ``value`` by ``divisor`` and raise when the result would be fractional."""

    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by expert TP size={divisor}")
    return value // divisor


def _apply_grouped_expert_swiglu_sharded_factory(
    original_sh_ten: ShardedTensor, sharded_offsets: Tuple, singleton_local_shards: bool = False
) -> ShardedTensorFactory:
    """Split grouped-expert SwiGLU tensors along the fused hidden axis for checkpointing."""

    if original_sh_ten.axis_fragmentations is None:
        raise ValueError(
            "Grouped-expert SwiGLU sharding requires regular-grid sharded tensor metadata."
        )

    swiglu_shard_axis = 1
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    local_axis_size = original_shape[swiglu_shard_axis]
    global_axis = swiglu_shard_axis + prepend_axis_num
    assert original_sh_ten.global_offset[global_axis] % local_axis_size == 0
    rank_offset = original_sh_ten.global_offset[global_axis] // local_axis_size
    axis_frag = original_sh_ten.axis_fragmentations[global_axis]

    preserved_rank_offsets = []
    for axis, local_axis_shape in enumerate(original_shape):
        if axis == swiglu_shard_axis:
            continue
        global_axis_idx = axis + prepend_axis_num
        axis_fragm = original_sh_ten.axis_fragmentations[global_axis_idx]
        if axis_fragm <= 1:
            continue
        global_offset = original_sh_ten.global_offset[global_axis_idx]
        assert global_offset % local_axis_shape == 0
        preserved_rank_offsets.append(
            (global_axis_idx, global_offset // local_axis_shape, axis_fragm)
        )

    @torch.no_grad()
    def sh_ten_build_fn(key: str, tensor: torch.Tensor, replica_id, flattened_range):
        del flattened_range

        if singleton_local_shards:
            offset_w = (global_axis, rank_offset, axis_frag)
            offset_v = (global_axis, rank_offset, axis_frag)
            w_key = f"{key}_w"
            v_key = f"{key}_v"
        else:
            offset_w = (global_axis, rank_offset, axis_frag * 2)
            offset_v = (global_axis, rank_offset + axis_frag, axis_frag * 2)
            w_key = key
            v_key = key

        tensor_w, tensor_v = torch.chunk(tensor, 2, dim=swiglu_shard_axis)
        rank_offsets = (*sharded_offsets, *preserved_rank_offsets)
        return [
            ShardedTensor.from_rank_offsets(
                w_key,
                tensor_w,
                *rank_offsets,
                offset_w,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            ShardedTensor.from_rank_offsets(
                v_key,
                tensor_v,
                *rank_offsets,
                offset_v,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        ]

    def sh_ten_merge_fn(sub_state_dict):
        try:
            return torch.cat(sub_state_dict, dim=swiglu_shard_axis)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            logger.warning(
                "CUDA OutOfMemoryError encountered during grouped-expert SwiGLU merge. "
                "Switching to CPU merge. (Error: %s)",
                exc,
            )
            merged_sub_state_dict = torch.cat(
                [tensor.cpu() for tensor in sub_state_dict], dim=swiglu_shard_axis
            )
            torch.cuda.empty_cache()
            return merged_sub_state_dict

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )


def _append_rank_offset(
    rank_offsets: List[Tuple[int, int, int]], axis: int, rank: int, axis_fragments: int
) -> None:
    """Append a sharding offset, combining fragmentations when the axis is already sharded."""

    if axis_fragments <= 1:
        return

    for index, (existing_axis, existing_rank, existing_fragments) in enumerate(rank_offsets):
        if existing_axis != axis:
            continue
        rank_offsets[index] = (
            axis,
            existing_rank * axis_fragments + rank,
            existing_fragments * axis_fragments,
        )
        return

    rank_offsets.append((axis, rank, axis_fragments))


def _make_grouped_expert_sharded_tensor(
    tensor: torch.Tensor,
    key: str,
    *,
    tp_axis: Optional[int],
    sharded_offsets: Tuple,
    pg_collection: ProcessGroupCollection | None,
    ep_size_fallback: int = 1,
    etp_size_fallback: int = 1,
) -> ShardedTensor:
    """Build a sharded tensor for packed grouped-expert weights.

    Grouped-expert LoRA weights shard two independent local axes: the packed
    expert axis across EP and the adapter matrix axis across expert TP.
    """

    prepend_axis_num = len(sharded_offsets)
    rank_offsets = list(sharded_offsets)

    ep_group = _get_process_group(pg_collection, "ep")
    ep_size = _process_group_size(ep_group, ep_size_fallback)
    _append_rank_offset(rank_offsets, prepend_axis_num, _process_group_rank(ep_group), ep_size)

    if tp_axis is not None:
        etp_group = _get_tensor_parallel_group(pg_collection, is_expert=True)
        etp_size = _process_group_size(etp_group, etp_size_fallback)
        _append_rank_offset(
            rank_offsets, prepend_axis_num + tp_axis, _process_group_rank(etp_group), etp_size
        )

    expt_dp_group = _get_process_group(pg_collection, "expt_dp")
    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *rank_offsets,
        replica_id=(0, 0, _process_group_rank(expt_dp_group)),
        prepend_axis_num=prepend_axis_num,
    )
