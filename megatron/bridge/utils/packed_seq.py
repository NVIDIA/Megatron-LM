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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/training/utils/packed_seq_utils.py
# Source license: Apache-2.0.

from __future__ import annotations

import inspect
import math

import torch
from packaging.version import Version as PkgVersion

import megatron.core
from megatron.core import mpu, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_batch_on_this_cp_rank

_MIN_MCORE_THD_CP_VERSION = PkgVersion("0.18.0")


def get_thd_cp_partition_indices(
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    cp_group: torch.distributed.ProcessGroup,
    device: torch.device,
) -> torch.Tensor:
    """Return MCore's context-parallel partition indices for a THD stream.

    Args:
        cu_seqlens: Physical cumulative sequence offsets for the packed stream.
        total_tokens: Total padded token count before CP partitioning.
        cp_group: Context-parallel process group.
        device: Device on which the returned indices will be consumed.

    Returns:
        Long tensor containing this CP rank's indices into the full stream.

    Raises:
        RuntimeError: If the installed Megatron-Core version does not expose
            THD partitioning through ``get_batch_on_this_cp_rank``.
    """
    mcore_version = PkgVersion(megatron.core.__version__)
    supports_thd_partitioning = (
        "is_hybrid_cp" in inspect.signature(get_batch_on_this_cp_rank).parameters
    )
    if mcore_version < _MIN_MCORE_THD_CP_VERSION or not supports_thd_partitioning:
        raise RuntimeError(
            "THD context-parallel partitioning requires Megatron-Core >= 0.18.0 with "
            "get_batch_on_this_cp_rank(..., is_hybrid_cp=...); "
            f"found {megatron.core.__version__}. Please upgrade Megatron-Core."
        )

    cu_seqlens = cu_seqlens.to(device=device)
    index_batch = {
        "tokens": torch.arange(total_tokens, device=device, dtype=torch.long).unsqueeze(0),
        "cu_seqlens": cu_seqlens.unsqueeze(0) if cu_seqlens.dim() == 1 else cu_seqlens,
        "cu_seqlens_padded": None,
    }
    partitioned_batch = get_batch_on_this_cp_rank(
        index_batch, is_hybrid_cp=False, cp_group=cp_group
    )
    return partitioned_batch["tokens"].squeeze(0).to(device=device, dtype=torch.long)


def get_packed_seq_q_cu_seqlens(
    packed_seq_params: PackedSeqParams,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return unpadded and physical query cumulative offsets.

    Args:
        packed_seq_params: MCore THD sequence metadata.

    Returns:
        Unpadded query offsets and physical offsets. Physical offsets use the
        padded metadata when available and otherwise fall back to unpadded offsets.
    """
    cu_seqlens = packed_seq_params.cu_seqlens_q
    cu_seqlens_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens
    return cu_seqlens, cu_seqlens_padded


def get_packed_seq_cp_partition_indices(
    packed_seq_params: PackedSeqParams,
    *,
    total_tokens: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    cp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Return MCore's partition indices for packed CP.

    Args:
        packed_seq_params: MCore THD metadata for the full packed stream.
        total_tokens: Total padded token count before CP partitioning.
        cp_size: Context-parallel world size.
        cp_rank: Context-parallel rank.
        device: Device on which the returned indices will be consumed.
        cp_group: Context-parallel process group. Uses MCore parallel state when omitted.

    Returns:
        Long tensor containing this CP rank's indices into the full stream.

    Raises:
        ValueError: If packed query boundaries are unavailable or the requested
            rank and size do not match the context-parallel group.
    """
    _, cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if cu_seqlens is None:
        raise ValueError("Packed CP partitioning requires cu_seqlens_q metadata.")

    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid context-parallel rank {cp_rank} for size {cp_size}.")
    if cp_group is not None and (cp_group.size() != cp_size or cp_group.rank() != cp_rank):
        raise ValueError(
            f"Context-parallel group has rank {cp_group.rank()} and size {cp_group.size()}, "
            f"but rank {cp_rank} and size {cp_size} were requested."
        )
    if cp_size == 1:
        return torch.arange(total_tokens, device=device, dtype=torch.long)
    if cp_group is None:
        cp_group = parallel_state.get_context_parallel_group()
        if cp_group.size() != cp_size or cp_group.rank() != cp_rank:
            raise ValueError(
                f"Context-parallel group has rank {cp_group.rank()} and size {cp_group.size()}, "
                f"but rank {cp_rank} and size {cp_size} were requested."
            )
    return get_thd_cp_partition_indices(
        cu_seqlens, total_tokens=total_tokens, cp_group=cp_group, device=device
    )


def preprocess_packed_seqs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pre_process: bool = True,
    pg_collection: ProcessGroupCollection | None = None,
    use_fp8_padding: bool = False,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """Preprocess packed sequences into THD layout.

    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets
    first and last chunks, GPU1 gets second and second last chunks, and so on),
    for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    # Ensure boolean dtype for correct advanced indexing (bool → mask select,
    # int → fancy index which silently corrupts data when values are 0/1).
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    if pg_collection is not None:
        tp_size = pg_collection.tp.size()
        cp_size = pg_collection.cp.size()
        cp_rank = pg_collection.cp.rank()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    if use_fp8_padding:
        total_align = align_size * 128
        align_size = math.lcm(16, align_size)

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    if use_fp8_padding:
        pad_size_last = (total_align - cu_seqlens_padded[-1] % total_align) % total_align
        cu_seqlens_padded[-1] += pad_size_last
        seqlens_in_batch_padded[-1] += pad_size_last

    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization.
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()

    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i, attention_mask[i]]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            d = input_ids[i, attention_mask[i]]
            first_start = half_seqlen * cp_rank
            first_end = min(half_seqlen * (cp_rank + 1), d.shape[0])
            first_len = max(first_end - first_start, 0)
            if first_len > 0:
                input_ids_rmpad[start_idx : start_idx + first_len] = d[first_start:first_end]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = min(seqlen_padded_i - half_seqlen * cp_rank, d.shape[0])
            remain_len = max(remain_end - remain_start, 0)
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )

    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params
