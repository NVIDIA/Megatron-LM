# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Ported verbatim from DeepSeek DualPipe: https://github.com/deepseek-ai/DualPipe

from typing import List, Tuple

import torch
import torch.distributed as dist

TENSOR_SHAPES: List[Tuple[int]] = None
TENSOR_DTYPE: torch.dtype = None


def set_p2p_tensor_shapes(shapes: List[Tuple[int]]):
    """Set the shapes of the tensors exchanged between pipeline ranks."""
    global TENSOR_SHAPES
    TENSOR_SHAPES = shapes


def set_p2p_tensor_dtype(dtype: torch.dtype):
    """Set the dtype of the tensors exchanged between pipeline ranks."""
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype


def build_from_tensor_shapes():
    """Allocate empty receive buffers matching the configured P2P shapes and dtype."""
    return [
        torch.empty(s, dtype=TENSOR_DTYPE, device="cuda", requires_grad=True) for s in TENSOR_SHAPES
    ]


def append_irecv(ops: List[dist.P2POp], src: int, group: dist.ProcessGroup) -> List[torch.Tensor]:
    """Append irecv ops (one per configured tensor shape) from ``src`` to ``ops``."""
    tensors = build_from_tensor_shapes()
    src = dist.distributed_c10d.get_global_rank(group, src)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv, tensor, src))
    return tensors


def append_isend(
    ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int, group: dist.ProcessGroup
) -> None:
    """Append isend ops for ``tensors`` to ``dst`` to ``ops``."""
    dst = dist.distributed_c10d.get_global_rank(group, dst)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))
