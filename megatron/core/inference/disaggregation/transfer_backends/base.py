# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KV transfer backend registry and the buffer geometry shared by backends.

Backends are selected explicitly by the caller's launcher configuration,
never from the environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch

from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout
from megatron.core.inference.disaggregation.mamba_reshard import MambaShardLayout

KVTransportBackend = Any


def construct_kv_transfer_backend_class(name: str) -> KVTransportBackend:
    """Return the backend class registered under ``name``."""

    normalized = name.lower().replace("_", "-")
    if normalized == "nixl":
        from .nixl import NixlTransferBackend

        return NixlTransferBackend
    if normalized == "nccl":
        from .nccl import NcclTransferBackend

        return NcclTransferBackend
    raise ValueError("Unsupported KV transfer backend %r; expected 'nixl' or 'nccl'." % name)


@dataclass
class BufferGeometry:
    """Address geometry of one registered paged buffer.

    Each (outer, block) pair is one contiguous slice; the outer stride skips
    over the full block pool for that outer index. ``layout`` is the canonical
    KV shard layout when the buffer is a KV cache; Mamba pools carry their
    typed layout separately.
    """

    buf_ptr: int
    element_size: int
    device_id: int
    blocks_axis: int
    num_blocks: int
    num_outer: int
    bytes_per_slice: int
    outer_stride_bytes: int
    heads_per_partition: Optional[int]
    head_dim: Optional[int]
    tokens_per_block: Optional[int]
    layout: Optional[KVShardLayout]


def compute_buffer_geometry(
    memory_buffer: torch.Tensor,
    expected_num_blocks: int,
    *,
    backend_name: str,
    tp_size: Optional[int] = None,
    tp_rank: Optional[int] = None,
    num_kv_heads_global: Optional[int] = None,
    heads_per_partition: Optional[int] = None,
    head_dim: Optional[int] = None,
    tokens_per_block: Optional[int] = None,
    global_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
    pp_rank: Optional[int] = None,
    num_layers_global: Optional[int] = None,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    mamba_layout: Optional[MambaShardLayout] = None,
    mamba_state_kind: Optional[str] = None,
) -> BufferGeometry:
    """Locate the blocks axis, derive the slice strides, and validate the
    canonical KV layout when the full geometry is provided.

    Shared by every transfer backend so they agree on addressing and on the
    exported metadata schema. The inference KV layout is [2, L, B, T, H, d].
    """
    if (mamba_layout is None) != (mamba_state_kind is None):
        raise ValueError("mamba_layout and mamba_state_kind must be provided together")
    if mamba_state_kind not in (None, "conv", "ssm"):
        raise ValueError("mamba_state_kind must be 'conv' or 'ssm'")

    layout_capable = (
        None
        not in (
            global_rank,
            tp_size,
            tp_rank,
            pp_size,
            pp_rank,
            num_layers_global,
            num_kv_heads_global,
            heads_per_partition,
            head_dim,
            tokens_per_block,
            layer_start,
            layer_end,
        )
        and heads_per_partition * tp_size == num_kv_heads_global
    )

    shape = list(memory_buffer.shape)
    candidates = [i for i, dim in enumerate(shape) if dim == expected_num_blocks]
    if not candidates:
        raise RuntimeError(
            f"{backend_name}: no axis in memory_buffer shape {shape} matches "
            f"expected_num_blocks={expected_num_blocks}. Layout is unrecognized; "
            "bug in caller or new Megatron tensor shape."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"{backend_name}: ambiguous blocks axis in shape {shape} "
            f"(expected_num_blocks={expected_num_blocks} matches multiple axes "
            f"{candidates}). Caller must pass a more distinctive value."
        )
    blocks_axis = candidates[0]

    elements_per_slice = 1
    for dim in shape[blocks_axis + 1 :]:
        elements_per_slice *= dim
    element_size = memory_buffer.element_size()
    bytes_per_slice = element_size * elements_per_slice
    num_outer = 1
    for dim in shape[:blocks_axis]:
        num_outer *= dim

    layout = None
    if layout_capable:
        layout = KVShardLayout(
            num_layers=int(num_layers_global),
            num_heads=int(num_kv_heads_global),
            tp_size=int(tp_size),
            tp_rank=int(tp_rank),
            pp_size=int(pp_size),
            pp_rank=int(pp_rank),
            global_rank=int(global_rank),
            layer_start=int(layer_start),
            num_local_layers=int(layer_end) - int(layer_start),
        )
        if blocks_axis != 2:
            raise ValueError("inference KV transfers require the [2, L, B, T, H, d] layout")
        if layout.local_num_heads() != heads_per_partition:
            raise ValueError(
                "heads_per_partition does not match the canonical KV layout: "
                f"{heads_per_partition} vs {layout.local_num_heads()}"
            )
        if num_outer % layout.local_num_layers() != 0:
            raise ValueError(
                f"num_outer={num_outer} is not divisible by local layers="
                f"{layout.local_num_layers()}"
            )
    if layout is not None and mamba_layout is not None:
        raise ValueError("a transfer backend cannot have both KV and Mamba layouts")

    return BufferGeometry(
        buf_ptr=memory_buffer.data_ptr(),
        element_size=element_size,
        device_id=memory_buffer.device.index if memory_buffer.is_cuda else 0,
        blocks_axis=blocks_axis,
        num_blocks=expected_num_blocks,
        num_outer=num_outer,
        bytes_per_slice=bytes_per_slice,
        outer_stride_bytes=expected_num_blocks * bytes_per_slice,
        heads_per_partition=heads_per_partition,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        layout=layout,
    )


def export_geometry_meta(geometry: BufferGeometry, mamba_layout=None) -> dict:
    """The wire schema shared by every backend's export_meta."""
    meta = {
        "base_addr": geometry.buf_ptr,
        "outer_stride_bytes": geometry.outer_stride_bytes,
        "device_id": geometry.device_id,
        "num_outer": geometry.num_outer,
        "bytes_per_slice": geometry.bytes_per_slice,
        "blocks_axis": geometry.blocks_axis,
        "num_blocks": geometry.num_blocks,
        "heads_per_partition": geometry.heads_per_partition,
        "head_dim": geometry.head_dim,
        "tokens_per_block": geometry.tokens_per_block,
        "element_size": geometry.element_size,
    }
    if geometry.layout is not None:
        layer_start, layer_end = geometry.layout.layer_range()
        meta.update(
            {
                "global_rank": geometry.layout.global_rank,
                "tp_size": geometry.layout.tp_size,
                "tp_rank": geometry.layout.tp_rank,
                "pp_size": geometry.layout.pp_size,
                "pp_rank": geometry.layout.pp_rank,
                "num_layers_global": geometry.layout.num_layers,
                "num_kv_heads_global": geometry.layout.num_heads,
                "layer_start": layer_start,
                "layer_end": layer_end,
            }
        )
    if mamba_layout is not None:
        meta["mamba_layout"] = asdict(mamba_layout)
    return meta
