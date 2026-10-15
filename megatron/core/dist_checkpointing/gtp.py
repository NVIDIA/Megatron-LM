# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local checkpoint fragments for GTP fused projections (no parameter collectives)."""

from dataclasses import replace
from typing import List, Optional

import torch

from .mapping import LocalNonpersistentObject, ReplicaId, ShardedTensor, ShardedTensorFactory


def untrimmed_gtp_shard(entry: ShardedTensor) -> torch.Tensor:
    """Return checkpoint storage, including padding, without losing FP8 dequantization."""
    buffer = getattr(entry, 'gtp_pad_buffer', None)
    return entry.data if buffer is None else buffer


def attach_optimizer_factory(
    factory: ShardedTensorFactory, weight: torch.Tensor
) -> ShardedTensorFactory:
    """Bind physical optimizer tensors to the model's ordered local logical fragments."""
    from .optimizer import _make_sharded_optimizer_factory

    parts = [part for part in factory.build() if isinstance(part, ShardedTensor)]
    factory.optimizer_factory = _make_sharded_optimizer_factory(
        parts, weight, factory.key, physical_numel=weight.numel(), replica_id=factory.replica_id
    )
    return factory


def _gtp_split_tensor_factory(
    orig_sh_ten: ShardedTensor,
    split_sections: List[int],
    split_names: List[str],
    gtp_remat_rank: int,
    tp_rank: int,
    tp_size: int,
    *,
    weight: Optional[torch.Tensor] = None,
    glu: bool = False,
    singleton_local_shards: bool = False,
) -> ShardedTensorFactory:
    """Save intersections of a physical GTP shard with TP-local semantic sections.

    Derived from GDP's local-fragment factory. SSM sections use separate keys; GLU
    sections retain the ordinary gate-major/up-major checkpoint coordinates. Model
    fragments are views. Optimizer companions map flat DP fragments to these same
    coordinates, without gathering either model values or optimizer state.
    """
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    data = untrimmed_gtp_shard(orig_sh_ten)
    metadata = orig_sh_ten.without_data()
    prepend = metadata.prepend_axis_num
    local_rows, *trailing_shape = data.shape
    trailing_shape = tuple(trailing_shape)
    if weight is not None:
        assert tuple(data.shape) == tuple(weight.shape), (data.shape, weight.shape)
    if len(split_sections) != len(split_names) or any(size <= 0 for size in split_sections):
        raise ValueError('Split names must match positive section sizes')
    if glu and (len(split_sections) != 2 or split_sections[0] != split_sections[1]):
        raise ValueError('GLU requires equal gate and up sections')
    shard_start = gtp_remat_rank * local_rows
    shard_end = min(shard_start + local_rows, sum(split_sections))
    valid_rows = max(0, shard_end - shard_start)
    base_key = orig_sh_ten.key
    pieces = []
    section_start = 0
    for index, (section_size, section_name) in enumerate(zip(split_sections, split_names)):
        piece_start = max(section_start, shard_start)
        piece_end = min(section_start + section_size, shard_end)
        if piece_start < piece_end:
            offset = tp_rank * section_size + piece_start - section_start
            global_rows = tp_size * section_size
            suffix = f'.{section_name}'
            if glu:
                suffix = f'_{section_name}' if singleton_local_shards else ''
                if not singleton_local_shards:
                    offset += index * global_rows
                    global_rows *= 2
            shape = (piece_end - piece_start,) + trailing_shape
            template = replace(
                metadata,
                key=base_key + suffix,
                local_shape=shape,
                global_shape=metadata.global_shape[:prepend] + (global_rows,) + trailing_shape,
                global_offset=metadata.global_offset[:prepend]
                + (offset,)
                + (0,) * len(trailing_shape),
                axis_fragmentations=None,
                flattened_range=None,
                allow_shape_mismatch=False,
            )
            pieces.append((slice(piece_start - shard_start, piece_end - shard_start), template))
        section_start += section_size

    @torch.no_grad()
    def build(key: str, tensor: torch.Tensor, replica_id: ReplicaId, flattened_range):
        assert flattened_range is None, 'Use the physical optimizer companion for flat fragments'
        if not pieces:
            return [LocalNonpersistentObject(tensor.new_empty((0,) + trailing_shape))]
        return [
            replace(
                template,
                key=key + template.key[len(base_key) :],
                data=tensor[rows],
                dtype=tensor.dtype,
                replica_id=replica_id,
            )
            for rows, template in pieces
        ]

    @torch.no_grad()
    def merge(parts):
        merged = parts[0].new_empty((local_rows,) + trailing_shape)
        if valid_rows:
            merged[:valid_rows] = torch.cat(parts)
        merged[valid_rows:].zero_()
        return merged

    # Preserve GDP's legacy optimizer behavior unless a source-bound companion is requested.
    factory = ShardedTensorFactory(
        base_key, data.view_as(data), build, merge, orig_sh_ten.replica_id
    )
    if weight is not None:
        attach_optimizer_factory(factory, weight)
    return factory


def split_gtp_projection(
    entry: ShardedTensor,
    weight: torch.Tensor,
    sections: list[int],
    names: list[str],
    tp_group: torch.distributed.ProcessGroup | None,
    *,
    glu: bool = False,
    singleton: bool = False,
) -> ShardedTensorFactory | None:
    """Return a local GTP split factory, or None for ordinary TP parameters."""
    from megatron.core.tensor_parallel import gtp_api
    from megatron.core.utils import get_pg_rank, get_pg_size

    if not gtp_api.HAVE_GTP or not gtp_api.is_gtp_param(weight):
        return None
    physical_rows = weight.shape[0]
    logical_rows = physical_rows * weight.group.size() - int(getattr(weight, 'pad_length', 0))
    if sum(sections) != logical_rows:
        raise ValueError(f'Projection sections {sections} do not cover {logical_rows} logical rows')
    return _gtp_split_tensor_factory(
        entry,
        sections,
        names,
        weight.group.rank(),
        get_pg_rank(tp_group),
        get_pg_size(tp_group),
        weight=weight,
        glu=glu,
        singleton_local_shards=singleton,
    )
