# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint mappings for GTP weight shards."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ReplicaId,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.transformer.utils import cat_with_oom_fallback


def make_gtp_swiglu_sharded_factory(
    original_sh_ten: ShardedTensor,
    sharded_offsets: tuple,
    singleton_local_shards: bool,
    gtp_size: int,
) -> ShardedTensorFactory:
    """Map contiguous GTP slices of a TP-local gate/up pair to checkpoint regions.

    Unpadded shards use two equal local tiles; padded shards intersect the
    logical gate/up regions. Merging restores the physical local row slice.
    """
    if getattr(original_sh_ten, "gtp_pad_length", 0):
        return _make_padded_swiglu_factory(
            original_sh_ten, sharded_offsets, singleton_local_shards, gtp_size
        )
    prepend_axis_num = len(sharded_offsets)
    local_axis_size = original_sh_ten.local_shape[0]
    rank_offset = original_sh_ten.global_offset[prepend_axis_num] // local_axis_size
    axis_frag = original_sh_ten.axis_fragmentations[prepend_axis_num]
    tp_rank, gtp_rank = divmod(rank_offset, gtp_size)
    tp_size = axis_frag // gtp_size

    @torch.no_grad()
    def build(
        key: str, tensor: torch.Tensor, replica_id: ReplicaId, flattened_range: slice | None
    ) -> list[ShardedTensor]:
        assert local_axis_size % 2 == 0, (
            "GTP checkpoint mapping requires two equal row tiles; "
            f"got {local_axis_size} local rows."
        )
        pieces = []
        for tile_idx, tile in enumerate(torch.chunk(tensor, 2, dim=0)):
            # Two tiles per rank; each TP-local gate/up half contains gtp_size tiles.
            half_idx, half_offset = divmod(2 * gtp_rank + tile_idx, gtp_size)
            # Position within the selected gate/up half across all TP ranks.
            tile_rank = tp_rank * gtp_size + half_offset
            if singleton_local_shards:
                tile_key = key + ("_w" if half_idx == 0 else "_v")
                tile_frag = axis_frag
            else:
                tile_key = key
                # A shared key stores all gate tiles before all up tiles.
                tile_rank += half_idx * tp_size * gtp_size
                tile_frag = axis_frag * 2
            pieces.append(
                ShardedTensor.from_rank_offsets(
                    tile_key,
                    tile,
                    *sharded_offsets,
                    (prepend_axis_num, tile_rank, tile_frag),
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                )
            )
        return pieces

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        build,
        cat_with_oom_fallback,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )


@dataclass(kw_only=True)
class GTPCheckpointFactory(ShardedTensorFactory):
    """Keep physical shard metadata until serialization.

    ``data`` is the complete local tensor. ``physical_shard`` contains metadata
    only, so clearing factory data also releases its reference to the weight.
    """

    physical_shard: ShardedTensor
    logical_shape: tuple[int, ...] = ()
    tp_axis: int = 0
    pad_length: int = 0
    is_expert: bool = False


@dataclass(frozen=True)
class _CheckpointRows:
    start: int
    stop: int
    key_suffix: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]


def physical_gtp_shard(
    shard: ShardedTensor | ShardedTensorFactory,
) -> ShardedTensor | ShardedTensorFactory:
    """Get full local rows for consumers that perform their own semantic split."""
    if not isinstance(shard, GTPCheckpointFactory):
        return shard
    physical = replace(
        shard.physical_shard,
        data=shard.data,
        dtype=shard.data.dtype if shard.data is not None else shard.physical_shard.dtype,
    )
    physical.gtp_pad_length = shard.pad_length
    return physical


def _row_factory(
    shard: ShardedTensor, rows: list[_CheckpointRows], **layout
) -> GTPCheckpointFactory:
    """Serialize an ordered real-row prefix and rebuild its padded local tensor."""
    template = shard.without_data()
    valid_rows = 0
    for region in rows:
        assert region.start == valid_rows and region.stop > region.start
        valid_rows = region.stop
    assert valid_rows <= template.local_shape[0]

    @torch.no_grad()
    def build(key, tensor, replica_id, flattened_range):
        if flattened_range is not None:
            raise NotImplementedError('Padded GTP checkpointing requires complete local tensors.')
        if tuple(tensor.shape) != template.local_shape:
            raise ValueError(
                f'{key}: expected physical shape {template.local_shape}, got {tensor.shape}.'
            )
        # Retain the caller's destination locally; it is never serialized.
        pieces = {'buffer': LocalNonpersistentObject(tensor)}
        for index, region in enumerate(rows):
            data = tensor[region.start : region.stop]
            pieces[str(index)] = replace(
                template,
                key=key + region.key_suffix,
                data=data,
                dtype=data.dtype,
                local_shape=tuple(data.shape),
                global_shape=region.global_shape,
                global_offset=region.global_offset,
                axis_fragmentations=None,
                replica_id=replica_id,
            )
        return pieces

    @torch.no_grad()
    def merge(parts):
        result = parts['buffer']
        for index, region in enumerate(rows):
            result[region.start : region.stop].copy_(parts[str(index)])
        result[valid_rows:].zero_()
        return result

    return GTPCheckpointFactory(
        shard.key, shard.data, build, merge, shard.replica_id, physical_shard=template, **layout
    )


def make_gtp_checkpoint_factory(
    shard: ShardedTensor,
    source_param: torch.Tensor,
    tp_axis: int,
    tp_rank: int,
    tp_size: int,
    gtp_rank: int,
    gtp_size: int,
) -> ShardedTensor | GTPCheckpointFactory:
    """Map physical GTP rows to unpadded global coordinates without gathering."""
    pad_length = getattr(shard, 'gtp_pad_length', 0)
    if not pad_length:
        return shard
    local_rows = shard.local_shape[0]
    logical_rows = local_rows * gtp_size - pad_length
    if not 0 < logical_rows < local_rows * gtp_size:
        raise ValueError(f'Invalid GTP padding {pad_length} for {shard.key}.')
    valid_rows = min(local_rows, max(0, logical_rows - gtp_rank * local_rows))
    axis = shard.prepend_axis_num
    shape, offset = list(shard.global_shape), list(shard.global_offset)
    shape[axis] = logical_rows * (tp_size if tp_axis == 0 else 1)
    offset[axis] = (tp_rank * logical_rows if tp_axis == 0 else 0) + gtp_rank * local_rows
    rows = [_CheckpointRows(0, valid_rows, '', tuple(shape), tuple(offset))] if valid_rows else []
    return _row_factory(
        shard,
        rows,
        logical_shape=tuple(shape),
        tp_axis=tp_axis,
        pad_length=pad_length,
        is_expert=not getattr(source_param, 'allreduce', True),
    )


def _make_padded_swiglu_factory(
    physical: ShardedTensor, sharded_offsets: tuple, singleton: bool, gtp_size: int
) -> GTPCheckpointFactory:
    """Give the shared row factory the two semantic halves of padded SwiGLU."""
    pad_length = physical.gtp_pad_length
    axis = len(sharded_offsets)
    tp_size = physical.axis_fragmentations[axis] // gtp_size
    local_rows = physical.local_shape[0]
    logical_rows = local_rows * gtp_size - pad_length
    if logical_rows <= 0 or logical_rows % 2:
        raise ValueError(f'{physical.key}: gate/up logical row counts must be equal.')
    rank_offset = physical.global_offset[axis] // local_rows
    tp_rank, gtp_rank = divmod(rank_offset, gtp_size)
    half_rows = logical_rows // 2
    start = gtp_rank * local_rows
    rows = []
    for half in range(2):
        first = max(start, half * half_rows)
        last = min(start + local_rows, (half + 1) * half_rows)
        if first >= last:
            continue
        shape = [1] * axis + list(physical.local_shape)
        offset = [0] * len(shape)
        for dim, rank, fragmentation in sharded_offsets:
            shape[dim], offset[dim] = fragmentation, rank
        shape[axis] = half_rows * tp_size * (1 if singleton else 2)
        offset[axis] = tp_rank * half_rows + first - half * half_rows
        if not singleton:
            offset[axis] += half * half_rows * tp_size
        suffix = ('_w' if half == 0 else '_v') if singleton else ''
        rows.append(
            _CheckpointRows(first - start, last - start, suffix, tuple(shape), tuple(offset))
        )
    return _row_factory(physical, rows, pad_length=pad_length)


def allow_legacy_gtp_padding(
    factories: list[GTPCheckpointFactory],
    checkpoint_metadata: dict[str, ShardedTensor],
    checkpoint_dir: str | Path,
    pad_for_alignment: int,
    legacy_tp_size: int | None = None,
) -> None:
    """Allow tail-padded legacy weights only when their source row layout is known."""
    saved_legacy_args = None
    legacy_args_loaded = False
    for factory in factories:
        if factory.key not in checkpoint_metadata:
            continue
        source_shape = checkpoint_metadata[factory.key].global_shape
        if source_shape == factory.logical_shape:
            continue
        axis = factory.physical_shard.prepend_axis_num
        if (
            len(source_shape) == len(factory.logical_shape)
            and source_shape[:axis] == factory.logical_shape[:axis]
            and source_shape[axis + 1 :] == factory.logical_shape[axis + 1 :]
            and source_shape[axis] > factory.logical_shape[axis]
            and pad_for_alignment > 1
            and source_shape[axis] % pad_for_alignment == 0
        ):
            if factory.tp_axis == 0:
                # A legacy checkpoint may contain padding BETWEEN TP partitions.
                # Its global shape alone cannot distinguish that from tail-only
                # padding, even when the destination uses TP1. Require source evidence.
                source_tp_size = legacy_tp_size
                if source_tp_size is None:
                    from megatron.core.dist_checkpointing.serialization import (
                        load_common_state_dict,
                    )

                    if not legacy_args_loaded:
                        saved_legacy_args = load_common_state_dict(str(checkpoint_dir)).get('args')
                        legacy_args_loaded = True
                    if isinstance(saved_legacy_args, dict):
                        get_arg = saved_legacy_args.get
                    else:

                        def get_arg(name):
                            return getattr(saved_legacy_args, name, None)

                    source_tp_size = get_arg('tensor_model_parallel_size')
                    if factory.is_expert:
                        source_tp_size = get_arg('expert_tensor_parallel_size') or source_tp_size
                if source_tp_size != 1:
                    raise ValueError(
                        f'{factory.key}: legacy row-sharded GTP padding requires known source TP1; '
                        'convert checkpoints with multiple or unknown source TP partitions '
                        'using their original layout before loading logical GTP weights.'
                    )
            factory.physical_shard.allow_shape_mismatch = True
