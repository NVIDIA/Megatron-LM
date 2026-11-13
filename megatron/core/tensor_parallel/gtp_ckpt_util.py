# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint mappings for GTP weight shards."""

import torch

from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensor, ShardedTensorFactory
from megatron.core.transformer.utils import cat_with_oom_fallback


def make_gtp_swiglu_sharded_factory(
    original_sh_ten: ShardedTensor,
    sharded_offsets: tuple,
    singleton_local_shards: bool,
    gtp_size: int,
) -> ShardedTensorFactory:
    """Map contiguous GTP slices of a TP-local gate/up pair to checkpoint regions.

    Each local tile can belong to either half. Only unpadded shards with even
    local row counts are supported; merging restores the original row slice.
    """
    prepend_axis_num = len(sharded_offsets)
    local_axis_size = original_sh_ten.local_shape[0]
    rank_offset = original_sh_ten.global_offset[prepend_axis_num] // local_axis_size
    axis_frag = original_sh_ten.axis_fragmentations[prepend_axis_num]
    pad_length = getattr(original_sh_ten, "gtp_pad_length", 0)
    tp_rank, gtp_rank = divmod(rank_offset, gtp_size)
    tp_size = axis_frag // gtp_size

    @torch.no_grad()
    def build(
        key: str, tensor: torch.Tensor, replica_id: ReplicaId, flattened_range: slice | None
    ) -> list[ShardedTensor]:
        assert not pad_length, "SwiGLU checkpoint mapping requires unpadded GTP FC1 shards."
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
