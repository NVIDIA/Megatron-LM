# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from typing import Optional

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.transformer.utils import cat_with_oom_fallback
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: list[int], split_names: list[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    return ShardedTensorFactory(
        orig_sh_ten.key,
        orig_sh_ten.data,
        sh_ten_build_fn,
        cat_with_oom_fallback,
        orig_sh_ten.replica_id,
    )


def _gtp_gather_rows_for_save(
    sh_ten: ShardedTensor,
    key: str,
    weight,
    target_rows: int,
    tp_group,
    dp_cp_group,
    sharded_offsets,
) -> ShardedTensor:
    """All-gather a GTP-sharded fused projection back to the logical TP-local tensor.

    Fused projections (Mamba and GatedDeltaNet ``in_proj``) are checkpointed as semantic
    sections whose boundaries do not line up with GTP slice boundaries, so a per-shard save
    would write a layout that depends on the save-time GTP degree. Gather the shards back to
    TP-local width (stripping the trailing alignment-pad rows) before the section split; the
    checkpoint layout then matches a non-GTP run. The cost is one all-gather per
    ``sharded_state_dict()`` call — including load-time target-dict construction, which is
    safe (all GTP peers build the dict together) but must stay out of per-iteration paths.

    The gathered tensor is replicated across the GTP peers as well as DP/CP. The GTP rank is
    folded into ``replica_id`` so DCP writer election stays correct even when ``dp_cp_group``
    excludes the GTP axis (explicit pg_collection grids pass ``pg_collection.dp_cp``, where
    GTP peers share a rank).
    """
    gtp_remat_group = weight.group
    gtp_rank = torch.distributed.get_rank(gtp_remat_group)
    local = sh_ten.data.contiguous()
    gathered = torch.empty(
        (local.shape[0] * torch.distributed.get_world_size(gtp_remat_group),) + local.shape[1:],
        dtype=local.dtype,
        device=local.device,
    )
    torch.distributed.all_gather_into_tensor(gathered, local, group=gtp_remat_group)
    if gathered.shape[0] > target_rows:
        # GTP alignment padding always sits at the tail of the last shard.
        gathered = gathered[:target_rows].contiguous()
    return make_tp_sharded_tensor_for_checkpoint(
        gathered,
        key,
        tp_axis=0,
        replica_id=(0, gtp_rank, torch.distributed.get_rank(dp_cp_group)),
        prepend_offsets=sharded_offsets,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def _gtp_slice_rows_on_load(factory: ShardedTensorFactory, weight) -> ShardedTensorFactory:
    """Wrap ``factory.merge_fn`` to slice the merged TP-local tensor back to this GTP shard.

    Load-side inverse of :func:`_gtp_gather_rows_for_save`: the checkpoint stores the full
    TP-local projection (pad stripped) under the per-section keys, and the default merge cats
    them back to the unpadded TP-local width. Mirror GTP initialization: zero-pad up to
    ``gtp_local_size * gtp_remat_size``, then select this rank's rows. The alignment-pad rows
    are re-zeroed rather than round-tripped.
    """
    gtp_remat_group = weight.group
    gtp_rank = torch.distributed.get_rank(gtp_remat_group)
    gtp_remat_size = torch.distributed.get_world_size(gtp_remat_group)
    gtp_local_size = weight.data.size(0)
    original_merge_fn = factory.merge_fn

    @torch.no_grad()
    def _gtp_slice_after_cat(sub_state_dict):
        full = original_merge_fn(sub_state_dict)
        if full.dim() != 2:
            # Fail loudly instead of padding/slicing a flattened buffer: only the unflattened
            # 2-D model-weight factory is supported (optimizer state resolves through the
            # per-shard rebuild, never through this merge).
            raise NotImplementedError(
                "GTP fused-projection merge expects the unflattened 2-D projection; got "
                f"a {full.dim()}-D tensor (flattened factories are unsupported)"
            )
        pad_rows = gtp_local_size * gtp_remat_size - full.shape[0]
        if pad_rows > 0:
            full = torch.nn.functional.pad(full, (0, 0, 0, pad_rows))
        start = gtp_rank * gtp_local_size
        return full[start : start + gtp_local_size].contiguous()

    return replace(factory, merge_fn=_gtp_slice_after_cat)
