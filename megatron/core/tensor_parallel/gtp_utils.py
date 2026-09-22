# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Utilities for Generalized Tensor Parallelism (GTP).

A fused projection (Mamba/GatedDeltaNet ``in_proj``, a gated MLP's ``fc1``) is one weight
whose dim0 carries several semantic sections. GTP shards dim0, and the section boundaries
do not line up with the shard boundaries — so the checkpoint must be written and read in
the LOGICAL layout (gather to TP-local for the section split on save, slice this rank's
contiguous rows back out on load). These two helpers implement that round trip; every
fused-projection module wires them around its own section-split factory.

These helpers preserve GTP's existing storage mapping: each shard holds a contiguous row
slice of the logical TP-local tensor. Runtime all-gather therefore reconstructs the logical
order without a permutation.
"""

from dataclasses import replace

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.dist_checkpointing.optimizer import _make_sharded_optimizer_factory
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


def gtp_entry_backlink(entry: ShardedTensor | ShardedTensorFactory) -> torch.Tensor | None:
    """Resolve a dequantized or padding-trimmed entry to its live parameter.

    Native-FP8 copies carry ``_gtp_dequant_src`` on the tensor. Trimmed entries carry
    ``gtp_pad_src`` on the ShardedTensor because slicing and detach drop tensor
    attributes. Both refer to the live parameter, including padded native FP8.
    """
    src = getattr(getattr(entry, 'data', None), '_gtp_dequant_src', None)
    if src is None:
        src = getattr(entry, 'gtp_pad_src', None)
    return src


def untrimmed_gtp_shard(sh_ten: ShardedTensor) -> torch.Tensor:
    """Return the full data buffer behind a padding-trimmed checkpoint entry.

    Collectives need equal shard sizes. For native FP8, ``gtp_pad_buffer`` holds
    the dequantized buffer; the live parameter in ``gtp_pad_src`` has different
    storage and must not replace it. Optimizer entries can use a single backlink
    because their identity and data buffers coincide.
    """
    src = getattr(sh_ten, "gtp_pad_buffer", None)
    if src is None:
        src = getattr(sh_ten, "gtp_pad_src", None)
    return sh_ten.data if src is None else src


@torch.no_grad()
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

    Fused projections are checkpointed as semantic sections whose boundaries do not line
    up with GTP slice boundaries, so a per-shard save would write a layout that depends on
    the save-time GTP degree. Gather the shards back to TP-local width (stripping the
    trailing alignment-pad rows) before the section split; the checkpoint layout then
    matches a non-GTP run. The cost is one all-gather per ``sharded_state_dict()`` call —
    including load-time target-dict construction, which is safe (all GTP peers build the
    dict together) but must stay out of per-iteration paths.

    The gathered tensor is replicated across the GTP peers as well as DP/CP. The GTP rank
    is folded into ``replica_id`` so DCP writer election stays correct even when
    ``dp_cp_group`` excludes the GTP axis (explicit pg_collection grids pass
    ``pg_collection.dp_cp``, where GTP peers share a rank).
    """
    gtp_remat_group = weight.group
    gtp_rank = torch.distributed.get_rank(gtp_remat_group)
    local = untrimmed_gtp_shard(sh_ten).contiguous()
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
    TP-local projection (pad stripped) under the per-section keys, and the default merge
    cats them back to the unpadded TP-local width. Mirror GTP initialization: zero-pad up
    to ``gtp_local_size * gtp_remat_size``, then select this rank's rows. The
    alignment-pad rows are re-zeroed rather than round-tripped.
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
            # Fail loudly instead of padding/slicing a flattened buffer: only the
            # unflattened 2-D model-weight factory is supported. Optimizer state uses
            # the physical-parameter companion rather than this model merge.
            raise NotImplementedError(
                "GTP fused-projection merge expects the unflattened 2-D projection; got "
                f"a {full.dim()}-D tensor (flattened factories are unsupported)"
            )
        pad_rows = gtp_local_size * gtp_remat_size - full.shape[0]
        if pad_rows > 0:
            full = torch.nn.functional.pad(full, (0, 0, 0, pad_rows))
        start = gtp_rank * gtp_local_size
        return full[start : start + gtp_local_size].contiguous()

    return replace(
        factory,
        merge_fn=_gtp_slice_after_cat,
        # Physical optimizer shards cover distinct offsets, so only DP replicas
        # remain; the model factory's middle coordinate elects gathered GTP copies.
        optimizer_factory=_fused_projection_optimizer_factory(
            factory,
            weight,
            shard_offset=gtp_rank * weight.numel(),
            replica_id=(factory.replica_id[0], 0, factory.replica_id[2]),
        ),
    )


def _fused_projection_optimizer_factory(
    factory: ShardedTensorFactory, weight: torch.Tensor, *, shard_offset: int = 0, replica_id=None
) -> ShardedTensorFactory:
    """Map physical optimizer slices into a fused projection's semantic checkpoint keys.

    Model checkpoint factories may hold a gathered/dequantized tensor instead of the
    live parameter. Optimizers need the parameter identity and must transform their own
    data, including arbitrary flat DP fragments, without requiring GTP collectives.
    Capture only the logical section metadata; intersect each physical input slice with
    those sections at build time. Padding is omitted on save and restored as zeros.

    Flat fragments are represented as ordinary rectangular shards (partial boundary
    rows and complete interior rows), since DCP no longer accepts flattened tensors.
    This helper supports the 1-D biases and 2-D weights of row-split projections.
    """
    if weight.ndim not in (1, 2):
        raise ValueError("Fused projection optimizer factory expects a 1-D or 2-D parameter")
    parts = factory.build()
    if any(len(part.local_shape) != weight.ndim for part in parts):
        raise ValueError("Expected row-ordered fused projection section metadata")
    return _make_sharded_optimizer_factory(
        parts,
        weight,
        factory.key,
        physical_numel=weight.numel(),
        shard_offset=shard_offset,
        replica_id=factory.replica_id if replica_id is None else replica_id,
    )
