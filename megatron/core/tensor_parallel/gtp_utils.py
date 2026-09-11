# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Utilities for Generalized Tensor Parallelism (GTP).

A fused projection (Mamba/GatedDeltaNet ``in_proj``, a gated MLP's ``fc1``) is one weight
whose dim0 carries several semantic sections. GTP shards dim0, and the section boundaries
do not line up with the shard boundaries — so the checkpoint must be written and read in
the LOGICAL layout (gather to TP-local for the section split on save, slice this rank's
contiguous rows back out on load). These two helpers implement that round trip; every
fused-projection module wires them around its own section-split factory.

The same pair also pins the storage mapping: a GTP shard holds a CONTIGUOUS row slice of
the logical TP-local tensor. The all-gathered weight is therefore already in logical
order and needs no runtime permutation at the consume sites.
"""

from dataclasses import replace

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensorFactory
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


def gtp_entry_backlink(entry: ShardedTensor | ShardedTensorFactory) -> torch.Tensor | None:
    """Return the live GTP param a model checkpoint entry stands for, or None.

    ``get_param_id_to_sharded_param_map`` / ``param_to_sharded_metadata`` key model entries by
    ``id()`` of the tensor the entry carries. Two GTP cases deliberately put a DIFFERENT tensor
    there, so identity alone cannot match them back to the live parameter:

    - Native FP8: the entry holds a dequantized BF16 COPY of the param, tagged
      ``_gtp_dequant_src`` on the copy (a tensor attribute).
    - Alignment padding: the entry holds a shard whose pad tail was trimmed so the checkpoint
      stays in logical layout, tagged ``gtp_pad_src`` on the ShardedTensor. It must live on the
      ShardedTensor and not on its data, because the torch strategy does
      ``sh_ten.data = sh_ten.data.detach()``, which drops tensor attributes.

    An FP8 param that is ALSO padded has both hops in play, so ``gtp_pad_src`` is set to the live
    param rather than to the dequantized copy -- otherwise the trimmed rank would resolve to a
    copy that the ``_gtp_dequant_src`` lookup can no longer see, and its optimizer state would
    silently vanish.

    Callers must not fold these two ``getattr`` results together with ``or``: the values are
    TENSORS, and ``or`` evaluates ``bool()`` on them, raising "Boolean value of Tensor with more
    than one element is ambiguous" for every native-FP8 entry.
    """
    src = getattr(getattr(entry, 'data', None), '_gtp_dequant_src', None)
    if src is None:
        src = getattr(entry, 'gtp_pad_src', None)
    return src


def untrimmed_gtp_shard(sh_ten: ShardedTensor) -> torch.Tensor:
    """Return the full padded GTP shard behind a checkpoint entry's data.

    ``make_tp_sharded_tensor_for_checkpoint`` keeps alignment padding out of the checkpoint by
    handing back a shard whose pad tail is trimmed -- which makes it SHORTER on the trailing GTP
    rank only. Any all-gather over the GTP group must use the untrimmed shard, or it sizes its
    output off the local length and the trailing rank builds a tensor ``gtp_remat_size * keep``
    rows tall instead of the TP-local projection (and the ranks disagree on the collective, so
    the job hangs rather than failing). Callers strip the pad again after gathering.

    Use ``gtp_pad_buffer`` (what the shard was sliced FROM), not ``gtp_pad_src`` (the live param,
    kept for identity matching). For a native-FP8 param those differ: every other rank
    contributes the dequantized BF16 copy, so contributing the FP8 param here would make the
    collective's inputs disagree in dtype.
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
    physical_shape = tuple(weight.shape)
    physical_numel = weight.numel()
    if len(physical_shape) not in (1, 2):
        raise ValueError("Fused projection optimizer factory expects a 1-D or 2-D parameter")
    templates = [part.without_data() for part in factory.build()]
    base_key = factory.key
    for part in templates:
        if len(part.local_shape) != len(physical_shape) or not part.key.startswith(base_key):
            raise ValueError("Expected row-ordered fused projection section metadata")
    if replica_id is None:
        replica_id = factory.replica_id

    @torch.no_grad()
    def build(key, data, replica_id, flattened_range):
        if flattened_range is None:
            start, stop = 0, physical_numel
        else:
            start, stop = flattened_range.start, flattened_range.stop
            if (
                flattened_range.step not in (None, 1)
                or start is None
                or stop is None
                or not 0 <= start <= stop <= physical_numel
            ):
                raise ValueError(f"Invalid optimizer fragment: {flattened_range}")
        if data.numel() != stop - start:
            raise ValueError("Optimizer data size does not match the physical parameter slice")
        flat = data.detach().reshape(-1)
        logical_start, logical_stop = shard_offset + start, shard_offset + stop
        section_start = 0
        chunks = []
        saved_numel = 0
        for template in templates:
            section_numel = template.local_shape[0]
            columns = template.local_shape[1] if len(physical_shape) == 2 else 1
            section_numel *= columns
            begin = max(logical_start, section_start)
            end = min(logical_stop, section_start + section_numel)
            cursor = begin
            while cursor < end:
                row, col = divmod(cursor - section_start, columns)
                remaining = end - cursor
                if col or remaining < columns:
                    rows, width = 1, min(columns - col, remaining)
                else:
                    rows, width = remaining // columns, columns
                count = rows * width
                shape = (rows, width) if len(physical_shape) == 2 else (rows,)
                offset = list(template.global_offset)
                offset[template.prepend_axis_num] += row
                if len(physical_shape) == 2:
                    offset[template.prepend_axis_num + 1] += col
                # Materialize prepended singleton axes for irregular DCP chunks:
                # the storage adapter then sees equal size/offset dimensionality.
                shape = (1,) * template.prepend_axis_num + shape
                chunk = flat[cursor - logical_start : cursor - logical_start + count].view(shape)
                chunks.append(
                    replace(
                        template,
                        key=key + template.key[len(base_key) :],
                        data=chunk,
                        dtype=data.dtype,
                        local_shape=shape,
                        global_offset=tuple(offset),
                        axis_fragmentations=None,
                        prepend_axis_num=0,
                        flattened_range=None,
                        replica_id=replica_id,
                    )
                )
                saved_numel += count
                cursor += count
            section_start += section_numel
        return {
            "chunks": chunks,
            "layout": LocalNonpersistentObject(
                (tuple(data.shape), data.numel() - saved_numel, data.new_empty(0))
            ),
        }

    @torch.no_grad()
    def merge(state):
        shape, padding, prototype = state["layout"]
        chunks = state["chunks"]
        flat = torch.cat([chunk.reshape(-1) for chunk in chunks]) if chunks else prototype
        if padding:
            flat = torch.cat((flat, flat.new_zeros(padding)))
        return flat.reshape(shape)

    return ShardedTensorFactory(factory.key, weight, build, merge, replica_id)
