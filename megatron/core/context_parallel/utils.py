# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context-parallel batch partitioning helpers."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

import torch

from megatron.core.packed_seq_params import PackedSeqParams

from .layout import CPLayout, THDCPLayoutPlan, _build_thd_zigzag_metadata, build_thd_cp_layout_plan


@dataclass(eq=False)
class ContextParallelBatch:
    """Batch views and reusable metadata prepared for the requested CP layouts.

    Entries with the same layout key in ``batches_by_layout`` and
    ``packed_seq_params_by_layout`` describe the same token ordering. Dense attention masks are
    the exception: their query rows remain zigzag because that is the layout consumed by softmax
    attention. Accessors default to ``boundary_layout``, and ``thd_plan`` connects the two packed
    token orderings.
    """

    boundary_layout: CPLayout
    batches_by_layout: dict[CPLayout, Dict[str, Any]]
    packed_seq_params_by_layout: dict[CPLayout, PackedSeqParams | None]
    thd_plan: THDCPLayoutPlan | None = None

    @classmethod
    def from_single_layout(
        cls, layout: CPLayout, batch: Dict[str, Any], packed_seq_params: PackedSeqParams | None
    ) -> "ContextParallelBatch":
        """Wrap an already-partitioned batch that has one physical CP layout."""
        return cls(
            boundary_layout=layout,
            batches_by_layout={layout: batch},
            packed_seq_params_by_layout={layout: packed_seq_params},
        )

    def get_batch(self, layout: CPLayout | None = None) -> Dict[str, Any]:
        """Return the batch view for a layout, defaulting to the boundary layout."""
        if layout is None:
            layout = self.boundary_layout
        return self.batches_by_layout[layout]

    def get_packed_seq_params(self, layout: CPLayout | None = None) -> PackedSeqParams | None:
        """Return packed metadata for a layout, defaulting to the boundary layout."""
        if layout is None:
            layout = self.boundary_layout
        return self.packed_seq_params_by_layout[layout]


def _get_batch_on_this_cp_rank_contiguous(
    batch: Dict[str, Any], cp_group: torch.distributed.ProcessGroup
) -> Dict[str, Any]:
    """Use contiguous CP shards while keeping dense attention-mask queries zigzag."""
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)

    sequence_keys = ('tokens', 'labels', 'loss_mask', 'position_ids')
    if cp_size == 1:
        return batch

    for key, val in batch.items():
        if val is None:
            continue
        if key == 'attention_mask':
            seq_dim = 2
            if val.shape[seq_dim] % (2 * cp_size) != 0:
                raise ValueError(
                    "The attention-mask sequence length must be divisible by 2 * CP size, "
                    f"got {val.shape[seq_dim]} and CP size {cp_size}."
                )
            segment_len = val.shape[seq_dim] // (2 * cp_size)
            front = val.narrow(seq_dim, cp_rank * segment_len, segment_len)
            back = val.narrow(seq_dim, (2 * cp_size - cp_rank - 1) * segment_len, segment_len)
            batch[key] = torch.cat((front, back), dim=seq_dim)
            continue

        if key not in sequence_keys:
            continue

        seq_dim = 1
        if val.shape[seq_dim] % (2 * cp_size) != 0:
            raise ValueError(
                "The sequence length must be divisible by 2 * CP size so the contiguous shard "
                "can be redistributed to zigzag attention, "
                f"got {val.shape[seq_dim]} and CP size {cp_size} for {key!r}."
            )
        local_seq_len = val.shape[seq_dim] // cp_size
        batch[key] = val.narrow(seq_dim, cp_rank * local_seq_len, local_seq_len).contiguous()

    return batch


def _get_batch_on_this_cp_rank_padded_zigzag(
    batch: Dict[str, Any],
    cp_group: torch.distributed.ProcessGroup,
    rank_order_indices: torch.Tensor,
    target_cu_seqlens_padded: torch.Tensor,
) -> Dict[str, Any]:
    """Build a packed zigzag shard padded for TE context-parallel attention."""
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    if cp_size == 1:
        return batch

    sequence_tensor = None
    for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
        sequence_tensor = batch.get(key)
        if sequence_tensor is not None:
            break
    cu_seqlens = batch['cu_seqlens'].squeeze(0)
    cu_seqlens_padded = batch.get('cu_seqlens_padded')
    if cu_seqlens_padded is not None:
        cu_seqlens_padded = cu_seqlens_padded.squeeze(0)
    physical_cu_seqlens = cu_seqlens if cu_seqlens_padded is None else cu_seqlens_padded
    if sequence_tensor is not None:
        source_positions = torch.arange(
            sequence_tensor.size(1), dtype=physical_cu_seqlens.dtype, device=sequence_tensor.device
        )
        sequence_ids = torch.searchsorted(physical_cu_seqlens[1:], source_positions, right=True)
        valid_ends = physical_cu_seqlens[:-1] + cu_seqlens[1:] - cu_seqlens[:-1]
        source_valid = source_positions < valid_ends.index_select(0, sequence_ids)
        index = rank_order_indices.view(cp_size, -1)[cp_rank]
        valid_index = index.clamp_min(0)
        padding = (index < 0) | ~source_valid.index_select(0, valid_index)
        for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
            tensor = batch.get(key)
            if tensor is not None:
                local_tensor = tensor.index_select(1, valid_index)
                batch[key] = local_tensor.masked_fill(padding.view(1, -1), 0)
    batch['cu_seqlens_padded'] = target_cu_seqlens_padded.unsqueeze(0)
    if batch.get('max_seqlen') is not None:
        max_seqlen = (target_cu_seqlens_padded[1:] - target_cu_seqlens_padded[:-1]).max()
        batch['max_seqlen'] = max_seqlen.reshape_as(batch['max_seqlen'])
    return batch


def _build_packed_seq_params(
    batch: Dict[str, Any],
    layout: CPLayout,
    cp_size: int,
    tokens_per_sample: int | None,
    use_logical_qkv_seqlens: bool = False,
    pad_between_seqs: bool | None = None,
) -> PackedSeqParams | None:
    """Build packed sequence metadata for one physical CP layout."""
    cu_seqlens = batch.get('cu_seqlens')
    if cu_seqlens is None:
        return None

    cu_seqlens = cu_seqlens.squeeze(0)
    cu_seqlens_padded = batch.get('cu_seqlens_padded')
    if cu_seqlens_padded is not None:
        cu_seqlens_padded = cu_seqlens_padded.squeeze(0)
    physical_cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
    qkv_cu_seqlens = (
        cu_seqlens
        if (layout == "contiguous" and cp_size > 1) or use_logical_qkv_seqlens
        else physical_cu_seqlens
    )

    max_seqlen = int(batch['max_seqlen'].item())
    local_cp_size = batch.get('local_cp_size')
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=qkv_cu_seqlens,
        cu_seqlens_kv=qkv_cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        local_cp_size=int(local_cp_size.item()) if local_cp_size is not None else None,
        cp_group=batch.get('hybrid_cp_group'),
        total_tokens=int(physical_cu_seqlens[-1].item()),
        tokens_per_sample=tokens_per_sample,
        pad_between_seqs=pad_between_seqs,
    )


def get_batches_on_this_cp_rank(
    batch: Dict[str, Any],
    boundary_layout: CPLayout,
    is_hybrid_cp: bool,
    cp_group: torch.distributed.ProcessGroup,
    additional_layouts: Iterable[CPLayout] = (),
    hybrid_cp_group_func: Callable[[int], torch.distributed.ProcessGroup] | None = None,
    use_per_sequence_balancing: bool = False,
    sequence_parallel: bool = False,
    tp_group: torch.distributed.ProcessGroup | None = None,
    tp_cp_group: torch.distributed.ProcessGroup | None = None,
    tokens_per_sample: int | None = None,
) -> ContextParallelBatch:
    """Partition a batch and prepare metadata for the requested CP layouts.

    The input is already broadcast over TP but has not yet been partitioned over CP. A caller
    may request additional physical views when different consumers need different layouts. The
    boundary view is always included and is the default returned by ``ContextParallelBatch``.

    Packed non-hybrid CP batches need special handling when layout conversion requires a padded
    zigzag view or the standard sharder would balance the flattened sample instead of each
    sequence. The same rank ordering is used to build the zigzag batch tensors and their
    ``PackedSeqParams``. When both layouts are requested, it also defines the
    activation-conversion plan. All other cases use the standard batch sharder.
    """
    from megatron.core.utils import get_batch_on_this_cp_rank

    requested_layouts = set(additional_layouts)
    requested_layouts.add(boundary_layout)
    cp_size = torch.distributed.get_world_size(cp_group)

    build_packed_zigzag_view = (
        not is_hybrid_cp
        and cp_size > 1
        and batch.get('cu_seqlens') is not None
        and "zigzag" in requested_layouts
        and (use_per_sequence_balancing or "contiguous" in requested_layouts)
    )
    build_thd_plan = build_packed_zigzag_view and "contiguous" in requested_layouts

    if build_packed_zigzag_view:
        # Build the padded zigzag ordering directly from the unsharded batch metadata.
        cu_seqlens = batch['cu_seqlens'].squeeze(0)
        cu_seqlens_padded = batch.get('cu_seqlens_padded')
        if cu_seqlens_padded is not None:
            cu_seqlens_padded = cu_seqlens_padded.squeeze(0)
        tp_size = tp_group.size() if sequence_parallel else 1
        zigzag_metadata = _build_thd_zigzag_metadata(
            cu_seqlens, cu_seqlens_padded, cp_size, tp_size
        )

        batches_by_layout = {
            "zigzag": _get_batch_on_this_cp_rank_padded_zigzag(
                dict(batch),
                cp_group=cp_group,
                rank_order_indices=zigzag_metadata.rank_order_indices,
                target_cu_seqlens_padded=zigzag_metadata.cu_seqlens_padded,
            )
        }
        if build_thd_plan:
            batches_by_layout["contiguous"] = get_batch_on_this_cp_rank(
                dict(batch), is_hybrid_cp=False, cp_group=cp_group, use_contiguous_cp=True
            )
        packed_seq_params_by_layout = {
            layout: _build_packed_seq_params(
                layout_batch,
                layout,
                cp_size,
                tokens_per_sample,
                use_logical_qkv_seqlens=layout == "zigzag",
                pad_between_seqs=(zigzag_metadata.pad_between_seqs if layout == "zigzag" else None),
            )
            for layout, layout_batch in batches_by_layout.items()
        }

        thd_plan = None
        if build_thd_plan:
            # The route is a separate artifact connecting the two completed layout views.
            contiguous_packed_seq_params = packed_seq_params_by_layout["contiguous"]
            assert contiguous_packed_seq_params is not None
            assert contiguous_packed_seq_params.total_tokens is not None
            thd_plan = build_thd_cp_layout_plan(
                zigzag_metadata.rank_order_indices,
                contiguous_packed_seq_params.total_tokens,
                cp_group,
                sequence_parallel,
                tp_group,
                tp_cp_group,
            )
        return ContextParallelBatch(
            boundary_layout=boundary_layout,
            batches_by_layout=batches_by_layout,
            packed_seq_params_by_layout=packed_seq_params_by_layout,
            thd_plan=thd_plan,
        )

    has_sequence_data = any(
        batch.get(key) is not None
        for key in ('tokens', 'labels', 'loss_mask', 'position_ids', 'attention_mask')
    )
    if has_sequence_data:
        # Copy the dictionary because the CP sharder replaces sequence-valued entries in place.
        batches_by_layout = {
            layout: get_batch_on_this_cp_rank(
                dict(batch),
                is_hybrid_cp=is_hybrid_cp,
                cp_group=cp_group,
                hybrid_cp_group_func=hybrid_cp_group_func,
                use_per_sequence_balancing=use_per_sequence_balancing,
                use_contiguous_cp=layout == "contiguous",
            )
            for layout in requested_layouts
        }
    else:
        # Intermediate PP stages receive activations rather than token-aligned tensors.
        batches_by_layout = {layout: dict(batch) for layout in requested_layouts}

    # No shared conversion route exists in this path, so build metadata from each physical batch
    # view rather than deriving one layout's metadata from the other.
    packed_seq_params_by_layout = {
        layout: _build_packed_seq_params(
            batches_by_layout[layout], layout, cp_size, tokens_per_sample
        )
        for layout in requested_layouts
    }

    return ContextParallelBatch(
        boundary_layout=boundary_layout,
        batches_by_layout=batches_by_layout,
        packed_seq_params_by_layout=packed_seq_params_by_layout,
    )
