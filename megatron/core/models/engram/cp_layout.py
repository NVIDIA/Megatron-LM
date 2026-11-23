# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Context-parallel sequence layout helpers for the n-gram memory.

The n-gram memory is the one part of the layer that is *not* position local: a hash window and
the causal convolution both read positions that may live on another CP rank. These helpers
restore the global sequence and re-select a rank's own positions, which is enough to make the
memory correct under CP, in the two layouts a layer's hidden states can be in:

* ``zigzag``: Megatron's causal context parallelism splits a sequence into ``2 * cp_size``
  chunks and gives rank ``r`` the pair ``(chunk[r], chunk[2 * cp_size - 1 - r])``, so every rank
  carries a similar amount of causal work. A packed row applies the same split inside each
  document.
* ``contiguous``: rank ``r`` owns the ``r``-th of ``cp_size`` equal consecutive slices of the
  row, packed or not, so a document may straddle ranks.

Both are the standard layouts, so the helpers deliberately mirror the partitioning of
``megatron/core/utils.py`` and ``megatron/core/context_parallel`` rather than inventing
another convention.
"""

from __future__ import annotations

import torch
from torch import Tensor

from megatron.core.utils import get_pg_rank, get_pg_size

ZIGZAG_CP_LAYOUT = "zigzag"
CONTIGUOUS_CP_LAYOUT = "contiguous"


def cp_layout_of_layer_type(config, layer_config_type: type) -> str:
    """Return the context-parallel layout of the hidden states a layer of a given type receives.

    The rule is HybridStack's: the attention-type layers run in ``attention_cp_layout`` and
    every other layer type in ``linear_cp_layout``. A config that carries
    ``_linear_cp_layout_explicit`` and leaves ``linear_cp_layout`` implicit runs the whole stack
    in its ``cp_partition_mode`` instead; a config without that field always follows the first
    rule. A missing layout field reads as zigzag.
    """
    if getattr(config, "_linear_cp_layout_explicit", True):
        # Imported here: the hybrid layer table imports every layer config.
        from megatron.core.models.hybrid.layers.utils import Symbols

        name = (
            "attention_cp_layout"
            if layer_config_type in Symbols.ATTENTION_LAYER_CONFIGS
            else "linear_cp_layout"
        )
    else:
        name = "cp_partition_mode"
    return getattr(config, name, None) or ZIGZAG_CP_LAYOUT


def resolve_layer_cp_layout(config) -> str:
    """Return the context-parallel layout of a layer from the layer's own config."""
    return cp_layout_of_layer_type(config, type(config))


def stack_cp_layouts(config) -> frozenset[str]:
    """Return the context-parallel layouts the layers of the stack run in.

    The stack-wide counterpart of :func:`resolve_layer_cp_layout`.
    """
    if getattr(config, "_linear_cp_layout_explicit", True):
        names = ("attention_cp_layout", "linear_cp_layout")
    else:
        names = ("cp_partition_mode",)
    return frozenset(getattr(config, name, None) or ZIGZAG_CP_LAYOUT for name in names)


def stack_mixes_cp_layouts(config) -> bool:
    """Whether the stack converts hidden states between zigzag and contiguous layers."""
    return len(stack_cp_layouts(config)) > 1


class _DifferentiableCPAllGather(torch.autograd.Function):
    """All-gather along the sequence dimension whose backward scatters the gradient back."""

    @staticmethod
    def forward(ctx, tensor: Tensor, group, sequence_dim: int):
        """All-gather ``tensor`` along ``sequence_dim`` over the CP group."""
        ctx.group = group
        ctx.sequence_dim = sequence_dim
        ctx.cp_size = get_pg_size(group)
        ctx.cp_rank = get_pg_rank(group)
        gathered = [torch.empty_like(tensor) for _ in range(ctx.cp_size)]
        torch.distributed.all_gather(gathered, tensor.contiguous(), group=group)
        return torch.cat(gathered, dim=sequence_dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Reduce-scatter the gradient back to this rank's sequence chunk."""
        # Reduce-scatter, not just "take my slice". Every rank convolves the gathered
        # sequence, so every rank produces gradient for *other* ranks' positions -- the
        # convolution's left context reaches back across the chunk boundary. Dropping those
        # contributions silently loses part of the gradient, and an end-to-end loss
        # comparison does not catch it because the lost part is small.
        reduced = grad_output.contiguous()
        torch.distributed.all_reduce(reduced, group=ctx.group)
        shards = reduced.chunk(ctx.cp_size, dim=ctx.sequence_dim)
        return shards[ctx.cp_rank].contiguous(), None, None


def restore_zigzag(tensor: Tensor, cp_size: int, sequence_dim: int) -> Tensor:
    """Reorder a rank-major all-gather into the global sequence order.

    ``tensor`` is the concatenation of every rank's pair of chunks, in rank order. Rank ``r``
    contributed ``chunk[r]`` followed by ``chunk[2 * cp_size - 1 - r]``.
    """
    if cp_size == 1:
        return tensor
    per_rank = tensor.chunk(cp_size, dim=sequence_dim)
    chunks: list[Tensor | None] = [None] * (2 * cp_size)
    for rank, shard in enumerate(per_rank):
        first, second = shard.chunk(2, dim=sequence_dim)
        chunks[rank] = first
        chunks[2 * cp_size - rank - 1] = second
    return torch.cat(chunks, dim=sequence_dim)


def select_zigzag(tensor: Tensor, cp_size: int, cp_rank: int, sequence_dim: int) -> Tensor:
    """Take this rank's two causal-balanced chunks out of a global-order sequence."""
    if cp_size == 1:
        return tensor
    chunks = tensor.chunk(2 * cp_size, dim=sequence_dim)
    return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]), dim=sequence_dim)


def select_contiguous(tensor: Tensor, cp_size: int, cp_rank: int, sequence_dim: int) -> Tensor:
    """Take this rank's slice out of a global-order sequence in the contiguous layout.

    The inverse needs no helper: a rank-major all-gather of the slices is the global sequence.
    """
    if cp_size == 1:
        return tensor
    sequence_length = tensor.shape[sequence_dim]
    if sequence_length % cp_size != 0:
        raise ValueError(
            "Engram contiguous context parallelism requires the sequence length to be "
            f"divisible by cp_size ({cp_size}); got {sequence_length}."
        )
    local_length = sequence_length // cp_size
    return tensor.narrow(sequence_dim, cp_rank * local_length, local_length)


def thd_partition_index(cu_seqlens: Tensor, cp_size: int, cp_rank: int) -> Tensor:
    """Global positions owned by ``cp_rank`` when packed rows are zigzagged per document.

    Packed (THD) context parallelism does not zigzag the row as a whole: it applies the same
    load-balanced split *inside each document*, which is why every document length must be
    divisible by ``2 * cp_size``. The layout therefore changes with the batch's document
    count, and the sequence cannot be restored from a fixed chunk map the way an unpacked row
    can. Mirrors Transformer Engine's ``thd_get_partitioned_indices``.
    """
    boundaries = cu_seqlens.to(torch.int64).tolist()
    spans = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        length = end - start
        if length == 0:
            continue
        if length % (2 * cp_size) != 0:
            raise ValueError(
                f"Engram context-parallel packed rows require every document length to be "
                f"divisible by 2 * cp_size ({2 * cp_size}); got {length}."
            )
        chunk = length // (2 * cp_size)
        first = start + cp_rank * chunk
        second = start + (2 * cp_size - cp_rank - 1) * chunk
        spans.append(torch.arange(first, first + chunk, device=cu_seqlens.device))
        spans.append(torch.arange(second, second + chunk, device=cu_seqlens.device))
    if not spans:
        return torch.empty(0, dtype=torch.int64, device=cu_seqlens.device)
    return torch.cat(spans)


def gather_sequence_thd(tensor: Tensor, group, cu_seqlens: Tensor, total_length: int) -> Tensor:
    """Restore the packed row's global order from this rank's per-document zigzag shards."""
    cp_size = get_pg_size(group)
    if cp_size == 1:
        return tensor
    gathered = _DifferentiableCPAllGather.apply(tensor, group, 0)
    index = torch.cat([thd_partition_index(cu_seqlens, cp_size, rank) for rank in range(cp_size)])
    restored = gathered.new_zeros((total_length, *gathered.shape[1:]))
    return restored.index_copy(0, index, gathered)


def scatter_sequence_thd(tensor: Tensor, group, cu_seqlens: Tensor) -> Tensor:
    """Take this rank's per-document zigzag positions out of a globally ordered packed row."""
    cp_size = get_pg_size(group)
    if cp_size == 1:
        return tensor
    index = thd_partition_index(cu_seqlens, cp_size, get_pg_rank(group))
    return tensor.index_select(0, index)
