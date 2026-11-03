# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compact narrow CP compressor projections with a unique-source backward map."""

import torch
from torch import Tensor


def cp_projection_maps(source_indices, valid_groups, full_halo_rows, halo_rows, local_rows):
    """Build shared forward/inverse maps, excluding all logically padded groups.

    Compression groups do not overlap for CSA2 r1/r2. Each participating source
    therefore has exactly one compact row, so backward gathers instead of using
    an atomic scatter. Invalid compact rows never overwrite a real inverse entry.
    """
    sources = source_indices.flatten() - (full_halo_rows - halo_rows)
    valid = valid_groups[:, None].expand_as(source_indices).flatten()
    sources = sources.masked_fill(~valid, -1)
    inverse = torch.full((halo_rows + local_rows,), -1, dtype=torch.int64, device=sources.device)
    rows = torch.arange(sources.numel(), device=sources.device).masked_fill(~valid, -1)
    if inverse.numel():
        inverse.scatter_reduce_(0, sources.clamp_min(0), rows, reduce="amax", include_self=True)
    return sources, inverse


def _compact_projection(local, halo, sources):
    rows = sources - halo.shape[0]
    shape = (-1,) + (1,) * (local.ndim - 1)
    if local.shape[0] == 0 and halo.shape[0] == 0:
        return local.new_zeros((sources.numel(), *local.shape[1:]))
    if local.shape[0] == 0:
        values = halo.index_select(0, sources.clamp(0, halo.shape[0] - 1))
    elif halo.shape[0] == 0:
        values = local.index_select(0, rows.clamp(0, local.shape[0] - 1))
    else:
        values = torch.where(
            (rows < 0).reshape(shape),
            halo.index_select(0, sources.clamp(0, halo.shape[0] - 1)),
            local.index_select(0, rows.clamp(0, local.shape[0] - 1)),
        )
    return values.masked_fill((sources < 0).reshape(shape), 0)


def _restore_projection_gradient(gradient, inverse):
    if gradient.shape[0] == 0:
        return gradient.new_zeros((inverse.numel(), *gradient.shape[1:]))
    shape = (-1,) + (1,) * (gradient.ndim - 1)
    return gradient.index_select(0, inverse.clamp_min(0)).masked_fill(
        (inverse < 0).reshape(shape), 0
    )


_compiled_compact = torch.compile(_compact_projection, fullgraph=True)
_compiled_restore = torch.compile(_restore_projection_gradient, fullgraph=True)


class _CompactCPProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, halo, sources, inverse):
        ctx.save_for_backward(inverse)
        ctx.halo_rows = halo.shape[0]
        implementation = _compiled_compact if local.is_cuda else _compact_projection
        return implementation(local, halo, sources)

    @staticmethod
    def backward(ctx, gradient):
        (inverse,) = ctx.saved_tensors
        implementation = _compiled_restore if gradient.is_cuda else _restore_projection_gradient
        restored = implementation(gradient, inverse)
        halo, local = restored.split((ctx.halo_rows, inverse.numel() - ctx.halo_rows))
        return local, halo, None, None


def compact_cp_projection(local: Tensor, halo: Tensor, sources: Tensor, inverse: Tensor) -> Tensor:
    """Compact projected KV or gate, saving only the shared integer inverse map."""
    return _CompactCPProjection.apply(local, halo, sources, inverse)
