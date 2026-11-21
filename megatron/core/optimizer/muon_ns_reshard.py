# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resharded distributed Newton-Schulz for wide GTP-sharded matrices.

GTP shards dim 0 (out_features), so a ``tp_mode="distributed"`` chain on a GTP-only weight
runs on the row shard ``[out/G, in]`` and its Gram is over dim 1 (in_features). For a wide
weight (``out < in``) that is the LONG dim: a ``10240 x 40960`` MLP fc2 all-reduces a
``40960^2`` Gram per step, which is why ``auto`` runs it duplicated.

With ``distributed_reshard_wide`` such a matrix is re-sharded once per Newton-Schulz from
dim-0 shards ``[out/G, in]`` to dim-1 shards ``[out, in/G]`` with one ``all_to_all_single``
of the fp32 shard over the GTP group. The chain then runs with ``partition_dim=1``, so the
Gram is over dim 0, the SHORT dim (``10240^2``), and a second all_to_all restores the dim-0
shards. The step is unchanged; only the layout of ``X`` changes. Same math as the
whole-matrix Newton-Schulz of the duplicated path; the rounding differs (the Gram is summed
per column block, then all-reduced).

- GTP dim-0 padding (``pad_length`` trailing rows): after the first all_to_all every rank
  holds ALL rows of its column block, so the pad rows are a trailing slice. They are stripped
  before the Newton-Schulz and restored as zeros before the reverse all_to_all, as on the
  duplicated path.
- ``in`` not divisible by the group: the shard is zero-padded on dim 1 to a multiple of ``G``.
  Zero columns change neither the normalization nor the Gram and stay zero through every
  step, so the padding is exact; it is cut off after the reverse exchange.
- Muon's scale factor comes from the logical ``(out, in)`` shape (the caller passes it).

Only a GTP-only weight that is wide in its logical (pad-stripped) shape is resharded; the
TP + GTP two-step path keeps its layout.
"""

from typing import Optional

import torch

from megatron.core.utils import get_pg_size


def local_cols(cols: int, group_size: int) -> int:
    """Width of one rank's dim-1 shard: ``ceil(cols / group_size)`` (dim 1 is zero-padded to
    a multiple of the group)."""
    return -(-cols // group_size)


def reshard_exchange_bytes(shard_rows: int, cols: int, group_size: int, elem_size: int = 4) -> int:
    """Bytes one rank sends (and receives) in ONE of the two all_to_all exchanges: its
    ``[shard_rows, cols]`` shard (dim 1 zero-padded to a multiple of the group) minus the slab
    it keeps. ``elem_size`` 4: the exchanged momentum is fp32."""
    padded = local_cols(cols, group_size) * group_size
    return shard_rows * padded * elem_size * (group_size - 1) // group_size


def rows_to_cols(shard: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
    """All-to-all a dim-0 shard into a dim-1 shard.

    ``shard`` is rank ``r``'s ``[R, K]`` row block (rows ``r R .. (r + 1) R`` of the full
    ``[G R, K]`` matrix). Returns ``[G R, Kc]`` with ``Kc = ceil(K / G)``: rank ``r``'s column
    block (columns ``r Kc .. (r + 1) Kc`` of the full matrix zero-padded to ``G Kc`` columns),
    complete over all rows.
    """
    group_size = get_pg_size(group)
    rows, cols = shard.shape
    kc = local_cols(cols, group_size)
    if kc * group_size != cols:
        shard = torch.nn.functional.pad(shard, (0, kc * group_size - cols))
    # [G, R, Kc]: slab s (this rank's rows of column block s) goes to rank s.
    send = shard.reshape(rows, group_size, kc).transpose(0, 1).contiguous()
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group)
    # recv[t] = rank t's rows of this rank's column block: stacking them in rank order gives
    # the full row range.
    return recv.view(group_size * rows, kc)


def cols_to_rows(
    block: torch.Tensor, group: torch.distributed.ProcessGroup, cols: Optional[int] = None
) -> torch.Tensor:
    """Inverse of :func:`rows_to_cols`: ``[G R, Kc]`` column block -> ``[R, cols]`` row shard
    (``cols`` defaults to ``G Kc``; a smaller value cuts the dim-1 zero padding)."""
    group_size = get_pg_size(group)
    full_rows, kc = block.shape
    rows = full_rows // group_size
    # Slab t (rows t R .. (t + 1) R of this rank's column block) goes to rank t.
    send = block.contiguous().view(group_size, rows, kc)
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group)
    # recv[s] = this rank's rows of column block s: put the blocks side by side.
    out = recv.transpose(0, 1).reshape(rows, group_size * kc)
    if cols is not None and cols != group_size * kc:
        out = out[:, :cols].contiguous()
    return out
