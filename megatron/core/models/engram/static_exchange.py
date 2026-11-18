# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact Engram row lookup over a static-shape exchange, capturable in CUDA graphs.

Every request of hash column ``c`` addresses the column's bucket range
``[offset_c, offset_c + size_c)`` of one row-sharded table. Owner ``k`` owns the balanced
contiguous rows ``[start_k, end_k)``, so a request of column ``c`` can only reach owners whose
rows intersect that range. ``L_k`` lists those columns for owner ``k``; a block of
``|L_k| * T`` slots per (requester, owner) pair is then an exact capacity for any data, and
every exchange size depends only on the token count ``T`` and the table layout. Nothing is
planned on the host per batch.

Slots are token-major inside each owner block: ``slot(k, t, j) = T * offset_k + t * |L_k| + j``
for the ``j``-th column of ``L_k``. An unowned slot carries row ID ``-1``. The owner's stable
row sort therefore visits each row's requests in (source rank, original request index)
order, the order of the variable-split NCCL/HybridEP paths, for any data.

Backward sums each owner row's BF16 gradients sequentially in FP32 from ``+0.0`` in that
order, rounds once to the parameter dtype and, under Megatron's
``gradient_accumulation_fusion`` contract, adds the result straight into ``main_grad``. This
is bit-identical to ``_embedding_row_gradients`` followed by DDP's ``main_grad.add_`` (the
MXFP8 path and BF16 tables in deterministic mode); no table-sized gradient is materialized.

Without column ranges every owner receives every request (``L_k`` = all columns), which is
still exact but ``P`` times larger. All ranks of the shard group must look up the same number
of tokens per call.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# Sort key of unowned slots: larger than any owner-local row, so they sort last.
SENTINEL = 1 << 62


if HAVE_TRITON:

    @triton.jit
    def _owner_of(address, base, remainder, large_end, small_rows, SHARDS: tl.constexpr):
        """Owner and first row of a global row's shard (balanced partition), clamped."""
        # Ranks below `remainder` own base + 1 rows (up to `large_end`), the rest own base.
        owner = tl.where(
            address < large_end,
            address // (base + 1),
            remainder + (address - large_end) // small_rows,
        )
        owner = tl.minimum(tl.maximum(owner, 0), SHARDS - 1)
        return owner, owner * base + tl.minimum(owner, remainder)

    @triton.jit
    def _static_pack_kernel(
        addresses,
        address_stride,
        metadata,
        send_ids,
        inverse,
        violations,
        tokens,
        base,
        remainder,
        large_end,
        small_rows,
        COLUMNS: tl.constexpr,
        SLOTS: tl.constexpr,
        SHARDS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
        WRITE_INVERSE: tl.constexpr,
    ):
        """Write every static send slot (row ID or -1) and the inverse map of each request."""
        # Offsets into the packed int64 metadata (see StaticLookupLayout.metadata).
        owner_offsets = metadata + 3 * SLOTS
        owner_widths = owner_offsets + SHARDS
        column_starts = owner_widths + SHARDS
        column_ends = column_starts + COLUMNS
        column_index = column_ends + COLUMNS

        token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        token_valid = token < tokens
        token = token.to(tl.int64)

        # Requests (t, c): range validation and the slot that will hold each result.
        column = tl.arange(0, BLOCK_C)
        column_valid = column < COLUMNS
        mask = token_valid[:, None] & column_valid[None, :]
        address = tl.load(
            addresses + token[:, None] * address_stride + column[None, :], mask=mask, other=0
        )
        low = tl.load(column_starts + column, mask=column_valid, other=0)
        high = tl.load(column_ends + column, mask=column_valid, other=0)
        in_range = (address >= low[None, :]) & (address < high[None, :])
        bad = tl.sum(tl.sum((mask & ~in_range).to(tl.int32), axis=1), axis=0)
        if bad > 0:
            tl.atomic_add(violations, bad)
        if WRITE_INVERSE:
            owner, _ = _owner_of(address, base, remainder, large_end, small_rows, SHARDS)
            index = tl.load(column_index + owner * COLUMNS + column[None, :], mask=mask, other=0)
            offset = tl.load(owner_offsets + owner, mask=mask, other=0)
            width = tl.load(owner_widths + owner, mask=mask, other=0)
            slot = tl.where(in_range, offset * tokens + token[:, None] * width + index, 0)
            tl.store(inverse + token[:, None] * COLUMNS + column[None, :], slot, mask=mask)

        # Send slots (k, t, j): the owner-local row if column L_k[j] of token t lives on k.
        virtual = tl.arange(0, BLOCK_S)
        virtual_valid = virtual < SLOTS
        slot_owner = tl.load(metadata + virtual, mask=virtual_valid, other=0)
        slot_column = tl.load(metadata + SLOTS + virtual, mask=virtual_valid, other=0)
        slot_index = tl.load(metadata + 2 * SLOTS + virtual, mask=virtual_valid, other=0)
        slot_mask = token_valid[:, None] & virtual_valid[None, :]
        address = tl.load(
            addresses + token[:, None] * address_stride + slot_column[None, :],
            mask=slot_mask,
            other=0,
        )
        low = tl.load(column_starts + slot_column, mask=virtual_valid, other=0)
        high = tl.load(column_ends + slot_column, mask=virtual_valid, other=0)
        owner, start = _owner_of(address, base, remainder, large_end, small_rows, SHARDS)
        owned = (address >= low[None, :]) & (address < high[None, :])
        owned = owned & (owner == slot_owner[None, :])
        offset = tl.load(owner_offsets + slot_owner, mask=virtual_valid, other=0)
        width = tl.load(owner_widths + slot_owner, mask=virtual_valid, other=0)
        destination = offset[None, :] * tokens + token[:, None] * width[None, :]
        tl.store(
            send_ids + destination + slot_index[None, :],
            tl.where(owned, address - start, -1),
            mask=slot_mask,
        )

    @triton.jit
    def _static_gather_kernel(
        ids,
        data,
        scales,
        out,
        keys,
        count,
        sentinel,
        WIDTH: tl.constexpr,
        SCALE_WIDTH: tl.constexpr,
        DATA_STRIDE: tl.constexpr,
        SCALE_STRIDE: tl.constexpr,
        OUT_STRIDE: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_SCALE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        HAS_SCALES: tl.constexpr,
        WRITE_KEYS: tl.constexpr,
    ):
        """Copy owner rows (MXFP8 bytes + E8M0 scales, or BF16 values) into request slots."""
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        in_bounds = rows < count
        row_id = tl.load(ids + rows, mask=in_bounds, other=-1)
        valid = row_id >= 0
        if WRITE_KEYS:
            tl.store(keys + rows, tl.where(valid, row_id, sentinel), mask=in_bounds)
        # Unowned slots stay unwritten: their requester never reads them.
        source = tl.where(valid, row_id, 0)
        destination = rows.to(tl.int64) * OUT_STRIDE
        cols = tl.arange(0, BLOCK_W)
        mask = valid[:, None] & (cols[None, :] < WIDTH)
        values = tl.load(data + source[:, None] * DATA_STRIDE + cols[None, :], mask=mask)
        tl.store(out + destination[:, None] + cols[None, :], values, mask=mask)
        if HAS_SCALES:
            scale_cols = tl.arange(0, BLOCK_SCALE)
            scale_mask = valid[:, None] & (scale_cols[None, :] < SCALE_WIDTH)
            values = tl.load(
                scales + source[:, None] * SCALE_STRIDE + scale_cols[None, :], mask=scale_mask
            )
            tl.store(out + destination[:, None] + WIDTH + scale_cols[None, :], values, scale_mask)

    @triton.jit
    def _static_scatter_kernel(
        grads,
        inverse,
        out,
        count,
        WIDTH: tl.constexpr,
        GRAD_STRIDE: tl.constexpr,
        OUT_STRIDE: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Place each request's gradient row in its send slot (unique destinations)."""
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        in_bounds = rows < count
        destination = tl.load(inverse + rows, mask=in_bounds, other=0)
        cols = tl.arange(0, BLOCK_W)
        mask = in_bounds[:, None] & (cols[None, :] < WIDTH)
        values = tl.load(
            grads + rows.to(tl.int64)[:, None] * GRAD_STRIDE + cols[None, :], mask=mask
        )
        tl.store(out + destination[:, None] * OUT_STRIDE + cols[None, :], values, mask=mask)

    @triton.jit
    def _accumulate_chunk(
        total,
        row,
        sorted_keys,
        order,
        grads,
        start,
        count,
        sentinel,
        cols,
        col_mask,
        GRAD_STRIDE: tl.constexpr,
        ROWS: tl.constexpr,
    ):
        """Add the sorted positions [start, start + ROWS) that belong to ``row``, in order.

        All index loads, then all row loads, then the adds: no load is issued behind an add,
        so the chunk's rows are in flight together. Positions past the segment are loaded
        (they index received rows) but never added.
        """
        keys = ()
        sources = ()
        for step in tl.static_range(ROWS):
            position = start + step
            in_bounds = position < count
            keys = keys + (tl.load(sorted_keys + position, mask=in_bounds, other=sentinel),)
            sources = sources + (tl.load(order + position, mask=in_bounds, other=0),)
        values = ()
        for step in tl.static_range(ROWS):
            values = values + (
                tl.load(grads + sources[step] * GRAD_STRIDE + cols, mask=col_mask, other=0.0),
            )
        for step in tl.static_range(ROWS):
            total = tl.where(keys[step] == row, total + values[step].to(tl.float32), total)
        return total

    @triton.jit
    def _static_accumulate_kernel(
        target,
        sorted_keys,
        order,
        grads,
        count,
        sentinel,
        WIDTH: tl.constexpr,
        TARGET_STRIDE: tl.constexpr,
        GRAD_STRIDE: tl.constexpr,
        BLOCK_W: tl.constexpr,
        FIRST_ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        WEIGHT_DTYPE: tl.constexpr,
        ACCUMULATE: tl.constexpr,
    ):
        """One writer per sorted row segment: sequential FP32 sum, one weight-dtype rounding."""
        first = tl.program_id(0)
        row = tl.load(sorted_keys + first)
        previous = tl.load(sorted_keys + first - 1, mask=first > 0, other=-1)
        if (row != sentinel) & (row != previous):
            cols = tl.arange(0, BLOCK_W)
            col_mask = cols < WIDTH
            total = tl.zeros((BLOCK_W,), dtype=tl.float32)
            # Most segments are short; hot rows (e.g. all padding tokens) continue in wide
            # chunks. A small first chunk keeps the kernel light for the common case.
            total = _accumulate_chunk(
                total,
                row,
                sorted_keys,
                order,
                grads,
                first,
                count,
                sentinel,
                cols,
                col_mask,
                GRAD_STRIDE,
                FIRST_ROWS,
            )
            start = first + FIRST_ROWS
            more = tl.load(sorted_keys + start, mask=start < count, other=sentinel) == row
            while more:
                following = tl.load(
                    sorted_keys + start + BLOCK_ROWS,
                    mask=start + BLOCK_ROWS < count,
                    other=sentinel,
                )
                total = _accumulate_chunk(
                    total,
                    row,
                    sorted_keys,
                    order,
                    grads,
                    start,
                    count,
                    sentinel,
                    cols,
                    col_mask,
                    GRAD_STRIDE,
                    BLOCK_ROWS,
                )
                start += BLOCK_ROWS
                more = following == row
            result = total.to(WEIGHT_DTYPE).to(tl.float32)
            pointer = target + row * TARGET_STRIDE + cols
            if ACCUMULATE:
                result = tl.load(pointer, mask=col_mask).to(tl.float32) + result
            tl.store(pointer, result.to(target.dtype.element_ty), mask=col_mask)


_TRITON_FLOAT_TYPES = (torch.float32, torch.bfloat16, torch.float16)


def _use_fused(*tensors: Tensor) -> bool:
    return HAVE_TRITON and all(tensor.is_cuda for tensor in tensors)


class StaticLookupLayout:
    """Host-side plan of the static exchange for one row-sharded table.

    Args:
        rows: Global (logical) rows of the table.
        shard_ranges: Balanced contiguous ``[start, end)`` row range of every shard rank.
        rank: This rank in the shard group.
        column_ranges: ``(offset, size)`` of every hash column's bucket range, or None to
            send every request to every owner (one column covering the table).
    """

    def __init__(
        self,
        rows: int,
        shard_ranges: Sequence[tuple[int, int]],
        rank: int,
        column_ranges: Optional[Sequence[tuple[int, int]]] = None,
    ) -> None:
        self.rows = int(rows)
        self.shards = len(shard_ranges)
        self.rank = rank
        self.owner_base, self.owner_remainder = divmod(self.rows, self.shards)
        for shard, (start, end) in enumerate(shard_ranges):
            # The kernels compute owners in closed form; it must match the partition.
            expected = shard * self.owner_base + min(shard, self.owner_remainder)
            if (start, end) != (
                expected,
                expected + self.owner_base + (shard < self.owner_remainder),
            ):
                raise ValueError("Engram static exchange requires the balanced row partition")
        self.has_column_ranges = column_ranges is not None
        if column_ranges is None:
            column_ranges = ((0, self.rows),)
        self.column_ranges = tuple((int(offset), int(size)) for offset, size in column_ranges)
        for offset, size in self.column_ranges:
            if size <= 0 or offset < 0 or offset + size > self.rows:
                raise ValueError(
                    f"Engram column range [{offset}, {offset + size}) must be non-empty and lie "
                    f"inside the table's {self.rows} rows"
                )
        self.columns = len(self.column_ranges)
        self.owner_columns = tuple(
            tuple(
                column
                for column, (offset, size) in enumerate(self.column_ranges)
                if offset < end and offset + size > start
            )
            for start, end in shard_ranges
        )
        self.widths = tuple(len(columns) for columns in self.owner_columns)
        self.slots = sum(self.widths)
        self.offsets = tuple(sum(self.widths[:shard]) for shard in range(self.shards))

    def send_splits(self, tokens: int) -> list[int]:
        """Requests sent to each owner: its static block of ``|L_k| * T`` slots."""
        return [width * tokens for width in self.widths]

    def recv_splits(self, tokens: int) -> list[int]:
        """Requests received from each requester: ``|L_rank| * T`` slots."""
        return [self.widths[self.rank] * tokens] * self.shards

    def metadata(self) -> Tensor:
        """Packed int64 tables read by the pack kernel (CPU; callers move it to the device).

        ``[slot owner, slot column, slot index] (3 x slots), owner slot offset, owner width
        (shards each), column start, column end (columns each), column index (shards x
        columns, -1 where the owner does not serve the column)``.
        """
        slot_owner, slot_column, slot_index = [], [], []
        index = [[-1] * self.columns for _ in range(self.shards)]
        for shard, columns in enumerate(self.owner_columns):
            for position, column in enumerate(columns):
                slot_owner.append(shard)
                slot_column.append(column)
                slot_index.append(position)
                index[shard][column] = position
        values = [
            *slot_owner,
            *slot_column,
            *slot_index,
            *self.offsets,
            *self.widths,
            *(offset for offset, _ in self.column_ranges),
            *(offset + size for offset, size in self.column_ranges),
            *(position for row in index for position in row),
        ]
        return torch.tensor(values, dtype=torch.int64)


def static_requests(hash_ids: Tensor, layout: StaticLookupLayout) -> tuple[int, Tensor]:
    """Return the token count and the ``[T, C]`` int64 address matrix of one lookup."""
    if hash_ids.ndim == 0 or hash_ids.shape[-1] != 1:
        raise ValueError("Engram static lookups take hash IDs [..., 1] for exactly one table")
    if layout.has_column_ranges and (hash_ids.ndim < 2 or hash_ids.shape[-2] != layout.columns):
        raise ValueError(
            f"Engram static lookup expects {layout.columns} hash columns [..., columns, 1], "
            f"got shape {tuple(hash_ids.shape)}"
        )
    tokens = hash_ids.numel() // layout.columns
    addresses = hash_ids.reshape(tokens, layout.columns)
    if addresses.dtype != torch.int64:
        addresses = addresses.to(torch.int64)
    if layout.columns > 1 and addresses.stride(1) != 1:
        addresses = addresses.contiguous()
    return tokens, addresses


def static_pack(
    addresses: Tensor,
    layout: StaticLookupLayout,
    metadata: Tensor,
    *,
    write_inverse: bool,
    use_fused: bool = True,
) -> tuple[Tensor, Optional[Tensor], Tensor]:
    """Build the static send buffer of row IDs (-1 when unowned) and each request's slot.

    Returns ``send_ids`` int64 ``[T * slots]``, ``inverse`` int64 ``[T * C]`` (or None) and an
    int32 ``[1]`` count of requests outside their column range.
    """
    tokens, columns = addresses.shape
    device = addresses.device
    send_ids = torch.empty(tokens * layout.slots, dtype=torch.int64, device=device)
    inverse = (
        torch.empty(tokens * columns, dtype=torch.int64, device=device) if write_inverse else None
    )
    violations = torch.zeros(1, dtype=torch.int32, device=device)
    if not tokens:
        return send_ids, inverse, violations
    if use_fused and _use_fused(addresses, metadata):
        block_t = 16
        _static_pack_kernel[(triton.cdiv(tokens, block_t),)](
            addresses,
            addresses.stride(0),
            metadata,
            send_ids,
            inverse if inverse is not None else send_ids,
            violations,
            tokens,
            layout.owner_base,
            layout.owner_remainder,
            (layout.owner_base + 1) * layout.owner_remainder,
            max(layout.owner_base, 1),
            columns,
            layout.slots,
            layout.shards,
            block_t,
            triton.next_power_of_2(columns),
            triton.next_power_of_2(layout.slots),
            write_inverse,
            num_warps=4,
        )
        return send_ids, inverse, violations

    # Reference implementation (CPU / no Triton).
    base, remainder = layout.owner_base, layout.owner_remainder
    starts = torch.tensor([offset for offset, _ in layout.column_ranges], device=device)
    ends = torch.tensor([offset + size for offset, size in layout.column_ranges], device=device)
    in_range = (addresses >= starts) & (addresses < ends)
    violations += (~in_range).sum().to(torch.int32)
    large_end = (base + 1) * remainder
    owner = torch.where(
        addresses < large_end,
        addresses // (base + 1),
        remainder + (addresses - large_end) // max(base, 1),
    ).clamp(0, layout.shards - 1)
    local = addresses - (owner * base + owner.clamp_max(remainder))
    blocks = []
    for shard, owned_columns in enumerate(layout.owner_columns):
        if owned_columns:
            selected = list(owned_columns)
            valid = in_range[:, selected] & (owner[:, selected] == shard)
            blocks.append(torch.where(valid, local[:, selected], -1).reshape(-1))
    send_ids.copy_(torch.cat(blocks))
    if inverse is not None:
        table = layout.metadata().to(device)
        offset = 3 * layout.slots
        offsets = table[offset : offset + layout.shards]
        widths = table[offset + layout.shards : offset + 2 * layout.shards]
        index = table[offset + 2 * layout.shards + 2 * columns :].view(layout.shards, columns)
        column = torch.arange(columns, device=device)
        token = torch.arange(tokens, device=device)[:, None]
        slot = offsets[owner] * tokens + token * widths[owner] + index[owner, column]
        inverse.copy_(torch.where(in_range, slot, 0).reshape(-1))
    return send_ids, inverse, violations


def static_gather(
    ids: Tensor,
    data: Tensor,
    scales: Optional[Tensor],
    width: int,
    *,
    write_keys: bool,
    use_fused: bool = True,
) -> tuple[Tensor, Optional[Tensor]]:
    """Gather owner rows for received slots; -1 slots are left unwritten.

    With ``scales`` (MXFP8) each output row is ``width`` data bytes followed by
    ``width // 32`` E8M0 scale bytes, the layout ``_mxfp8_lookup_activation`` reads;
    otherwise rows have ``data``'s dtype. Optionally also returns the backward sort keys
    (row ID, or ``SENTINEL`` for unowned slots).
    """
    scale_width = width // 32 if scales is not None else 0
    payload = torch.empty((ids.numel(), width + scale_width), dtype=data.dtype, device=ids.device)
    keys = torch.empty_like(ids) if write_keys else None
    if not ids.numel():
        return payload, keys
    if use_fused and _use_fused(ids, data) and data.stride(1) == 1:
        if scales is not None and scales.stride(1) != 1:
            raise ValueError("Engram MXFP8 scales must be row-major")
        block_rows = 16
        _static_gather_kernel[(triton.cdiv(ids.numel(), block_rows),)](
            ids,
            data,
            scales if scales is not None else data,
            payload,
            keys if keys is not None else ids,
            ids.numel(),
            SENTINEL,
            width,
            scale_width,
            data.stride(0),
            scales.stride(0) if scales is not None else 0,
            payload.stride(0),
            triton.next_power_of_2(width),
            triton.next_power_of_2(max(scale_width, 1)),
            block_rows,
            scales is not None,
            write_keys,
            num_warps=4,
        )
        return payload, keys

    # Reference implementation: unowned slots receive row 0 (never read).
    source = ids.clamp_min(0)
    payload[:, :width].copy_(data.index_select(0, source)[:, :width])
    if scales is not None:
        payload[:, width:].copy_(scales[:, :scale_width].index_select(0, source))
    if keys is not None:
        keys.copy_(torch.where(ids >= 0, ids, SENTINEL))
    return payload, keys


def static_scatter(grads: Tensor, inverse: Tensor, slots: int, *, use_fused: bool = True) -> Tensor:
    """Return ``[slots, D]`` with each request's gradient row in its slot (others undefined)."""
    out = grads.new_empty((slots, grads.shape[1]))
    if not grads.shape[0]:
        return out
    if use_fused and _use_fused(grads, inverse) and grads.stride(1) == 1:
        block_rows = 16
        _static_scatter_kernel[(triton.cdiv(grads.shape[0], block_rows),)](
            grads,
            inverse,
            out,
            grads.shape[0],
            grads.shape[1],
            grads.stride(0),
            out.stride(0),
            triton.next_power_of_2(grads.shape[1]),
            block_rows,
            num_warps=4,
        )
        return out
    out.index_copy_(0, inverse, grads)
    return out


def static_accumulate(
    target: Tensor,
    sorted_keys: Tensor,
    order: Tensor,
    grads: Tensor,
    weight_dtype: torch.dtype,
    *,
    accumulate: bool,
    use_fused: bool = True,
) -> None:
    """Reduce each owned row's gradients in sorted order into ``target`` (in place).

    Per row: a sequential FP32 sum from ``+0.0``, rounded once to ``weight_dtype``. With
    ``accumulate`` the rounded value is added to ``target`` (``main_grad``) in FP32 and stored
    in ``target``'s dtype; otherwise it is written (a zero-initialized dense gradient).
    Rows without requests are not touched.
    """
    count = sorted_keys.numel()
    if not count:
        return
    if (
        use_fused
        and _use_fused(target, sorted_keys, grads)
        and target.dtype in _TRITON_FLOAT_TYPES
        and grads.dtype in _TRITON_FLOAT_TYPES
        and weight_dtype in _TRITON_FLOAT_TYPES
        and target.stride(1) == grads.stride(1) == 1
    ):
        _static_accumulate_kernel[(count,)](
            target,
            sorted_keys,
            order,
            grads,
            count,
            SENTINEL,
            target.shape[1],
            target.stride(0),
            grads.stride(0),
            triton.next_power_of_2(target.shape[1]),
            2,
            16,
            {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[
                weight_dtype
            ],
            accumulate,
            # Measured on GB300 (proxy owner, D=256): one warp keeps the ~100K mostly short
            # segments at high occupancy; 16-row chunks halve the time of 3K-row hot segments.
            num_warps=1,
        )
        return

    # Reference implementation, as _embedding_row_gradients (host-synchronizing).
    owned = int((sorted_keys != SENTINEL).sum())
    if not owned:
        return
    keys, positions = sorted_keys[:owned], order[:owned]
    accumulation = torch.float64 if grads.dtype == torch.float64 else torch.float32
    values = grads.index_select(0, positions).to(accumulation)
    rows, lengths = torch.unique_consecutive(keys, return_counts=True)
    sums = torch.segment_reduce(values, 'sum', lengths=lengths).to(weight_dtype)
    if accumulate:
        sums = target.index_select(0, rows).to(accumulation) + sums.to(accumulation)
    target.index_copy_(0, rows, sums.to(target.dtype))


def accumulates_into_main_grad(weight: Tensor, config) -> bool:
    """Megatron's fused wgrad-accumulation contract (see VocabParallelEmbedding).

    DDP owns ``main_grad`` and skips its own add when ``grad_added_to_main_grad`` is set; the
    autograd gradient is then a storage-free ZeroTensor.
    """
    return bool(
        getattr(config, "gradient_accumulation_fusion", True)
        and weight.requires_grad
        and getattr(weight, "main_grad", None) is not None
        and hasattr(weight, "grad_added_to_main_grad")
        and not getattr(weight, "zero_out_wgrad", False)
        and not hasattr(weight, "__fsdp_param__")
    )


def check_equal_tokens(tokens: int, group, device: torch.device) -> None:
    """Assert on the device that every shard rank looks up ``tokens`` tokens.

    Both sides of each static all-to-all size their blocks with ``|L_k| * T``. Skipped while
    a CUDA graph is being captured: all ranks capture the same iteration after eager warmup,
    and replays reuse the captured shapes.
    """
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return
    bounds = torch.full((2,), tokens, dtype=torch.int64, device=device)
    bounds[1:].neg_()
    torch.distributed.all_reduce(bounds, op=torch.distributed.ReduceOp.MAX, group=group)
    torch._assert_async(
        bounds[0] + bounds[1] == 0,
        "Engram communication_backend='static' requires the same token count on every rank "
        "of the Engram shard group",
    )


def _mxfp8_storage(weight: Tensor):
    if not is_mxfp8tensor(weight):
        raise RuntimeError("Move Engram FP8 tables to CUDA before performing lookups")
    storage = weight if hasattr(weight, "_rowwise_data") else weight.data
    if getattr(storage, "_with_gemm_swizzled_scales", False):
        raise ValueError("Engram FP8 table lookup requires unswizzled rowwise scales")
    return storage


class StaticEngramLookup(torch.autograd.Function):
    """Exact lookup of one row-sharded table with static exchange shapes.

    ``output_width`` selects the MXFP8 GEMM-input activation (as ``lookup_quantized``);
    None returns rows ``[*hash_ids.shape, D]`` in the table dtype. With one shard there is no
    exchange: the owner gather is already in request order.
    """

    @staticmethod
    def forward(ctx, module, hash_ids, output_width, needs_wgrad, needs_table_grad, weight):
        """Pack, exchange IDs, gather, return payloads; save the owner row order."""
        layout, group, width = module.static_layout, module.ep_group, module.embedding_dim
        tokens, addresses = static_requests(hash_ids, layout)
        use_fused = _use_fused(addresses)
        message = "engram.lookup.static.forward"
        nvtx_range_push(message)
        try:
            shards = layout.shards
            if shards > 1:
                check_equal_tokens(tokens, group, addresses.device)
            send_ids, inverse, violations = static_pack(
                addresses,
                layout,
                module.static_lookup_metadata,
                write_inverse=shards > 1,
                use_fused=use_fused,
            )
            torch._assert_async(
                violations[0] == 0,
                "Engram static lookup: a hash ID lies outside its column's bucket range",
            )
            if shards > 1:
                recv_ids = send_ids.new_empty(layout.widths[layout.rank] * tokens * shards)
                torch.distributed.all_to_all_single(
                    recv_ids,
                    send_ids,
                    output_split_sizes=layout.recv_splits(tokens),
                    input_split_sizes=layout.send_splits(tokens),
                    group=group,
                )
            else:
                recv_ids = send_ids
            if output_width is not None:
                storage = _mxfp8_storage(weight)
                data, scales = storage._rowwise_data, storage._rowwise_scale_inv
            else:
                data, scales = weight, None
            payload, keys = static_gather(
                recv_ids, data, scales, width, write_keys=needs_table_grad, use_fused=use_fused
            )
            if shards > 1:
                returned = payload.new_empty((tokens * layout.slots, payload.shape[1]))
                torch.distributed.all_to_all_single(
                    returned,
                    payload,
                    output_split_sizes=layout.send_splits(tokens),
                    input_split_sizes=layout.recv_splits(tokens),
                    group=group,
                )
            else:
                returned = payload
            ctx.needs_table_grad = needs_table_grad
            if needs_table_grad:
                # The row order depends only on the received IDs; sort it before backward.
                sorted_keys, order = torch.sort(keys, stable=True)
                ctx.save_for_backward(inverse, sorted_keys, order)
                ctx.weight, ctx.layout, ctx.group = weight, layout, group
                ctx.tokens, ctx.requests, ctx.width = tokens, hash_ids.numel(), width
                ctx.quantized = output_width is not None
                ctx.into_main_grad = accumulates_into_main_grad(weight, module.config)
                ctx.use_fused = use_fused
            if output_width is not None:
                # Deferred: distributed_embedding imports this module.
                from megatron.core.models.engram import distributed_embedding

                return distributed_embedding._mxfp8_lookup_activation(
                    returned, width, output_width, weight.dtype, needs_wgrad, inverse=inverse
                )
            rows = returned if inverse is None else returned.index_select(0, inverse)
            return rows.view(*hash_ids.shape, width)
        finally:
            nvtx_range_pop(message)

    @staticmethod
    def backward(ctx, grad_output):
        """Return row gradients to their owners and reduce them per row, in request order."""
        if not ctx.needs_table_grad or not ctx.needs_input_grad[5]:
            return None, None, None, None, None, None
        message = "engram.lookup.static.backward"
        nvtx_range_push(message)
        try:
            inverse, sorted_keys, order = ctx.saved_tensors
            weight, layout, tokens, width = ctx.weight, ctx.layout, ctx.tokens, ctx.width
            if ctx.quantized:
                # Discard the GEMM's token alignment padding.
                grad = grad_output.reshape(-1)[: ctx.requests * width].view(ctx.requests, width)
            else:
                grad = grad_output.reshape(ctx.requests, width)
            if grad.stride(1) != 1:
                grad = grad.contiguous()
            if layout.shards > 1:
                # Same wire dtype and per-row order as the variable-split backends.
                send = static_scatter(grad, inverse, tokens * layout.slots, use_fused=ctx.use_fused)
                grad = send.new_empty((sorted_keys.numel(), width))
                torch.distributed.all_to_all_single(
                    grad,
                    send,
                    output_split_sizes=layout.recv_splits(tokens),
                    input_split_sizes=layout.send_splits(tokens),
                    group=ctx.group,
                )
            if ctx.into_main_grad:
                static_accumulate(
                    weight.main_grad,
                    sorted_keys,
                    order,
                    grad,
                    weight.dtype,
                    accumulate=True,
                    use_fused=ctx.use_fused,
                )
                weight.grad_added_to_main_grad = True
                grad_weight = torch._efficientzerotensor(
                    weight.shape, dtype=weight.dtype, device=weight.device
                )
            else:
                grad_weight = torch.zeros(weight.shape, dtype=weight.dtype, device=weight.device)
                static_accumulate(
                    grad_weight,
                    sorted_keys,
                    order,
                    grad,
                    weight.dtype,
                    accumulate=False,
                    use_fused=ctx.use_fused,
                )
            return None, None, None, None, None, grad_weight
        finally:
            nvtx_range_pop(message)
