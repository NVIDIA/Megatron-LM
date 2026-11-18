# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Sinkhorn optimizer step: momentum, Nesterov, balancing and update in K + 2 passes.

``Sinkhorn.step`` (``emerging_optimizers.py``) uses ``fused_sinkhorn_step`` for an FP32 CUDA
matrix when ``MCORE_SINKHORN_FUSED=1`` (read once at import, off by default) and Triton is
available; every other parameter, and every parameter with the switch off, takes
Emerging-Optimizers' step, which stays the reference. Per parameter the reference computes,
with beta the momentum, L the rows that belong to the matrix (``row_shard_local_rows``;
padding rows after them are not part of it), C the columns and K = (steps - 1) // 2 (none
when that is negative: the reference loops over ``range((steps - 1) // 2)``):

    m = lerp(m, grad, 1 - beta);  G = lerp(grad, m, beta)               (Nesterov)
    U = G[:L];  r = row L2 norms of U;  S = sum(r), all-reduced over the row-shard group
    rows with r <= threshold * S / N are zeroed (N: global rows, or L)
    U /= r + eps;  K times { c = column sums of U^2, all-reduced;  U /= sqrt(c) + eps;
                             U /= (row norms of U) + eps }
    p[:L] += -lr * ((U * sqrt(C)) * scale)

Every normalization multiplies rows or columns by scalars, so U = diag(a) G diag(b). The fused
step keeps the row factors a (L floats) and column factors b (C floats) and never materializes
U, G or any other full-size temporary:

1. One pass over all rows updates the momentum in place and writes the row norms of G and
   per-program partial sums of them (rows < L).
2. A fixed-order sum gives S, which is all-reduced; the first sweep turns the norms into
   a = 0 (masked) or 1 / (r + eps) and writes per-program column partial sums of U^2.
3. After the fixed-order sum and all-reduce of the column sums, b <- b / (sqrt(c) + eps) (with
   the rebalancing below), and the next sweep does the row step, a <- a / (||U_i|| + eps),
   followed by the next column partials on the rows it just normalized.
4. The last sweep does the final row step and updates p in place.

The factors are defined up to a common scale (a / t and b * t give the same U) and, left alone,
drift apart: after a row step each live row of U has unit norm, so the column sums average R / C
(R: live rows of the whole matrix), each column step divides b by about sqrt(R / C) and the next
row step multiplies a by as much. For an Engram table (48M x 256) that is 2^8.8 per pair, which
overflows a in rows of small norm from K = 8 on. So each column step also multiplies b by m, the
power of two nearest to sqrt(mean(c)) but at least 1, and the next sweep divides a by m as it
loads it. Powers of two scale exactly: U and every rounding stay as without the rebalancing
wherever that stays in the FP32 range, while a stays near 1 / (r + eps) and b near 1 (up to the
balancing itself and a factor of at most sqrt(2) per pair from rounding m to a power of two).
m stays 1 for means below 1, which come from few live rows (R < C, where the factors drift the
other way by sqrt(C / R) per pair, at most 2^6.5) or from rows whose norm is not far above eps,
where a smaller m would push a, already near 1 / eps, toward overflow. It is also 1 for means
from 2^127 on, infinity and NaN; finite statistics stay far below that, since every row step
leaves rows of norm at most 1 (up to rounding), so that mean(c) <= R / C.

Sweeps re-read grad and the updated momentum and recompute G with the same operations as
PyTorch's lerp, so momentum and G are bit-identical to the reference (torch.lerp is
``self + w * (end - self)`` for |w| < 0.5 and ``end - (end - self) * (1 - w)`` otherwise; nvcc
contracts both into one fused multiply-add, which the kernels spell out with ``tl.fma``, with
the weight rounded to FP32 as PyTorch does). The threshold, eps, sqrt(C), the scale factor and
-lr are rounded to FP32 like PyTorch's scalar arguments (division by the row count multiplies
by its FP32 reciprocal, as PyTorch's division by a Python number does), the norms use
correctly rounded square roots and divisions, and the update is one fused multiply-add like
``p.add_(update, alpha=-lr)``. U itself is a product of factors instead of a chain of
divisions, and the sums run in another order than PyTorch's reductions, so the update agrees
with the reference to FP32 rounding (a few ulps), not bit for bit. The exception is a row whose
norm lies within a few ulps of the masking threshold: S and the row norms are rounded
differently, so one path may mask it and the other not, which changes that row's update by
O(1) and, through the column sums, the other rows' by about 1 / (2 R). Padding rows (L <= row <
rows) keep p unchanged and get the momentum update, like the reference.

Degenerate inputs follow the reference. Masked rows are +0, as the reference's masked_fill
makes them (G * 0 would be -0 where G < 0 and turn a -0.0 parameter into +0.0). A column of
tiny nonzero values whose squares underflow is rescaled by 1 / eps like in the reference. A
factor whose update overflows to infinity keeps its previous value. With the rebalancing that
takes a row or column whose squares underflow again after such a rescaling: an all-zero column,
which stays zero (1 / (0 + eps) per iteration would overflow its factor and turn the zeros into
NaN), or entries below roughly 3e-43 times their row's norm (with eps = 1e-20), which stay about
zero where the reference rescales them once more. NaN or infinity in the gradient or momentum
propagate as in the reference (the Nesterov value is NaN there and every parameter of the matrix
becomes NaN), and a row norm that overflows masks every row (zero update).

Collectives: exactly the reference's, in the same order, dtype and shape (ProcessGroupWrapper
compares shapes across ranks under TORCH_DISTRIBUTED_DEBUG=DETAIL), so a rank that runs the
reference interoperates with one that runs the fused step: one 0-dim FP32 all-reduce of S,
then K all-reduces of the [1, C] column sums, over the row-shard group, only when that group
has more than one rank. Nothing synchronizes with the host. Launch configurations depend only
on the shapes, every reduction has a fixed order and there are no atomics, so the step is
deterministic run to run. Extra memory: 2 L + at most 2^23 FP32 values (about 80 MB for a
6M x 256 shard) instead of the reference's four full-size FP32 temporaries.
"""

import contextlib
import functools
import math
import struct
import warnings
from typing import Optional

import torch

from megatron.core.env_flags import env_flag

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # Triton ships with the PyTorch container; keep CPU-only imports working.
    HAVE_TRITON = False

# Read once at import; Sinkhorn.step dispatches on it (and on HAVE_TRITON).
SINKHORN_FUSED = env_flag("MCORE_SINKHORN_FUSED", False)
if SINKHORN_FUSED and not HAVE_TRITON:
    warnings.warn("MCORE_SINKHORN_FUSED=1 but Triton is unavailable; Sinkhorn keeps its reference")

# Launch configuration; a function of the shapes only, so every sum has a fixed order. Tuned
# on GB300 for 6M x 256 shards (all passes run at 7.2-7.3 TB/s, the HBM streaming rate).
# Register tile of a pass: BLOCK_R rows x BLOCK_C (the padded column count) FP32 values, on one
# warp per _ELEMENTS_PER_WARP of them.
_TILE_ELEMENTS = 2048
_ELEMENTS_PER_WARP = 512
_MAX_WARPS = 16
# Widest padded row a pass keeps in registers; wider matrices take the reference.
_MAX_BLOCK_C = 8192
# Each program takes a contiguous chunk of at least _MIN_TILES_PER_PROGRAM tiles, so that a
# pass has at most _MAX_PROGRAMS programs (many short programs keep the last wave short) and
# the sweeps' column partials at most _MAX_PARTIALS values. Loads are software-pipelined over
# up to _LOOP_STAGES tiles, fewer where the stage buffers of the three streamed matrices would
# not fit the device's shared memory (stages change speed only, not the arithmetic).
_MAX_PROGRAMS = 32768
_MAX_PARTIALS = 2**23
_MIN_TILES_PER_PROGRAM = 4
_LOOP_STAGES = 3
# Fixed-order sum of partial rows: rows per program, and its register tile.
_SUM_ROWS = 128
_SUM_BLOCK_R = 16
_SUM_BLOCK_C = 256
# The column-factor update is one program (its rebalancing needs the mean of all column sums),
# with one warp per _FACTOR_COLS_PER_WARP columns.
_FACTOR_COLS_PER_WARP = 256
# Matrices with this many elements index with int64 offsets.
_INT32_LIMIT = 2**31


def _f32(value: float) -> float:
    """``value`` rounded to the nearest FP32 number, as PyTorch converts scalar arguments."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


if HAVE_TRITON:

    @triton.jit
    def _lerp(start, end, weight, one_minus_weight, SMALL_WEIGHT: tl.constexpr):
        """``torch.lerp(start, end, weight)`` for FP32, as PyTorch's CUDA kernel rounds it."""
        diff = end - start
        if SMALL_WEIGHT:
            return tl.fma(weight, diff, start)
        return tl.fma(-diff, one_minus_weight, end)

    @triton.jit
    def _balanced(nesterov, row_factor, col_factor):
        """U = (G * a) * b, and +0 in masked rows (a == 0) like the reference's masked_fill."""
        a = row_factor[:, None]
        return tl.where(a == 0.0, 0.0, (nesterov * a) * col_factor[None, :])

    @triton.jit
    def _keep_finite(new, old):
        """A factor whose update overflowed keeps its value (e.g. an all-zero column's)."""
        return tl.where(new == float("inf"), old, new)

    @triton.jit
    def _sinkhorn_momentum_kernel(
        grad,
        momentum,
        norms,
        norm_partials,
        num_rows,
        local_rows,
        num_cols,
        rows_per_program,
        momentum_weight,
        momentum_one_minus,
        nesterov_weight,
        nesterov_one_minus,
        MOMENTUM_SMALL: tl.constexpr,
        NESTEROV_SMALL: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        INT64: tl.constexpr,
        STAGES: tl.constexpr,
    ):
        """Pass 1: momentum update of all rows, row norms of G and their partial sum (rows < L).

        Each program owns ``rows_per_program`` consecutive rows and writes one partial sum.
        """
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        col_ok = cols < num_cols
        total = tl.zeros((BLOCK_R,), tl.float32)
        first = pid * rows_per_program
        last = tl.minimum(first + rows_per_program, num_rows)
        for start in tl.range(first, last, BLOCK_R, num_stages=STAGES):
            rows = start + tl.arange(0, BLOCK_R)
            row_ok = rows < last
            if INT64:
                offsets = rows.to(tl.int64)[:, None] * num_cols + cols[None, :]
            else:
                offsets = rows[:, None] * num_cols + cols[None, :]
            ok = row_ok[:, None] & col_ok[None, :]
            g = tl.load(grad + offsets, ok, other=0.0)
            m = tl.load(momentum + offsets, ok, other=0.0)
            m = _lerp(m, g, momentum_weight, momentum_one_minus, MOMENTUM_SMALL)
            tl.store(momentum + offsets, m, ok)
            nesterov = _lerp(g, m, nesterov_weight, nesterov_one_minus, NESTEROV_SMALL)
            norm = tl.sqrt_rn(tl.sum(nesterov * nesterov, axis=1))
            live = row_ok & (rows < local_rows)
            tl.store(norms + rows, norm, live)
            total += tl.where(live, norm, 0.0)
        tl.store(norm_partials + pid, tl.sum(total, axis=0))

    @triton.jit
    def _sinkhorn_sweep_kernel(
        grad,
        momentum,
        param,
        row_factors,
        new_row_factors,
        col_factors,
        gauge,
        norm_sum,
        col_partials,
        local_rows,
        num_cols,
        rows_per_program,
        nesterov_weight,
        nesterov_one_minus,
        eps,
        threshold,
        inv_mean_rows,
        sqrt_cols,
        scale,
        neg_lr,
        NESTEROV_SMALL: tl.constexpr,
        FIRST: tl.constexpr,
        ROW_STEP: tl.constexpr,
        COLUMN_SUMS: tl.constexpr,
        APPLY: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        INT64: tl.constexpr,
        STAGES: tl.constexpr,
    ):
        """One pass over rows < L: [masking | row step], then [column partials | update of p].

        FIRST turns the row norms in ``row_factors`` into the first row factors (0 for masked
        rows); ROW_STEP multiplies the row factors by ``gauge`` (1 / m of the last column
        step, which multiplied the column factors by m) and normalizes the rows of U with the
        current column factors. Without APPLY the new row factors go to ``new_row_factors``
        (another buffer: the warps that share a row would race on an in-place update);
        COLUMN_SUMS writes this program's partial column sums of U^2 (rows in order, one
        register tile at a time).
        """
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        col_ok = cols < num_cols
        b = tl.load(col_factors + cols, col_ok, other=0.0)
        if FIRST:
            # PyTorch: threshold * (S / N), the division as a multiply by FP32(1 / N).
            cut = threshold * (tl.load(norm_sum) * inv_mean_rows)
        if ROW_STEP:
            # A power of two: a * rebalance is exact and (G * a) * b keeps its bits.
            rebalance = tl.load(gauge)
        acc = tl.zeros((BLOCK_R, BLOCK_C), tl.float32)
        first = pid * rows_per_program
        last = tl.minimum(first + rows_per_program, local_rows)
        for start in tl.range(first, last, BLOCK_R, num_stages=STAGES):
            rows = start + tl.arange(0, BLOCK_R)
            row_ok = rows < last
            if INT64:
                offsets = rows.to(tl.int64)[:, None] * num_cols + cols[None, :]
            else:
                offsets = rows[:, None] * num_cols + cols[None, :]
            ok = row_ok[:, None] & col_ok[None, :]
            g = tl.load(grad + offsets, ok, other=0.0)
            m = tl.load(momentum + offsets, ok, other=0.0)
            nesterov = _lerp(g, m, nesterov_weight, nesterov_one_minus, NESTEROV_SMALL)
            a = tl.load(row_factors + rows, row_ok, other=0.0)
            if FIRST:
                one = tl.full((BLOCK_R,), 1.0, tl.float32)
                a = tl.where(a <= cut, 0.0, tl.div_rn(one, a + eps))
            if ROW_STEP:
                a = a * rebalance
            u = _balanced(nesterov, a, b)
            if ROW_STEP:
                norm = tl.sqrt_rn(tl.sum(u * u, axis=1))
                a = _keep_finite(tl.div_rn(a, norm + eps), a)
                u = _balanced(nesterov, a, b)
            if not APPLY:
                tl.store(new_row_factors + rows, a, row_ok)
            if COLUMN_SUMS:
                acc += u * u
            if APPLY:
                p = tl.load(param + offsets, ok, other=0.0)
                tl.store(param + offsets, tl.fma(neg_lr, (u * sqrt_cols) * scale, p), ok)
        if COLUMN_SUMS:
            tl.store(col_partials + pid * num_cols + cols, tl.sum(acc, axis=0), col_ok)

    @triton.jit
    def _sum_rows_kernel(
        src,
        dst,
        num_rows,
        num_cols,
        ROWS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """dst[i] = sum of src rows [i * ROWS, (i + 1) * ROWS), in a fixed order."""
        pid = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        col_ok = cols < num_cols
        acc = tl.zeros((BLOCK_R, BLOCK_C), tl.float32)
        for start in tl.static_range(0, ROWS, BLOCK_R):
            rows = pid * ROWS + start + tl.arange(0, BLOCK_R)
            ok = (rows < num_rows)[:, None] & col_ok[None, :]
            acc += tl.load(src + rows.to(tl.int64)[:, None] * num_cols + cols[None, :], ok, 0.0)
        tl.store(dst + pid * num_cols + cols, tl.sum(acc, axis=0), col_ok)

    @triton.jit
    def _sinkhorn_col_factor_kernel(
        col_sums, col_factors, gauge, num_cols, eps, BLOCK: tl.constexpr
    ):
        """Column step on the factors, b <- (b / (sqrt(c) + eps)) * m, in one program.

        m rebalances the factors (see the module docstring). With e the binary exponent of
        sqrt(mean(c)) * sqrt(2), 2^e is the power of two nearest to sqrt(mean(c)); m = 2^e
        where 1 <= e <= 63 and m = 1 otherwise, that is for means below about 2 (including
        zero), means from about 2^127 on, infinity and NaN. ``gauge`` receives 1 / m for the
        next sweep's row factors. Every rank of a row-shard group computes the same m from
        the same all-reduced sums.
        """
        cols = tl.arange(0, BLOCK)
        ok = cols < num_cols
        sums = tl.load(col_sums + cols, ok, other=0.0)
        b = tl.load(col_factors + cols, ok, other=1.0)
        # The binary exponent of sqrt(mean) * sqrt(2) is log2(sqrt(mean)) rounded to nearest.
        root = tl.sqrt_rn(tl.sum(sums, axis=0) / num_cols) * 1.4142135623730951
        exponent = ((root.to(tl.int32, bitcast=True) >> 23) & 255) - 127
        exponent = tl.where((exponent > 0) & (exponent < 64), exponent, 0)
        rebalance = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
        tl.store(gauge, ((127 - exponent) << 23).to(tl.float32, bitcast=True))
        new = tl.div_rn(b, tl.sqrt_rn(sums) + eps) * rebalance
        tl.store(col_factors + cols, _keep_finite(new, b), ok)


def _like(tensor: torch.Tensor) -> dict:
    """Keyword arguments of an FP32 buffer on ``tensor``'s device."""
    return {"dtype": torch.float32, "device": tensor.device}


def _tile(num_cols: int) -> tuple:
    """(BLOCK_R, BLOCK_C, warps) of the register tile of a matrix with ``num_cols`` columns."""
    block_c = max(16, triton.next_power_of_2(num_cols))
    block_r = max(1, _TILE_ELEMENTS // block_c)
    return block_r, block_c, max(1, min(_MAX_WARPS, block_r * block_c // _ELEMENTS_PER_WARP))


@functools.lru_cache(maxsize=None)
def _stages(tile_bytes: int, device_index: int) -> int:
    """Pipeline stages whose buffers for grad, momentum and param fit in shared memory."""
    shared = triton.runtime.driver.active.utils.get_device_properties(device_index)[
        "max_shared_mem"
    ]
    stages = _LOOP_STAGES
    while stages > 1 and 3 * (stages - 1) * tile_bytes > shared - 16384:
        stages -= 1
    return stages


def _rows_per_program(num_rows: int, block_r: int, max_programs: int) -> int:
    """Rows of each program's contiguous chunk (whole tiles); a function of the shape only."""
    tiles = triton.cdiv(max(num_rows, 1), block_r)
    return block_r * max(_MIN_TILES_PER_PROGRAM, triton.cdiv(tiles, max_programs))


def _fixed_order_sum(partials: torch.Tensor, out: torch.Tensor) -> None:
    """Sum the rows of [n, C] ``partials`` into ``out`` ([C]) in an order fixed by n and C."""
    rows, cols = partials.shape
    src = partials
    while True:
        programs = triton.cdiv(rows, _SUM_ROWS)
        dst = out if programs == 1 else torch.empty(programs, cols, **_like(out))
        _sum_rows_kernel[(programs, triton.cdiv(cols, _SUM_BLOCK_C))](
            src,
            dst,
            rows,
            cols,
            ROWS=_SUM_ROWS,
            BLOCK_R=_SUM_BLOCK_R,
            BLOCK_C=min(_SUM_BLOCK_C, max(16, triton.next_power_of_2(cols))),
            num_warps=4,
        )
        if programs == 1:
            return
        src, rows = dst, programs


def fused_step_supported(param: torch.Tensor, grad: torch.Tensor, momentum: torch.Tensor) -> bool:
    """Whether the fused kernels can update ``param`` (tensor properties only).

    FP32, 2D, on one CUDA device, equal shapes, contiguous, non-overlapping, with rows that
    fit a register tile. The optimizer checks its own settings (see ``Sinkhorn.step``).
    """
    if not HAVE_TRITON or momentum is None:
        return False
    tensors = (param, grad, momentum)
    if any(
        not isinstance(t, torch.Tensor)
        or not t.is_cuda
        or t.dtype != torch.float32
        or t.dim() != 2
        or t.shape != param.shape
        or t.device != param.device
        or not t.is_contiguous()
        for t in tensors
    ):
        return False
    if triton.next_power_of_2(param.shape[1]) > _MAX_BLOCK_C:
        return False
    nbytes = param.numel() * 4
    spans = sorted((t.data_ptr(), t.data_ptr() + nbytes) for t in tensors)
    return all(end <= start for (_, end), (start, _) in zip(spans, spans[1:]))


def fused_sinkhorn_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    eps: float,
    num_normalization_steps: int,
    zero_row_threshold: float,
    extra_scale_factor: float,
    local_rows: int,
    mean_rows: int,
    row_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """One Sinkhorn optimizer step of one matrix, in place (see the module docstring).

    Args:
        param: [rows, C] FP32 parameter, updated in rows < ``local_rows``.
        grad: its FP32 gradient (read only).
        momentum_buffer: FP32 momentum, updated in all rows.
        lr, momentum, eps, num_normalization_steps, zero_row_threshold, extra_scale_factor:
            the parameter group's Sinkhorn settings. Like the reference's loop, the step runs
            max(0, (num_normalization_steps - 1) // 2) column/row pairs after the first row
            step, for any integer.
        local_rows: L, the leading rows that belong to the (logical) matrix.
        mean_rows: divisor of the summed row norms for the masking threshold (the global rows
            of a row-sharded matrix, else L).
        row_group: all-reduce group of a row-sharded matrix with more than one rank, else None.
    """
    if not fused_step_supported(param, grad, momentum_buffer):
        raise ValueError("fused_sinkhorn_step needs contiguous FP32 CUDA matrices of one shape")
    if not 0 <= local_rows <= param.shape[0]:
        raise ValueError(f"local_rows {local_rows} outside [0, {param.shape[0]}]")
    device = (
        contextlib.nullcontext()
        if param.device.index == torch.cuda.current_device()
        else torch.cuda.device(param.device)
    )
    with device:
        _Step(param, grad, momentum_buffer, momentum, local_rows, row_group).run(
            lr=lr,
            eps=eps,
            # range((steps - 1) // 2) in the reference: no pair below 1 step.
            pairs=max(0, (int(num_normalization_steps) - 1) // 2),
            zero_row_threshold=zero_row_threshold,
            extra_scale_factor=extra_scale_factor,
            mean_rows=mean_rows,
        )


class _Step:
    """Launches of one fused step: pass 1, then the sweeps with their column-sum reductions."""

    def __init__(self, param, grad, momentum_buffer, momentum, local_rows, row_group):
        self.param, self.grad, self.momentum = param, grad, momentum_buffer
        self.num_rows, self.num_cols = param.shape
        self.local_rows, self.row_group = local_rows, row_group
        self.momentum_weight = _f32(1 - momentum)
        self.nesterov_weight = _f32(momentum)
        block_r, block_c, warps = _tile(self.num_cols)
        self.common = dict(
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            INT64=self.num_rows * self.num_cols >= _INT32_LIMIT,
            STAGES=_stages(block_r * block_c * 4, param.device.index),
            num_warps=warps,
        )
        self.like = _like(param)

    def momentum_pass(self, norms: torch.Tensor, norm_sum: torch.Tensor) -> None:
        """Momentum of all rows, row norms of G for rows < L, and S into ``norm_sum``."""
        if not self.num_rows:
            norm_sum.zero_()
            return
        rows_per_program = _rows_per_program(self.num_rows, self.common["BLOCK_R"], _MAX_PROGRAMS)
        programs = triton.cdiv(self.num_rows, rows_per_program)
        partials = torch.empty(programs, 1, **self.like)
        _sinkhorn_momentum_kernel[(programs,)](
            self.grad,
            self.momentum,
            norms,
            partials,
            self.num_rows,
            self.local_rows,
            self.num_cols,
            rows_per_program,
            self.momentum_weight,
            _f32(1.0 - self.momentum_weight),
            self.nesterov_weight,
            _f32(1.0 - self.nesterov_weight),
            MOMENTUM_SMALL=abs(self.momentum_weight) < 0.5,
            NESTEROV_SMALL=abs(self.nesterov_weight) < 0.5,
            **self.common,
        )
        _fixed_order_sum(partials, norm_sum)

    def run(self, lr, eps, pairs, zero_row_threshold, extra_scale_factor, mean_rows) -> None:
        """The whole step: K = ``pairs`` column/row normalization pairs after the first row step."""
        # Row norms from pass 1, then the row factors a; each sweep reads one buffer and
        # writes the other.
        row_factors = [torch.empty(self.local_rows, **self.like) for _ in range(2)]
        norm_sum = torch.empty(1, **self.like)
        self.momentum_pass(row_factors[0], norm_sum)
        # The all-reduces take the reference's shapes, 0-dim and [1, C] (see the module docstring).
        if self.row_group is not None:
            torch.distributed.all_reduce(norm_sum.view(()), group=self.row_group)

        col_factors = torch.ones(self.num_cols, **self.like)
        # 1 / m of the last column step's rebalancing, for the next sweep's row factors.
        gauge = torch.ones(1, **self.like)
        col_sums = torch.empty(self.num_cols, **self.like)
        factor_block = max(16, triton.next_power_of_2(self.num_cols))
        max_programs = max(1, min(_MAX_PROGRAMS, _MAX_PARTIALS // self.num_cols))
        rows_per_program = _rows_per_program(self.local_rows, self.common["BLOCK_R"], max_programs)
        programs = triton.cdiv(max(self.local_rows, 1), rows_per_program)
        col_partials = torch.empty(programs, self.num_cols, **self.like)
        eps = _f32(eps)
        scalars = (
            eps,
            _f32(zero_row_threshold),
            # PyTorch divides by a Python number as a multiply by its FP32 reciprocal.
            _f32(1.0 / _f32(mean_rows)) if mean_rows else float("inf"),
            _f32(math.sqrt(self.num_cols)),
            _f32(extra_scale_factor),
            _f32(-lr),
        )

        def sweep(first: bool, row_step: bool, column_sums: bool, apply: bool) -> None:
            if self.local_rows:
                _sinkhorn_sweep_kernel[(programs,)](
                    self.grad,
                    self.momentum,
                    self.param,
                    row_factors[0],
                    row_factors[1],
                    col_factors,
                    gauge,
                    norm_sum,
                    col_partials,
                    self.local_rows,
                    self.num_cols,
                    rows_per_program,
                    self.nesterov_weight,
                    _f32(1.0 - self.nesterov_weight),
                    *scalars,
                    NESTEROV_SMALL=abs(self.nesterov_weight) < 0.5,
                    FIRST=first,
                    ROW_STEP=row_step,
                    COLUMN_SUMS=column_sums,
                    APPLY=apply,
                    **self.common,
                )
                row_factors.reverse()
            if not column_sums:
                return
            if self.local_rows:
                _fixed_order_sum(col_partials, col_sums)
            else:
                col_sums.zero_()
            if self.row_group is not None:
                torch.distributed.all_reduce(col_sums.view(1, -1), group=self.row_group)
            _sinkhorn_col_factor_kernel[(1,)](
                col_sums,
                col_factors,
                gauge,
                self.num_cols,
                eps,
                BLOCK=factor_block,
                num_warps=max(1, min(_MAX_WARPS, factor_block // _FACTOR_COLS_PER_WARP)),
            )

        if pairs == 0:
            sweep(first=True, row_step=False, column_sums=False, apply=True)
            return
        sweep(first=True, row_step=False, column_sums=True, apply=False)
        for _ in range(pairs - 1):
            sweep(first=False, row_step=True, column_sums=True, apply=False)
        sweep(first=False, row_step=True, column_sums=False, apply=True)
