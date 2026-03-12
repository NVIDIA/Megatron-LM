import math
import operator
from dataclasses import dataclass
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T, dsl_user_op


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    """Create accumulator tensor with M-N view layout."""
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@dsl_user_op
def fmax(
    a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None
) -> Float32:
    """Compute maximum of two float values using NVVM intrinsic."""
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def fmax_reduce(x: cute.TensorSSA, init_val: float | Float32 | None = None) -> Float32:
    """Reduce tensor to maximum value."""
    res = cute.make_fragment(x.shape, Float32)
    res.store(x)
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = fmax(local_max[0], res[i + 0])
        local_max[1] = fmax(local_max[1], res[i + 1])
        local_max[2] = fmax(local_max[2], res[i + 2])
        local_max[3] = fmax(local_max[3], res[i + 3])
    local_max[0] = fmax(local_max[0], local_max[1])
    local_max[2] = fmax(local_max[2], local_max[3])
    local_max[0] = fmax(local_max[0], local_max[2])
    return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)


@cute.jit
def fadd_reduce(x: cute.TensorSSA, init_val: float | Float32 | None = None) -> Float32:
    """Reduce tensor by summing all elements."""
    if const_expr(init_val is None):
        init_val = Float32.zero
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    """Perform warp-level reduction using shuffle operations."""
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@dataclass
class OnlineCrossEntropy:
    """Online algorithm for softmax + cross entropy over vocab chunks (N tiles).

    Maintains running max/sum for softmax and writes label logits during vocab iteration.
    """

    num_rows: cutlass.Constexpr[int]
    num_cols: cutlass.Constexpr[int]
    row_max: cute.Tensor  # [num_rows] in RMEM - Softmax max
    row_sum: cute.Tensor  # [num_rows] in RMEM - Softmax sum

    @staticmethod
    def create(num_rows: cutlass.Constexpr[int], num_cols: cutlass.Constexpr[int]):
        """Create instance. Call reset() before use."""
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return OnlineCrossEntropy(num_rows, num_cols, row_max, row_sum)

    def reset(self) -> None:
        """Initialize max to -inf and sum to 0."""
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(Float32.zero)

    @cute.jit
    def update(
        self,
        acc_mn_tile: cute.Tensor,
        labels_per_thread: cute.Tensor,
        tCcAcc_mn_logprobs: cute.Tensor,
        gLogprobs: cute.Tensor,
        row_block_start: cutlass.Int32,
        num_tokens: cutlass.Int32,
        vocab_tile_start: cutlass.Int32,
        vocab_tile_size: cutlass.Int32,
        split_vocab_start: cutlass.Int32,
        vocab_offset: cutlass.Int32,
        ignore_index: cutlass.Int64,
        is_first: bool = False,
    ) -> None:
        """Update max/sum for softmax and write label logits to gLogprobs."""
        # Convert local CTA vocab range to global for label comparison
        global_split_start = vocab_offset + split_vocab_start
        global_split_end = vocab_offset + vocab_tile_start + vocab_tile_size

        for r in cutlass.range(self.num_rows, unroll_full=True):
            acc_row = acc_mn_tile[r, None].load()

            label = labels_per_thread[r]
            coord_m_local = tCcAcc_mn_logprobs[r, 0][0]
            coord_m_global: cutlass.Int32 = row_block_start + coord_m_local
            m_is_valid: cutlass.Boolean = coord_m_global < num_tokens

            acc_row_masked = cute.make_rmem_tensor(acc_row.shape, Float32)
            for v in cutlass.range(self.num_cols, unroll_full=True):
                coord_n_local = tCcAcc_mn_logprobs[r, v][1]

                # Mask OOB elements for softmax computation
                oob = coord_n_local >= vocab_tile_size
                acc_row_masked[v] = -Float32.inf if oob else acc_row[v]

                # Store label logit if this position matches the label
                # Use global position (vocab_offset + local) for comparison with global label
                if m_is_valid:
                    global_position = vocab_offset + vocab_tile_start + coord_n_local
                    if (
                        (label >= global_split_start)
                        and (label < global_split_end)
                        and (global_position == label)
                        and (label != ignore_index)
                    ):
                        gLogprobs[coord_m_local] = acc_row[v]

            # ==================== Softmax Online Update ====================
            acc_row_masked_ssa = acc_row_masked.load()
            row_max_cur = fmax_reduce(acc_row_masked_ssa)
            row_max_cur = warp_reduce(row_max_cur, cute.arch.fmax, width=4)

            if is_first:
                self.row_max[r] = row_max_cur
                exp_row = cute.exp(acc_row_masked_ssa - row_max_cur)
                row_sum_cur = fadd_reduce(exp_row)
                row_sum_cur = warp_reduce(row_sum_cur, operator.add, width=4)
                self.row_sum[r] = row_sum_cur
            else:
                max_old = self.row_max[r]
                max_new = cute.arch.fmax(max_old, row_max_cur)
                self.row_max[r] = max_new
                coeff = cute.exp(max_old - max_new)
                exp_row = cute.exp(acc_row_masked_ssa - max_new)
                row_sum_new = fadd_reduce(exp_row)
                row_sum_new = warp_reduce(row_sum_new, operator.add, width=4)
                self.row_sum[r] = coeff * self.row_sum[r] + row_sum_new
