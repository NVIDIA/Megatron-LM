# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact histogram top-k using public CuTe DSL operations, without custom PTX."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack

THREADS = 1024
BINS = 1024
CAPACITY = 1024  # Maximum crossing-bucket candidates, independent of the output top-k.


@cute.jit
def cta_prefix(value: cutlass.Int32, scratch: cute.Tensor, thread: cutlass.Int32):
    """Inclusive scan of one count per thread, via warp scans and warp totals."""
    lane = thread % 32
    warp = thread // 32
    total = value
    for stage in cutlass.range_constexpr(5):
        step = 1 << stage
        previous = cute.arch.shuffle_sync_up(total, step, mask_and_clamp=0)
        if lane >= step:
            total += previous
    if lane == 31:
        scratch[warp] = total
    cute.arch.barrier()

    if warp == 0:
        warp_total = scratch[lane]
        initial = warp_total
        for stage in cutlass.range_constexpr(5):
            step = 1 << stage
            previous = cute.arch.shuffle_sync_up(warp_total, step, mask_and_clamp=0)
            if lane >= step:
                warp_total += previous
        scratch[lane] = warp_total - initial
    cute.arch.barrier()
    result = total + scratch[warp]
    cute.arch.barrier()
    return result


class HistogramTopK:
    """Select exact top-k using a histogram and deterministic crossing-bucket ranks."""

    def __init__(self, width: int, topk: int):
        if width % 4 or not 0 < width <= 65536:
            raise ValueError("Row stride must be divisible by four and at most 65536")
        self.width = width
        self.topk = topk
        self.packs = (width + THREADS * 4 - 1) // (THREADS * 4)
        self.cached = self.packs <= 8

    @cute.kernel
    def kernel(
        self, scores: cute.Tensor, lengths: cute.Tensor, output: cute.Tensor, flags: cute.Tensor
    ):
        """Select one bounded row, marking rows that need an exact radix fallback."""
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        n = lengths[row]
        allocator = utils.SmemAllocator()
        histogram = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((BINS,)), byte_alignment=16
        )
        warp_scan = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,)), byte_alignment=16
        )
        range_min = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,)), byte_alignment=16
        )
        range_max = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,)), byte_alignment=16
        )
        control = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((3,)), byte_alignment=16
        )
        candidate_values = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((CAPACITY,)), byte_alignment=16
        )
        candidate_indices = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((CAPACITY,)), byte_alignment=16
        )

        for slot in cutlass.range_constexpr((self.topk + THREADS - 1) // THREADS):
            index = tid + slot * THREADS
            if index < self.topk:
                output[row, index] = -1
        if n <= self.topk:
            for slot in cutlass.range_constexpr((self.topk + THREADS - 1) // THREADS):
                index = tid + slot * THREADS
                if index < self.topk and index < n:
                    output[row, index] = index
            if tid == 0:
                flags[row] = 0
        else:
            values = cute.make_rmem_tensor((self.packs, 4), cutlass.Float32)
            low = cutlass.Float32.inf
            high = -cutlass.Float32.inf
            for pack in cutlass.range_constexpr(self.packs):
                for sub in cutlass.range_constexpr(4):
                    index = pack * THREADS * 4 + tid * 4 + sub
                    value = cutlass.Float32(0)
                    if index < n:
                        value = scores[row, index]
                        low = cute.arch.fmin(low, value)
                        high = cute.arch.fmax(high, value)
                    if cutlass.const_expr(self.cached):
                        values[pack, sub] = value
            for stage in cutlass.range_constexpr(5):
                step = 1 << stage
                low = cute.arch.fmin(low, cute.arch.shuffle_sync_bfly(low, step))
                high = cute.arch.fmax(high, cute.arch.shuffle_sync_bfly(high, step))
            if lane == 0:
                range_min[warp] = low
                range_max[warp] = high
            cute.arch.barrier()
            if warp == 0:
                low = range_min[lane]
                high = range_max[lane]
                for stage in cutlass.range_constexpr(5):
                    step = 1 << stage
                    low = cute.arch.fmin(low, cute.arch.shuffle_sync_bfly(low, step))
                    high = cute.arch.fmax(high, cute.arch.shuffle_sync_bfly(high, step))
                if lane == 0:
                    range_min[0] = low
                    range_max[0] = high
            cute.arch.barrier()
            low = range_min[0]
            high = range_max[0]
            span = high - low
            scale = cutlass.Float32(BINS) / span
            finite = (
                (low > -cutlass.Float32.inf)
                and (high < cutlass.Float32.inf)
                and (span < cutlass.Float32.inf)
            )
            if finite and span == 0:
                for slot in cutlass.range_constexpr((self.topk + THREADS - 1) // THREADS):
                    index = tid + slot * THREADS
                    if index < self.topk:
                        output[row, index] = index
                if tid == 0:
                    flags[row] = 0
            elif finite and span > 0 and scale < cutlass.Float32.inf:
                histogram[tid] = 0
                cute.arch.barrier()
                for pack in cutlass.range_constexpr(self.packs):
                    for sub in cutlass.range_constexpr(4):
                        index = pack * THREADS * 4 + tid * 4 + sub
                        if index < n:
                            value = cutlass.Float32(0)
                            if cutlass.const_expr(self.cached):
                                value = values[pack, sub]
                            else:
                                value = scores[row, index]
                            digit = cutlass.Int32((value - low) * scale)
                            if digit < 0:
                                digit = cutlass.Int32(0)
                            if digit >= BINS:
                                digit = cutlass.Int32(BINS - 1)
                            cute.arch.atomic_add(
                                histogram.iterator + digit,
                                cutlass.Int32(1),
                                sem="relaxed",
                                scope="cta",
                            )
                cute.arch.barrier()
                digit = BINS - 1 - tid
                count = histogram[digit]
                prefix = cta_prefix(count, warp_scan, tid)
                if prefix >= self.topk and prefix - count < self.topk:
                    control[0] = digit
                    control[1] = prefix - count
                    control[2] = count
                cute.arch.barrier()
                crossing = control[0]
                above = control[1]
                candidates = control[2]
                if candidates <= CAPACITY:
                    greater_mask = cutlass.Uint64(0)
                    crossing_mask = cutlass.Uint64(0)
                    for pack in cutlass.range_constexpr(self.packs):
                        for sub in cutlass.range_constexpr(4):
                            index = pack * THREADS * 4 + tid * 4 + sub
                            if index < n:
                                value = cutlass.Float32(0)
                                if cutlass.const_expr(self.cached):
                                    value = values[pack, sub]
                                else:
                                    value = scores[row, index]
                                digit = cutlass.Int32((value - low) * scale)
                                if digit < 0:
                                    digit = cutlass.Int32(0)
                                if digit >= BINS:
                                    digit = cutlass.Int32(BINS - 1)
                                if digit > crossing:
                                    greater_mask |= cutlass.Uint64(1) << (pack * 4 + sub)
                                if digit == crossing:
                                    crossing_mask |= cutlass.Uint64(1) << (pack * 4 + sub)
                    greater_count = cutlass.Int32(cute.arch.popc(greater_mask))
                    crossing_count = cutlass.Int32(cute.arch.popc(crossing_mask))
                    greater_offset = cta_prefix(greater_count, warp_scan, tid) - greater_count
                    crossing_offset = cta_prefix(crossing_count, warp_scan, tid) - crossing_count
                    for pack in cutlass.range_constexpr(self.packs):
                        for sub in cutlass.range_constexpr(4):
                            bit = cutlass.Uint64(1) << (pack * 4 + sub)
                            index = pack * THREADS * 4 + tid * 4 + sub
                            if (greater_mask & bit) != 0:
                                output[row, greater_offset] = index
                                greater_offset += 1
                            if (crossing_mask & bit) != 0:
                                candidate_values[crossing_offset] = scores[row, index]
                                candidate_indices[crossing_offset] = index
                                crossing_offset += 1
                    cute.arch.barrier()
                    if tid < candidates:
                        value = candidate_values[tid]
                        index = candidate_indices[tid]
                        rank = cutlass.Int32(0)
                        for other in range(candidates):
                            other_value = candidate_values[other]
                            other_index = candidate_indices[other]
                            if other_value > value or (
                                other_value == value and other_index < index
                            ):
                                rank += 1
                        if rank < self.topk - above:
                            output[row, above + rank] = index
                    if tid == 0:
                        flags[row] = 0
                else:
                    if tid == 0:
                        flags[row] = 1
            else:
                if tid == 0:
                    flags[row] = 1

    @cute.jit
    def __call__(self, scores, lengths, output, flags, stream: cuda.CUstream):
        """Launch one CTA per row on the supplied CUDA stream."""
        self.kernel(scores, lengths, output, flags).launch(
            grid=(scores.shape[0], 1, 1), block=(THREADS, 1, 1), stream=stream
        )


def _compile_kernel(scores, lengths, output, flags):
    """Compile the shape-specialized CuTe launch before CUDA Graph capture."""
    topk = HistogramTopK(scores.shape[1], output.shape[1])
    return cute.compile(
        topk, *(from_dlpack(t) for t in (scores, lengths, output, flags)), cuda.CUstream(0)
    )
