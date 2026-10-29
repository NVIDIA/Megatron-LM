# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact CuTe DSL top-k with Triton radix fallback and long-context chunk merging."""

import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
import triton
import triton.language as tl
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


@triton.jit
def radix_topk(Scores, Lengths, Indices, WIDTH: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    """Select exact ordered float keys, resolving cutoff ties by position."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    length = tl.minimum(tl.load(Lengths + row), WIDTH)
    valid = offsets < length
    values = tl.load(Scores + row * WIDTH + offsets, mask=valid, other=float("-inf"))
    bits = values.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    active = valid
    remaining = tl.minimum(K, length)
    prefix = tl.full((), 0, tl.uint32)
    bins = tl.arange(0, 256)
    for shift in tl.static_range(24, -1, -8):
        digits = ((ordered >> shift) & 255).to(tl.int32)
        counts = tl.histogram(tl.where(active, digits, 256), 256)
        cumulative = tl.cumsum(counts, 0, reverse=True)
        crossing = tl.max(tl.where(cumulative >= remaining, bins, -1), 0)
        above = tl.sum(tl.where(bins > crossing, counts, 0), 0)
        remaining -= above
        active &= digits == crossing
        prefix |= crossing.to(tl.uint32) << shift
    greater = valid & (ordered > prefix)
    equal = valid & (ordered == prefix)
    count = tl.sum(greater.to(tl.int32), 0)
    equal_rank = tl.cumsum(equal.to(tl.int32), 0)
    selected = greater | (equal & (equal_rank <= tl.minimum(K, length) - count))
    rank = tl.cumsum(selected.to(tl.int32), 0) - 1
    slots = tl.arange(0, K)
    tl.store(Indices + row * K + slots, -1)
    tl.debug_barrier()
    tl.store(Indices + row * K + rank, offsets, mask=selected & (rank < K))


@triton.jit
def fallback_if_needed(
    Scores, Lengths, Indices, Flags, WIDTH: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    """Repair only rows rejected by the CuTe histogram fast path."""
    if tl.load(Flags + tl.program_id(0)):
        radix_topk(Scores, Lengths, Indices, WIDTH, K, BLOCK)


class CuteTopK:
    """Compile a bounded CuTe selection launch with preallocated exact fallback storage."""

    def __init__(
        self, scores: torch.Tensor, lengths: torch.Tensor, output: torch.Tensor, flags: torch.Tensor
    ) -> None:
        self.scores, self.lengths, self.output, self.flags = (scores, lengths, output, flags)
        self.arguments = tuple(from_dlpack(t) for t in (scores, lengths, output, flags))
        self.compiled = _compile_kernel(scores, lengths, output, flags)

    def __call__(self) -> None:
        """Run the prepared CuTe selection and conditional exact fallback."""
        self.compiled(*self.arguments, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        fallback_if_needed[(self.scores.shape[0],)](
            self.scores,
            self.lengths,
            self.output,
            self.flags,
            WIDTH=self.scores.shape[1],
            K=self.output.shape[1],
            BLOCK=triton.next_power_of_2(self.scores.shape[1]),
            num_warps=16,
            enable_fp_fusion=False,
        )


@triton.jit
def chunk_lengths(Lengths, Local, Candidate, CHUNKS: tl.constexpr, K: tl.constexpr):
    """Compute each bounded row length and its exact candidate-union size."""
    row = tl.program_id(0)
    n = tl.load(Lengths + row)
    total = tl.full((), 0, tl.int32)
    for chunk in tl.static_range(CHUNKS):
        length = tl.minimum(tl.maximum(n - chunk * 65536, 0), 65536)
        tl.store(Local + row * CHUNKS + chunk, length)
        total += tl.minimum(length, K)
    tl.store(Candidate + row, total)


@triton.jit
def gather_candidates(
    Scores, Local, Candidates, WIDTH: tl.constexpr, CHUNKS: tl.constexpr, K: tl.constexpr
):
    """Gather original scores for the union of disjoint chunk-local top-k supports."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    slots = tl.arange(0, K)
    index = tl.load(Local + (row * CHUNKS + chunk) * K + slots)
    value = tl.load(Scores + row * WIDTH + chunk * 65536 + index, index >= 0, float("-inf"))
    tl.store(Candidates + (row * CHUNKS + chunk) * K + slots, value)


@triton.jit
def remap(Local, Selected, Out, CHUNKS: tl.constexpr, K: tl.constexpr):
    """Map selected candidate slots back to logical key positions."""
    row = tl.program_id(0)
    slots = tl.arange(0, K)
    selected = tl.load(Selected + row * K + slots)
    local = tl.load(Local + row * CHUNKS * K + selected, selected >= 0, -1)
    tl.store(
        Out + row * K + slots,
        tl.where((selected >= 0) & (local >= 0), selected // K * 65536 + local, -1),
    )


class CuteTopKWorkspace:
    """Use disjoint bounded row chunks, then exact top-k of their union.

    The score stride must include complete 65536-element chunks for long contexts.
    Every buffer and CuTe launch is prepared before graph capture.
    """

    def __init__(self, scores: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor) -> None:
        self.batch, self.width = scores.shape
        self.k = indices.shape[1]
        self.chunks = math.ceil(self.width / 65536)
        self.indices = indices
        self.lengths = lengths
        self.scores = scores
        if self.chunks == 1:
            self.flags = torch.empty(self.batch, device=scores.device, dtype=torch.int32)
            self.select = CuteTopK(scores, lengths, indices, self.flags)
        else:
            if self.width % 65536:
                raise ValueError(
                    "Long-context score stride must be padded to complete 65536-element chunks"
                )
            self.local_lengths = torch.empty(
                self.batch * self.chunks, device=scores.device, dtype=torch.int32
            )
            self.candidate_lengths = torch.empty(
                self.batch, device=scores.device, dtype=torch.int32
            )
            self.local_indices = torch.empty(
                self.batch * self.chunks, self.k, device=scores.device, dtype=torch.int32
            )
            self.candidates = torch.empty(
                self.batch, self.chunks * self.k, device=scores.device, dtype=torch.float32
            )
            self.selected = torch.empty_like(indices)
            self.flags = torch.empty_like(self.local_lengths)
            self.merge_flags = torch.empty_like(lengths)
            self.select = CuteTopK(
                scores.view(self.batch * self.chunks, 65536),
                self.local_lengths,
                self.local_indices,
                self.flags,
            )
            self.merge = CuteTopK(
                self.candidates, self.candidate_lengths, self.selected, self.merge_flags
            )

    def __call__(self) -> None:
        """Select into stable output storage, merging chunk supports when needed."""
        if self.chunks == 1:
            self.select()
        else:
            chunk_lengths[(self.batch,)](
                self.lengths,
                self.local_lengths,
                self.candidate_lengths,
                CHUNKS=self.chunks,
                K=self.k,
            )
            self.select()
            gather_candidates[(self.batch, self.chunks)](
                self.scores,
                self.local_indices,
                self.candidates,
                WIDTH=self.width,
                CHUNKS=self.chunks,
                K=self.k,
            )
            self.merge()
            remap[(self.batch,)](
                self.local_indices, self.selected, self.indices, CHUNKS=self.chunks, K=self.k
            )
