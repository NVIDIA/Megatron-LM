# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CuTeDSL copy kernels for MoE paged stash.

Drop-in alternatives to ``paged_stash_copy_kernel`` (Triton) with the same contract: allocate
``ceil(num_tokens / page_size)`` pages from the CUDA free list, fall back to the pinned host
free list, otherwise raise ``overflow``; scatter token rows into the allocated pages; record
the page IDs in ``page_record`` and the advanced head in ``new_free_list_head``.

Two implementations, both selected by :func:`run` from the tensor geometry:

* :func:`run_direct` -- one CTA per token row; threads split the row into 16B vectors. Handles
  any 16B-aligned row width.
* :func:`run_tma` -- flat byte stream moved with ``cp.async.bulk``, pipelined through
  mbarriers. Uses 32 threads per CTA, so it leaves more SM capacity for the compute this
  overlaps with, but requires the CTA tiling to divide evenly into pages.

Opt in from ``PagedTensor.offload_to_stash`` with ``MEGATRON_PAGED_STASH_CUTE=1``; set
``MEGATRON_PAGED_STASH_CUTE_TMA=0`` to force the direct kernel.
"""

import os

import cuda.bindings.driver as _drv
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import SmemAllocator

_VBYTES = 16  # bytes moved per vector access

# (threads_per_cta, rows_per_cta) selected per token count.
_DEFAULT_CFG = (64, 1)
_CFG = {64: (128, 1), 256: (128, 1), 1024: (128, 1), 4096: (128, 1)}

_CACHE = {}


def _pick_threads(nvec, preferred):
    """CTA width that divides ``nvec`` exactly, never wider than ``preferred``.

    ``_CFG`` is tuned on token count alone, but the per-thread vector count is
    ``nvec // threads``.  A width that does not divide ``nvec`` silently drops the
    tail of every row (e.g. nvec=448 with 128 threads copies only 384 vectors), and
    a width wider than ``nvec`` copies nothing at all.  Narrow to the widest exact
    divisor instead so the row is always fully covered.
    """
    for threads in (preferred, 128, 64, 32):
        if threads <= preferred and threads <= nvec and nvec % threads == 0:
            return threads
    return max(nvec, 1)


@cute.kernel
def _stash_kernel(
    mSrc: cute.Tensor,  # (T, NVEC, 16) u8
    mCuda: cute.Tensor,  # (CAP, NVEC, 16) u8
    mHost: cute.Tensor,  # (CAP, NVEC, 16) u8
    mN: cute.Tensor,  # (1,)  i64
    mFLC: cute.Tensor,  # (P,)  i64
    mFLH: cute.Tensor,  # (P,)  i64
    mHead: cute.Tensor,  # (2,)  i64
    mTail: cute.Tensor,  # (2,)  i64
    mCap: cute.Tensor,  # (2,)  i64
    mOvfIn: cute.Tensor,  # (1,)  i64
    oPR: cute.Tensor,  # (P,)  i64
    oOvf: cute.Tensor,  # (1,)  i64
    oHS: cute.Tensor,  # (1,)  i64
    oSP: cute.Tensor,  # (1,)  i64
    oNH: cute.Tensor,  # (2,)  i64
    oNHA: cute.Tensor,  # (2,)  i64
    page_size: cutlass.Constexpr,
    pages: cutlass.Constexpr,
    record_pages: cutlass.Constexpr,
    nvec: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    r0 = cutlass.Int32(bidx) * rows
    vpr = nvec // threads  # vectors per thread per row
    vpt = vpr * rows  # vectors per thread total

    # --- stage every cold global read up front so they overlap in one latency ---
    # 1) the token rows themselves (grid covers T exactly, so r0+k < T always)
    frs = []
    for k in cutlass.range_constexpr(rows):
        for j in cutlass.range_constexpr(vpr):
            g = mSrc[(r0 + k, j * threads + tidx, None)]
            f = cute.make_fragment_like(g)
            cute.autovec_copy(g, f)
            frs.append(f)

    # 2) allocator metadata in SMEM.  Do not stage entire free lists here: in
    # production they may contain tens of thousands of pages, which exceeds the
    # per-CTA shared-memory limit.  The selected page ID is read directly below.
    smem = utils.SmemAllocator()
    sBuf = smem.allocate_tensor(cutlass.Int64, cute.make_layout((8,)), byte_alignment=16)

    # 3) allocator metadata -- issued by ONE lane only.  Every warp reading these
    #    eight scalars itself cost 8 extra L1/L2 broadcast requests per warp,
    #    which dominated the LSU request count for the whole kernel.
    mbase = 0
    if tidx == 0:
        sBuf[mbase + 0] = mN[0]
        sBuf[mbase + 1] = mOvfIn[0]
        sBuf[mbase + 2] = mHead[0]
        sBuf[mbase + 3] = mHead[1]
        sBuf[mbase + 4] = mTail[0]
        sBuf[mbase + 5] = mTail[1]
        sBuf[mbase + 6] = mCap[0]
        sBuf[mbase + 7] = mCap[1]

    cute.arch.barrier()

    n = sBuf[mbase + 0]
    ov0 = sBuf[mbase + 1]
    h0 = sBuf[mbase + 2]
    h1 = sBuf[mbase + 3]
    t0 = sBuf[mbase + 4]
    t1 = sBuf[mbase + 5]
    c0 = sBuf[mbase + 6]
    c1 = sBuf[mbase + 7]

    required = (n + (page_size - 1)) // page_size

    one = cutlass.Int32(1)
    zero = cutlass.Int32(0)
    c_ok = one if (t0 - h0) >= required else zero
    h_ok = one if (t1 - h1) >= required else zero
    no_ov = one if ov0 == 0 else zero
    fit = c_ok + h_ok - c_ok * h_ok
    active = no_ov * fit
    do_cuda = active * c_ok
    do_host = active - do_cuda
    ovf_set = no_ov * (one - fit)

    # Index each free list with its OWN head/capacity.  A shared (head, capacity)
    # pair taken from the selected buffer overruns the other list whenever the two
    # differ in length -- the CUDA and host stashes are sized independently -- and
    # a zero capacity (no host buffer configured) turns the modulo into a
    # division by zero that yields an arbitrary offset.
    c0s = c0 if c0 > 0 else cutlass.Int64(1)
    c1s = c1 if c1 > 0 else cutlass.Int64(1)

    slot = r0 // page_size
    sub = r0 - slot * page_size
    fidx_c = cutlass.Int32((h0 + cutlass.Int64(slot)) % c0s)
    fidx_h = cutlass.Int32((h1 + cutlass.Int64(slot)) % c1s)
    dpage = mFLC[fidx_c] if do_cuda == one else mFLH[fidx_h]
    # Keep the destination row 64-bit.  A production stash holds millions of rows
    # of several KiB, so row * hidden_size overruns Int32 (18 GiB for the 7168-byte
    # activation) and an Int32 row index wraps the address into unrelated memory.
    drow0 = dpage * cutlass.Int64(page_size) + cutlass.Int64(sub)

    if active == one:
        if do_cuda == one:
            for k in cutlass.range_constexpr(rows):
                if cutlass.Int64(r0 + k) < n:
                    for j in cutlass.range_constexpr(vpr):
                        d = mCuda[(drow0 + cutlass.Int64(k), j * threads + tidx, None)]
                        cute.autovec_copy(frs[k * vpr + j], d)
        else:
            for k in cutlass.range_constexpr(rows):
                if cutlass.Int64(r0 + k) < n:
                    for j in cutlass.range_constexpr(vpr):
                        d = mHost[(drow0 + cutlass.Int64(k), j * threads + tidx, None)]
                        cute.autovec_copy(frs[k * vpr + j], d)

    if bidx == 0:
        # page_record is sized for this activation's maximum token count, not
        # for every page in the much larger shared stash free list.
        nchunk = (record_pages + threads - 1) // threads
        for c in cutlass.range_constexpr(nchunk):
            pidx = tidx + c * threads
            if pidx < record_pages:
                s64 = cutlass.Int64(pidx)
                rec_c = cutlass.Int32((h0 + s64) % c0s)
                rec_h = cutlass.Int32((h1 + s64) % c1s)
                pg = mFLC[rec_c] if do_cuda == one else mFLH[rec_h]
                gated = pg if active == one else cutlass.Int64(0)
                oPR[pidx] = gated if s64 < required else cutlass.Int64(0)

        if tidx == 0:
            nh0 = h0 + required * cutlass.Int64(do_cuda)
            nh1 = h1 + required * cutlass.Int64(do_host)
            oNH[0] = nh0
            oNH[1] = nh1
            oNHA[0] = nh0
            oNHA[1] = nh1
            oOvf[0] = ov0 + cutlass.Int64(ovf_set)
            # host_spill is a single flag shared by every stash in the iteration;
            # the manager zeroes it once per reset and expects it to stay set.
            # Writing do_host unconditionally clears it on the next CUDA-resident
            # stash, so only ever raise it here.
            if do_host == one:
                oHS[0] = cutlass.Int64(1)
            oSP[0] = cutlass.Int64(do_host + ovf_set)


@cute.jit
def _launch(
    mSrc: cute.Tensor,
    mCuda: cute.Tensor,
    mHost: cute.Tensor,
    mN: cute.Tensor,
    mFLC: cute.Tensor,
    mFLH: cute.Tensor,
    mHead: cute.Tensor,
    mTail: cute.Tensor,
    mCap: cute.Tensor,
    mOvfIn: cute.Tensor,
    oPR: cute.Tensor,
    oOvf: cute.Tensor,
    oHS: cute.Tensor,
    oSP: cute.Tensor,
    oNH: cute.Tensor,
    oNHA: cute.Tensor,
    stream,
):
    T = mSrc.shape[0]
    nvec = mSrc.shape[1]
    cap_rows = mCuda.shape[0]
    pages = mFLC.shape[0]
    record_pages = oPR.shape[0]
    page_size = cap_rows // pages
    threads, rows = _CFG.get(T, _DEFAULT_CFG)
    threads = _pick_threads(nvec, threads)

    _stash_kernel(
        mSrc,
        mCuda,
        mHost,
        mN,
        mFLC,
        mFLH,
        mHead,
        mTail,
        mCap,
        mOvfIn,
        oPR,
        oOvf,
        oHS,
        oSP,
        oNH,
        oNHA,
        page_size,
        pages,
        record_pages,
        nvec,
        threads,
        rows,
    ).launch(grid=((T + rows - 1) // rows, 1, 1), block=(threads, 1, 1), smem=8 * 8, stream=stream)


def run_direct(
    source,
    num_tokens,
    free_list_cuda,
    free_list_host,
    free_list_head,
    free_list_tail,
    free_list_capacity,
    overflow_initial,
    cuda_stash,
    host_stash,
    page_record,
    overflow,
    host_spill,
    spilled_to_host,
    new_free_list_head,
    free_list_head_after,
):
    T = source.shape[0]
    H = source.shape[1]
    CAP = cuda_stash.shape[0]
    NVEC = H // _VBYTES

    src3 = source.view(T, NVEC, _VBYTES)
    cud3 = cuda_stash.view(CAP, NVEC, _VBYTES)
    # Host spill capacity is independent from CUDA stash capacity.  It is
    # commonly smaller, so preserve its own row count when forming the vector
    # view; the kernel touches it only on the host-spill branch.
    hos3 = host_stash.view(host_stash.shape[0], NVEC, _VBYTES)

    args = (
        from_dlpack(src3, assumed_align=16),
        from_dlpack(cud3, assumed_align=16),
        from_dlpack(hos3, assumed_align=16),
        from_dlpack(num_tokens, assumed_align=8),
        from_dlpack(free_list_cuda, assumed_align=8),
        from_dlpack(free_list_host, assumed_align=8),
        from_dlpack(free_list_head, assumed_align=8),
        from_dlpack(free_list_tail, assumed_align=8),
        from_dlpack(free_list_capacity, assumed_align=8),
        from_dlpack(overflow_initial, assumed_align=8),
        from_dlpack(page_record, assumed_align=8),
        from_dlpack(overflow, assumed_align=8),
        from_dlpack(host_spill, assumed_align=8),
        from_dlpack(spilled_to_host, assumed_align=8),
        from_dlpack(new_free_list_head, assumed_align=8),
        from_dlpack(free_list_head_after, assumed_align=8),
    )

    stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)

    # Every tensor extent referenced by the CuTe JIT body is specialization
    # state.  In particular, page_record and host stash vary across layers
    # while the source/CUDA-stash shape can remain identical.
    key = (
        T,
        H,
        CAP,
        host_stash.shape[0],
        free_list_cuda.shape[0],
        free_list_host.shape[0],
        page_record.shape[0],
    )
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_launch, *args, stream)
        _CACHE[key] = fn
    fn(*args, stream)


LOAD_POLICY = "L2::evict_first"
STORE_POLICY = "L2::evict_last"


@dsl_user_op
def bulk_g2s(smem_addr, gmem_addr, nbytes, mbar_addr, *, loc=None, ip=None):
    asm = (
        "{\n"
        ".reg .b64 gsrc;\n"
        ".reg .b64 pol;\n"
        "cvta.to.global.u64 gsrc, $1;\n"
        f"createpolicy.fractional.{LOAD_POLICY}.b64 pol, 1.0;\n"
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [$0], [gsrc], $2, [$3], pol;\n"
        "}\n"
    )
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
            cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_s2g(gmem_addr, smem_addr, nbytes, *, loc=None, ip=None):
    asm = (
        "{\n"
        ".reg .b64 gdst;\n"
        ".reg .b64 pol;\n"
        "cvta.to.global.u64 gdst, $0;\n"
        f"createpolicy.fractional.{STORE_POLICY}.b64 pol, 1.0;\n"
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"
        " [gdst], [$1], $2, pol;\n"
        "}\n"
    )
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def copy_plan(total_bytes):
    if total_bytes >= 16 * 1024 * 1024:
        tile_bytes, tiles_per_cta = 2048, 16
    elif total_bytes >= 4 * 1024 * 1024:
        tile_bytes, tiles_per_cta = 4096, 4
    elif total_bytes >= 1024 * 1024:
        tile_bytes, tiles_per_cta = 4096, 2
    else:
        tile_bytes, tiles_per_cta = 2048, 1

    max_stages, store_lag = 6, 2
    stages = min(max_stages, tiles_per_cta + store_lag)
    lag = min(store_lag, stages - 1)
    return tile_bytes, tiles_per_cta, stages, lag


class PagedStashFused:
    """Bulk-copy (TMA) stash for one activation geometry.

    Every extent the kernel needs is baked in at trace time, so an instance is
    only valid for the exact ``(t_rows, hidden_bytes, page_size, record_pages)``
    it was built with.  Use :func:`tma_supported` before selecting this path.
    """

    def __init__(self, t_rows, hidden_bytes, page_size, record_pages):
        self.t_rows = t_rows
        self.hidden_bytes = hidden_bytes
        self.page_size = page_size
        self.page_bytes = page_size * hidden_bytes
        self.record_pages = record_pages
        self.total_bytes = t_rows * hidden_bytes
        self.tile_bytes, self.tiles_per_cta, self.stages, self.store_lag = copy_plan(
            self.total_bytes
        )
        self.load_ahead = min(self.stages - self.store_lag, self.tiles_per_cta)
        self.bytes_per_cta = self.tile_bytes * self.tiles_per_cta
        self.num_ctas = self.total_bytes // self.bytes_per_cta

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        num_tokens: cute.Tensor,
        free_list_cuda: cute.Tensor,
        free_list_host: cute.Tensor,
        free_list_head: cute.Tensor,
        free_list_tail: cute.Tensor,
        free_list_capacity: cute.Tensor,
        overflow_initial: cute.Tensor,
        cuda_stash: cute.Tensor,
        host_stash: cute.Tensor,
        page_record: cute.Tensor,
        overflow: cute.Tensor,
        host_spill: cute.Tensor,
        spilled_to_host: cute.Tensor,
        new_free_list_head: cute.Tensor,
        free_list_head_after: cute.Tensor,
        stream,
    ):
        self.kernel(
            source,
            num_tokens,
            free_list_cuda,
            free_list_host,
            free_list_head,
            free_list_tail,
            free_list_capacity,
            overflow_initial,
            cuda_stash,
            host_stash,
            page_record,
            overflow,
            host_spill,
            spilled_to_host,
            new_free_list_head,
            free_list_head_after,
        ).launch(grid=(self.num_ctas, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        num_tokens: cute.Tensor,
        free_list_cuda: cute.Tensor,
        free_list_host: cute.Tensor,
        free_list_head: cute.Tensor,
        free_list_tail: cute.Tensor,
        free_list_capacity: cute.Tensor,
        overflow_initial: cute.Tensor,
        cuda_stash: cute.Tensor,
        host_stash: cute.Tensor,
        page_record: cute.Tensor,
        overflow: cute.Tensor,
        host_spill: cute.Tensor,
        spilled_to_host: cute.Tensor,
        new_free_list_head: cute.Tensor,
        free_list_head_after: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = SmemAllocator()
        barriers = smem.allocate_array(cutlass.Int64, 2 * self.stages, byte_alignment=8)
        buffers = smem.allocate(self.stages * self.tile_bytes, 128)

        load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=barriers,
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tile_bytes,
        )
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.store_lag + 1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.ThreadBlock),
        )

        n = cutlass.Int32(num_tokens[0])
        required = (n + self.page_size - 1) // self.page_size
        incoming = cutlass.Int64(overflow_initial[0])
        head_cuda = cutlass.Int32(free_list_head[0])
        head_host = cutlass.Int32(free_list_head[1])
        cap_cuda = cutlass.Int32(free_list_capacity[0])
        cap_host = cutlass.Int32(free_list_capacity[1])
        # An unconfigured host stash reports capacity 0; keep the modulus non-zero
        # so the (never selected) host index stays inside the substituted list.
        cap_cuda_s = cap_cuda if cap_cuda > 0 else cutlass.Int32(1)
        cap_host_s = cap_host if cap_host > 0 else cutlass.Int32(1)
        cuda_ok = cutlass.Int32(free_list_tail[0]) - head_cuda >= required
        host_ok = cutlass.Int32(free_list_tail[1]) - head_host >= required
        can_allocate = cuda_ok | host_ok
        success = (incoming == 0) & can_allocate
        use_host = (cuda_ok == False) & host_ok
        active_host = success & use_host
        active_cuda = success & (use_host == False)
        allocation_failed = (incoming == 0) & (can_allocate == False)

        # Byte offsets exceed 2 GiB for production activations; keep them 64-bit.
        base = cutlass.Int64(bidx) * self.bytes_per_cta
        active_bytes = cutlass.Int64(n) * self.hidden_bytes
        copy_this_cta = success & (base < active_bytes)

        if copy_this_cta:
            source_base = cutlass.Int64(source.iterator.toint()) + base
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stages
            )

            for j in cutlass.range_constexpr(self.load_ahead):
                load_pipeline.producer_acquire(producer_state)
                with cute.arch.elect_one():
                    bulk_g2s(
                        (buffers + j * self.tile_bytes).toint(),
                        source_base + j * self.tile_bytes,
                        self.tile_bytes,
                        load_pipeline.producer_get_barrier(producer_state).toint(),
                    )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()

            slot = cutlass.Int32(base // self.page_bytes)
            offset_in_page = base % self.page_bytes
            cuda_slot = (head_cuda + slot) % cap_cuda_s
            host_slot = (head_host + slot) % cap_host_s
            cuda_page = cutlass.Int64(free_list_cuda[cuda_slot])
            host_page = cutlass.Int64(free_list_host[host_slot])
            host_i64 = cutlass.Int64(use_host)
            cuda_i64 = cutlass.Int64(1) - host_i64
            page = cuda_page * cuda_i64 + host_page * host_i64
            stash_base = (
                cutlass.Int64(cuda_stash.iterator.toint()) * cuda_i64
                + cutlass.Int64(host_stash.iterator.toint()) * host_i64
                + page * self.page_bytes
                + offset_in_page
            )

            for i in cutlass.range_constexpr(self.tiles_per_cta):
                stage = i % self.stages
                load_pipeline.consumer_wait(consumer_state)
                with cute.arch.elect_one():
                    bulk_s2g(
                        stash_base + i * self.tile_bytes,
                        (buffers + stage * self.tile_bytes).toint(),
                        self.tile_bytes,
                    )
                    store_pipeline.producer_commit()
                    store_pipeline.producer_acquire()
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

                next_tile = i + self.load_ahead
                if next_tile < self.tiles_per_cta:
                    load_pipeline.producer_acquire(producer_state)
                    next_stage = next_tile % self.stages
                    with cute.arch.elect_one():
                        bulk_g2s(
                            (buffers + next_stage * self.tile_bytes).toint(),
                            source_base + next_tile * self.tile_bytes,
                            self.tile_bytes,
                            load_pipeline.producer_get_barrier(producer_state).toint(),
                        )
                    load_pipeline.producer_commit(producer_state)
                    producer_state.advance()

            with cute.arch.elect_one():
                store_pipeline.producer_tail()

        if bidx == 0:
            # page_record holds one entry per page of THIS activation, which is
            # unrelated to the CTA width; walk it in warp-sized chunks.
            nchunk = (self.record_pages + 31) // 32
            for k in cutlass.range_constexpr(nchunk):
                i = cutlass.Int32(tidx + k * 32)
                if i < self.record_pages:
                    page_record[i] = cutlass.Int64(0)
                    if success & (i < required):
                        cuda_index = (head_cuda + i) % cap_cuda_s
                        host_index = (head_host + i) % cap_host_s
                        cuda_page = cutlass.Int64(free_list_cuda[cuda_index])
                        host_page = cutlass.Int64(free_list_host[host_index])
                        host_i64 = cutlass.Int64(use_host)
                        page_record[i] = (
                            cuda_page * (cutlass.Int64(1) - host_i64) + host_page * host_i64
                        )

            if tidx == 0:
                overflow[0] = incoming + cutlass.Int64(allocation_failed)
                # Shared sticky flag across every stash in the iteration: raise
                # only, never clear (the manager zeroes it once per reset).
                if active_host:
                    host_spill[0] = cutlass.Int64(1)
                spilled_to_host[0] = cutlass.Int64(active_host | allocation_failed)

                new_cuda_head = cutlass.Int64(head_cuda) + cutlass.Int64(required) * cutlass.Int64(
                    active_cuda
                )
                new_host_head = cutlass.Int64(head_host) + cutlass.Int64(required) * cutlass.Int64(
                    active_host
                )
                new_free_list_head[0] = new_cuda_head
                new_free_list_head[1] = new_host_head
                free_list_head_after[0] = new_cuda_head
                free_list_head_after[1] = new_host_head


_COMPILED = {}


def tma_supported(t_rows, hidden_bytes, page_size):
    """Whether :class:`PagedStashFused` can copy this geometry correctly.

    Each CTA issues one contiguous ``bytes_per_cta`` run of bulk copies from a
    single page, so the activation must tile exactly into CTAs and each CTA run
    must stay inside one page.
    """
    if t_rows <= 0 or hidden_bytes <= 0 or page_size <= 0:
        return False
    if hidden_bytes % _VBYTES != 0:
        return False
    total_bytes = t_rows * hidden_bytes
    tile_bytes, tiles_per_cta, _, _ = copy_plan(total_bytes)
    bytes_per_cta = tile_bytes * tiles_per_cta
    if total_bytes % bytes_per_cta != 0:
        return False
    if (page_size * hidden_bytes) % bytes_per_cta != 0:
        return False
    return total_bytes // bytes_per_cta >= 1


@torch.no_grad()
def run_tma(
    source,
    num_tokens,
    free_list_cuda,
    free_list_host,
    free_list_head,
    free_list_tail,
    free_list_capacity,
    overflow_initial,
    cuda_stash,
    host_stash,
    page_record,
    overflow,
    host_spill,
    spilled_to_host,
    new_free_list_head,
    free_list_head_after,
):
    t_rows = source.shape[0]
    hidden_bytes = source.shape[1] * source.element_size()
    page_size = cuda_stash.shape[0] // free_list_cuda.shape[0]
    record_pages = page_record.shape[0]
    tensors = (
        source,
        num_tokens,
        free_list_cuda,
        free_list_host,
        free_list_head,
        free_list_tail,
        free_list_capacity,
        overflow_initial,
        cuda_stash,
        host_stash,
        page_record,
        overflow,
        host_spill,
        spilled_to_host,
        new_free_list_head,
        free_list_head_after,
    )
    cute_tensors = tuple(from_dlpack(t, assumed_align=16) for t in tensors)
    stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)
    # The kernel bakes in every extent it touches, so the row count alone does
    # not identify a compiled variant.
    key = (t_rows, hidden_bytes, page_size, record_pages)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            PagedStashFused(t_rows, hidden_bytes, page_size, record_pages), *cute_tensors, stream
        )
        _COMPILED[key] = compiled
    compiled(*cute_tensors, stream)


def run(
    source,
    num_tokens,
    free_list_cuda,
    free_list_host,
    free_list_head,
    free_list_tail,
    free_list_capacity,
    overflow_initial,
    cuda_stash,
    host_stash,
    page_record,
    overflow,
    host_spill,
    spilled_to_host,
    new_free_list_head,
    free_list_head_after,
):
    # With no pinned host stash configured the caller passes the CUDA buffer as
    # host_dst but leaves the host free list empty.  Substitute the CUDA list so
    # every index the kernel computes lands in real memory; the host branch is
    # unreachable because its availability (tail - head) is zero.
    if free_list_host.numel() == 0:
        free_list_host = free_list_cuda

    args = (
        source,
        num_tokens,
        free_list_cuda,
        free_list_host,
        free_list_head,
        free_list_tail,
        free_list_capacity,
        overflow_initial,
        cuda_stash,
        host_stash,
        page_record,
        overflow,
        host_spill,
        spilled_to_host,
        new_free_list_head,
        free_list_head_after,
    )

    t_rows = source.shape[0]
    hidden_bytes = source.shape[1] * source.element_size()
    stash_bytes = cuda_stash.shape[1] * cuda_stash.element_size()
    pages = free_list_cuda.shape[0]
    page_size = cuda_stash.shape[0] // pages if pages > 0 else 0
    # Both branches are CuTeDSL; this only picks between them.  The bulk-copy
    # kernel is preferred: it drives the TMA engines from 32 threads per CTA
    # instead of 64-128 threads issuing vector load/store, so it leaves far more
    # SM capacity to the compute the pack stream overlaps with.  Standalone it is
    # within a few percent of run_direct (531 vs 519 us summed over the DSv3-671B
    # activations).  run_direct handles everything the bulk path cannot.
    #
    # Selecting a kernel by row count alone is what crashed: every activation with
    # that row count was fed to a variant traced for a different row width, so the
    # bulk copies walked far outside the stash buffer.  Gate on the full geometry.
    if (
        os.getenv("MEGATRON_PAGED_STASH_CUTE_TMA", "1") == "1"
        and hidden_bytes == stash_bytes
        and pages * page_size == cuda_stash.shape[0]
        and tma_supported(t_rows, hidden_bytes, page_size)
    ):
        run_tma(*args)
    else:
        run_direct(*args)
