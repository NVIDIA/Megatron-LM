# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact top-k with a Triton inline-PTX fast path and a Triton radix fallback."""

import triton
import triton.language as tl


@triton.jit
def chunk_lengths(Lengths, ChunkLengths, CandidateLengths, CHUNKS: tl.constexpr, K: tl.constexpr):
    """Build valid chunk lengths and the total candidate count per row."""
    row = tl.program_id(0)
    n = tl.load(Lengths + row)
    total = tl.full((), 0, tl.int32)
    for chunk in tl.static_range(CHUNKS):
        length = tl.minimum(tl.maximum(n - chunk * 65536, 0), 65536)
        tl.store(ChunkLengths + row * CHUNKS + chunk, length)
        total += tl.minimum(length, K)
    tl.store(CandidateLengths + row, total)


@triton.jit
def gather_chunk_candidates(
    Scores, ChunkIndices, Candidates, WIDTH: tl.constexpr, CHUNKS: tl.constexpr, K: tl.constexpr
):
    """Gather original scores for the union of chunk-local top-k supports."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    slots = tl.arange(0, K)
    index = tl.load(ChunkIndices + (row * CHUNKS + chunk) * K + slots)
    score = tl.load(Scores + row * WIDTH + chunk * 65536 + index, index >= 0, float("-inf"))
    tl.store(Candidates + (row * CHUNKS + chunk) * K + slots, score)


@triton.jit
def remap_chunk_indices(
    ChunkIndices, Selected, Lengths, Out, CHUNKS: tl.constexpr, K: tl.constexpr
):
    """Map selected candidate slots back to global logical key positions."""
    row = tl.program_id(0)
    slots = tl.arange(0, K)
    selected = tl.load(Selected + row * K + slots)
    local = tl.load(ChunkIndices + row * CHUNKS * K + selected, selected >= 0, -1)
    result = selected // K * 65536 + local
    n = tl.load(Lengths + row)
    tl.store(
        Out + row * K + slots, tl.where((selected >= 0) & (local >= 0) & (result < n), result, -1)
    )


@triton.jit
def radix_topk(Scores, Lengths, Indices, WIDTH: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    """Select an exact support by ordered float bits, resolving cutoff ties by position."""
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
    num_greater = tl.sum(greater.to(tl.int32), 0)
    equal_rank = tl.cumsum(equal.to(tl.int32), 0)
    selected = greater | (equal & (equal_rank <= tl.minimum(K, length) - num_greater))
    output_rank = tl.cumsum(selected.to(tl.int32), 0) - 1
    slots = tl.arange(0, K)
    tl.store(Indices + row * K + slots, -1)
    tl.debug_barrier()
    tl.store(
        Indices + row * K + output_rank, offsets.to(tl.int32), mask=selected & (output_rank < K)
    )


@triton.constexpr_function
def fast_topk_asm(width: int, k: int) -> str:
    # The fast path gives each CUDA thread one logical lane. Keep this geometry
    # consistent with exact_topk's launch and the warp-prefix shared storage.
    """Build the histogram and exact crossing-bucket selection program for one CTA."""
    threads, bins, cap = 1024, 1024, 1024
    packs = (width + threads * 4 - 1) // (threads * 4)
    scalars = packs * 4
    cached = scalars <= 32
    wide = scalars > 32
    kind = "64" if wide else "32"
    # Keep static storage below 16 KiB: a fallback with 32 KiB dynamic scratch
    # must fit CUDA's default 48 KiB limit when Triton does not request opt-in.
    # Large crossing buckets still use the exact radix fallback.
    WMN, WMX, WA, WB, CTRL, KEY, IDX = 4096, 4224, 4352, 4480, 4608, 4672, 8768
    lines = [
        "{",
        ".shared .align 16 .b8 topk_smem[12864];",
        ".reg .u64 src,dst,addr,mask_a64,mask_b64,rev64;",
        ".reg .b32 v<" + str(scalars if cached else 4) + ">;",
        ".reg .b32 lo,hi,mn,mx,scale,f,tmp;",
        ".reg .u32 tid,lane,warp,n,na,idx,pos,base,saddr,haddr,tmpa,tmpb,digit,cross,above,m,need;",
        (
            ".reg .u32 ca,cb,oa,ob,ua,ub,orig_a,orig_b,off_a,off_b,mask_a,mask_b,bit,key,ki,ii,kj,"
            "ij,rank,j,candidate;"
        ),
        ".reg .pred p,q,valid,eq;",
        "mov.u64 src,$1; mov.u64 dst,$2; mov.u32 n,$3; mov.u32 tid,$4; mov.u32 $0,0;",
        "and.b32 lane,tid,31; shr.u32 warp,tid,5; mov.u32 saddr,topk_smem;",
        f"setp.le.u32 p,n,{k}; @p bra TOPK_DIRECT;",
        "add.u32 na,n,3; and.b32 na,na,0xfffffffc;",
        "mov.b32 mn,0f7f800000; mov.b32 mx,0fff800000;",
    ]

    def emit(*xs):
        lines.extend(xs)

    def load_pack(pack):
        vs = [f"v{pack*4+j}" if cached else f"v{j}" for j in range(4)]
        emit(
            f"mad.lo.u32 idx,tid,4,{pack*threads*4};",
            "mul.wide.u32 addr,idx,4; add.u64 addr,addr,src;",
            "setp.lt.u32 p,idx,na;",
        )
        for v in vs:
            emit(f"mov.b32 {v},0fff800000;")
        emit("@p ld.global.v4.b32 {" + ",".join(vs) + "},[addr];")
        return vs

    for pack in range(packs):
        vs = load_pack(pack)
        for sub, v in enumerate(vs):
            emit(
                f"add.u32 pos,idx,{sub}; setp.lt.u32 valid,pos,n;",
                f"@valid min.f32 mn,mn,{v}; @valid max.f32 mx,mx,{v};",
            )
    # Reduce per-warp minima/maxima, then broadcast the CTA range.
    for delta in (16, 8, 4, 2, 1):
        emit(
            f"shfl.sync.bfly.b32 tmp,mn,{delta},31,0xffffffff; min.f32 mn,mn,tmp;",
            f"shfl.sync.bfly.b32 tmp,mx,{delta},31,0xffffffff; max.f32 mx,mx,tmp;",
        )
    emit(
        "setp.eq.u32 p,lane,0;",
        f"mad.lo.u32 haddr,warp,4,{WMN}; add.u32 haddr,haddr,saddr; @p st.shared.b32 [haddr],mn;",
        f"mad.lo.u32 haddr,warp,4,{WMX}; add.u32 haddr,haddr,saddr; @p st.shared.b32 [haddr],mx;",
        "bar.sync 0; setp.ne.u32 p,warp,0; @p bra TOPK_RANGE_DONE;",
        f"mad.lo.u32 haddr,lane,4,{WMN}; add.u32 haddr,haddr,saddr; ld.shared.b32 mn,[haddr];",
        f"mad.lo.u32 haddr,lane,4,{WMX}; add.u32 haddr,haddr,saddr; ld.shared.b32 mx,[haddr];",
    )
    for delta in (16, 8, 4, 2, 1):
        emit(
            f"shfl.sync.bfly.b32 tmp,mn,{delta},31,0xffffffff; min.f32 mn,mn,tmp;",
            f"shfl.sync.bfly.b32 tmp,mx,{delta},31,0xffffffff; max.f32 mx,mx,tmp;",
        )
    emit(
        "setp.eq.u32 p,lane,0;",
        f"add.u32 haddr,saddr,{CTRL+12}; @p st.shared.b32 [haddr],mn;",
        f"add.u32 haddr,saddr,{CTRL+16}; @p st.shared.b32 [haddr],mx;",
        "TOPK_RANGE_DONE: bar.sync 0;",
        f"add.u32 haddr,saddr,{CTRL+12}; ld.shared.b32 lo,[haddr];",
        f"add.u32 haddr,saddr,{CTRL+16}; ld.shared.b32 hi,[haddr];",
        "setp.eq.f32 p,lo,hi; @p bra TOPK_DIRECT;",
        "testp.finite.f32 p,lo; testp.finite.f32 q,hi; and.pred p,p,q; @!p bra TOPK_FALLBACK;",
        (
            "sub.rn.f32 f,hi,lo; testp.finite.f32 p,f; @!p bra TOPK_FALLBACK; div.approx.f32 scale,"
            "0f44800000,f; testp.finite.f32 p,scale; @!p bra TOPK_FALLBACK;"
        ),
        "mad.lo.u32 haddr,tid,4,saddr; st.shared.u32 [haddr],0; bar.sync 0;",
    )

    def classify(v, offset):
        emit(
            f"mad.lo.u32 pos,tid,4,{offset}; setp.lt.u32 valid,pos,n;",
            f"sub.rn.f32 f,{v},lo; mul.rn.f32 f,f,scale; cvt.rzi.s32.f32 digit,f;",
            f"max.s32 digit,digit,0; min.s32 digit,digit,{bins-1};",
        )

    for pack in range(packs):
        vs = [f"v{pack*4+j}" for j in range(4)] if cached else load_pack(pack)
        for sub, v in enumerate(vs):
            classify(v, pack * threads * 4 + sub)
            emit("mad.lo.u32 haddr,digit,4,saddr; @valid atom.shared.add.u32 tmp,[haddr],1;")
    emit(
        "bar.sync 0;",
        f"sub.u32 digit,{bins-1},tid;",
        "mad.lo.u32 haddr,digit,4,saddr; ld.shared.u32 ca,[haddr]; mov.u32 oa,ca;",
    )

    def scan_up(value, temp):
        for delta in (1, 2, 4, 8, 16):
            emit(
                f"shfl.sync.up.b32 {temp},{value},{delta},0,0xffffffff;",
                f"setp.ge.u32 p,lane,{delta}; @p add.u32 {value},{value},{temp};",
            )

    def cta_prefix(pair, tag):
        scan_up("ca", "tmpa")
        if pair:
            scan_up("cb", "tmpb")
        emit(
            "setp.eq.u32 p,lane,31;",
            (
                f"mad.lo.u32 haddr,warp,4,{WA}; add.u32 haddr,haddr,saddr; @p st.shared.u32 "
                "[haddr],ca;"
            ),
        )
        if pair:
            emit(
                (
                    f"mad.lo.u32 haddr,warp,4,{WB}; add.u32 haddr,haddr,saddr; @p st.shared.u32 "
                    "[haddr],cb;"
                )
            )
        emit(
            f"bar.sync 0; setp.ne.u32 p,warp,0; @p bra {tag}_PREFIX_DONE;",
            (
                f"mad.lo.u32 haddr,lane,4,{WA}; add.u32 haddr,haddr,saddr; ld.shared.u32 ua,"
                "[haddr]; mov.u32 orig_a,ua;"
            ),
        )
        if pair:
            emit(
                (
                    f"mad.lo.u32 haddr,lane,4,{WB}; add.u32 haddr,haddr,saddr; ld.shared.u32 ub,"
                    "[haddr]; mov.u32 orig_b,ub;"
                )
            )
        scan_up("ua", "tmpa")
        if pair:
            scan_up("ub", "tmpb")
        emit(
            "sub.u32 ua,ua,orig_a;",
            f"mad.lo.u32 haddr,lane,4,{WA}; add.u32 haddr,haddr,saddr; st.shared.u32 [haddr],ua;",
        )
        if pair:
            emit(
                "sub.u32 ub,ub,orig_b;",
                (
                    f"mad.lo.u32 haddr,lane,4,{WB}; add.u32 haddr,haddr,saddr; st.shared.u32 "
                    "[haddr],ub;"
                ),
            )
        emit(
            f"{tag}_PREFIX_DONE: bar.sync 0;",
            (
                f"mad.lo.u32 haddr,warp,4,{WA}; add.u32 haddr,haddr,saddr; ld.shared.u32 ua,"
                "[haddr]; add.u32 ca,ca,ua;"
            ),
        )
        if pair:
            emit(
                (
                    f"mad.lo.u32 haddr,warp,4,{WB}; add.u32 haddr,haddr,saddr; ld.shared.u32 ub,"
                    "[haddr]; add.u32 cb,cb,ub;"
                )
            )

    cta_prefix(False, "HIST")
    emit(
        f"setp.ge.u32 p,ca,{k}; sub.u32 tmp,ca,oa; setp.lt.u32 q,tmp,{k}; and.pred p,p,q;",
        f"add.u32 haddr,saddr,{CTRL}; @p st.shared.u32 [haddr],digit;",
        f"add.u32 haddr,saddr,{CTRL+4}; @p st.shared.u32 [haddr],tmp;",
        f"add.u32 haddr,saddr,{CTRL+8}; @p st.shared.u32 [haddr],oa;",
        "bar.sync 0;",
        f"add.u32 haddr,saddr,{CTRL}; ld.shared.u32 cross,[haddr];",
        f"add.u32 haddr,saddr,{CTRL+4}; ld.shared.u32 above,[haddr];",
        f"add.u32 haddr,saddr,{CTRL+8}; ld.shared.u32 m,[haddr];",
        f"setp.gt.u32 p,m,{cap}; @p bra TOPK_FALLBACK;",
        f"sub.u32 need,{k},above;",
        f"mov.b{kind} " + ("mask_a64" if wide else "mask_a") + ",0;",
        f"mov.b{kind} " + ("mask_b64" if wide else "mask_b") + ",0;",
    )
    ma, mb = ("mask_a64", "mask_b64") if wide else ("mask_a", "mask_b")
    for pack in range(packs):
        vs = [f"v{pack*4+j}" for j in range(4)] if cached else load_pack(pack)
        for sub, v in enumerate(vs):
            classify(v, pack * threads * 4 + sub)
            emit(
                "setp.gt.u32 p,digit,cross; and.pred p,p,valid;",
                f"@p or.b{kind} {ma},{ma},{1<<(pack*4+sub)};",
                "setp.eq.u32 p,digit,cross; and.pred p,p,valid;",
                f"@p or.b{kind} {mb},{mb},{1<<(pack*4+sub)};",
            )
    emit(f"popc.b{kind} ca,{ma}; popc.b{kind} cb,{mb}; mov.u32 oa,ca; mov.u32 ob,cb;")
    cta_prefix(True, "EMIT")
    emit(
        "sub.u32 off_a,ca,oa; sub.u32 off_b,cb,ob;",
        f"TOPK_EMIT_A: setp.eq.u{kind} p,{ma},0; @p bra TOPK_EMIT_A_DONE;",
    )

    def get_bit(mask):
        if wide:
            emit(f"brev.b64 rev64,{mask}; clz.b64 bit,rev64;")
        else:
            emit(f"brev.b32 tmp,{mask}; clz.b32 bit,tmp;")
        emit(
            (
                "shr.u32 idx,bit,2; mad.lo.u32 idx,idx,1024,tid; shl.b32 idx,idx,2; and.b32 tmp,"
                "bit,3; add.u32 idx,idx,tmp;"
            )
        )

    get_bit(ma)
    emit(
        (
            "mul.wide.u32 addr,off_a,4; add.u64 addr,addr,dst; st.global.u32 [addr],idx; add.u32 "
            "off_a,off_a,1;"
        ),
        f"add.u{kind} " + ("rev64" if wide else "tmp") + f",{ma},-1;",
        f"and.b{kind} {ma},{ma}," + ("rev64" if wide else "tmp") + ";",
        "bra TOPK_EMIT_A;",
        "TOPK_EMIT_A_DONE:",
        f"TOPK_GATHER: setp.eq.u{kind} p,{mb},0; @p bra TOPK_GATHER_DONE;",
    )
    get_bit(mb)
    emit(
        "mul.wide.u32 addr,idx,4; add.u64 addr,addr,src; ld.global.b32 key,[addr];",
        "setp.lt.s32 p,key,0; selp.b32 tmp,0xffffffff,0x80000000,p; xor.b32 key,key,tmp;",
        f"mad.lo.u32 haddr,off_b,4,{KEY}; add.u32 haddr,haddr,saddr; st.shared.u32 [haddr],key;",
        f"mad.lo.u32 haddr,off_b,4,{IDX}; add.u32 haddr,haddr,saddr; st.shared.u32 [haddr],idx;",
        "add.u32 off_b,off_b,1;",
        f"add.u{kind} " + ("rev64" if wide else "tmp") + f",{mb},-1;",
        f"and.b{kind} {mb},{mb}," + ("rev64" if wide else "tmp") + ";",
        "bra TOPK_GATHER; TOPK_GATHER_DONE: bar.sync 0; mov.u32 candidate,tid;",
        "TOPK_CANDIDATE: setp.ge.u32 p,candidate,m; @p bra TOPK_CANDIDATE_DONE;",
        f"mad.lo.u32 haddr,candidate,4,{KEY}; add.u32 haddr,haddr,saddr; ld.shared.u32 ki,[haddr];",
        f"mad.lo.u32 haddr,candidate,4,{IDX}; add.u32 haddr,haddr,saddr; ld.shared.u32 ii,[haddr];",
        "mov.u32 rank,0; mov.u32 j,0;",
        "TOPK_RANK: setp.ge.u32 p,j,m; @p bra TOPK_RANK_DONE;",
        f"mad.lo.u32 haddr,j,4,{KEY}; add.u32 haddr,haddr,saddr; ld.shared.u32 kj,[haddr];",
        f"mad.lo.u32 haddr,j,4,{IDX}; add.u32 haddr,haddr,saddr; ld.shared.u32 ij,[haddr];",
        (
            "setp.gt.u32 p,kj,ki; setp.eq.u32 eq,kj,ki; setp.lt.u32 q,ij,ii; and.pred eq,eq,q; "
            "or.pred p,p,eq;"
        ),
        "selp.u32 tmp,1,0,p; add.u32 rank,rank,tmp; add.u32 j,j,1; bra TOPK_RANK;",
        (
            "TOPK_RANK_DONE: setp.lt.u32 p,rank,need; add.u32 pos,above,rank; mul.wide.u32 addr,"
            "pos,4; add.u64 addr,addr,dst; @p st.global.u32 [addr],ii;"
        ),
        "add.u32 candidate,candidate,1024; bra TOPK_CANDIDATE;",
        "TOPK_CANDIDATE_DONE: bra TOPK_EXIT;",
        "TOPK_DIRECT:",
    )
    for slot in range((k + threads - 1) // threads):
        emit(
            (
                f"add.u32 pos,tid,{slot*threads}; setp.lt.u32 valid,pos,n; selp.u32 idx,pos,"
                "0xffffffff,valid;"
            ),
            (
                f"setp.lt.u32 p,pos,{k}; mul.wide.u32 addr,pos,4; add.u64 addr,addr,dst; @p "
                "st.global.u32 [addr],idx;"
            ),
        )
    emit("bra TOPK_EXIT; TOPK_FALLBACK: mov.u32 $0,1; TOPK_EXIT: }")
    return "\n".join(lines)


@triton.jit
def exact_topk(Scores, Lengths, Indices, Flags, WIDTH: tl.constexpr, K: tl.constexpr):
    """Select each valid row prefix with the PTX path and exact radix fallback."""
    tl.static_assert(WIDTH % 4 == 0 and WIDTH > 0 and WIDTH <= 65536)
    tl.static_assert(K == 512 or K == 1024 or K == 2048)
    row = tl.program_id(0)
    tid = tl.arange(0, 1024)
    n = tl.minimum(tl.load(Lengths + row), WIDTH)
    flag = tl.inline_asm_elementwise(
        fast_topk_asm(WIDTH, K),
        constraints="=r,l,l,r,r,~{memory}",
        args=[(Scores + row * WIDTH).to(tl.uint64), (Indices + row * K).to(tl.uint64), n, tid],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    fallback = tl.max(flag, 0)
    tl.store(Flags + row, fallback)
    if fallback:
        radix_topk(Scores, Lengths, Indices, WIDTH=WIDTH, K=K, BLOCK=triton.next_power_of_2(WIDTH))
