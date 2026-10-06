# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused shared-prefix (tree/forest) attention — flash-composed passes with exact backward.

Self-contained port of the optimized kernel suite from the aresk_shared_prefix sandbox (opt1–opt11
on branch opt1-plancache-tritonmerge @ 3aa3faf03; bench/parity harness lives there under
examples/shared_prefix_attention). Public entry points:

- ``flash_composed_forest_attention_fused(q, k, v, node_start, node_len, node_parent)`` — general
  DFS-preorder forest attention as few flash varlen passes merged by online-softmax LSE, with the
  EXACT backward (merged-output substitution: plain autograd drops the inter-pass normalizer
  coupling because flash's LSE output carries no gradient).
- ``flash_composed_forest_attention(q, k, v, forest)`` — multi-group star forests
  (``[(offset, prefix_len, completion_lens), ...]``), fused into one plan.

Env knobs (defaults tuned on GB200): NRL_SP_CHAINFIRST (hybrid chain plan), NRL_SP_QSLICE
(zero-copy q views), NRL_SP_STREAMS (stream overlap), and NRL_SP_COMBINE (cross-pass
consolidation, default off). Experimental Triton KV gather, backward glue, and dQ assembly have
separate, default-off opt-ins so each can be parity-qualified independently.
NRL_SP_DETERMINISTIC_BACKWARD is a default-off diagnostic that selects FlashAttention's
deterministic backward. The retained Triton LSE merge is production-disabled pending a full-model
corruption fix.
GB200 net vs block-diagonal at equal work: star-like/balanced trees 1.59x training / 1.56x
logprob; deep branched trees 1.00x / 1.01x at a 1.10x FLOP ceiling (91-92% kernel efficiency).
"""

import os
from contextlib import nullcontext
from itertools import pairwise
from typing import List

import torch

from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp

# --- Optimization: plan caching + fused (Triton) LSE merge ------------------------------------
# The fused kernel's overhead vs raw flash is NOT attention math; it is (a) rebuilding the pass
# plan (pure-Python node loops + Python-int index lists -> H2D copies) on EVERY call -- the same
# bin layout recurs across all ~50 layers of a step -- and (b) the eager online-softmax merge,
# which upcasts every pass output to fp32 and round-trips full [total, np, hn] tensors through
# memory several times per forward. (a) is fixed by an LRU plan cache keyed on the node arrays;
# (b) by a single Triton kernel that reads each pass's output/LSE once and writes the merged
# output (+ final LSE) once, fp32 math in-register, bf16 out. The LSE merge is fail-closed below;
# other Triton optimizations have independent default-off gates and automatically fall back when
# Triton is unavailable.
try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


# Default OFF in this production copy: the Triton merge kernel is bit-correct in isolation
# (kernel-vs-eager 3e-4 on captured in-model pass tensors) but nondeterministically corrupts
# the cross-pass rows when run inside the full HybridModel process (0.2-0.3 rel logits error,
# Heisenbug: any sync/instrumentation in the pass region masks it; streams/tile-config/
# sync-before-merge all ruled out — suspected Triton runtime interaction with TE/mamba
# kernels in-process). The eager merge is stably exact in-model and the end-to-end cost is
# small (fwd 3.54x vs flex 3.11x with eager merge). Fail closed if a sandbox happens to export
# the old opt-in: known-corrupt execution must not silently enter a production training run.
def _resolve_fused_merge_setting():
    requested = os.environ.get("NRL_SP_FUSED_MERGE", "0") not in ("0", "", "false", "False")
    if requested:
        raise RuntimeError(
            "NRL_SP_FUSED_MERGE is disabled in the production shared-prefix port: "
            "the Triton merge has known nondeterministic full-HybridModel corruption"
        )
    return False


def _resolve_experimental_triton_setting(env_name):
    """Resolve a default-off opt-in for one independently parity-qualified Triton path.

    These kernels are intentionally experimental. Reject ambiguous values so a launcher typo
    cannot silently turn one on before its target topology has passed numerical parity.
    """
    value = os.environ.get(env_name, "0")
    normalized = value.lower()
    if normalized in ("", "0", "false"):
        return False
    if normalized in ("1", "true"):
        return True
    raise RuntimeError(f"{env_name} must be one of 0, 1, false, or true; got {value!r}")


def _resolve_deterministic_backward_setting():
    """Resolve the opt-in deterministic FlashAttention backward diagnostic.

    Reject unknown values instead of silently selecting the faster nondeterministic path when a
    launcher misspells the diagnostic setting.
    """
    value = os.environ.get("NRL_SP_DETERMINISTIC_BACKWARD", "0")
    normalized = value.lower()
    if normalized in ("0", "false"):
        return False
    if normalized in ("1", "true"):
        return True
    raise RuntimeError(
        "NRL_SP_DETERMINISTIC_BACKWARD must be one of 0, 1, false, or true; " f"got {value!r}"
    )


_SP_FUSED_MERGE = _resolve_fused_merge_setting()
_SP_FUSED_KV_GATHER = _resolve_experimental_triton_setting("NRL_SP_FUSED_KV_GATHER")
_SP_FUSED_BACKWARD_GLUE = _resolve_experimental_triton_setting("NRL_SP_FUSED_BACKWARD_GLUE")
_SP_FUSED_DQ_ASSEMBLY = _resolve_experimental_triton_setting("NRL_SP_FUSED_DQ_ASSEMBLY")
_SP_DETERMINISTIC_BACKWARD = _resolve_deterministic_backward_setting()
# merge/dq-assembly kernel tile config (swept on GB200; override for other parts)
_SP_MERGE_BT = int(os.environ.get("NRL_SP_MERGE_BT", "16"))
_SP_MERGE_WARPS = int(os.environ.get("NRL_SP_MERGE_WARPS", "8"))

_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 128


if HAVE_TRITON:

    @triton.jit
    def _sp_merge_fwd_kernel(
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,  # pass outputs, [rows_p, np, HN] (dtype of q)
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,  # pass LSEs, fp32 [np, rows_p]
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,  # int32 [total]: token -> row in pass (or -1)
        r0,
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,  # rows_p per pass (for LSE stride)
        out_ptr,  # merged output [total, np, HN] (dtype of q)
        lsef_ptr,  # final LSE fp32 [np, total]
        n_passes,
        total,
        np_: tl.constexpr,
        HN: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # One program merges a BLOCK_T-token tile for one head: amortizes scheduling over
        # 524k-programs-of-tiny-work (the v1 grid), keeps o loads coalesced along HN, and gives
        # the compiler ILP across the tile. [BLOCK_T, HN] fp32 tile state.
        tb = tl.program_id(0)
        h = tl.program_id(1)
        t = tb * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < total
        offs = tl.arange(0, HN)
        m = tl.full([BLOCK_T], float("-inf"), dtype=tl.float32)
        s = tl.zeros([BLOCK_T], dtype=tl.float32)
        acc = tl.zeros([BLOCK_T, HN], dtype=tl.float32)
        for p in tl.static_range(7):
            if p < n_passes:
                if p == 0:
                    idx_ptr, o_ptr, l_ptr, rows = i0, o0, l0, r0
                elif p == 1:
                    idx_ptr, o_ptr, l_ptr, rows = i1, o1, l1, r1
                elif p == 2:
                    idx_ptr, o_ptr, l_ptr, rows = i2, o2, l2, r2
                elif p == 3:
                    idx_ptr, o_ptr, l_ptr, rows = i3, o3, l3, r3
                elif p == 4:
                    idx_ptr, o_ptr, l_ptr, rows = i4, o4, l4, r4
                elif p == 5:
                    idx_ptr, o_ptr, l_ptr, rows = i5, o5, l5, r5
                else:
                    idx_ptr, o_ptr, l_ptr, rows = i6, o6, l6, r6
                r = tl.load(idx_ptr + t, mask=tmask, other=-1)
                hit = (r >= 0) & tmask
                lse = tl.load(l_ptr + h * rows + r, mask=hit, other=float("-inf"))
                o = tl.load(
                    o_ptr + (r[:, None] * np_ + h) * HN + offs[None, :],
                    mask=hit[:, None],
                    other=0.0,
                ).to(tl.float32)
                m_new = tl.maximum(m, lse)
                m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                scale_old = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_safe))
                w = tl.where(hit, tl.exp(lse - m_safe), 0.0)
                acc = acc * scale_old[:, None] + o * w[:, None]
                s = s * scale_old + w
                m = m_new
        out = acc / tl.where(s == 0.0, 1.0, s)[:, None]
        tl.store(
            out_ptr + (t[:, None] * np_ + h) * HN + offs[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=tmask[:, None],
        )
        tl.store(lsef_ptr + h * total + t, m + tl.log(s), mask=tmask)


if HAVE_TRITON:

    @triton.jit
    def _sp_scale_gather_kernel(
        do_ptr,  # [total, np, HN] upstream grad (q dtype)
        lse_ptr,  # fp32 [np, rows] this pass's LSE
        lsef_ptr,  # fp32 [np, total] merged LSE
        qidx_ptr,  # int64 [rows] pass-row -> token (identity pass passes arange)
        dox_ptr,  # out [rows, np, HN] (q dtype): w * do[qidx]
        rows,
        total,
        np_: tl.constexpr,
        HN: tl.constexpr,
    ):
        rb = tl.program_id(0)
        h = tl.program_id(1)
        BLOCK_R: tl.constexpr = 16
        r = rb * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < rows
        offs = tl.arange(0, HN)
        t = tl.load(qidx_ptr + r, mask=rmask, other=0)
        lse_p = tl.load(lse_ptr + h * rows + r, mask=rmask, other=0.0)
        w = tl.exp(lse_p - tl.load(lsef_ptr + h * total + t, mask=rmask, other=0.0))
        # zero-K padding rows (q-slice cross pass) report LSE=+inf: their weight must be 0,
        # not inf, so their (zero) flash grads stay zero instead of turning NaN.
        w = tl.where(lse_p > 1e30, 0.0, w)
        do = tl.load(
            do_ptr + (t[:, None] * np_ + h) * HN + offs[None, :], mask=rmask[:, None], other=0.0
        ).to(tl.float32)
        tl.store(
            dox_ptr + (r[:, None] * np_ + h) * HN + offs[None, :],
            (do * w[:, None]).to(dox_ptr.dtype.element_ty),
            mask=rmask[:, None],
        )

    @triton.jit
    def _sp_scatter_accum_kernel(
        dst_ptr,  # fp32 [total, n, HN] accumulator
        src_ptr,  # [rows, n, HN] pass grad (q dtype)
        idx_ptr,  # int64 [rows] pass-row -> token
        n_rows,
        n_: tl.constexpr,
        HN: tl.constexpr,
    ):
        rb = tl.program_id(0)
        h = tl.program_id(1)
        BLOCK_R: tl.constexpr = 16
        r = rb * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < n_rows
        offs = tl.arange(0, HN)
        t = tl.load(idx_ptr + r, mask=rmask, other=0)
        add = tl.load(
            src_ptr + (r[:, None] * n_ + h) * HN + offs[None, :], mask=rmask[:, None], other=0.0
        ).to(tl.float32)
        # atomic: with the consolidated cross pass a deep token owns one row PER ancestor level,
        # so multiple programs may target the same destination row.
        tl.atomic_add(
            dst_ptr + (t[:, None] * n_ + h) * HN + offs[None, :], add, mask=rmask[:, None]
        )


if HAVE_TRITON:

    @triton.jit
    def _sp_dq_merge_kernel(
        d0,
        d1,
        d2,
        d3,
        d4,
        d5,
        d6,  # per-pass dq contributions [rows_p, np, HN] (q dtype)
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,  # int32 [total]: token -> row in pass (or -1)
        out_ptr,  # final dq [total, np, HN] (q dtype)
        n_passes,
        total,
        np_: tl.constexpr,
        HN: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # opt10: dq final assembly as a gather-side sum over slots (mirror of the fwd merge,
        # minus the LSE weighting — each pass's dqx is already its finished contribution).
        # Replaces the fp32 [total, np, HN] accumulator + per-pass scatter/adds + final cast:
        # every dqx is read once, dq written once, fp32 math in-register.
        tb = tl.program_id(0)
        h = tl.program_id(1)
        t = tb * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < total
        offs = tl.arange(0, HN)
        acc = tl.zeros([BLOCK_T, HN], dtype=tl.float32)
        for p in tl.static_range(7):
            if p < n_passes:
                if p == 0:
                    idx_ptr, d_ptr = i0, d0
                elif p == 1:
                    idx_ptr, d_ptr = i1, d1
                elif p == 2:
                    idx_ptr, d_ptr = i2, d2
                elif p == 3:
                    idx_ptr, d_ptr = i3, d3
                elif p == 4:
                    idx_ptr, d_ptr = i4, d4
                elif p == 5:
                    idx_ptr, d_ptr = i5, d5
                else:
                    idx_ptr, d_ptr = i6, d6
                r = tl.load(idx_ptr + t, mask=tmask, other=-1)
                hit = (r >= 0) & tmask
                acc += tl.load(
                    d_ptr + (r[:, None] * np_ + h) * HN + offs[None, :],
                    mask=hit[:, None],
                    other=0.0,
                ).to(tl.float32)
        tl.store(
            out_ptr + (t[:, None] * np_ + h) * HN + offs[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=tmask[:, None],
        )

    @triton.jit
    def _sp_gather_kv_kernel(
        k_ptr,
        v_ptr,  # [total, ng, HN] sources (q dtype)
        idx_ptr,  # int64 [rows] pass-row -> token
        kx_ptr,
        vx_ptr,  # [rows, ng, HN] destinations
        k_stride_t,
        k_stride_h,
        k_stride_d,
        v_stride_t,
        v_stride_h,
        v_stride_d,
        ng: tl.constexpr,
        HN: tl.constexpr,
    ):
        r = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, HN)
        t = tl.load(idx_ptr + r)
        tl.store(
            kx_ptr + (r * ng + h) * HN + offs,
            tl.load(k_ptr + t * k_stride_t + h * k_stride_h + offs * k_stride_d),
        )
        tl.store(
            vx_ptr + (r * ng + h) * HN + offs,
            tl.load(v_ptr + t * v_stride_t + h * v_stride_h + offs * v_stride_d),
        )


# Round-3: overlap the independent per-pass flash calls on side CUDA streams (they only join at
# the LSE merge / the gradient scatters), and gather K+V through one fused kernel into
# plan-cached workspace buffers (no per-call allocations, one index read for both tensors).
# NRL_SP_STREAMS=0 disables the stream overlap (kernels still fused).
_SP_STREAMS = os.environ.get("NRL_SP_STREAMS", "1") not in ("0", "", "false", "False")
# Consolidate all per-level cross passes into one flash call (any depth => 2 flash calls total).
# Requires the Triton merge path (the eager merge cannot handle a token owning multiple rows of
# one pass); plans are cached per effective mode so runtime flag flips stay correct.
_SP_COMBINE_CROSS = os.environ.get("NRL_SP_COMBINE", "1") not in ("0", "", "false", "False")
_SP_STREAM_POOL: List = []
_SP_STREAM_POOL_N = 4


def _sp_streams():
    if not _SP_STREAMS or not torch.cuda.is_available():
        return None
    if not _SP_STREAM_POOL:
        _SP_STREAM_POOL.extend(torch.cuda.Stream() for _ in range(_SP_STREAM_POOL_N))
    return _SP_STREAM_POOL


def _sp_fused_kv_gather_effective():
    return HAVE_TRITON and _SP_FUSED_KV_GATHER


def _sp_fused_backward_glue_effective():
    return HAVE_TRITON and _SP_FUSED_BACKWARD_GLUE


def _sp_fused_dq_assembly_effective(slot_pass):
    # The kernel has seven statically-unrolled slot arguments. Deeper trees must retain the
    # eager accumulator instead of tripping the kernel's assertion after an explicit opt-in.
    return HAVE_TRITON and _SP_FUSED_DQ_ASSEMBLY and len(slot_pass) <= 7


def _gather_kv(k, v, k_idx):
    """Fused K+V gather (one index read, both tensors) via Triton.

    K/V commonly arrive as strided views of Megatron's interleaved mixed-QKV projection, so the
    source strides must be explicit. Treating them as packed ``[total, ng, hn]`` tensors silently
    reads neighboring Q/K fields on cross-attention passes. The outputs are newly allocated and
    contiguous; the non-Triton path falls back to two stride-aware ``index_select`` calls.
    """
    if _sp_fused_kv_gather_effective():
        rows = k_idx.numel()
        ng, hn = k.shape[1], k.shape[2]
        kx = torch.empty(rows, ng, hn, dtype=k.dtype, device=k.device)
        vx = torch.empty(rows, ng, hn, dtype=v.dtype, device=v.device)
        _sp_gather_kv_kernel[(rows, ng)](
            k,
            v,
            k_idx,
            kx,
            vx,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            ng=ng,
            HN=hn,
        )
        return kx, vx
    return k.index_select(0, k_idx), v.index_select(0, k_idx)


def _sp_combine_effective():
    return _SP_COMBINE_CROSS and HAVE_TRITON and _SP_FUSED_MERGE


def _plan_key(node_start, node_len, node_parent, device, *, full_context: bool = False):
    return (
        tuple(int(x) for x in node_start),
        tuple(int(x) for x in node_len),
        tuple(int(x) for x in node_parent),
        str(device),
        _sp_combine_effective(),
        _SP_CHAINFIRST,
        _SP_QSLICE,
        full_context,
    )


def _validate_forest_dfs_preorder(node_start, node_len, node_parent):
    """Reject layouts whose node spans cannot represent a contiguous DFS-preorder forest."""
    ns = [int(x) for x in node_start]
    nl = [int(x) for x in node_len]
    par = [int(x) for x in node_parent]
    if len(ns) != len(nl) or len(ns) != len(par):
        raise ValueError("forest node_start, node_len, and node_parent must have equal lengths")

    cursor = 0
    for i, (start, length) in enumerate(zip(ns, nl)):
        if start != cursor:
            raise ValueError(
                f"forest node {i} start={start} != expected {cursor} "
                "(spans must be contiguous in array order)"
            )
        if length <= 0:
            raise ValueError(f"forest node {i} has non-positive length {length}")
        cursor += length

    stack: List[int] = []
    for i, parent in enumerate(par):
        if parent == -1:
            stack = [i]
            continue
        while stack and stack[-1] != parent:
            stack.pop()
        if not stack or stack[-1] != parent:
            raise ValueError(
                f"forest/tree layout is not DFS-preorder at node {i} (parent {parent}): the fused "
                "tree attention requires each node's subtree to be the contiguous run after it. "
                "Emit nodes in DFS preorder (parent, then each child's full subtree)."
            )
        stack.append(i)


def _forest_attention_plan_cached(
    node_start, node_len, node_parent, device, *, full_context: bool = False
):
    """Cached ``(total, passes, inv_maps)`` for a bin layout. The same layout is reused by every
    attention layer of the step (and often across steps), so the Python plan construction and the
    token->pass-row inverse maps (for the fused merge) are built once. ``inv_maps[p]`` is an int32
    ``[total]`` tensor mapping token -> its row in pass ``p`` (-1 if the token is not a query of
    that pass; pass 0 -- the self pass -- is the identity)."""
    _validate_forest_dfs_preorder(node_start, node_len, node_parent)
    key = _plan_key(node_start, node_len, node_parent, device, full_context=full_context)
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    plan = None
    if full_context:
        plan = _forest_attention_plan_full_context(node_start, node_len, node_parent, device)
    elif _SP_CHAINFIRST:
        plan = _forest_attention_plan_chainfirst(node_start, node_len, node_parent, device)
    if plan is None:
        plan = _forest_attention_plan(node_start, node_len, node_parent, device)
    total, passes = plan
    # never consolidate slice-form cross passes: combining would re-materialize the q gather
    # (and its backward scatter) that the _QSlice views exist to avoid.
    if (
        _sp_combine_effective()
        and len(passes) > 2
        and not any(isinstance(p[0], _QSlice) for p in passes)
    ):
        # Consolidate every per-depth-level cross pass into ONE flash_varlen call: varlen just
        # needs per-sequence contiguous q/k slices, and each (ancestor-span <- descendant-run)
        # pair is one sequence regardless of which level it came from. Any tree then costs
        # exactly 2 flash calls (self + cross) instead of depth+1 -- fewer launches, bigger
        # kernels. The merge/backward kernels are unchanged: they operate per SLOT (a token's
        # entry at one ancestor level), and cross slots simply share the combined pass's
        # output/LSE tensors with different row maps.
        self_pass = passes[0]
        qpos_parts, kpos_parts, cuq, cuk = [], [], [0], [0]
        level_row_start = []  # row offset of each level's block in the combined pass
        for q_idx, k_idx, cu_q, cu_k, _mxq, _mxk, _c in passes[1:]:
            level_row_start.append(cuq[-1])
            qpos_parts.append(q_idx)
            kpos_parts.append(k_idx)
            base_q, base_k = cuq[-1], cuk[-1]
            cuq.extend((cu_q[1:].to(torch.long) + base_q).tolist())
            cuk.extend((cu_k[1:].to(torch.long) + base_k).tolist())
        qpos = torch.cat(qpos_parts)
        kpos = torch.cat(kpos_parts)
        mxq = max(cuq[i + 1] - cuq[i] for i in range(len(cuq) - 1))
        mxk = max(cuk[i + 1] - cuk[i] for i in range(len(cuk) - 1))
        combined = (
            qpos,
            kpos,
            torch.tensor(cuq, dtype=torch.int32, device=device),
            torch.tensor(cuk, dtype=torch.int32, device=device),
            mxq,
            mxk,
            False,
        )
        # slots: slot 0 = self pass; slot j>=1 = the level-(j-1) rows INSIDE the combined pass.
        slot_pass = [0] + [1] * (len(passes) - 1)
        slot_inv = [torch.arange(total, dtype=torch.int32, device=device)]
        for li, (q_idx, *_rest) in enumerate(passes[1:]):
            inv = torch.full((total,), -1, dtype=torch.int32, device=device)
            inv[q_idx] = (
                torch.arange(q_idx.numel(), dtype=torch.int32, device=device) + level_row_start[li]
            )
            slot_inv.append(inv)
        passes = [self_pass, combined]
        identity = torch.arange(total, dtype=torch.long, device=device)
        qidx64 = [identity, qpos]
        entry = (total, passes, slot_inv, qidx64, slot_pass)
    else:
        inv_maps, qidx64 = [], []
        identity = torch.arange(total, dtype=torch.long, device=device)
        for q_idx, *_rest in passes:
            if q_idx is None:
                inv = torch.arange(total, dtype=torch.int32, device=device)
                qidx64.append(identity)
            elif isinstance(q_idx, _QSlice):
                # row r of the pass <-> token lo+r; gap tokens are padding rows, not queries.
                inv = torch.full((total,), -1, dtype=torch.int32, device=device)
                inv[q_idx.lo : q_idx.hi] = torch.arange(
                    q_idx.hi - q_idx.lo, dtype=torch.int32, device=device
                )
                for g0, g1 in q_idx.gaps:
                    inv[g0:g1] = -1
                qidx64.append(torch.arange(q_idx.lo, q_idx.hi, dtype=torch.long, device=device))
            else:
                inv = torch.full((total,), -1, dtype=torch.int32, device=device)
                inv[q_idx] = torch.arange(q_idx.numel(), dtype=torch.int32, device=device)
                qidx64.append(q_idx)
            inv_maps.append(inv)
        entry = (total, passes, inv_maps, qidx64, list(range(len(passes))))
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = entry
    return entry


def _merge_passes_triton(slot_pass, inv_maps, outs, lses, total, np_, hn, dtype, device):
    """One-kernel online-softmax merge across SLOTS. A slot is one attended-set contribution for
    a token (its own node, or one ancestor level); ``slot_pass[s]`` says which pass's output/LSE
    tensors slot ``s`` reads (with the consolidated cross pass, every cross slot shares pass 1's
    tensors under a different row map). Returns (o_merged [total, np, hn] in ``dtype``, lse_final
    fp32 [np, total])."""
    MAXP = 7
    n = len(slot_pass)
    assert n <= MAXP, f"fused merge supports <= {MAXP} slots (got {n}); deepen tl.static_range"
    outs = [o.contiguous() for o in outs]
    lses = [l.contiguous() for l in lses]
    o_args, l_args, i_args, r_args = [], [], [], []
    for s in range(MAXP):
        p = slot_pass[s] if s < n else slot_pass[0]
        o_args.append(outs[p])
        l_args.append(lses[p])
        i_args.append(inv_maps[s] if s < n else inv_maps[0])
        r_args.append(lses[p].shape[1])
    out = torch.empty(total, np_, hn, dtype=dtype, device=device)
    lse_final = torch.empty(np_, total, dtype=torch.float32, device=device)
    bt, wp = _SP_MERGE_BT, _SP_MERGE_WARPS
    _sp_merge_fwd_kernel[((total + bt - 1) // bt, np_)](
        *o_args,
        *l_args,
        *i_args,
        *r_args,
        out,
        lse_final,
        n,
        total,
        np_=np_,
        HN=hn,
        BLOCK_T=bt,
        num_warps=wp,
    )
    return out, lse_final


def _merge_dq_triton(slot_pass, inv_maps, dqxs, total, np_, hn, dtype, device):
    """dq final assembly across slots: dq[t] = sum over slots s of dqxs[slot_pass[s]][inv_s[t]].
    One kernel, dqx tensors read once, dq written once in ``dtype`` (no fp32 accumulator)."""
    MAXP = 7
    n = len(slot_pass)
    assert n <= MAXP
    d_args, i_args = [], []
    for s in range(MAXP):
        p = slot_pass[s] if s < n else slot_pass[0]
        d_args.append(dqxs[p])
        i_args.append(inv_maps[s] if s < n else inv_maps[0])
    dq = torch.empty(total, np_, hn, dtype=dtype, device=device)
    bt, wp = _SP_MERGE_BT, _SP_MERGE_WARPS
    _sp_dq_merge_kernel[((total + bt - 1) // bt, np_)](
        *d_args, *i_args, dq, n, total, np_=np_, HN=hn, BLOCK_T=bt, num_warps=wp
    )
    return dq


# Chain-first plan (NRL_SP_CHAINFIRST=1, default on, auto-fallback): when the layout emits each
# node's continuation child immediately after it (chain-first DFS), maximal parent-adjacent runs
# ("chains") behave as plain CAUSAL sequences — a chain token's causal prefix within the run is
# exactly its in-chain ancestors. The self pass then uses per-CHAIN (not per-node) causal
# sequences, absorbing all within-chain cross attention (for spine-dominated trees that deletes
# most cross rows). Each chain with ancestors ABOVE its head attends one contiguous k-range
# [path_start, chain_start) — fat, flash-friendly K instead of per-level skinny spans — packed
# into a single non-causal cross pass. Falls back to the per-level plan when any cross-needing
# chain's ancestor range is non-contiguous (e.g. interior non-first children).
_SP_CHAINFIRST = os.environ.get("NRL_SP_CHAINFIRST", "1") not in ("0", "", "false", "False")

# opt8: when the chain-first cross pass's query rows form one contiguous layout range (they do
# per tree: branches+siblings all sit after the spine), pass a zero-copy VIEW q[lo:hi] to flash
# instead of index_select (on branched_mc that gather+scatter round-trips ~176MB per fwd+bwd).
# Gaps between trees in multi-tree bins are covered by zero-length-K padding sequences: flash
# returns out=0 / dq=0 / LSE=+inf for those rows (probe-verified), the merge excludes them via
# inv=-1, and the backward scale kernel guards LSE=+inf -> weight 0.
_SP_QSLICE = os.environ.get("NRL_SP_QSLICE", "1") not in ("0", "", "false", "False")


class _QSlice:
    """Marker for a cross pass whose q rows are the contiguous token range [lo, hi) (row r of
    the pass <-> token lo+r), with ``gaps`` = token sub-ranges inside [lo, hi) that are only
    zero-K padding sequences (not real queries of the pass)."""

    __slots__ = ("lo", "hi", "gaps")

    def __init__(self, lo, hi, gaps):
        self.lo, self.hi, self.gaps = lo, hi, gaps

    def numel(self):
        """Return the number of query rows in the contiguous slice."""
        return self.hi - self.lo


def _sel_rows(t, q_idx):
    """Resolve a pass's q-row selector: None => identity, _QSlice => zero-copy view,
    tensor => gather."""
    if q_idx is None:
        return t
    if isinstance(q_idx, _QSlice):
        return t[q_idx.lo : q_idx.hi]
    return t.index_select(0, q_idx)


def _forest_attention_plan_full_context(node_start, node_len, node_parent, device):
    """Give each node its full ancestor KV path in one bottom-right causal pass.

    This trades repeated narrow GQA KV rows for removing partial-output rounding
    and the forward LSE merge. It preserves the existing backward's KV scatter.
    It does not emulate the original dense TE context-parallel arithmetic.
    """
    starts = [int(x) for x in node_start]
    lengths = [int(x) for x in node_len]
    parents = [int(x) for x in node_parent]
    kv_parts, cu_q, cu_k = [], [0], [0]
    for node, length in enumerate(lengths):
        if length == 0:
            continue
        path = []
        ancestor = node
        while ancestor != -1:
            path.append(ancestor)
            ancestor = parents[ancestor]
        path.reverse()
        kv_parts.extend(
            torch.arange(starts[i], starts[i] + lengths[i], device=device) for i in path
        )
        cu_q.append(cu_q[-1] + length)
        cu_k.append(cu_k[-1] + sum(lengths[i] for i in path))
    assert kv_parts, "empty forests must be handled before attention"
    maximum_q = max(b - a for a, b in pairwise(cu_q))
    maximum_k = max(b - a for a, b in pairwise(cu_k))
    return cu_q[-1], [
        (
            None,
            torch.cat(kv_parts),
            torch.tensor(cu_q, dtype=torch.int32, device=device),
            torch.tensor(cu_k, dtype=torch.int32, device=device),
            maximum_q,
            maximum_k,
            True,
        )
    ]


def _forest_attention_plan_chainfirst(node_start, node_len, node_parent, device):
    """Hybrid chain decomposition (opt9; supersedes both the pure chain-first plan and the
    per-level fallback). Always applicable:

    - SELF pass: one CAUSAL sequence per maximal parent-adjacent chain. A chain is a pure path
      (only one child can start at its parent's end), so a chain token's causal prefix is exactly
      its in-chain ancestors — correct for any DFS-preorder layout.
    - FAT cross pass: each chain attends the maximal CONTIGUOUS prefix of its head's ancestor
      path (walking down from the tree root while spans stay adjacent) — one fat-K sequence.
      Chain-first layouts have fully contiguous ancestor paths, so this is their only cross pass.
    - SKINNY cross passes: ancestors after the contiguity break (e.g. interior non-first children
      in balanced trees) get one per-ancestor sequence each, grouped by break-index so every q row
      appears at most once per pass (the merge-slot invariant)."""
    ns = [int(x) for x in node_start]
    nl = [int(x) for x in node_len]
    par = [int(x) for x in node_parent]
    N = len(ns)
    total = max((ns[i] + nl[i] for i in range(N)), default=0)

    adj = [par[i] != -1 and ns[i] == ns[par[i]] + nl[par[i]] for i in range(N)]
    chain_head = list(range(N))
    for i in range(N):
        if adj[i]:
            chain_head[i] = chain_head[par[i]]

    heads = sorted(set(chain_head))
    chain_end = {}
    for i in range(N):
        h = chain_head[i]
        chain_end[h] = max(chain_end.get(h, 0), ns[i] + nl[i])

    fat = []  # (q0, q1, a0, a1): contiguous ancestor-prefix sequences
    skinny = {}  # break-index -> list of (q0, q1, s0, s1) single-ancestor sequences
    for h in heads:
        if par[h] == -1:
            continue
        path, x = [], par[h]
        while x != -1:
            path.append(x)
            x = par[x]
        path.reverse()  # tree root first
        y0 = ns[path[0]]
        y = y0 + nl[path[0]]
        i = 1
        while i < len(path) and ns[path[i]] == y:
            y += nl[path[i]]
            i += 1
        q0, q1 = ns[h], chain_end[h]
        fat.append((q0, q1, y0, y))
        for j, a in enumerate(path[i:]):
            skinny.setdefault(j, []).append((q0, q1, ns[a], ns[a] + nl[a]))

    def _i32(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    def _i64(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    passes = []
    # self pass: one causal sequence per CHAIN (q/k identity over [0, total)).
    cu = [0]
    for h in heads:
        cu.append(cu[-1] + (chain_end[h] - ns[h]))
    mx = max((chain_end[h] - ns[h] for h in heads), default=0)
    passes.append((None, None, _i32(cu), _i32(cu), mx, mx, True))

    def _build_cross(seqs):
        seqs.sort(key=lambda c: c[0])
        if _SP_QSLICE:
            # contiguous q view [qlo, qhi): real sequences + zero-K padding over the gaps.
            qlo, qhi = seqs[0][0], seqs[-1][1]
            kpos, cuq, cuk, gaps = [], [0], [0], []
            cur, mxq = qlo, 0
            for q0, q1, a0, a1 in seqs:
                if q0 > cur:  # gap: rows exist in the view but attend nothing
                    gaps.append((cur, q0))
                    cuq.append(cuq[-1] + (q0 - cur))
                    cuk.append(cuk[-1])
                    mxq = max(mxq, q0 - cur)
                kpos.append(torch.arange(a0, a1, dtype=torch.long, device=device))
                cuq.append(cuq[-1] + (q1 - q0))
                cuk.append(cuk[-1] + (a1 - a0))
                mxq = max(mxq, q1 - q0)
                cur = q1
            mxk = max(c[3] - c[2] for c in seqs)
            return (_QSlice(qlo, qhi, gaps), torch.cat(kpos), _i32(cuq), _i32(cuk), mxq, mxk, False)
        qpos, kpos, cuq, cuk = [], [], [0], [0]
        for q0, q1, a0, a1 in seqs:
            qpos.append(torch.arange(q0, q1, dtype=torch.long, device=device))
            kpos.append(torch.arange(a0, a1, dtype=torch.long, device=device))
            cuq.append(cuq[-1] + (q1 - q0))
            cuk.append(cuk[-1] + (a1 - a0))
        mxq = max(c[1] - c[0] for c in seqs)
        mxk = max(c[3] - c[2] for c in seqs)
        return (torch.cat(qpos), torch.cat(kpos), _i32(cuq), _i32(cuk), mxq, mxk, False)

    if fat:
        passes.append(_build_cross(fat))
    for j in sorted(skinny):
        passes.append(_build_cross(skinny[j]))
    return total, passes


def _forest_attention_plan(node_start, node_len, node_parent, device):
    """Decompose a forest/tree into the flash passes the composed attention runs.

    Returns ``(total, passes)`` where ``passes`` is a list of
    ``(q_idx, k_idx, cu_q, cu_k, max_q, max_k, causal)``:
      * one SELF pass -- every node attends its own span causally (block-diagonal varlen over all
        tokens), and
      * one CROSS pass per ancestor depth level L -- tokens strictly below depth L attend their
        level-L ancestor span, non-causally (DFS contiguity makes each ancestor's descendant tokens
        a contiguous run).
    A token attends ``{own node, causal} ∪ {each ancestor node, full}`` -- the union of the passes
    it
    appears in as a query. ``max_depth + 1`` passes total, independent of group count: depth-1
    forest
    (stars) ⇒ self + 1 cross; arbitrary-depth trees ⇒ ``depth + 1``. ``node_*`` give the structure
    (parents precede children; a subtree is a contiguous DFS run).
    """
    ns = [int(x) for x in node_start]
    nl = [int(x) for x in node_len]
    par = [int(x) for x in node_parent]
    N = len(ns)
    total = max((ns[i] + nl[i] for i in range(N)), default=0)

    # PRECONDITION: node arrays must be DFS-preorder (each node's subtree is the contiguous run
    # immediately after it). ``subtree_end`` below relies on this -- a non-DFS layout would silently
    # attend the WRONG ancestor spans (branches lose interior-ancestor context -> corrupt logprobs).
    # PackedTreeLayout only checks contiguity + parent<i, which do NOT imply DFS; validate here
    # (O(N),
    # negligible vs attention) so a mis-ordered layout fails loudly instead of training on garbage.
    _validate_forest_dfs_preorder(ns, nl, par)

    depth = [0] * N
    for i in range(N):
        depth[i] = 0 if par[i] == -1 else depth[par[i]] + 1
    d_max = max(depth, default=-1)
    subtree_end = [ns[i] + nl[i] for i in range(N)]
    for i in range(N):
        j = i + 1
        while j < N and depth[j] > depth[i]:
            subtree_end[i] = ns[j] + nl[j]
            j += 1

    def _i32(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    def _i64(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    passes = []
    # self pass: one block-diagonal causal varlen over every node's own span. Its q/k indices are
    # the
    # identity over [0, total), so they are left as ``None`` -- the kernel then uses q/k/v
    # directly and
    # skips a full-tensor gather/scatter every forward and backward (a real cost on big packed
    # bins).
    cu = [0]
    for i in range(N):
        cu.append(cu[-1] + nl[i])
    mnl = max(nl, default=0)
    passes.append((None, None, _i32(cu), _i32(cu), mnl, mnl, True))

    # cross passes: one per ancestor depth level.
    for L in range(d_max):
        qpos: List[int] = []
        kpos: List[int] = []
        cuq, cuk = [0], [0]
        for a in range(N):
            if depth[a] != L:
                continue
            qs, qe = (ns[a] + nl[a], subtree_end[a])  # strict descendants of a (contiguous, DFS)
            if qe <= qs:
                continue
            qpos.extend(range(qs, qe))
            kpos.extend(range(ns[a], ns[a] + nl[a]))
            cuq.append(cuq[-1] + (qe - qs))
            cuk.append(cuk[-1] + nl[a])
        if not qpos:
            continue
        mxq = max(cuq[i + 1] - cuq[i] for i in range(len(cuq) - 1))
        mxk = max(cuk[i + 1] - cuk[i] for i in range(len(cuk) - 1))
        passes.append((_i64(qpos), _i64(kpos), _i32(cuq), _i32(cuk), mxq, mxk, False))
    return total, passes


class _ComposedForestAttn(torch.autograd.Function):
    """Composed forest/tree attention with an EXACT backward.

    The forward runs the ``_forest_attention_plan`` passes with flash and merges them by online
    softmax (LSE) -- the union-softmax identity, so the forward is exact. The backward is the
    delicate part: each pass's flash output is a sub-attention, and the *naive*
    autograd-through-flash
    drops the inter-pass normalizer-coupling term (flash exposes no gradient through its LSE),
    giving
    ~15% wrong q/k grads. We fix it by calling the low-level ``_flash_attn_varlen_backward`` per
    pass
    with ``dout = w_pass * do`` AND substituting the MERGED output ``o`` for the pass's own output:
    flash uses ``out`` only to form the row-delta ``D = rowsum(dout ∘ out)``, which is exactly the
    softmax ``G`` term, so feeding the merged ``o`` injects the global normalizer that was missing.
    The result is the exact union-softmax score gradient ``P_ij (v_j·do − o·do)`` for every pass --
    dq, dk, dv all correct -- with no custom kernel (see TREE_PACKING_DESIGN.md §5)."""

    @staticmethod
    def forward(ctx, q, k, v, node_start, node_len, node_parent, scale, full_context):
        # q: [total, np, hn]; k, v: [total, ng, hn]; scale already resolved to a float.
        """Evaluate independent causal and ancestor passes, then merge their softmax outputs."""
        from flash_attn import flash_attn_varlen_func

        total, passes, inv_maps, qidx64, slot_pass = _forest_attention_plan_cached(
            node_start, node_len, node_parent, q.device, full_context=full_context
        )
        np_, hn = q.shape[1], q.shape[2]
        streams = _sp_streams()
        outs, lses = [None] * len(passes), [None] * len(passes)
        if streams is not None and len(passes) > 1:
            # passes are independent until the merge: fan them out on side streams so the small
            # cross passes hide under the big self pass. Their outputs are consumed back on the
            # current stream after the join events (record_stream keeps the allocator honest).
            cur = torch.cuda.current_stream()
            ev_in = torch.cuda.Event()
            ev_in.record(cur)
            join = []
            for i, (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal) in enumerate(passes):
                st = streams[i % len(streams)]
                st.wait_event(ev_in)
                with torch.cuda.stream(st):
                    qx = _sel_rows(q, q_idx)
                    if k_idx is None:
                        kx, vx = k, v
                    else:
                        kx, vx = _gather_kv(k, v, k_idx)
                    o, lse, _ = flash_attn_varlen_func(
                        qx,
                        kx,
                        vx,
                        cu_q,
                        cu_k,
                        mxq,
                        mxk,
                        softmax_scale=scale,
                        causal=causal,
                        return_attn_probs=True,
                    )
                    o.record_stream(cur)
                    lse.record_stream(cur)
                    ev = torch.cuda.Event()
                    ev.record(st)
                    join.append(ev)
                outs[i], lses[i] = o, lse
            for ev in join:
                cur.wait_event(ev)
        else:
            for i, (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal) in enumerate(passes):
                qx = _sel_rows(q, q_idx)  # None => identity (self pass)
                if k_idx is None:
                    kx, vx = k, v
                else:
                    kx, vx = _gather_kv(k, v, k_idx)
                o, lse, _ = flash_attn_varlen_func(
                    qx,
                    kx,
                    vx,
                    cu_q,
                    cu_k,
                    mxq,
                    mxk,
                    softmax_scale=scale,
                    causal=causal,
                    return_attn_probs=True,
                )  # o [Σq, np, hn], lse [np, Σq]
                outs[i], lses[i] = o, lse

        if _SP_FUSED_MERGE and HAVE_TRITON and len(slot_pass) <= 7:
            # single-kernel online-softmax merge: reads each pass output/LSE once, writes the
            # merged output + final LSE once (fp32 in-register), replacing the eager fp32
            # upcast/mul/index_add round-trips below.
            o_merged, lse_final = _merge_passes_triton(
                slot_pass, inv_maps, outs, lses, total, np_, hn, q.dtype, q.device
            )
        else:
            # merged LSE per (head, token): logsumexp over every pass the token queries in.
            lse_final = torch.full(
                (np_, total), float("-inf"), device=q.device, dtype=torch.float32
            )
            for (q_idx, *_), lse in zip(passes, lses):
                ls = lse.float()
                if isinstance(q_idx, _QSlice):
                    # zero-K padding rows report LSE=+inf: neutralize before merging.
                    ls = torch.where(torch.isinf(ls), torch.full_like(ls, float("-inf")), ls)
                    lse_final[:, q_idx.lo : q_idx.hi] = torch.logaddexp(
                        lse_final[:, q_idx.lo : q_idx.hi], ls
                    )
                elif q_idx is None:
                    lse_final = torch.logaddexp(lse_final, ls)
                else:
                    lse_final[:, q_idx] = torch.logaddexp(lse_final[:, q_idx], ls)
            # merged output: sum_pass w_pass * o_pass, w_pass = exp(lse_pass - lse_final).
            o_merged = torch.zeros(total, np_, hn, device=q.device, dtype=torch.float32)
            for (q_idx, *_), o, lse in zip(passes, outs, lses):
                ls = lse.float()
                if isinstance(q_idx, _QSlice):
                    ls = torch.where(torch.isinf(ls), torch.full_like(ls, float("-inf")), ls)
                    lf = lse_final[:, q_idx.lo : q_idx.hi]
                    contrib = torch.exp(ls - lf).transpose(0, 1).unsqueeze(-1) * o.float()
                    o_merged[q_idx.lo : q_idx.hi] += contrib
                    continue
                lf = lse_final if q_idx is None else lse_final.index_select(1, q_idx)
                contrib = torch.exp(ls - lf).transpose(0, 1).unsqueeze(-1) * o.float()
                if q_idx is None:
                    o_merged = o_merged + contrib
                else:
                    o_merged.index_add_(0, q_idx, contrib)
            o_merged = o_merged.to(q.dtype)

        ctx.save_for_backward(q, k, v, o_merged)
        ctx.passes = passes
        ctx.lses = lses
        ctx.lse_final = lse_final
        ctx.scale = scale
        ctx.qidx64 = qidx64
        ctx.inv_maps = inv_maps
        ctx.slot_pass = slot_pass
        return o_merged

    @staticmethod
    def backward(ctx, do):
        """Accumulate pass gradients using the merged output for the global softmax correction."""
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

        q, k, v, o_merged = ctx.saved_tensors
        lse_final, scale = ctx.lse_final, ctx.scale
        do = do.contiguous()
        total, np_, hn = q.shape[0], q.shape[1], q.shape[2]
        have_cached_row_maps = getattr(ctx, "qidx64", None) is not None
        # Fused glue initializes KV gradients from an identity self pass. A full-context
        # plan gathers repeated ancestor KV in its first pass and needs the general scatter.
        use_backward_glue = (
            _sp_fused_backward_glue_effective()
            and have_cached_row_maps
            and ctx.passes[0][1] is None
        )
        use_dq_assembly = _sp_fused_dq_assembly_effective(ctx.slot_pass) and have_cached_row_maps

        if use_backward_glue:
            # Fused backward glue: (a) per-pass dout scaling w*do fused with the query gather in
            # one kernel (the eager path materialized an fp32 exp/mul chain + an index_select per
            # pass); (b) the self pass (always pass 0, identity indices over all tokens) INITIALIZES
            # the fp32 accumulators instead of zeros+add; cross passes scatter-accumulate through a
            # cast-fused kernel (no per-pass .float() temporaries). The flash calls are unchanged
            # -- the exact-backward trick (merged o substituted for the pass output) is preserved.
            ng = k.shape[1]
            streams = _sp_streams()
            cur = torch.cuda.current_stream()
            results = [None] * len(ctx.passes)
            join = [None] * len(ctx.passes)
            ev_in = None
            if streams is not None and len(ctx.passes) > 1:
                ev_in = torch.cuda.Event()
                ev_in.record(cur)
            for i, ((q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal), lse) in enumerate(
                zip(ctx.passes, ctx.lses)
            ):
                st = None if ev_in is None else streams[i % len(streams)]
                stream_ctx = torch.cuda.stream(st) if st is not None else nullcontext()
                if st is not None:
                    st.wait_event(ev_in)
                with stream_ctx:
                    qidx = ctx.qidx64[i]
                    rows = qidx.numel()
                    qx = _sel_rows(q, q_idx)
                    if k_idx is None:
                        kx, vx = k, v
                    else:
                        kx, vx = _gather_kv(k, v, k_idx)
                    ox = _sel_rows(o_merged, q_idx)
                    dox = torch.empty(rows, np_, hn, dtype=q.dtype, device=q.device)
                    _sp_scale_gather_kernel[((rows + 15) // 16, np_)](
                        do, lse.contiguous(), lse_final, qidx, dox, rows, total, np_=np_, HN=hn
                    )
                    dqx, dkx, dvx = (
                        torch.empty_like(qx),
                        torch.empty_like(kx),
                        torch.empty_like(vx),
                    )
                    _flash_attn_varlen_backward(
                        dox,
                        qx,
                        kx,
                        vx,
                        ox,
                        lse,
                        dqx,
                        dkx,
                        dvx,
                        cu_q,
                        cu_k,
                        mxq,
                        mxk,
                        0.0,
                        scale,
                        causal,
                        -1,
                        -1,
                        0.0,
                        None,
                        _SP_DETERMINISTIC_BACKWARD,
                        None,
                        False,
                    )
                    results[i] = (dqx, dkx, dvx, qidx, k_idx, rows, q_idx)
                    if st is not None:
                        dqx.record_stream(cur)
                        dkx.record_stream(cur)
                        dvx.record_stream(cur)
                        ev = torch.cuda.Event()
                        ev.record(st)
                        join[i] = ev
            # k/v accumulate in pass order on the current stream (scatters are read-modify-write
            # on shared fp32 accumulators, ng=8 so the buffers are small); dq is assembled by one
            # gather-side merge kernel over all pass contributions (opt10) once every pass joins.
            dk = dv = None
            dqxs = [None] * len(ctx.passes)
            for i, res in enumerate(results):
                if join[i] is not None:
                    cur.wait_event(join[i])
                dqx, dkx, dvx, qidx, k_idx, rows, q_idx = res
                dqxs[i] = dqx
                if i == 0:
                    # self pass: identity over all tokens -> direct init, no zeros/scatter.
                    dk = dkx.float()
                    dv = dvx.float()
                else:
                    kro = k_idx.numel()
                    _sp_scatter_accum_kernel[((kro + 15) // 16, ng)](
                        dk, dkx, k_idx, kro, n_=ng, HN=hn
                    )
                    _sp_scatter_accum_kernel[((kro + 15) // 16, ng)](
                        dv, dvx, k_idx, kro, n_=ng, HN=hn
                    )
            if use_dq_assembly:
                dq = _merge_dq_triton(
                    ctx.slot_pass, ctx.inv_maps, dqxs, total, np_, hn, q.dtype, q.device
                )
            else:
                dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
                for dqx, result in zip(dqxs, results):
                    q_idx = result[-1]
                    if q_idx is None:
                        dq += dqx.float()
                    elif isinstance(q_idx, _QSlice):
                        dq[q_idx.lo : q_idx.hi] += dqx.float()
                    else:
                        dq.index_add_(0, q_idx, dqx.float())
                dq = dq.to(q.dtype)
            return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None

        dq = None if use_dq_assembly else torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dqxs = [] if use_dq_assembly else None
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        for (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal), lse in zip(ctx.passes, ctx.lses):
            qx = _sel_rows(q, q_idx)  # q_idx None => identity (self pass)
            if k_idx is None:
                kx, vx = k, v
            else:
                kx, vx = _gather_kv(k, v, k_idx)
            ox = _sel_rows(o_merged, q_idx)  # MERGED output -> exact
            if isinstance(q_idx, _QSlice):
                lf = lse_final[:, q_idx.lo : q_idx.hi]
                dox_full = do[q_idx.lo : q_idx.hi]
                ls = lse.float()
                ls = torch.where(torch.isinf(ls), torch.full_like(ls, float("-inf")), ls)
                w = torch.exp(ls - lf)  # 0 on zero-K padding rows
            else:
                lf = lse_final if q_idx is None else lse_final.index_select(1, q_idx)
                dox_full = do if q_idx is None else do.index_select(0, q_idx)
                w = torch.exp(lse.float() - lf)
            dox = (w.transpose(0, 1).unsqueeze(-1) * dox_full).to(q.dtype)
            dqx, dkx, dvx = (torch.empty_like(qx), torch.empty_like(kx), torch.empty_like(vx))
            _flash_attn_varlen_backward(
                dox,
                qx,
                kx,
                vx,
                ox,
                lse,
                dqx,
                dkx,
                dvx,
                cu_q,
                cu_k,
                mxq,
                mxk,
                0.0,
                scale,
                causal,
                -1,
                -1,
                0.0,
                None,
                _SP_DETERMINISTIC_BACKWARD,
                None,
                False,
            )
            if use_dq_assembly:
                dqxs.append(dqx)
            else:
                if q_idx is None:
                    dq += dqx.float()
                elif isinstance(q_idx, _QSlice):
                    dq[q_idx.lo : q_idx.hi] += dqx.float()
                else:
                    dq.index_add_(0, q_idx, dqx.float())
            if k_idx is None:
                dk += dkx.float()
                dv += dvx.float()
            else:
                dk.index_add_(0, k_idx, dkx.float())
                dv.index_add_(0, k_idx, dvx.float())
        if use_dq_assembly:
            dq = _merge_dq_triton(
                ctx.slot_pass, ctx.inv_maps, dqxs, total, np_, hn, q.dtype, q.device
            )
        else:
            dq = dq.to(q.dtype)
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


def flash_composed_forest_attention_fused(
    query, key, value, node_start, node_len, node_parent, scale=None, *, full_context: bool = False
):
    """Level-decomposed forest/tree attention, fused to ``max_depth + 1`` flash passes per bin.

    A token attends ``{its own node, causally} ∪ {each ancestor node, fully}``. Decomposed by
    ancestor
    depth level (see :func:`_forest_attention_plan`) and merged by online softmax, so cost is
    independent of group count (depth-1 stars ⇒ 1 self + 1 cross pass) -- vs the per-group loop's
    ``~3 * #groups`` launches. Forward AND backward are exact via :class:`_ComposedForestAttn`.

    ``query`` is ``[sq, b, np, hn]`` and ``key``/``value`` ``[sq, b, ng, hn]`` (b == 1). Trailing
    pad
    positions (beyond the last node) get zero outputs and are never read downstream. Returns
    ``[sq, b, np*hn]``; same default scale (1/sqrt(hn)) as the flex/loop paths.
    """
    sq, b, np_, hn = query.shape
    assert b == 1, "shared-prefix packing uses a single packed sequence (b == 1)"
    total = max((int(s) + int(l) for s, l in zip(node_start, node_len)), default=0)
    scale = scale if scale is not None else hn**-0.5
    out = _ComposedForestAttn.apply(
        query[:total, 0],
        key[:total, 0],
        value[:total, 0],
        node_start,
        node_len,
        node_parent,
        scale,
        full_context,
    )  # [total, np, hn]
    if sq > total:
        out = torch.cat([out, out.new_zeros(sq - total, np_, hn)], dim=0)
    return out.reshape(sq, 1, np_ * hn).contiguous()


def _forest_to_nodes(forest):
    """Expand a depth-1 ``forest`` list into flat node arrays (PackedTreeLayout structure).

    ``forest`` is ``[(token_offset, prefix_len, completion_lens), ...]``. Each group becomes a root
    node (the prefix span, parent -1) followed by one child node per completion. Node order is DFS
    (root then its children), as the fused kernel requires.
    """
    node_start, node_len, node_parent = [], [], []
    for off, prefix_len, completion_lens in forest:
        root = len(node_start)
        node_start.append(int(off))
        node_len.append(int(prefix_len))
        node_parent.append(-1)
        pos = int(off) + int(prefix_len)
        for c in completion_lens:
            node_start.append(pos)
            node_len.append(int(c))
            node_parent.append(root)
            pos += int(c)
    return node_start, node_len, node_parent


def flash_composed_forest_attention(
    query, key, value, forest, scale=None, *, full_context: bool = False
):
    """Forest (multi-group / Case-1) shared-prefix attention -- fused, level-decomposed.

    ``forest`` is a list of ``(token_offset, prefix_len, completion_lens)``, one per group packed
    into this bin. Expanded to flat node arrays and dispatched to
    :func:`flash_composed_forest_attention_fused`, which fuses ALL groups into ``max_depth + 1``
    flash calls (depth-1 forest -> 1 self + 1 cross + merge) regardless of group count -- vs the
    old per-group loop's ``~3 * #groups`` launches, which scaled badly with many small groups.
    See :func:`flash_composed_forest_attention_loop` for the reference per-group form (kept for
    parity testing).
    """
    node_start, node_len, node_parent = _forest_to_nodes(forest)
    return flash_composed_forest_attention_fused(
        query, key, value, node_start, node_len, node_parent, scale=scale, full_context=full_context
    )


def _undo_cp_zigzag(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Convert rank-major CP zigzag chunks into canonical global token order.

    Standard context parallelism gives rank ``r`` chunks ``(r, 2*C-r-1)``.  An
    all-to-all from sequence to head parallelism concatenates those rank-local pairs in
    rank order, i.e. ``0,last,1,last-1,...``.  Mamba uses the same permutation before
    its sequential scan; forest attention needs it for its global node offsets too.
    """
    if cp_size < 1 or input_.shape[0] % (2 * cp_size):
        raise ValueError("CP zigzag sequence length must be divisible by 2 * CP size")
    chunks = torch.chunk(input_, chunks=2 * cp_size, dim=0)
    order = [2 * index for index in range(cp_size)] + [
        2 * cp_size - 2 * index - 1 for index in range(cp_size)
    ]
    return torch.cat([chunks[index] for index in order], dim=0)


def _redo_cp_zigzag(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Convert canonical global token order back to rank-major CP zigzag chunks."""
    if cp_size < 1 or input_.shape[0] % (2 * cp_size):
        raise ValueError("CP zigzag sequence length must be divisible by 2 * CP size")
    chunks = torch.chunk(input_, chunks=2 * cp_size, dim=0)
    order = [None] * (2 * cp_size)
    order[::2] = range(cp_size)
    order[1::2] = reversed(range(cp_size, 2 * cp_size))
    return torch.cat([chunks[index] for index in order], dim=0)


def _cp_local_kv_head_slice(query_heads: int, kv_heads: int, cp_size: int, cp_rank: int) -> slice:
    """Return the global KV-head span serving one CP rank's contiguous Q-head span.

    The fused flash primitive supports GQA, but its local Q-to-KV grouping must exactly
    match the global grouping after Q heads are sharded over CP.  Some valid layouts
    (including 32 Q heads / 2 KV heads / CP4) replicate a single KV head on multiple CP
    ranks.  Layouts whose CP boundary cuts unequal pieces from several KV groups fail
    closed rather than silently changing attention semantics.
    """
    if min(query_heads, kv_heads, cp_size) < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(
            f"invalid shared-prefix CP head geometry: {query_heads=}, {kv_heads=}, "
            f"{cp_size=}, {cp_rank=}"
        )
    if query_heads % cp_size:
        raise NotImplementedError(
            f"shared-prefix CP attention requires {query_heads=} divisible by {cp_size=}"
        )
    if query_heads % kv_heads:
        raise NotImplementedError(
            "shared-prefix CP attention requires an integral global Q/KV head ratio"
        )

    local_query_heads = query_heads // cp_size
    queries_per_kv = query_heads // kv_heads
    query_start = cp_rank * local_query_heads
    global_mapping = [
        query_index // queries_per_kv
        for query_index in range(query_start, query_start + local_query_heads)
    ]
    kv_start = global_mapping[0]
    kv_stop = global_mapping[-1] + 1
    local_kv_heads = kv_stop - kv_start
    if local_query_heads % local_kv_heads:
        raise NotImplementedError(
            "shared-prefix CP attention cannot express this Q/KV grouping with local flash GQA"
        )
    local_queries_per_kv = local_query_heads // local_kv_heads
    expected_mapping = [
        kv_start + local_index // local_queries_per_kv for local_index in range(local_query_heads)
    ]
    if global_mapping != expected_mapping:
        raise NotImplementedError(
            "shared-prefix CP attention boundary cuts unequal portions of multiple KV groups"
        )
    return slice(kv_start, kv_stop)


def _cp_kv_head_slices_for_destinations(
    query_heads: int, kv_heads: int, cp_size: int
) -> tuple[slice, ...]:
    """Return equal-width KV-head blocks in all-to-all destination order.

    Each destination owns one contiguous Q-head block.  Its KV block can overlap
    another destination's block for GQA, but the equal-split all-to-all requires
    every destination block to contain the same number of heads.
    """
    destination_slices = tuple(
        _cp_local_kv_head_slice(query_heads, kv_heads, cp_size, cp_rank)
        for cp_rank in range(cp_size)
    )
    destination_widths = tuple(
        head_slice.stop - head_slice.start for head_slice in destination_slices
    )
    if len(set(destination_widths)) != 1:
        raise NotImplementedError(
            "shared-prefix CP attention requires equal KV-head widths for every "
            f"all-to-all destination; got {destination_widths}"
        )
    return destination_slices


def _cp_pack_destination_head_slices(
    input_: torch.Tensor, destination_slices: tuple[slice, ...]
) -> torch.Tensor:
    """Pack destination-specific head blocks before an equal-split all-to-all.

    Overlapping slices are intentionally concatenated more than once.  Autograd
    accumulates their backward contributions into the shared source heads, which
    preserves GQA KV-gradient multiplicity without materializing every KV head on
    every destination.
    """
    if not destination_slices:
        raise ValueError("shared-prefix CP attention requires at least one destination")
    head_count = input_.shape[2]
    widths = []
    for head_slice in destination_slices:
        if (
            head_slice.step not in (None, 1)
            or head_slice.start is None
            or head_slice.stop is None
            or not 0 <= head_slice.start < head_slice.stop <= head_count
        ):
            raise ValueError(
                "shared-prefix CP attention destination slices must be non-empty, "
                "unit-stride, and within the KV-head dimension"
            )
        widths.append(head_slice.stop - head_slice.start)
    if len(set(widths)) != 1:
        raise NotImplementedError(
            "shared-prefix CP attention requires equal KV-head widths for every "
            f"all-to-all destination; got {tuple(widths)}"
        )
    return torch.cat([input_[:, :, head_slice, :] for head_slice in destination_slices], dim=2)


def _cp_sequence_to_head_parallel(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    *,
    destination_head_slices: tuple[slice, ...] | None = None,
) -> torch.Tensor:
    """Change ``[S/C,1,H,D]`` zigzag sequence shards to canonical head shards."""
    if input_.ndim != 4 or input_.shape[1] != 1:
        raise ValueError("shared-prefix CP attention requires Q/K/V shape [S/C,1,H,D]")
    cp_size = cp_group.size()
    if input_.shape[0] % 2:
        raise ValueError(
            "shared-prefix CP attention requires an even local sequence length for zigzag CP"
        )
    if destination_head_slices is not None:
        if len(destination_head_slices) != cp_size:
            raise ValueError(
                "shared-prefix CP attention requires one head slice per all-to-all destination"
            )
        input_ = _cp_pack_destination_head_slices(input_, destination_head_slices)
    elif input_.shape[2] % cp_size:
        raise NotImplementedError(
            "shared-prefix CP attention requires Q heads divisible by context parallel size"
        )

    local_sequence, batch, heads, head_dim = input_.shape
    exchanged = all_to_all_sp2hp(
        input_.reshape(local_sequence, batch, heads * head_dim), group=cp_group
    )
    global_sequence = local_sequence * cp_size
    local_heads = heads // cp_size
    exchanged = exchanged.reshape(global_sequence, batch, local_heads, head_dim)
    return _undo_cp_zigzag(exchanged, cp_size)


def _cp_head_to_sequence_parallel(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Invert :func:`_cp_sequence_to_head_parallel` for attention output."""
    if input_.ndim != 4 or input_.shape[1] != 1:
        raise ValueError("shared-prefix CP attention output must have shape [S,1,H/C,D]")
    cp_size = cp_group.size()
    global_sequence, batch, local_heads, head_dim = input_.shape
    if global_sequence % (2 * cp_size):
        raise ValueError("shared-prefix CP global sequence length must be divisible by 2 * CP size")
    rank_major = _redo_cp_zigzag(input_, cp_size)
    exchanged = all_to_all_hp2sp(
        rank_major.reshape(global_sequence, batch, local_heads * head_dim), group=cp_group
    )
    return exchanged.reshape(global_sequence // cp_size, batch, local_heads * cp_size, head_dim)


def flash_composed_forest_attention_cp(
    query,
    key,
    value,
    forest,
    *,
    cp_group: torch.distributed.ProcessGroup,
    scale=None,
    full_context: bool = False,
):
    """Exact forest attention for standard zigzag context-parallel sequence shards.

    Q heads are sharded across CP while each destination receives only its required
    K/V heads. Overlapping GQA slices remain differentiably replicated. All tensors
    are converted to canonical global sequence order, then the existing optimized
    exact-forward/exact-backward forest primitive runs on the local head shard. Its
    output is transformed back to the caller's CP-local zigzag token order. No global
    logits, hidden states, or full-KV activation bases are materialized.
    """
    cp_size = cp_group.size()
    if cp_size == 1:
        return flash_composed_forest_attention(
            query, key, value, forest, scale=scale, full_context=full_context
        )
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("shared-prefix CP attention requires Q/K/V rank-4 tensors")
    if any(tensor.shape[1] != 1 for tensor in (query, key, value)):
        raise ValueError("shared-prefix CP attention requires Q/K/V batch size 1")
    if not query.shape[0] == key.shape[0] == value.shape[0]:
        raise ValueError("shared-prefix CP attention requires aligned local Q/K/V sequences")
    if key.shape[2] != value.shape[2]:
        raise ValueError("shared-prefix CP attention requires equal K/V head counts")
    if not query.shape[3] == key.shape[3] == value.shape[3]:
        raise ValueError("shared-prefix CP attention requires equal Q/K/V head dimensions")
    if not query.dtype == key.dtype == value.dtype:
        raise ValueError("shared-prefix CP attention requires equal Q/K/V dtypes")
    if not query.device == key.device == value.device:
        raise ValueError("shared-prefix CP attention requires Q/K/V on the same device")

    query_heads = query.shape[2]
    kv_heads = key.shape[2]
    destination_kv_slices = _cp_kv_head_slices_for_destinations(query_heads, kv_heads, cp_size)
    query_global = _cp_sequence_to_head_parallel(query, cp_group)
    key_global = _cp_sequence_to_head_parallel(
        key, cp_group, destination_head_slices=destination_kv_slices
    )
    value_global = _cp_sequence_to_head_parallel(
        value, cp_group, destination_head_slices=destination_kv_slices
    )

    output = flash_composed_forest_attention(
        query_global, key_global, value_global, forest, scale=scale, full_context=full_context
    )
    output = output.reshape(query_global.shape[0], 1, query_global.shape[2], query_global.shape[3])
    output = _cp_head_to_sequence_parallel(output, cp_group)
    return output.reshape(output.shape[0], 1, -1).contiguous()
