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

Env knobs: NRL_SP_CHAINFIRST (hybrid chain plan), NRL_SP_QSLICE (zero-copy q views),
and NRL_SP_STREAMS (stream overlap). Unqualified Triton KV gather, backward glue,
and dQ assembly are deferred; their former opt-ins fail explicitly.
NRL_SP_DETERMINISTIC_BACKWARD is a default-off diagnostic that selects FlashAttention's
deterministic backward. The eager LSE merge is used in production. The legacy NRL_SP_FUSED_MERGE opt-in fails
closed because its removed Triton implementation has known full-model corruption.
GB200 net vs block-diagonal at equal work: star-like/balanced trees 1.59x training / 1.56x
logprob; deep branched trees 1.00x / 1.01x at a 1.10x FLOP ceiling (91-92% kernel efficiency).
"""

import os
from itertools import pairwise
from typing import List

import torch

from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp

# Cache pass plans because the same packed layout recurs across model layers.
# Unqualified Triton gather/backward variants are deferred from this production slice.


# Retain the legacy environment guard after removing the unusable Triton merge.
# The removed kernel was bit-correct in isolation
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
    """Fail explicitly if a deferred experimental attention variant is requested."""
    value = os.environ.get(env_name, "0")
    normalized = value.lower()
    if normalized in ("", "0", "false"):
        return False
    if normalized in ("1", "true"):
        raise RuntimeError(
            f"{env_name} is deferred from the production shared-prefix port; "
            "use the default eager attention path"
        )
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


_resolve_fused_merge_setting()
_resolve_experimental_triton_setting("NRL_SP_FUSED_KV_GATHER")
_resolve_experimental_triton_setting("NRL_SP_FUSED_BACKWARD_GLUE")
_resolve_experimental_triton_setting("NRL_SP_FUSED_DQ_ASSEMBLY")
_SP_DETERMINISTIC_BACKWARD = _resolve_deterministic_backward_setting()
for _deferred_option in ("NRL_SP_MERGE_BT", "NRL_SP_MERGE_WARPS"):
    if _deferred_option in os.environ:
        raise RuntimeError(
            f"{_deferred_option} configures deferred Triton attention assembly; "
            "remove it to use the default eager attention path"
        )

_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 128


# Overlap the independent per-pass FlashAttention calls on side CUDA streams.
_SP_STREAMS = os.environ.get("NRL_SP_STREAMS", "1") not in ("0", "", "false", "False")
_SP_STREAM_POOL: List = []
_SP_STREAM_POOL_N = 4


def _sp_streams():
    if not _SP_STREAMS or not torch.cuda.is_available():
        return None
    if not _SP_STREAM_POOL:
        _SP_STREAM_POOL.extend(torch.cuda.Stream() for _ in range(_SP_STREAM_POOL_N))
    return _SP_STREAM_POOL


def _gather_kv(k, v, k_idx):
    """Gather strided interleaved K/V projections into independent contiguous tensors."""
    return k.index_select(0, k_idx), v.index_select(0, k_idx)


def _plan_key(node_start, node_len, node_parent, device, *, full_context: bool = False):
    return (
        tuple(int(x) for x in node_start),
        tuple(int(x) for x in node_len),
        tuple(int(x) for x in node_parent),
        str(device),
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
    """Cache the exact pass plan reused across attention layers and training steps."""
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
    entry = (total, passes)
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = entry
    return entry


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
# returns out=0 / dq=0 / LSE=+inf for those rows; eager merge and backward use weight 0.
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

        total, passes = _forest_attention_plan_cached(
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

        lse_final = torch.full((np_, total), float("-inf"), device=q.device, dtype=torch.float32)
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
        return o_merged

    @staticmethod
    def backward(ctx, do):
        """Accumulate pass gradients using the merged output for the global softmax correction."""
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

        q, k, v, o_merged = ctx.saved_tensors
        lse_final, scale = ctx.lse_final, ctx.scale
        do = do.contiguous()
        total, np_, hn = q.shape[0], q.shape[1], q.shape[2]
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
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
