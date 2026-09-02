# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

from __future__ import annotations

import bisect
import importlib.util
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


_WRAPPER_CACHE: dict[tuple, object] = {}
_FUSED_LAUNCH_COUNT = 0
_BWD_WRAPPER_CACHE: dict[tuple, object] = {}
_FUSED_BWD_LAUNCH_COUNT = 0
# Calls this backend answered locally with zeros because no rank had any
# cross-rank state to exchange. No kernel runs, so the launch counters stay
# put; a separate counter lets a test tell 'handled, nothing to do' apart
# from 'fell back to Triton'.
_CP_NOOP_COUNT = 0
# Why calls fell back to Triton, keyed by the predicate that rejected them.
# Every rejection below is an anonymous `return None`, so without this a job
# that never reaches the fused kernels is indistinguishable from one that does
# -- it just runs slower, and reads months later as "the kernel is slow" rather
# than "the kernel never ran". A dict increment is tens of nanoseconds against
# the ~15 us the rest of this predicate costs, so it is always on.
_FALLBACK_REASONS: dict[str, int] = {}


def _reject(reason: str) -> None:
    """Record why this call is not taking the fused path, and refuse it."""
    _FALLBACK_REASONS[reason] = _FALLBACK_REASONS.get(reason, 0) + 1
    return None


def _backend_mode() -> str:
    value = os.environ.get(
        "MCORE_GDN_CP_CUTEDSL", os.environ.get("FLA_CP_CUTEDSL", "auto")
    ).lower()
    if value in {"0", "false", "no", "off"}:
        return "disabled"
    if value in {"1", "true", "yes", "on"}:
        return "enabled"
    if value == "auto":
        return value
    raise ValueError("MCORE_GDN_CP_CUTEDSL must be one of auto, 0, or 1")


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _rank_chains(global_cu_cpu, world_size: int) -> list[tuple[int, int]] | None:
    """``(pre_num_ranks, post_num_ranks)`` for every rank, from global offsets.

    Mirrors :func:`fla.ops.cp.context.get_cp_cu_seqlens` exactly, but for all
    ranks rather than the calling one.  Pure CPU integer work on a vector every
    rank already holds identically, so every rank computes the same list and a
    decision made from it is rank-invariant by construction.

    Returns ``None`` when the partitioning is degenerate (``total < world_size``),
    which is also rank-invariant.
    """
    # .tolist() converts in one C-level pass; a Python-level
    # [int(x) for x in tensor] costs several us on every call.
    cu = global_cu_cpu.tolist()
    total = cu[-1]
    part = total // world_size
    if part <= 0:
        return None
    ends, starts = cu[1:], cu[:-1]
    chains = []
    for r in range(world_size):
        lo, hi = part * r, part * r + part
        first = bisect.bisect_right(ends, lo)
        last = bisect.bisect_left(starts, hi)
        # first rank of this rank's FIRST local sequence -> pre_num_ranks
        pre = r - cu[first] // part
        # last rank of this rank's LAST local sequence -> post_num_ranks
        post = (cu[last] - 1) // part - r
        chains.append((pre, post))
    return chains


def _chain_mode(
    *,
    context: FLACPContext,
    cu_seqlens: torch.Tensor | None,
    T: int,
    rank: int,
    world_size: int,
) -> str | None:
    """Classify the CP chain for this call, identically on every rank.

    Memoized on the context object.  One ``FLACPContext`` is threaded to every
    layer of the model in both directions, and the classification is a pure
    function of ``(context, T)`` -- ``rank`` and ``world_size`` come from
    ``context.group``.  Recomputing it costs a ``.tolist()`` of the whole
    global offsets vector plus ``W`` bisects plus four CPU-tensor scalar
    extractions, and under THD the offsets change every step, so this is on the
    per-step critical path at every layer.  The memo lives on the context, so a
    fresh packing (which builds a fresh context) invalidates it for free.

    The consistency guard below therefore runs once per ``(context, T)`` rather
    than once per layer.  That keeps its teeth: ``build_cp_context`` clones the
    offsets it is handed, so a context's offsets do not change after
    construction, and the failure the guard was written for -- a caller
    recycling one offsets buffer across steps -- produces a *new* context each
    step and so a new memo.  It remains rank-invariant either way, since every
    rank memoizes the same answer from the same global vector.

    Returns ``"fused"`` when the fused kernels can run, ``"noop"`` when no rank
    has any cross-rank state to exchange (so the whole pre-process is a no-op
    and can be answered locally with zeros), or ``None`` to fall back.

    Both fused kernels run ONE chain shape: every rank pushes its summary to
    ranks ``rank+1..W-1`` and folds ranks ``0..rank-1`` (reversed in the
    backward).  Every request in a packed batch therefore runs at the same CP
    size, ``W``, regardless of how many ranks it actually spans -- a packed
    batch is expressed by *suppressing halves of the summary* rather than by
    shortening the chain.  See :func:`_emit_flags` for the rules and why they
    reproduce the Triton path's ``pre_num_ranks``-long fold exactly.

    So the only thing left to check here is that the chains are the ones
    ``get_cp_cu_seqlens`` would produce and that they lie inside the group.
    Since ``get_cp_cu_seqlens`` refuses an indivisible total, every chain it
    produces is in range and this classifier accepts every packing -- measured
    at 512/512 sampled packings for ``W`` in {2, 4, 8}.  The bound below is kept
    as a guard rather than a filter: it is what would catch a context built by
    some other route, and a wrong chain here is a wrong *number*, not a crash.

    Checking only the *local* rank's chain would not be sufficient: partitionings
    exist where one rank's chain is in range and another's is not, and a split
    decision deadlocks (Triton all-gathers, the fused path uses symmetric
    memory).  Hence the all-rank check.
    """
    if cu_seqlens is None:
        return None
    memo = getattr(context, "_cutedsl_chain_memo", None)
    if memo is None:
        memo = {}
        context._cutedsl_chain_memo = memo
    elif T in memo:
        return memo[T]
    mode = _classify_chain(
        context=context, cu_seqlens=cu_seqlens, T=T,
        rank=rank, world_size=world_size,
    )
    memo[T] = mode
    return mode


def _classify_chain(
    *,
    context: FLACPContext,
    cu_seqlens: torch.Tensor | None,
    T: int,
    rank: int,
    world_size: int,
) -> str | None:
    """Uncached body of :func:`_chain_mode`; see there for the reasoning."""
    cu_cpu = context.cu_seqlens_cpu
    if cu_cpu is None or len(cu_cpu) < 2:
        return None
    # The local window must cover the whole local tensor, or the producer
    # window computed below would not correspond to the tensor handed in.
    if int(cu_cpu[0]) != 0 or int(cu_cpu[-1]) != T:
        return None
    if len(cu_cpu) - 1 != len(cu_seqlens) - 1:
        return None

    global_cu = getattr(context, "global_cu_seqlens_cpu", None)
    if global_cu is None:
        # Older context without the global offsets: only the single-sequence
        # case can be decided rank-invariantly from local data alone.
        if getattr(context, "global_num_seqs", None) != 1 or len(cu_cpu) != 2:
            return None
        chains = [(r, world_size - r - 1) for r in range(world_size)]
    else:
        chains = _rank_chains(global_cu, world_size)
        if chains is None:
            return None

    if chains[rank] != (context.pre_num_ranks, context.post_num_ranks):
        # Reconstructing this rank's own chain from the global offsets must
        # reproduce what the context recorded; if it does not, the context is
        # internally inconsistent (e.g. its offsets tensor was mutated after
        # construction).  Raise rather than fall back: falling back would be a
        # *per-rank* decision, and one rank taking Triton's all-gather while
        # its peers enter the symmetric-memory kernel hangs the job. A crash is
        # recoverable; a hang is not.
        raise RuntimeError(
            "FLACPContext is inconsistent: rank {} records pre/post_num_ranks "
            "{} but its global_cu_seqlens_cpu implies {}. The offsets tensor "
            "was probably mutated after build_cp_context().".format(
                rank,
                (context.pre_num_ranks, context.post_num_ranks),
                chains[rank],
            )
        )
    # Every chain must name ranks that exist.  Unreachable for a context built
    # by `get_cp_cu_seqlens`, which refuses the indivisible total that was the
    # only way to violate this; kept as a guard against a hand-built context.
    for r, (pre, post) in enumerate(chains):
        if not (0 <= pre <= r and 0 <= post <= world_size - r - 1):
            return None
    if all(pre == 0 and post == 0 for pre, post in chains):
        return "noop"
    return "fused"


def _emit_flags(context: FLACPContext, *, forward: bool) -> tuple[bool, bool]:
    """Which halves of this rank's summary may reach its consumers.

    The kernels always run the whole-group chain, so a rank always pushes to
    every later rank (forward) and folds every earlier one.  What makes that
    correct for a packed batch is that a producer whose window does not carry
    a given half of ``h_out = M . h_in + X`` pushes ZEROS for it.

    Forward, for producer ``j``:

    * ``X_j`` is the contribution of the sub-sequence that continues onto
      ``j+1``, i.e. of ``j``'s LAST local sequence -- the window
      :func:`_boundary_window` hands the kernel.  If nothing continues past
      ``j`` (``post_num_ranks == 0``) there is no such sub-sequence and ``X_j``
      is zero.  The Triton producer expresses the same thing by skipping the
      kernel behind ``if not context.is_last_rank``, leaving its ``hm`` zeroed.

    * ``M_j`` is the transition of ``j``'s whole window, and is the true
      transition only when that window is one uninterrupted continuation: one
      local sequence (no boundary inside), entering from an earlier rank
      (``pre_num_ranks > 0``) and leaving to a later one.  Otherwise the scan
      resets somewhere in the window, which annihilates everything before the
      reset -- exactly what ``M_j = 0`` does to a consumer's fold.

    That zero is what bounds the fold.  Consumer ``r`` computes
    ``h <- M_j . h + X_j`` for ``j = 0 .. r-1``, and the lowest ``j`` with
    ``M_j = 0`` truncates the sum at ``j``.  When ``pre_num_ranks_r > 0`` that
    ``j`` is precisely ``r - pre_num_ranks_r``, the first rank of ``r``'s own
    first local sequence, so the full chain covers exactly the ``pre_num_ranks``
    sources Triton's fold does (``merge_fwd_bwd_kernel`` walks
    ``rank - num_ranks .. rank-1``) -- term for term.  When
    ``pre_num_ranks_r == 0`` the fold must produce zero, and it does for a
    different reason: ``r``'s first local sequence starts on ``r``'s own
    boundary, so ``r-1``'s last local sequence ended there, so ``post_{r-1}``
    is 0 and BOTH of ``r-1``'s halves are zero.

    ``tests/context_parallel/test_cp_chain_emit.py`` checks both branches
    against a directly-simulated scan.

    The backward is the mirror image: ``pre`` and ``post`` swap roles, the
    window is the FIRST local sequence, and the fold runs ``r+1 .. W-1``.
    """
    if context.pre_num_ranks is None or context.post_num_ranks is None:
        # A context built without chain metadata can only be the whole-group
        # single-sequence case, which suppresses nothing.  `_classify_chain`
        # never produces one -- it cross-checks both fields -- but the
        # standalone harness entry in `bwd_fused.py` does not go through it.
        return True, True
    pre = int(context.pre_num_ranks)
    post = int(context.post_num_ranks)
    if not forward:
        pre, post = post, pre
    single = context.cu_seqlens_cpu is None or len(context.cu_seqlens_cpu) == 2
    emit_h = post > 0
    return emit_h, emit_h and pre > 0 and single


def _boundary_window(context: FLACPContext, T: int, *, forward: bool) -> tuple[int, int]:
    """The one sub-sequence this rank must summarise, as ``[bos, eos)``.

    Only one local sequence can carry cross-rank state, and the Triton
    reference picks it the same way: the forward producer is called with
    ``cu_seqlens[-2:]`` (the LAST local sequence, the only one that can
    continue onto a later rank) and the backward with ``cu_seqlens[:2]`` (the
    FIRST, the only one that can be a continuation from an earlier rank).  See
    ``fla/ops/cp/chunk_delta_h.py``.

    Memoized on the context alongside the chain classification: the offsets are
    cloned at ``build_cp_context`` time and do not change afterwards, so this is
    a pure function of ``(context, forward)``.  Uncached it is two CPU-tensor
    scalar extractions (~1 us each) on the per-step path of every layer.
    """
    cu = context.cu_seqlens_cpu
    if cu is None:
        return 0, T
    memo = getattr(context, "_cutedsl_window_memo", None)
    if memo is None:
        memo = {}
        context._cutedsl_window_memo = memo
    else:
        window = memo.get(forward)
        if window is not None:
            return window
    bos, eos = (int(cu[-2]), int(cu[-1])) if forward else (int(cu[0]), int(cu[1]))
    if not (0 <= bos < eos <= T):
        raise ValueError(f"invalid boundary window [{bos}, {eos}) for T={T}")
    memo[forward] = (bos, eos)
    return bos, eos


def _window(tensor: torch.Tensor | None, bos: int, eos: int) -> torch.Tensor | None:
    """Producer-window view.

    ``B == 1`` and every operand is checked contiguous by the predicates below,
    so ``tensor[:, bos:eos]`` has unit stride on the last mode, ``H*K`` on the
    time mode and a size-1 leading mode -- i.e. it is contiguous by
    construction, and the ``.contiguous()`` this used to end with could only
    ever return ``self``.  It is dropped rather than kept "for safety": under
    THD this runs on 4 (forward) or 6 (backward) operands every step of every
    layer, and an ATen dispatch that provably cannot do anything is ~1 us of
    pure host time each.
    """
    if tensor is None or (bos == 0 and eos == tensor.shape[1]):
        return tensor
    # `narrow` rather than `[:, bos:eos]`: same view, one less slice object and
    # tuple to build per operand.
    return tensor.narrow(1, bos, eos - bos)


_GROUP_TOPOLOGY_CACHE: dict[object, tuple[int, int]] = {}


def _group_topology(group) -> tuple[int, int]:
    """``(world_size, rank)`` for a process group, memoized.

    Both are fixed for the lifetime of the group, and both are queried on every
    forward and backward of every layer.  ``dist.get_world_size`` /
    ``dist.get_rank`` each walk the distributed_c10d group bookkeeping, which is
    ~1-2 us apiece -- small until THD makes the surrounding kernel window half a
    shard, at which point the whole host path is a third of the pre-process.
    """
    topology = _GROUP_TOPOLOGY_CACHE.get(group)
    if topology is None:
        topology = (dist.get_world_size(group), dist.get_rank(group))
        _GROUP_TOPOLOGY_CACHE[group] = topology
    return topology


_RUNTIME_SUPPORT_CACHE: dict[tuple, bool] = {}


def _runtime_supported(*, device: torch.device, group) -> bool:
    """CuTeDSL toolchain, Blackwell device, and NCCL group checks.

    Memoized on ``(device, group)``: every term is a property of the toolchain,
    the GPU, or the process group, none of which change once the process is
    up.  Uncached it costs ~20-25 us warm (three ``find_spec`` path walks plus
    two device-property queries) on *every* forward and backward of *every*
    layer, which at production depth is milliseconds per step of pure host
    time.  The group is held in the key rather than its ``id()`` so a recycled
    address cannot alias a freed group's answer.
    """
    key = (device.type, device.index, group)
    cached = _RUNTIME_SUPPORT_CACHE.get(key)
    if cached is None:
        cached = (
            _has_module("cutlass")
            and _has_module("cuda.bindings.driver")
            and _has_module("torch.distributed._symmetric_memory")
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device) == (10, 0)
            and "B200" in torch.cuda.get_device_name(device)
            and str(dist.get_backend(group)).lower().endswith("nccl")
            and 2 <= dist.get_world_size(group) <= 8
        )
        _RUNTIME_SUPPORT_CACHE[key] = cached
    return cached


def _is_supported(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    bg: torch.Tensor | None,
    v: torch.Tensor | None,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    context: FLACPContext,
) -> str | None:
    """Return the chain mode to run, or ``None`` to fall back to Triton.

    Every refusal is attributed through :func:`_reject`; see
    :func:`get_cutedsl_fallback_reasons`.
    """
    if _backend_mode() == "disabled":
        return _reject("MCORE_GDN_CP_CUTEDSL=0")
    if context is None or getattr(context, "group", None) is None:
        return _reject("no CP context or group")
    if chunk_size != 64:
        return _reject(f"chunk_size={chunk_size} (only 64)")
    if g is None or gk is not None or bg is not None:
        return _reject("gate is not scalar g (gk/bg/none unsupported)")
    if v is not None:
        return _reject("separate v operand (DPLR-style) unsupported")
    if k.ndim != 4 or u.ndim != 4 or w.ndim != 4 or g.ndim != 3:
        return _reject("operand rank mismatch")

    B, T, H, K = k.shape
    HV, V = u.shape[2], u.shape[-1]
    if B != 1:
        return _reject(f"B={B} (only 1)")
    if T < 1:
        return _reject("empty local shard")
    if K != V or K not in {64, 128}:
        return _reject(f"K={K},V={V} (need K==V in 64,128)")
    if H < 1 or HV % H != 0 or HV % 4 != 0:
        return _reject(f"head geometry H={H},HV={HV}")
    if u.shape != (B, T, HV, V) or w.shape != (B, T, HV, K) or g.shape != (B, T, HV):
        return _reject("operand shape mismatch")
    if k.dtype != torch.bfloat16 or u.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return _reject(f"dtype {k.dtype} (only bfloat16)")
    if g.dtype not in {torch.bfloat16, torch.float32}:
        return _reject(f"gate dtype {g.dtype}")
    # `g` joins the contiguity check that used to cover only k/u/w. The window
    # slice below no longer calls `.contiguous()`, so a non-contiguous operand
    # must be refused here rather than silently repacked; production `g` comes
    # from `chunk_local_cumsum` and is always contiguous, so this refuses
    # nothing real and a refusal is a correct Triton call either way.
    if not (k.is_contiguous() and u.is_contiguous() and w.is_contiguous()
            and g.is_contiguous()):
        return _reject("non-contiguous operand")
    device = k.device
    if u.device != device or w.device != device or g.device != device:
        return _reject("operands on different devices")

    world_size, rank = _group_topology(context.group)
    # Cheap first: the chain classification is integer work over a CPU vector
    # (~0.1 us) and is what rejects most calls, while _runtime_supported costs
    # ~50 us in a process that has never dispatched -- three uncached
    # importlib.util.find_spec path walks, because `cutlass` is only imported
    # on the first successful dispatch.
    mode = _chain_mode(
        context=context,
        cu_seqlens=cu_seqlens,
        T=T,
        rank=rank,
        world_size=world_size,
    )
    if mode is None:
        return _reject("chain not classifiable")
    # "noop" launches nothing, but it also *skips the NCCL all-gather* the
    # Triton path does unconditionally. That is a behaviour change, so keep it
    # inside the same B200/NCCL contract as "fused" rather than letting this
    # backend alter results on hardware where it is meant to be inert. The
    # cheap chain check still runs first, so a refused call never pays for the
    # toolchain probe.
    if not (k.is_cuda and _runtime_supported(device=k.device, group=context.group)):
        return _reject("runtime unsupported (device/toolchain/backend/world size)")
    return mode


def _out_state(N, HV, K, V, device, *, live_slot):
    """The ``[N, HV, K, V]`` fp32 state buffer both hooks return.

    Every slot must read as zero except the one the fused kernel fills, and it
    fills that one completely.  So when there is a live slot only the others
    need clearing -- at ``N == 1`` (the single-sequence case, and the common
    THD one) that is no clearing at all, against an 8 MB memset per direction
    per layer per step.  ``live_slot=None`` means the kernel writes nothing
    here (the producer rank, and the ``noop`` chain) and everything is zeroed.
    """
    if live_slot is None:
        return torch.zeros(N, HV, K, V, dtype=torch.float32, device=device)
    state = torch.empty(N, HV, K, V, dtype=torch.float32, device=device)
    if N > 1:
        if live_slot == 0:
            state[1:].zero_()
        else:
            state[:-1].zero_()
    return state


@torch.no_grad()
def try_chunk_gated_delta_rule_fwd_h_pre_process_cutedsl(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    bg: torch.Tensor | None,
    v: torch.Tensor | None,
    chunk_size: int,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
    context: FLACPContext,
) -> torch.Tensor | None:
    mode = _is_supported(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        bg=bg,
        v=v,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        context=context,
    )
    if mode is None:
        return None

    _, T, H, K = k.shape
    HV, V = u.shape[2], u.shape[-1]
    world_size, rank = _group_topology(context.group)
    # One state per local sequence, matching the Triton wrapper. Only slot 0
    # can be a continuation from an earlier rank; the rest start fresh and stay
    # zero.
    N = len(context.cu_seqlens_cpu) - 1
    initial_state = _out_state(
        N, HV, K, V, k.device,
        # `_merge_role` writes every element of slot 0 on a merging rank
        # (every `hv`, every V-subtile `b`, rows 0..K), so pre-zeroing it there
        # is an 8 MB memset the kernel immediately overwrites. Rank 0 folds
        # nothing and never writes it, so it does need the zeros.
        live_slot=0 if (mode == "fused" and rank > 0) else None,
    )

    if mode == "noop":
        # No rank has cross-rank state to exchange: every sequence is complete
        # within the rank that holds it. The Triton path also produces nothing
        # here (`is_last_rank` skips the producer, `is_first_rank` skips the
        # merge). Decided identically on every rank, so skipping the collective
        # cannot desynchronise the group.
        global _CP_NOOP_COUNT
        _CP_NOOP_COUNT += 1
        if state_v_first:
            initial_state = initial_state.transpose(-1, -2).contiguous()
        return initial_state

    key = (
        id(context.group),
        k.device.index,
        H,
        HV,
        K,
        V,
        "g",
    )
    fused = _WRAPPER_CACHE.get(key)
    if fused is None:
        # Imported on the cache miss only: a module-level `from ... import` is
        # still a `__import__` call plus a getattr on every execution, and this
        # function runs twice per layer per step.
        from .fused_ws import CuteDSLFusedCPPreProcessWS

        fused = CuteDSLFusedCPPreProcessWS(
            context.group,
            H,
            HV,
            K,
            V,
            gate_mode="g",
            device=k.device,
            split=1,
        )
        _WRAPPER_CACHE[key] = fused

    # The kernel scans one contiguous window. For a packed batch that window is
    # the last local sequence -- the only one that can continue onto a later
    # rank -- exactly as the Triton producer's `cu_seqlens[-2:]`.
    bos, eos = _boundary_window(context, T, forward=True)
    # Write straight into slot 0 rather than into a staging buffer and copying:
    # `initial_state` is contiguous, so `[0]` is a contiguous [HV,K,V] fp32 view
    # of exactly the destination, and it is already zeroed -- which rank 0 needs,
    # since it folds nothing and so never writes h0_out.  The staging buffer cost
    # an extra HV*K*V fp32 zero-fill plus a device-to-device copy of the same
    # 8 MB, per layer per step.
    # `launch_validated`, not `__call__`: every shape, dtype, device and
    # contiguity condition the wrapper would re-check has just been checked by
    # `_is_supported` above, and `h0_out` is a contiguous fp32 [HV,K,V] view of
    # a buffer this function allocated. Re-deriving them costs ~15 us of host
    # time per call, which under THD is a real fraction of the pre-process.
    emit_h, emit_m = _emit_flags(context, forward=True)
    fused.launch_validated(
        _window(k, bos, eos),
        _window(u, bos, eos),
        _window(w, bos, eos),
        g=_window(g, bos, eos),
        h0_out=initial_state[0],
        emit_h=emit_h,
        emit_m=emit_m,
    )

    global _FUSED_LAUNCH_COUNT
    _FUSED_LAUNCH_COUNT += 1
    if state_v_first:
        initial_state = initial_state.transpose(-1, -2).contiguous()
    return initial_state


def get_cutedsl_fused_launch_count() -> int:
    return _FUSED_LAUNCH_COUNT


def _is_supported_bwd(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    bg: torch.Tensor | None,
    dht: torch.Tensor | None,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    context: FLACPContext,
) -> str | None:
    """Backward twin of :func:`_is_supported`.

    Every condition the fused backward would otherwise assert on is checked
    here, so an unsupported call falls back to Triton instead of raising.
    ``initial_state`` is deliberately not inspected: the CP backward
    pre-process ignores it and returns ``None`` in its place, matching the
    Triton path.
    """
    if _backend_mode() == "disabled":
        return _reject("bwd: MCORE_GDN_CP_CUTEDSL=0")
    if context is None or getattr(context, "group", None) is None:
        return _reject("bwd: no CP context or group")
    if chunk_size != 64:
        return _reject(f"bwd: chunk_size={chunk_size} (only 64)")
    if g is None or gk is not None or bg is not None:
        return _reject("bwd: gate is not scalar g (gk/bg/none unsupported)")
    if dht is not None:
        return _reject("bwd: caller supplied dht")
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 4 or do.ndim != 4 or dv.ndim != 4:
        return _reject("bwd: operand rank mismatch")
    if g.ndim != 3:
        return _reject("bwd: operand rank mismatch")

    B, T, H, K = q.shape
    HV, V = do.shape[2], do.shape[-1]
    if B != 1:
        return _reject(f"bwd: B={B} (only 1)")
    if T < 1:
        return _reject("bwd: empty local shard")
    if K != V or K not in {64, 128}:
        return _reject(f"bwd: K={K},V={V} (need K==V in 64,128)")
    if H < 1 or HV % H != 0 or HV % 4 != 0:
        return _reject(f"bwd: head geometry H={H},HV={HV}")
    if k.shape != (B, T, H, K) or w.shape != (B, T, HV, K):
        return _reject("bwd: operand shape mismatch")
    if do.shape != (B, T, HV, V) or dv.shape != (B, T, HV, V):
        return _reject("bwd: operand shape mismatch")
    if g.shape != (B, T, HV):
        return _reject("bwd: operand shape mismatch")
    if (q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16
            or w.dtype != torch.bfloat16 or do.dtype != torch.bfloat16
            or dv.dtype != torch.bfloat16):
        return _reject(f"bwd: dtype {q.dtype} (only bfloat16)")
    if g.dtype not in {torch.bfloat16, torch.float32}:
        return _reject(f"bwd: gate dtype {g.dtype}")
    # See the forward twin: `g` joins the contiguity check now that the window
    # slice is a pure view.
    if not (q.is_contiguous() and k.is_contiguous() and w.is_contiguous()
            and do.is_contiguous() and dv.is_contiguous()
            and g.is_contiguous()):
        return _reject("bwd: non-contiguous operand")
    device = q.device
    if (k.device != device or w.device != device or do.device != device
            or dv.device != device or g.device != device):
        return _reject("bwd: operands on different devices")

    world_size, rank = _group_topology(context.group)
    mode = _chain_mode(
        context=context,
        cu_seqlens=cu_seqlens,
        T=T,
        rank=rank,
        world_size=world_size,
    )
    if mode is None:
        return _reject("bwd: chain not classifiable")
    # See the forward twin: "noop" skips a collective, so it stays inside the
    # same contract instead of taking effect on unsupported hardware.
    if not (q.is_cuda and _runtime_supported(device=q.device, group=context.group)):
        return _reject("bwd: runtime unsupported (device/toolchain/backend/world size)")
    return mode


@torch.no_grad()
def try_chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    bg: torch.Tensor | None,
    scale: float | None,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
    dht: torch.Tensor | None,
    chunk_size: int,
    context: FLACPContext,
) -> torch.Tensor | None:
    mode = _is_supported_bwd(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        bg=bg,
        dht=dht,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        context=context,
    )
    if mode is None:
        return None

    _, T, H, K = q.shape
    HV, V = do.shape[2], do.shape[-1]
    world_size, rank = _group_topology(context.group)
    # One gradient state per local sequence, matching the Triton wrapper. Only
    # the LAST slot can carry gradient back from a later rank; the rest stay
    # zero. (Mirror image of the forward, which writes slot 0.)
    N = len(context.cu_seqlens_cpu) - 1
    dht_out = _out_state(
        N, HV, K, V, q.device,
        # Mirror of the forward: a merging rank's `_merge_role` fills the last
        # slot completely; the last rank folds nothing and needs the zeros.
        live_slot=N - 1 if (mode == "fused" and rank < world_size - 1) else None,
    )

    if mode == "noop":
        global _CP_NOOP_COUNT
        _CP_NOOP_COUNT += 1
        if state_v_first:
            dht_out = dht_out.transpose(-1, -2).contiguous()
        return dht_out

    # Keyed on the same tuple the wrapper class is chosen from, so the class
    # itself is only imported and selected on a cache miss (see the forward).
    key = (
        id(context.group),
        q.device.index,
        H,
        HV,
        K,
        V,
        "g",
    )
    fused = _BWD_WRAPPER_CACHE.get(key)
    if fused is None:
        from .bwd_fused import CuteDSLFusedCPBwdPreProcess
        from .bwd_ws import CuteDSLFusedCPBwdPreProcessWS, ws_supported

        # The warp-specialized Blackwell engine is 2.7x exact FLA at the
        # canonical shape versus 0.99x for the SM80 engine it replaces; it is
        # specialized for K == V == 128 and everything else keeps the non-WS
        # engine, which additionally supports K=64 and any 64-multiple V.
        wrapper_cls = (
            CuteDSLFusedCPBwdPreProcessWS if ws_supported(K, V, "g")
            else CuteDSLFusedCPBwdPreProcess
        )
        fused = wrapper_cls(
            context.group,
            H,
            HV,
            K,
            V,
            gate_mode="g",
            device=q.device,
        )
        _BWD_WRAPPER_CACHE[key] = fused

    # The reverse scan runs over the first local sequence -- the only one that
    # can be a continuation from an earlier rank -- matching the Triton
    # producer's `cu_seqlens[:2]`.
    bos, eos = _boundary_window(context, T, forward=False)
    # Mirror of the forward: write straight into the last slot.  `dht_out[-1]`
    # is a contiguous [HV,K,V] fp32 view of the destination and is already
    # zeroed, which the last rank needs -- it has no later rank to fold from and
    # so never writes dht_out.
    # See the forward: the predicate above has already established everything
    # `__call__` would re-assert.
    emit_h, emit_m = _emit_flags(context, forward=False)
    fused.launch_validated(
        _window(q, bos, eos),
        _window(k, bos, eos),
        _window(w, bos, eos),
        _window(do, bos, eos),
        _window(dv, bos, eos),
        g=_window(g, bos, eos),
        scale=K ** -0.5 if scale is None else float(scale),
        dht_out=dht_out[-1],
        emit_h=emit_h,
        emit_m=emit_m,
    )

    global _FUSED_BWD_LAUNCH_COUNT
    _FUSED_BWD_LAUNCH_COUNT += 1
    if state_v_first:
        dht_out = dht_out.transpose(-1, -2).contiguous()
    return dht_out


def get_cutedsl_fused_bwd_launch_count() -> int:
    return _FUSED_BWD_LAUNCH_COUNT


def get_cutedsl_cp_noop_count() -> int:
    return _CP_NOOP_COUNT


def get_cutedsl_fallback_reasons() -> dict[str, int]:
    """Why this process fell back to Triton, and how often, per cause.

    Empty means every CP pre-process took the fused path (or the noop path,
    which is also this backend). A nonempty dict names the predicate to fix:

        >>> from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_cp_cute import get_cutedsl_fallback_reasons
        >>> get_cutedsl_fallback_reasons()
        {'chain not classifiable': 1024}

    Worth asserting on in a benchmark. A fallback costs nothing but throughput,
    so nothing fails -- the run simply reports Triton numbers under a fused
    label, which is indistinguishable from the fused path being slow.
    """
    return dict(_FALLBACK_REASONS)


def reset_cutedsl_fallback_reasons() -> None:
    _FALLBACK_REASONS.clear()
