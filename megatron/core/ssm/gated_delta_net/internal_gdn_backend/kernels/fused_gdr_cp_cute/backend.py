# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

from __future__ import annotations

import importlib.util
import os
import threading
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


_WRAPPER_CACHE: dict[tuple, object] = {}
_WRAPPER_CACHE_LOCKS: dict[object, threading.Lock] = {}
_WRAPPER_CACHE_LOCKS_GUARD = threading.Lock()
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


def _make_raw_stream():
    """Return the current CUDA stream handle without constructing a Stream object."""
    raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if raw is None:

        def raw(_index):
            return torch.cuda.current_stream(_index).cuda_stream

    return raw


_raw_stream = _make_raw_stream()


def _reject(reason: str) -> None:
    """Record why this call is not taking the fused path, and refuse it."""
    _FALLBACK_REASONS[reason] = _FALLBACK_REASONS.get(reason, 0) + 1
    return None


def _wrapper_cache_key(group, device, H, HV, K, V) -> tuple:
    """Return the rank-stable ownership key for a mutable CP wrapper."""
    return (id(group), device.index, H, HV, K, V, "g")


class _SerializedWrapper:
    """Marshal one mutable symmetric-memory wrapper onto its first CUDA stream.

    A wrapper advances communication epochs in launch order. CP callers already
    submit collectives in the same order on every rank; marshalling every local
    call onto one stream avoids adding rank-local stream scheduling as another
    source of epoch order. The common owner-stream path uses the raw stream
    accessor and does not construct a temporary ``torch.cuda.Stream`` object.
    """

    def __init__(self, wrapper, device: torch.device):
        self._wrapper = wrapper
        self._device = torch.device(device)
        self._lock = threading.Lock()
        # The inner wrapper constructor may enqueue asynchronous CUDA
        # initialization. Bind that construction stream immediately, before the
        # wrapper can be published through the shared cache, so a racing first
        # launch on another stream is ordered after initialization.
        self._owner_handle = _raw_stream(self._device.index)
        self._owner_stream = torch.cuda.current_stream(self._device)
        self._external_streams = {}
        self._input_ready = None
        self._completion = None

    def launch_validated(self, *args, **kwargs):
        caller_handle = _raw_stream(self._device.index)
        with self._lock:
            kwargs["_stream_handle"] = self._owner_handle
            if caller_handle == self._owner_handle:
                return self._wrapper.launch_validated(*args, **kwargs)

            caller_stream = self._external_streams.get(caller_handle)
            if caller_stream is None:
                caller_stream = torch.cuda.ExternalStream(caller_handle, device=self._device)
                self._external_streams[caller_handle] = caller_stream
            if self._input_ready is None:
                self._input_ready = torch.cuda.Event()
                self._completion = torch.cuda.Event()

            self._input_ready.record(caller_stream)
            self._owner_stream.wait_event(self._input_ready)
            # The wrappers perform ATen preprocessing (padding, gate conversion,
            # and descriptor updates) before the compiled launch. Run the whole
            # call on the owner stream so the ready event orders those operations
            # together with the kernel that consumes their results.
            with torch.cuda.stream(self._owner_stream):
                result = self._wrapper.launch_validated(*args, **kwargs)
            self._completion.record(self._owner_stream)
            caller_stream.wait_event(self._completion)
            return result


def _rank_consistent_wrapper_init(group, device: torch.device, signature: tuple[int, ...]) -> bool:
    """Confirm every rank is about to construct the same wrapper.

    This collective runs only on a cache miss. It prevents two local host
    threads that win their process locks in different orders on different ranks
    from crossing symmetric-memory rendezvous for different wrapper shapes.
    """
    world_size = dist.get_world_size(group)
    local = torch.tensor(signature, dtype=torch.int64, device=device)
    gathered = torch.empty(world_size * len(signature), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    return bool((gathered.view(world_size, -1) == local).all().item())


def _wrapper_cache_lock(group) -> threading.Lock:
    """Return an initialization lock scoped to one process group."""
    with _WRAPPER_CACHE_LOCKS_GUARD:
        lock = _WRAPPER_CACHE_LOCKS.get(group)
        if lock is None:
            lock = threading.Lock()
            _WRAPPER_CACHE_LOCKS[group] = lock
        return lock


def _get_or_create_wrapper(
    cache: dict, key: tuple, factory, *, group, device: torch.device, signature: tuple[int, ...]
):
    """Construct a rendezvous-owning wrapper once and in rank-consistent order."""
    wrapper = cache.get(key)
    if wrapper is not None:
        return wrapper
    with _wrapper_cache_lock(group):
        wrapper = cache.get(key)
        if wrapper is None:
            if not _rank_consistent_wrapper_init(group, device, signature):
                raise RuntimeError(
                    "CuTeDSL CP wrapper initialization order differs across ranks; "
                    "distributed calls must enter the same direction and shape order."
                )
            wrapper = factory()
            cache[key] = wrapper
    return wrapper


def _cuda_graphs_enabled(context: FLACPContext) -> bool:
    """Read the rank-invariant model configuration copied onto the CP context."""
    return bool(getattr(context, "_cutedsl_cuda_graph_enabled", False))


def _backend_mode() -> str:
    value = os.environ.get("MCORE_GDN_CP_CUTEDSL", os.environ.get("FLA_CP_CUTEDSL", "auto")).lower()
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


def _chain_mode(
    *, context: FLACPContext, cu_seqlens: torch.Tensor | None, T: int, rank: int, world_size: int
) -> str | None:
    """Classify the CP chain for this call, identically on every rank.

    Memoized on the context object. One ``FLACPContext`` is threaded to every
    layer of the model in both directions, and the classification is a pure
    function of ``(context, T)`` -- ``rank`` and ``world_size`` come from
    ``context.group``.

    Without a host-side global offsets side channel, the only rank-invariant
    layout this Python dispatcher can prove is one global sequence spanning the
    full CP group. Dense ``B == 1`` and packed THD with exactly one global
    sequence set ``context.global_num_seqs == 1`` before entering this backend.
    Broader packed layouts fall back to the FLA/Triton CP path.

    Both fused kernels run one chain shape: every rank pushes its summary to
    ranks ``rank+1..W-1`` and folds ranks ``0..rank-1`` (reversed in the
    backward). For the supported single-sequence layout that is exactly FLA's
    full-chain CP state exchange.
    """
    if cu_seqlens is None:
        return None
    memo = getattr(context, "_cutedsl_chain_memo", None)
    if memo is None:
        memo = {}
        context._cutedsl_chain_memo = memo
    memo_key = (getattr(context, "_cutedsl_metadata_generation", 0), T)
    if memo_key in memo:
        return memo[memo_key]
    mode = _classify_chain(
        context=context, cu_seqlens=cu_seqlens, T=T, rank=rank, world_size=world_size
    )
    memo[memo_key] = mode
    return mode


def _classify_chain(
    *, context: FLACPContext, cu_seqlens: torch.Tensor | None, T: int, rank: int, world_size: int
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

    if getattr(context, "global_num_seqs", None) != 1 or len(cu_cpu) != 2:
        return None

    expected_chain = (rank, world_size - rank - 1)
    actual_chain = (context.pre_num_ranks, context.post_num_ranks)
    if actual_chain != expected_chain:
        # The only no-host-metadata case this backend can decide
        # rank-invariantly is one global sequence spanning the full CP group.
        # A mismatch means the context is not that layout, or the hint is
        # wrong. Raise instead of letting different ranks split between the
        # fused symmetric-memory path and FLA's all-gather path.
        raise RuntimeError(
            "FLACPContext is inconsistent with a single full-chain CP sequence: "
            "rank {} records pre/post_num_ranks {}, expected {}.".format(
                rank, actual_chain, expected_chain
            )
        )
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

    Memoized on the context alongside the chain classification. The generation,
    direction, and local length are part of the key so rebinding metadata cannot
    reuse a stale boundary. Uncached it is two CPU-tensor scalar extractions
    (~1 us each) on the per-step path of every layer.
    """
    cu = context.cu_seqlens_cpu
    if cu is None:
        return 0, T
    memo_key = (getattr(context, "_cutedsl_metadata_generation", 0), forward, T)
    memo = getattr(context, "_cutedsl_window_memo", None)
    if memo is None:
        memo = {}
        context._cutedsl_window_memo = memo
    else:
        window = memo.get(memo_key)
        if window is not None:
            return window
    bos, eos = (int(cu[-2]), int(cu[-1])) if forward else (int(cu[0]), int(cu[1]))
    if not (0 <= bos < eos <= T):
        raise ValueError(f"invalid boundary window [{bos}, {eos}) for T={T}")
    memo[memo_key] = (bos, eos)
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
    if not (k.is_contiguous() and u.is_contiguous() and w.is_contiguous() and g.is_contiguous()):
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
        context=context, cu_seqlens=cu_seqlens, T=T, rank=rank, world_size=world_size
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
    if _cuda_graphs_enabled(context):
        return _reject("CUDA graph configuration unsupported")
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
        N,
        HV,
        K,
        V,
        k.device,
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

    key = _wrapper_cache_key(context.group, k.device, H, HV, K, V)
    fused = _WRAPPER_CACHE.get(key)
    if fused is None:
        # Wrapper construction performs a symmetric-memory rendezvous; the
        # process-wide helper prevents two local host threads from racing it.
        from .fused_ws import CuteDSLFusedCPPreProcessWS

        fused = _get_or_create_wrapper(
            _WRAPPER_CACHE,
            key,
            lambda: _SerializedWrapper(
                CuteDSLFusedCPPreProcessWS(
                    context.group, H, HV, K, V, gate_mode="g", device=k.device, split=1
                ),
                k.device,
            ),
            group=context.group,
            device=k.device,
            signature=(0, H, HV, K, V),
        )

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
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or do.dtype != torch.bfloat16
        or dv.dtype != torch.bfloat16
    ):
        return _reject(f"bwd: dtype {q.dtype} (only bfloat16)")
    if g.dtype not in {torch.bfloat16, torch.float32}:
        return _reject(f"bwd: gate dtype {g.dtype}")
    # See the forward twin: `g` joins the contiguity check now that the window
    # slice is a pure view.
    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and w.is_contiguous()
        and do.is_contiguous()
        and dv.is_contiguous()
        and g.is_contiguous()
    ):
        return _reject("bwd: non-contiguous operand")
    device = q.device
    if (
        k.device != device
        or w.device != device
        or do.device != device
        or dv.device != device
        or g.device != device
    ):
        return _reject("bwd: operands on different devices")

    world_size, rank = _group_topology(context.group)
    mode = _chain_mode(
        context=context, cu_seqlens=cu_seqlens, T=T, rank=rank, world_size=world_size
    )
    if mode is None:
        return _reject("bwd: chain not classifiable")
    # See the forward twin: "noop" skips a collective, so it stays inside the
    # same contract instead of taking effect on unsupported hardware.
    if not (q.is_cuda and _runtime_supported(device=q.device, group=context.group)):
        return _reject("bwd: runtime unsupported (device/toolchain/backend/world size)")
    if _cuda_graphs_enabled(context):
        return _reject("bwd: CUDA graph configuration unsupported")
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
        N,
        HV,
        K,
        V,
        q.device,
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
    key = _wrapper_cache_key(context.group, q.device, H, HV, K, V)
    fused = _BWD_WRAPPER_CACHE.get(key)
    if fused is None:
        from .bwd_fused import CuteDSLFusedCPBwdPreProcess
        from .bwd_ws import CuteDSLFusedCPBwdPreProcessWS, ws_supported

        # The warp-specialized Blackwell engine is 2.7x exact FLA at the
        # canonical shape versus 0.99x for the SM80 engine it replaces. Other
        # shapes keep the non-WS engine.
        wrapper_cls = (
            CuteDSLFusedCPBwdPreProcessWS
            if ws_supported(K, V, "g")
            else CuteDSLFusedCPBwdPreProcess
        )
        fused = _get_or_create_wrapper(
            _BWD_WRAPPER_CACHE,
            key,
            lambda: _SerializedWrapper(
                wrapper_cls(context.group, H, HV, K, V, gate_mode="g", device=q.device), q.device
            ),
            group=context.group,
            device=q.device,
            signature=(1, H, HV, K, V),
        )

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
        scale=K**-0.5 if scale is None else float(scale),
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
