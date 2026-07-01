# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

from typing import Optional

from megatron.core.utils import internal_api

try:
    import deep_ep

    Buffer = getattr(deep_ep, "Buffer", None)
    ElasticBuffer = getattr(deep_ep, "ElasticBuffer", None)

    try:
        from deep_ep.utils import EventHandle, EventOverlap
    except ImportError:
        EventHandle = getattr(deep_ep, "EventHandle", None)
        EventOverlap = getattr(deep_ep, "EventOverlap", None)

    HAVE_DEEP_EP_LEGACY = (
        Buffer is not None and EventHandle is not None and EventOverlap is not None
    )
    HAVE_DEEP_EP_V2 = ElasticBuffer is not None
    HAVE_DEEP_EP = HAVE_DEEP_EP_LEGACY or HAVE_DEEP_EP_V2
except ImportError:
    Buffer = None
    ElasticBuffer = None
    EventHandle = None
    EventOverlap = None
    HAVE_DEEP_EP_LEGACY = False
    HAVE_DEEP_EP_V2 = False
    HAVE_DEEP_EP = False

import torch

_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP_LEGACY:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


# Pool of pre-allocated ElasticBuffers, keyed by (group, num_max_tokens_per_rank).
# Multiple distinct sizes are pinned at init (e.g. one for decode steps, one for
# prefill chunks) so the runtime can pick a small buffer for decode without
# violating DeepEP V2's symmetric-memory invariant — every rank picks the same
# pool entry by passing the same `num_max_tokens_per_rank`.
_deepep_v2_buffer_pool = {}

# The currently-active per-rank dispatch size, set by the engine before each
# step via `set_deepep_v2_active_dispatch_size`. The dispatcher uses this to
# pick which pre-allocated buffer to use, regardless of the local token count
# (which differs across ranks under chunked prefill). Must be identical across
# all ranks in the dispatch group at the moment of dispatch.
_deepep_v2_active_dispatch_size = None

_deepep_v2_num_sms = 0
_deepep_v2_num_sms_group = None
_deepep_v2_num_sms_num_experts = None
_deepep_v2_num_sms_num_topk = None

# Cumulative within-row-duplicate count across all DeepEP V2 dispatches when
# MCORE_DEEPEP_V2_DEBUG_TOPK_GRAPH=1. Written by the dispatch forward (no host
# sync, capture-safe). Read it after a synchronization point — e.g.,
# `torch.cuda.synchronize(); print(_DEEPEP_V2_DUP_COUNTER.item())`.
_DEEPEP_V2_DUP_COUNTER = None

# Set of (group_size, num_experts, num_topk, num_max_tokens_per_rank, hidden,
# use_expanded_layout) tuples already printed (MCORE_DEEPEP_V2_DEBUG_DISPATCH=1).
# We print one line per *distinct* dispatch shape so we can see whether
# different MoE layers (e.g. MTP vs main decoder) dispatch with different params.
_DEEPEP_V2_DEBUG_SEEN = set()

# Routing-imbalance probe (MCORE_DEEPEP_V2_DEBUG_BALANCE=1). Persistent device
# tensors holding running stats of (max-rank-load / mean-rank-load) per
# dispatch. Capture-safe: written by the dispatch forward inside captured
# graphs; readable from outside via host sync. See the probe code in
# DeepEPV2Dispatch.forward for the read recipe.
_DEEPEP_V2_BALANCE_MAX_RATIO_SUM = None  # cumulative sum of per-dispatch ratios
_DEEPEP_V2_BALANCE_MAX_RATIO_PEAK = None  # running peak ratio observed
_DEEPEP_V2_BALANCE_COUNT = None  # number of dispatches accumulated


def set_deepep_v2_active_dispatch_size(num_max_tokens_per_rank):
    """Tell the DeepEP V2 dispatcher which pinned buffer to use for the next dispatch(es).

    The engine MUST call this before each forward step, with a value identical
    across all ranks in the dispatch group (decode vs chunked-prefill). The
    DeepEP V2 ElasticBuffer is symmetric NVLink memory: cross-rank reads land
    on undefined offsets if ranks pick different buffers.

    Pass `None` to revert to "use the largest pinned buffer" (failsafe).
    """
    global _deepep_v2_active_dispatch_size
    _deepep_v2_active_dispatch_size = num_max_tokens_per_rank


def _select_deepep_v2_buffer(group: torch.distributed.ProcessGroup, requested_size: int):
    """Select the smallest pinned buffer in the pool that holds `requested_size`.

    Asserts that at least one buffer in the pool is large enough and tied to
    the same `group`. Lazy reallocation has been removed: any size that the
    runtime needs must be pre-allocated at init via `prepare_deepep_v2_buffer`,
    otherwise we raise — silent realloc with a different size on different
    ranks is exactly the bug pattern that triggers
    `dispatch_copy_epilogue.cuh:106`.
    """
    matching = [
        ((g, sz), buf)
        for (g, sz), buf in _deepep_v2_buffer_pool.items()
        if g is group and sz >= requested_size
    ]
    if not matching:
        sizes = sorted(sz for (g, sz) in _deepep_v2_buffer_pool if g is group)
        raise RuntimeError(
            f"No pre-allocated DeepEP V2 buffer can hold requested size "
            f"{requested_size} (pool has sizes {sizes} for this group). "
            f"Call prepare_deepep_v2_buffer at init with a size >= {requested_size}."
        )
    # Smallest fit — same correctness, less padding work.
    matching.sort(key=lambda kv: kv[0][1])
    return matching[0][1]


def _get_deepep_v2_buffer(
    group: torch.distributed.ProcessGroup, num_max_tokens_per_rank: int, hidden: int, num_topk: int
):
    """Return the pinned ElasticBuffer matching the requested per-rank size.

    Looks up the pool populated by `prepare_deepep_v2_buffer`. The
    `hidden`/`num_topk` arguments are kept for API compatibility but the
    pinned buffers were sized at init using these same values; runtime
    callers don't trigger reallocation.
    """
    return _select_deepep_v2_buffer(group, num_max_tokens_per_rank)


def prepare_deepep_v2_buffer(
    group: torch.distributed.ProcessGroup, num_max_tokens_per_rank: int, hidden: int, num_topk: int
) -> None:
    """Pin an ElasticBuffer of the given per-rank size in the global pool.

    Call once per distinct size the engine needs (typically two: a decode-cap
    sized buffer for fixed-shape decode steps, and a prefill-cap sized buffer
    for chunked-prefill steps). All calls must be made outside any CUDA graph
    capture region.
    """
    if not HAVE_DEEP_EP_V2:
        raise RuntimeError(
            "prepare_deepep_v2_buffer requires deep_ep with ElasticBuffer (V2). "
            "Install deep_ep from the epv2-release branch."
        )
    key = (group, num_max_tokens_per_rank)
    if key in _deepep_v2_buffer_pool:
        return  # already pinned
    _deepep_v2_buffer_pool[key] = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
    )


def _tokens_per_expert_from_psum(psum: torch.Tensor) -> torch.Tensor:
    """Recover per-expert token counts from the device-resident inclusive prefix sum.

    Avoids ``torch.tensor(handle.num_recv_tokens_per_expert_list)``, which is a
    host->device copy that breaks CUDA graph capture. Valid when
    ``expert_alignment == 1`` (alignment-padded == raw counts), which is the
    setting used for both the training and inference paths here.
    """
    return torch.diff(psum, prepend=psum.new_zeros(1))


def _get_deepep_v2_num_sms(
    buffer, group: torch.distributed.ProcessGroup, num_experts: int, num_topk: int
) -> int:
    """Get or calculate the SM count for the current DeepEP V2 layout.

    Defaults to ``buffer.get_theoretical_num_sms(num_experts, num_topk)`` which
    optimises the dispatch kernel's *standalone* throughput. On a chip with
    competing concurrent kernels (attention, Mamba, expert FFN) this can
    over-allocate SMs to dispatch and starve compute. Override with the
    ``MCORE_DEEPEP_V2_NUM_SMS`` env var to fix the count for tuning sweeps;
    typical good values on B200 are 8-24 (vs the auto-default of ~30 at
    EP=64, num_topk=22). Each new value triggers a JIT recompile, so pin
    ``EP_JIT_CACHE_DIR`` if sweeping.
    """
    import os as _os

    _override = _os.environ.get("MCORE_DEEPEP_V2_NUM_SMS")
    if _override is not None:
        return int(_override)

    global _deepep_v2_num_sms
    global _deepep_v2_num_sms_group
    global _deepep_v2_num_sms_num_experts
    global _deepep_v2_num_sms_num_topk

    if (
        _deepep_v2_num_sms == 0
        or _deepep_v2_num_sms_group != group
        or _deepep_v2_num_sms_num_experts != num_experts
        or _deepep_v2_num_sms_num_topk != num_topk
    ):
        _deepep_v2_num_sms = buffer.get_theoretical_num_sms(num_experts, num_topk)
        _deepep_v2_num_sms_group = group
        _deepep_v2_num_sms_num_experts = num_experts
        _deepep_v2_num_sms_num_topk = num_topk
    return _deepep_v2_num_sms


class DeepEPV2Dispatch(torch.autograd.Function):
    """Fused dispatch operation using DeepEP V2 ElasticBuffer."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_expanded_layout=False,
    ):
        # PASSTHROUGH for graph-capture diagnostic
        fake_recv = torch.zeros(
            x.shape[0] * num_experts // group.size() * token_indices.shape[1],
            x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )
        fake_tpe = torch.zeros(num_experts // group.size(), dtype=torch.int64, device=x.device)
        fake_handle = type(
            'FakeHandle',
            (),
            {
                'topk_idx': token_indices,
                'num_max_tokens_per_rank': x.shape[0],
                'num_sms': 0,
                'psum_num_recv_tokens_per_expert': torch.zeros_like(fake_tpe).cumsum(0).int(),
            },
        )()
        ctx.handle = fake_handle
        ctx.group = group
        return fake_recv, None, token_probs[:, 0:1], fake_tpe, fake_handle

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_expanded_layout=False,
    ):
        local_num_tokens, hidden = x.shape
        num_topk = token_indices.shape[1]
        # Pick the buffer based on the engine-set active dispatch size, not on
        # the local token count (which differs across ranks under chunked
        # prefill). The engine sets this via `set_deepep_v2_active_dispatch_size`
        # before each forward step. Falls back to "largest pinned buffer" so
        # paths that haven't been wired to set the active size still work
        # (just at the worst-case capacity).
        if _deepep_v2_active_dispatch_size is not None:
            requested_size = _deepep_v2_active_dispatch_size
        else:
            # Fallback: largest pinned buffer for this group. Same behaviour
            # as the previous single-buffer design.
            requested_size = max(
                (sz for (g, sz) in _deepep_v2_buffer_pool if g is group), default=local_num_tokens
            )
        buffer = _get_deepep_v2_buffer(group, requested_size, hidden, num_topk)
        # Use the buffer's allocated per-rank capacity rather than the local
        # token count: ElasticBuffer is symmetric NVLink memory, so every rank
        # MUST pass the same `num_max_tokens_per_rank` to `buffer.dispatch`,
        # otherwise the kernel templates with mismatched strides per rank and
        # cross-rank reads land on undefined offsets (manifests as duplicate
        # `dst_expert_idx` values tripping `dispatch_copy_epilogue.cuh:106`).
        # `prepare_deepep_v2_buffer` is sized at init time for the worst case
        # across decode + prefill, so all ranks see the same capacity.
        num_max_tokens_per_rank = buffer.num_max_tokens_per_rank
        assert local_num_tokens <= num_max_tokens_per_rank, (
            f"deepep_v2 dispatch local_num_tokens={local_num_tokens} exceeds "
            f"pre-allocated num_max_tokens_per_rank={num_max_tokens_per_rank}; "
            f"increase the cap in dynamic_context._deepep_v2_ep_dispatcher init."
        )
        num_sms = _get_deepep_v2_num_sms(buffer, group, num_experts, num_topk)
        # DEBUG: dump dispatch parameters once per (param-tuple) per rank, so
        # we see *every distinct* dispatch shape across MoE layers (MTP and
        # main decoder layers may differ). Enable with
        # MCORE_DEEPEP_V2_DEBUG_DISPATCH=1.
        import os as _os

        global _DEEPEP_V2_DEBUG_SEEN
        if _os.environ.get("MCORE_DEEPEP_V2_DEBUG_DISPATCH", "0") == "1":
            _key = (
                group.size(),
                num_experts,
                num_topk,
                local_num_tokens,
                num_max_tokens_per_rank,
                hidden,
                use_expanded_layout,
            )
            if _key not in _DEEPEP_V2_DEBUG_SEEN:
                _DEEPEP_V2_DEBUG_SEEN.add(_key)
                _rank = torch.distributed.get_rank()
                _group_rank = torch.distributed.get_rank(group=group)
                _idx_max = int(token_indices.max().item())
                _idx_min = int(token_indices.min().item())
                print(
                    f"[deepep_v2 debug] global_rank={_rank} group_rank={_group_rank} "
                    f"group_size={group.size()} num_experts={num_experts} "
                    f"num_topk={num_topk} local_num_tokens={local_num_tokens} "
                    f"num_max_tokens_per_rank={num_max_tokens_per_rank} "
                    f"hidden={hidden} use_expanded_layout={use_expanded_layout} "
                    f"token_indices.shape={tuple(token_indices.shape)} "
                    f"token_indices.dtype={token_indices.dtype} "
                    f"token_indices min={_idx_min} max={_idx_max} "
                    f"token_probs.dtype={token_probs.dtype}",
                    flush=True,
                )
        # Inference (use_expanded_layout=True) is the graph-capturable path: we
        # must avoid the dispatch-side host sync, and we recover tokens_per_expert
        # from the device-resident prefix-sum on the handle below.
        do_cpu_sync = False if use_expanded_layout else None
        # DEBUG: graph-capture-friendly per-token duplicate check on the
        # token_indices tensor that is about to be handed to DeepEP V2. We
        # accumulate the duplicate count into a persistent device tensor on
        # every dispatch (no host sync, survives capture/replay). Read it
        # from outside the graph via `_DEEPEP_V2_DUP_COUNTER`. Enable with
        # MCORE_DEEPEP_V2_DEBUG_TOPK_GRAPH=1.
        import os as _os

        if _os.environ.get("MCORE_DEEPEP_V2_DEBUG_TOPK_GRAPH", "0") == "1":
            _idx = token_indices
            _safe = _idx.clamp(min=0)
            _sorted, _ = _safe.sort(dim=-1)
            _dup_mask = _sorted[:, 1:] == _sorted[:, :-1]
            _dup_count = _dup_mask.any(dim=-1).to(torch.int64).sum()
            global _DEEPEP_V2_DUP_COUNTER
            if _DEEPEP_V2_DUP_COUNTER is None or _DEEPEP_V2_DUP_COUNTER.device != _idx.device:
                _DEEPEP_V2_DUP_COUNTER = torch.zeros(1, dtype=torch.int64, device=_idx.device)
            _DEEPEP_V2_DUP_COUNTER += _dup_count
        # DEBUG: routing-imbalance probe. Computes per-rank token load
        # (number of (token, slot) pairs whose target expert lives on each
        # rank in the EP group) and tracks max/mean ratio. At EP > a few,
        # imbalance bottlenecks the dispatch — the most-loaded rank gates
        # the step. Capture-safe: writes summary stats to persistent device
        # tensors. Enable with MCORE_DEEPEP_V2_DEBUG_BALANCE=1.
        # Outside the captured graph (e.g., between steps or at run end),
        # read on rank 0:
        #     from megatron.core.transformer.moe.fused_a2a import (
        #         _DEEPEP_V2_BALANCE_MAX_RATIO_SUM,
        #         _DEEPEP_V2_BALANCE_MAX_RATIO_PEAK,
        #         _DEEPEP_V2_BALANCE_COUNT,
        #     )
        #     n = _DEEPEP_V2_BALANCE_COUNT.item()
        #     avg = _DEEPEP_V2_BALANCE_MAX_RATIO_SUM.item() / max(n, 1)
        #     peak = _DEEPEP_V2_BALANCE_MAX_RATIO_PEAK.item()
        #     print(f"avg max/mean rank-load ratio = {avg:.2f}, peak = {peak:.2f} over {n} dispatches")
        if _os.environ.get("MCORE_DEEPEP_V2_DEBUG_BALANCE", "0") == "1":
            _idx = token_indices
            _ep_size = group.size()
            _num_local_experts = num_experts // _ep_size
            # Map each (token, slot) to its destination rank; the dummy `_ep_size`
            # bucket absorbs invalid (-1) slots and gets dropped after.
            _valid = _idx >= 0
            _dst_rank = torch.where(_valid, _idx // _num_local_experts, _idx.new_full((), _ep_size))
            # Capture-safe histogram via scatter_add_ on a pre-allocated tensor.
            # `torch.bincount` is NOT capture-safe (internally syncs to size the
            # output) even with `minlength`. scatter_add_ has fully static shape.
            _bins = torch.zeros(_ep_size + 1, dtype=torch.int64, device=_idx.device)
            _flat = _dst_rank.flatten().to(torch.int64)
            _bins.scatter_add_(0, _flat, torch.ones_like(_flat))
            _load = _bins[:_ep_size].to(torch.float32)
            _mean = _load.mean()
            _max = _load.max()
            # Avoid division by zero on empty steps (rare but possible during warmup).
            _ratio = torch.where(_mean > 0, _max / _mean, _max.new_zeros(()))
            global _DEEPEP_V2_BALANCE_MAX_RATIO_SUM
            global _DEEPEP_V2_BALANCE_MAX_RATIO_PEAK
            global _DEEPEP_V2_BALANCE_COUNT
            if (
                _DEEPEP_V2_BALANCE_MAX_RATIO_SUM is None
                or _DEEPEP_V2_BALANCE_MAX_RATIO_SUM.device != _idx.device
            ):
                _DEEPEP_V2_BALANCE_MAX_RATIO_SUM = torch.zeros(
                    1, dtype=torch.float32, device=_idx.device
                )
                _DEEPEP_V2_BALANCE_MAX_RATIO_PEAK = torch.zeros(
                    1, dtype=torch.float32, device=_idx.device
                )
                _DEEPEP_V2_BALANCE_COUNT = torch.zeros(1, dtype=torch.int64, device=_idx.device)
            _DEEPEP_V2_BALANCE_MAX_RATIO_SUM += _ratio
            _DEEPEP_V2_BALANCE_MAX_RATIO_PEAK = torch.maximum(
                _DEEPEP_V2_BALANCE_MAX_RATIO_PEAK, _ratio.unsqueeze(0)
            )
            _DEEPEP_V2_BALANCE_COUNT += 1
            # In eager (non-capturing) mode, also print every 50 dispatches
            # on rank 0 so you get live feedback during warmup or eager runs.
            if not torch.cuda.is_current_stream_capturing():
                _n_int = int(_DEEPEP_V2_BALANCE_COUNT.item())
                if (
                    _n_int > 0
                    and _n_int % 50 == 0
                    and torch.distributed.is_initialized()
                    and torch.distributed.get_rank() == 0
                ):
                    _avg = float(_DEEPEP_V2_BALANCE_MAX_RATIO_SUM.item()) / _n_int
                    _peak = float(_DEEPEP_V2_BALANCE_MAX_RATIO_PEAK.item())
                    _last = float(_ratio.item())
                    print(
                        f"[deepep_v2 balance] dispatches={_n_int} ep_size={_ep_size} "
                        f"last_max/mean={_last:.2f} avg_max/mean={_avg:.2f} peak_max/mean={_peak:.2f} "
                        f"(1.0=perfect balance, 2.0=hot rank has 2x mean, etc.)",
                        flush=True,
                    )
        recv_x, recv_token_indices, recv_token_probs, handle, event = buffer.dispatch(
            x.contiguous(),
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            expert_alignment=32,
            num_sms=num_sms,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            do_cpu_sync=do_cpu_sync,
            do_expand=use_expanded_layout,
        )
        if use_expanded_layout:
            assert recv_token_indices is None
            assert recv_token_probs is not None and recv_token_probs.dim() == 1

        if async_finish:
            event.current_stream_wait()

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.num_max_tokens_per_rank = num_max_tokens_per_rank
        ctx.hidden = hidden
        ctx.num_topk = num_topk
        ctx.num_sms = num_sms
        tokens_per_expert = _tokens_per_expert_from_psum(handle.psum_num_recv_tokens_per_expert)

        return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        buffer = _get_deepep_v2_buffer(
            ctx.group, ctx.num_max_tokens_per_rank, ctx.hidden, ctx.num_topk
        )
        grad_token_probs_for_combine = (
            grad_token_probs.float() if grad_token_probs is not None else None
        )
        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle=ctx.handle,
            topk_weights=grad_token_probs_for_combine,
            num_sms=ctx.num_sms,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None, None


class DeepEPV2Combine(torch.autograd.Function):
    """Fused combine operation using DeepEP V2 ElasticBuffer."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_expanded_layout=False,
    ):
        # PASSTHROUGH
        out = torch.zeros(
            handle.num_max_tokens_per_rank, x.shape[1], dtype=x.dtype, device=x.device
        )
        ctx.handle = handle
        ctx.group = group
        return out, None

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_expanded_layout=False,
    ):
        num_topk = handle.topk_idx.shape[1]
        hidden = x.shape[1]
        buffer = _get_deepep_v2_buffer(group, handle.num_max_tokens_per_rank, hidden, num_topk)
        num_sms = handle.num_sms
        combined_x, _, event = buffer.combine(
            x,
            handle=handle,
            num_sms=num_sms,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        if async_finish:
            event.current_stream_wait()

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.hidden = hidden
        ctx.num_topk = num_topk
        ctx.num_sms = num_sms
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        buffer = _get_deepep_v2_buffer(
            ctx.group, ctx.handle.num_max_tokens_per_rank, ctx.hidden, ctx.num_topk
        )
        grad_x, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            num_sms=ctx.num_sms,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            event.current_stream_wait()
        return grad_x, None, None, None, None, None


if HAVE_DEEP_EP_V2:

    def deepep_v2_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_expanded_layout=False,
    ):
        """Perform fused dispatch operation using DeepEP V2 ElasticBuffer."""
        return DeepEPV2Dispatch.apply(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
            use_expanded_layout,
        )

    def deepep_v2_combine(
        x,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_expanded_layout=False,
    ):
        """Perform fused combine operation using DeepEP V2 ElasticBuffer."""
        return DeepEPV2Combine.apply(
            x, group, handle, async_finish, allocate_on_comm_stream, use_expanded_layout
        )

else:
    deepep_v2_dispatch = None
    deepep_v2_combine = None


try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    num_sms_dispatch_api: Optional[int] = None,
    num_sms_combine_api: Optional[int] = None,
    num_blocks_permute: Optional[int] = None,
    num_blocks_unpermute: Optional[int] = None,
    fp8_dispatch: bool = False,
    num_sms_preprocessing_api: Optional[int] = None,
) -> None:
    '''
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group (torch.distributed.ProcessGroup):
            Process group for HybridEP all-to-all communication.
        hidden_dim (int):
            Hidden dimension of the input tensor.
        num_tokens (int):
            Maximum token count of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (Optional[int]):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (Optional[int]):
            Number of SMs used by the combine API.
        num_blocks_permute (Optional[int]):
            Number of blocks used by the permute part.
        num_blocks_unpermute (Optional[int]):
            Number of blocks used by the unpermute part.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
        num_sms_preprocessing_api (Optional[int]):
            Number of SMs used by the preprocessing (metadata scan) kernel.
    '''
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    kwargs = {}
    if num_sms_dispatch_api is not None:
        kwargs['num_sms_dispatch_api'] = num_sms_dispatch_api
    if num_sms_combine_api is not None:
        kwargs['num_sms_combine_api'] = num_sms_combine_api
    if num_blocks_permute is not None:
        kwargs['num_blocks_permute'] = num_blocks_permute
    if num_blocks_unpermute is not None:
        kwargs['num_blocks_unpermute'] = num_blocks_unpermute
    if num_sms_preprocessing_api is not None:
        kwargs['num_sms_preprocessing_api'] = num_sms_preprocessing_api
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=num_tokens,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        **kwargs,
    )


def reset_hybrid_ep_buffer():
    '''
    Reset the HybridEP buffer
    '''
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


class HybridEPDispatch(torch.autograd.Function):
    '''
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=None,
        num_sms_combine_api=None,
        num_blocks_permute=None,
        num_blocks_unpermute=None,
        fused=False,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
    ):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        if fused or num_blocks_permute is not None or num_blocks_unpermute is not None:
            import inspect
            import warnings

            sig = inspect.signature(HybridEPBuffer.dispatch_with_permute)
            if 'fuse_permute_dispatch' not in sig.parameters:
                warnings.warn(
                    "Current DeepEP version does not support fused permute dispatch or "
                    "num_blocks_permute/num_blocks_unpermute. Falling back to unfused "
                    "HybridEP dispatch.",
                    UserWarning,
                    stacklevel=2,
                )
                fused = False
                num_blocks_permute = None
                num_blocks_unpermute = None

        if _hybrid_ep_buffer is None:
            num_tokens, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                num_tokens,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                num_blocks_permute,
                num_blocks_unpermute,
                fp8_dispatch,
                num_sms_preprocessing_api,
            )
        # If we provide the num_permuted_tokens, we do not need to use sync to
        # wait for the data in pinned memory ready
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
            **({"fuse_permute_dispatch": fused} if fused else {}),
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.fused = fused
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        '''
        Backward pass of fused dispatch of the HybridEP backend
        '''
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            **({"fuse_unpermute_combine": ctx.fused} if ctx.fused else {}),
        )
        return (
            combined_hidden,
            None,
            combined_probs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@internal_api
class HybridEPCombine(torch.autograd.Function):
    '''
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None, fused=False):
        '''
        Forward pass of fused combine of the HybridEP backend
        '''
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=pad_multiple,
            **({"fuse_unpermute_combine": fused} if fused else {}),
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.fused = fused
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        '''
        Backward pass of fused combine of the HybridEP backend
        '''
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
            **({"fuse_permute_dispatch": ctx.fused} if ctx.fused else {}),
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=None,
        num_sms_combine_api=None,
        num_blocks_permute=None,
        num_blocks_unpermute=None,
        fused=False,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
    ):
        '''
        Perform fused dispatch for "permute + dispatch a2a + permute" using the
        HybridEP backend.

        Args:
            x (torch.Tensor):
                Input hidden states to dispatch.
            routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
            probs (torch.Tensor):
                Routing probabilities for each token-expert pair.
            group (torch.distributed.ProcessGroup):
                Process group used for communication.
            num_local_experts (int):
                Number of local experts.
            num_sms_dispatch_api (Optional[int]):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (Optional[int]):
                Number of SMs used by the combine API.
            num_blocks_permute (Optional[int]):
                Number of blocks used by the permute part.
            num_blocks_unpermute (Optional[int]):
                Number of blocks used by the unpermute part.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
            num_sms_preprocessing_api (int):
                Number of SMs used by the preprocessing (metadata scan) kernel.
        '''
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_blocks_permute,
            num_blocks_unpermute,
            fused,
            num_permuted_tokens,
            pad_multiple,
            num_sms_preprocessing_api,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple, fused=False):
        '''
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        '''
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple, fused)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None
