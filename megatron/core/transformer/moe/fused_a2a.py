# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
from typing import Optional

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

# DeepEP V2 introduces `ElasticBuffer` in place of `Buffer`
# (deepseek-ai/DeepEP PR #605, merged 2026-04-29). When installed, V2
# is preferred automatically: the MoE call sites below branch on
# `HAVE_DEEP_EP_V2` and translate the V1 5/6-tuple returns and layout
# kwargs to V2's 5-tuple dispatch / 3-tuple combine contract. When V2
# is absent, the legacy `Buffer` code path runs unchanged, so this is
# a backwards-compatible addition (mirrors the `HybridEPBuffer` probe
# already present below).
try:
    from deep_ep import ElasticBuffer

    HAVE_DEEP_EP_V2 = True
except ImportError:
    HAVE_DEEP_EP_V2 = False

# `EventHandle` / `EventOverlap` live in `deep_ep.utils` in V1. In V2
# (PR #605 onwards) `EventOverlap` is defined in `deep_ep.utils.event`
# but is not re-exported from the package `__init__`. Import with
# graceful fall-through so the module loads under either flavour.
if HAVE_DEEP_EP or HAVE_DEEP_EP_V2:
    try:
        from deep_ep.utils import EventHandle, EventOverlap  # noqa: F401
    except ImportError:
        from deep_ep.utils import EventHandle  # noqa: F401
        from deep_ep.utils.event import EventOverlap  # noqa: F401

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


# V2 MoE-shape ctor parameters. `num_max_tokens_per_rank` is baked into
# V2's JIT kernel template instantiation (dispatch.hpp:138,150 in
# DeepEP), so ranks that disagree compile incompatible binaries and
# the cross-node Gin barrier (kHybridDispatchTag0) can hang with
# `signal: 1, target: 2`. Pinning at construction ensures template
# homogeneity across ranks (matches DeepEP `tests/elastic/test_ep.py`
# reference harness; also used in the downstream reproduction at
# https://github.com/antonai-work/nemo-rl-deepep-v2-efa).
_V2_NUM_MAX_TOKENS_PER_RANK = int(
    os.environ.get("MCORE_DEEPEP_V2_MAX_TOKENS_PER_RANK", "8192")
)
_V2_HIDDEN = int(os.environ.get("MCORE_DEEPEP_V2_HIDDEN", "7168"))
_V2_NUM_TOPK = int(os.environ.get("MCORE_DEEPEP_V2_NUM_TOPK", "8"))


def _is_efa_environment() -> bool:
    """Detect AWS EFA fabric so we can prefer V2's MoE-shape ctor and
    auto-cap the Queue-Pair budget. The byte-pool ctor has been
    observed to hang the tag-6 Gin cross-node barrier on EFA; the
    MoE-shape ctor does not (validated downstream at
    https://github.com/antonai-work/nemo-rl-deepep-v2-efa)."""
    return (
        os.environ.get("FI_PROVIDER", "") == "efa"
        or os.environ.get("DEEP_EP_BACKEND", "") == "nccl"
    )


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Prefers DeepEP V2 `ElasticBuffer` when available, otherwise falls
    back to the legacy `Buffer`. The returned object exposes
    `.group`, `.num_nvl_bytes`, `.num_rdma_bytes` in both cases so
    cache invalidation below is compatible.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer or ElasticBuffer: Communication buffer
    """
    global _buffer
    if HAVE_DEEP_EP_V2:
        # V2 collapses the dispatch/combine Config pair into a single
        # hint call. Fall back gracefully if the hint itself fails.
        num_nvl_bytes, num_rdma_bytes = 0, 0
        try:
            hint_bytes = ElasticBuffer.get_buffer_size_hint(
                group=group,
                num_max_tokens_per_rank=_V2_NUM_MAX_TOKENS_PER_RANK,
                hidden=_V2_HIDDEN,
                num_topk=_V2_NUM_TOPK,
                use_fp8_dispatch=False,
                allow_hybrid_mode=True,
                allow_multiple_reduction=True,
            )
            # Book-keeping only; V2 doesn't split NVL / RDMA.
            num_nvl_bytes = hint_bytes
            num_rdma_bytes = hint_bytes
        except Exception:
            pass

        if (
            _buffer is None
            or _buffer.group != group
            or getattr(_buffer, "num_nvl_bytes", 0) < num_nvl_bytes
            or getattr(_buffer, "num_rdma_bytes", 0) < num_rdma_bytes
        ):
            # EFA QP auto-cap: pass num_allocated_qps=0 so V2 clamps
            # to the per-fabric safe ceiling. Explicit non-zero values
            # over-subscribe the aws-ofi-nccl 128-slot shared GIN ring
            # (CUDA 719 at dispatch.hpp:183).
            num_qps = 0 if _is_efa_environment() else 0
            _buffer = ElasticBuffer(
                group=group,
                num_max_tokens_per_rank=_V2_NUM_MAX_TOKENS_PER_RANK,
                hidden=_V2_HIDDEN,
                num_topk=_V2_NUM_TOPK,
                use_fp8_dispatch=False,
                allow_hybrid_mode=True,
                num_allocated_qps=num_qps,
            )
            # Emulate the V1 attributes used by the cache-invalidation
            # check above (ElasticBuffer doesn't expose them natively).
            _buffer.num_nvl_bytes = num_nvl_bytes
            _buffer.num_rdma_bytes = num_rdma_bytes
        return _buffer

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

        if HAVE_DEEP_EP_V2:
            # V2 `ElasticBuffer.dispatch` infers layout internally from
            # `topk_idx`, returns a 5-tuple `(recv_x, recv_topk_idx,
            # recv_topk_weights, EPHandle, event)`, and carries
            # `num_recv_tokens_per_expert_list` on the handle.
            # `async_finish` is renamed `async_with_compute_stream`.
            # V2 contract (buffer.hpp:483): `previous_event` requires
            # `allocate_on_comm_stream=True`; seed via `capture()` if
            # the caller didn't provide one.
            _prev_evt = None
            _alloc_on_comm = allocate_on_comm_stream
            if async_finish:
                try:
                    _prev_evt = buffer.capture()
                    _alloc_on_comm = True
                except Exception:
                    _prev_evt = None

            recv_x, recv_token_indices, recv_token_probs, handle, after_event = (
                buffer.dispatch(
                    x,
                    topk_idx=token_indices,
                    topk_weights=token_probs,
                    num_experts=num_experts,
                    num_max_tokens_per_rank=_V2_NUM_MAX_TOKENS_PER_RANK,
                    num_sms=0,
                    num_qps=0,
                    previous_event=_prev_evt,
                    async_with_compute_stream=async_finish,
                    allocate_on_comm_stream=_alloc_on_comm,
                    do_expand=False,
                )
            )

            if async_finish:
                # V2 returns a raw cuda Event when `async_with_compute_stream`
                # is set; wrap via EventOverlap for V1 stream-wait.
                EventOverlap(after_event).current_stream_wait() \
                    if after_event is not None else None

            ctx.group = group
            ctx.handle = handle
            ctx.async_finish = async_finish
            ctx.allocate_on_comm_stream = allocate_on_comm_stream

            num_recv_tokens_per_expert_list = getattr(
                handle, "num_recv_tokens_per_expert_list", None
            )
            if isinstance(num_recv_tokens_per_expert_list, torch.Tensor):
                num_recv_tokens_per_expert_list = (
                    num_recv_tokens_per_expert_list.tolist()
                )
            if num_recv_tokens_per_expert_list is None:
                num_recv_tokens_per_expert_list = []
            tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

            return (
                recv_x,
                recv_token_indices,
                recv_token_probs,
                tokens_per_expert,
                handle,
            )

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

        if HAVE_DEEP_EP_V2:
            # V2 combine returns `(grad_x, grad_topk_weights, event)`.
            # `num_sms=0` tells V2 to reuse `handle.num_sms` from the
            # forward dispatch — a mismatch triggers CUDA 719 at
            # csrc/jit/handle.hpp:86.
            _prev_evt = None
            _alloc_on_comm = ctx.allocate_on_comm_stream
            if ctx.async_finish:
                try:
                    _prev_evt = buffer.capture()
                    _alloc_on_comm = True
                except Exception:
                    _prev_evt = None
            grad_x, grad_token_probs, after_event = buffer.combine(
                grad_output.contiguous(),
                handle,
                topk_weights=grad_token_probs.float(),
                num_sms=0,
                num_qps=0,
                previous_event=_prev_evt,
                async_with_compute_stream=ctx.async_finish,
                allocate_on_comm_stream=_alloc_on_comm,
            )
            if ctx.async_finish and after_event is not None:
                EventOverlap(after_event).current_stream_wait()
            return grad_x, None, grad_token_probs, None, None, None, None

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
        buffer = get_buffer(group, get_hidden_bytes(x))

        if HAVE_DEEP_EP_V2:
            _prev_evt = None
            _alloc_on_comm = allocate_on_comm_stream
            if async_finish:
                try:
                    _prev_evt = buffer.capture()
                    _alloc_on_comm = True
                except Exception:
                    _prev_evt = None
            combined_x, _, after_event = buffer.combine(
                x,
                handle=handle,
                num_sms=0,
                num_qps=0,
                previous_event=_prev_evt,
                async_with_compute_stream=async_finish,
                allocate_on_comm_stream=_alloc_on_comm,
            )
            if async_finish and after_event is not None:
                EventOverlap(after_event).current_stream_wait()
            ctx.handle = handle
            ctx.group = group
            ctx.async_finish = async_finish
            ctx.allocate_on_comm_stream = allocate_on_comm_stream
            return combined_x, None

        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
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
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))

        if HAVE_DEEP_EP_V2:
            # V2 dispatch reuses the forward handle; backward receives a
            # 5-tuple (recv_x, recv_topk_idx, recv_topk_weights, handle,
            # event). We only need the first element (grad_x).
            # V2 `ElasticBuffer.dispatch` at elastic.py:768 calls
            # `get_theoretical_num_sms(num_experts, num_topk)` before
            # resolving `num_experts` from the handle, so we must pass
            # it explicitly when `num_sms=0`.
            _prev_evt = None
            _alloc_on_comm = ctx.allocate_on_comm_stream
            if ctx.async_finish:
                try:
                    _prev_evt = buffer.capture()
                    _alloc_on_comm = True
                except Exception:
                    _prev_evt = None
            _handle_num_experts = getattr(ctx.handle, "num_experts", None)
            grad_x, _, _, _, after_event = buffer.dispatch(
                grad_output.contiguous(),
                handle=ctx.handle,
                num_experts=_handle_num_experts,
                num_sms=0,
                num_qps=0,
                previous_event=_prev_evt,
                async_with_compute_stream=ctx.async_finish,
                allocate_on_comm_stream=_alloc_on_comm,
                do_expand=False,
            )
            if ctx.async_finish and after_event is not None:
                EventOverlap(after_event).current_stream_wait()
            return grad_x, None, None, None, None

        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
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


if HAVE_DEEP_EP or HAVE_DEEP_EP_V2:

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
        """Sets the number of SMs to use for DeepEP.

        Routes to `ElasticBuffer.set_num_sms` when DeepEP V2 is
        available (the V2 `Buffer` symbol may not exist), otherwise to
        legacy `Buffer.set_num_sms`.
        """
        if HAVE_DEEP_EP_V2:
            ElasticBuffer.set_num_sms(num_sms)
        else:
            Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


try:
    import hybrid_ep_cpp
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None
_HYBRID_EP_TOKEN_ALIGNMENT = 16
_HYBRID_EP_MIN_BUFFER_TOKENS = 512
_HYBRID_EP_IB_QP_MAX_DEPTH = 65535
_HYBRID_EP_IB_DISPATCH_DEPTH_PER_TOKEN = 3


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _hybrid_ep_num_nodes(group: torch.distributed.ProcessGroup) -> int:
    """Mirror HybridEP's NVLink-domain detection without constructing the full buffer."""
    ranks_per_nvlink_domain_env = os.getenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN")
    if ranks_per_nvlink_domain_env is not None:
        ranks_per_nvlink_domain = int(ranks_per_nvlink_domain_env)
    else:
        allocator = hybrid_ep_cpp.ExtendedMemoryAllocator()
        ranks_per_nvlink_domain = allocator.detect_accessible_ranks(group)

    assert group.size() % ranks_per_nvlink_domain == 0, (
        f"The number of ranks {group.size()} should be divisible by the number of ranks per "
        f"NVLink domain {ranks_per_nvlink_domain}."
    )
    return group.size() // ranks_per_nvlink_domain


def _hybrid_ep_uses_internode_rdma(group: torch.distributed.ProcessGroup) -> bool:
    if _hybrid_ep_buffer is not None and hasattr(_hybrid_ep_buffer, "num_of_nodes"):
        return _hybrid_ep_buffer.num_of_nodes > 1
    return _hybrid_ep_num_nodes(group) > 1


def _validate_hybrid_ep_ib_tx_depth(num_tokens: int, group: torch.distributed.ProcessGroup) -> None:
    buffer_tokens = max(
        _round_up_to_multiple(num_tokens, _HYBRID_EP_TOKEN_ALIGNMENT), _HYBRID_EP_MIN_BUFFER_TOKENS
    )
    tx_depth = _HYBRID_EP_IB_DISPATCH_DEPTH_PER_TOKEN * buffer_tokens + 1
    if tx_depth <= _HYBRID_EP_IB_QP_MAX_DEPTH:
        return

    if not _hybrid_ep_uses_internode_rdma(group):
        return

    max_supported_tokens = (
        ((_HYBRID_EP_IB_QP_MAX_DEPTH - 1) // _HYBRID_EP_IB_DISPATCH_DEPTH_PER_TOKEN)
        // _HYBRID_EP_TOKEN_ALIGNMENT
        * _HYBRID_EP_TOKEN_ALIGNMENT
    )
    raise ValueError(
        f"HybridEP InfiniBand dispatch queue pair depth ({tx_depth}) exceeds the hardware "
        f"limit of {_HYBRID_EP_IB_QP_MAX_DEPTH}. DeepEP computes this depth from the "
        f"tokens per rank rounded up to a {_HYBRID_EP_TOKEN_ALIGNMENT}-token buffer "
        f"alignment ({buffer_tokens}). Reduce sequence length or micro-batch size, or "
        f"increase Tensor Parallelism (TP) / Context Parallelism (CP), so tokens per rank "
        f"are at most {max_supported_tokens} for multi-node HybridEP."
    )


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

        num_tokens, hidden_dim = x.shape[-2:]
        _validate_hybrid_ep_ib_tx_depth(num_tokens, group)
        if _hybrid_ep_buffer is None:
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
