# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
from typing import Optional

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
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


if HAVE_DEEP_EP:

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


try:
    from transformer_engine.pytorch import ep as te_ep

    HAVE_TE_EP = True
except ImportError:
    HAVE_TE_EP = False

_TE_EP_MISSING_MSG = (
    "transformer_engine.pytorch.ep is unavailable. The 'ncclep' flex dispatcher backend "
    "requires a TransformerEngine build with NCCL EP support (NVTE_BUILD_WITH_NCCL_EP=1)."
)


def ensure_nccl_ep_bootstrapped(
    ep_group, num_experts, max_tokens_per_rank, recv_capacity_per_rank, hidden_dim, num_sms=0
):
    """Initialize the process-wide NCCL EP context once. Idempotent.

    Collective on ``ep_group``: TE's ``ep_bootstrap`` issues a barrier and borrows the
    group's NCCL communicator, so every rank must call this with identical arguments
    before the first dispatch. Reuses TransformerEngine's own one-time flag, so repeated
    calls (e.g. once per MoE layer) are no-ops.

    Args:
        ep_group (torch.distributed.ProcessGroup): The expert-parallel process group.
        num_experts (int): Total experts across ``ep_group`` (global, not per-rank).
        max_tokens_per_rank (int): Upper bound on local input tokens per forward. Must be
            even (NCCL EP requires ``num_tokens_per_rank * inner_dim % 4 == 0``).
        recv_capacity_per_rank (int): Per-rank receive-buffer capacity in tokens. Must be
            ``>= max_tokens_per_rank``; runtime overflow hard-traps (no soft drop).
        hidden_dim (int): Token hidden size.
        num_sms (int): SM cap passed to TE as ``max_num_sms`` (0 lets TE/NCCL choose).
    """
    if not HAVE_TE_EP:
        raise RuntimeError(_TE_EP_MISSING_MSG)
    if te_ep._BOOTSTRAPPED:  # reuse TE's own one-time guard; no parallel state to drift
        return
    te_ep.ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        max_num_sms=num_sms,
    )


def nccl_ep_finalize():
    """Tear down the NCCL EP context. Idempotent; safe when never bootstrapped.

    Releases the borrowed NCCL communicator and must run before the process group is
    destroyed.
    """
    if HAVE_TE_EP:
        te_ep.ep_finalize()


class NcclEpContext:
    """Per-MoE-layer NCCL EP routing context: one TE ``EpHandle`` + one ``EpBuffer``.

    Holds a single handle+buffer. Each concurrently in-flight microbatch needs its own:
    a second forward on the same handle before the first's backward corrupts the earlier
    backward (relevant once 1F1B pipelining is enabled; ``TODO(ncclep, PP>1)``).

    Args:
        top_k (int): Routing fan-out per token (TP-folded router_topk at the call site).
        max_tokens_per_rank (int): Upper bound on local input tokens per forward.
        recv_capacity_per_rank (int): Per-rank receive-buffer capacity in tokens.
        hidden_dim (int): Token hidden size.
        num_local_experts (int): Experts owned by this rank (``num_experts // ep_size``).
        alignment (int): Per-expert packing alignment for ``recv_tokens`` (grouped-GEMM
            tile; 0 = packed contiguously by actual count).
        ep_group (torch.distributed.ProcessGroup): Required when ``use_symm_mem=True``
            (symm-mem rendezvous is collective).
        use_symm_mem (bool): Use NCCL symmetric-memory payload buffers (zero-copy) vs. the
            HBM staged-copy path. Defaults to the HBM staged-copy path.
        payload_dtype (torch.dtype): Token dtype carried through dispatch/combine.
    """

    def __init__(
        self,
        top_k,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        num_local_experts,
        alignment=0,
        ep_group=None,
        use_symm_mem=False,
        payload_dtype=torch.bfloat16,
    ):
        if not HAVE_TE_EP:
            raise RuntimeError(_TE_EP_MISSING_MSG)
        if use_symm_mem and ep_group is None:
            raise ValueError("NcclEpContext(use_symm_mem=True) requires ep_group.")
        self.handle = te_ep.EpHandle(
            top_k=top_k,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            alignment=alignment,
            payload_dtype=payload_dtype,
        )
        self.buffer = te_ep.EpBuffer(
            self.handle, ep_group=ep_group if use_symm_mem else None, use_symm_mem=use_symm_mem
        )


if HAVE_TE_EP:

    def nccl_ep_dispatch(context, tokens, topk_idx, topk_weights):
        """Autograd-aware prepare + dispatch via TransformerEngine NCCL EP.

        Args:
            context (NcclEpContext): The per-layer EP routing context.
            tokens (torch.Tensor): Local input tokens ``[num_local_tokens, hidden]``
                (leading dims flattened by TE), ``payload_dtype``.
            topk_idx (torch.Tensor): ``int64`` ``[num_local_tokens, top_k]`` global expert
                ids per token.
            topk_weights (torch.Tensor): ``float32`` ``[num_local_tokens, top_k]`` weights.

        Returns:
            tuple: ``(recv_tokens, tokens_per_expert, dispatched_probs)``:
              * ``recv_tokens``: packed received tokens ``[recv_capacity_per_rank, hidden]``,
                grouped by local expert (no separate compaction step).
              * ``tokens_per_expert``: ``int32`` ``[num_local_experts]`` device tensor of
                received counts per local expert (feeds grouped GEMM as group sizes;
                alignment-padded, == actual when ``alignment=0``).
              * ``dispatched_probs``: ``float32`` ``[recv_capacity_per_rank]`` per-slot
                weights; apply them in the expert MLP (combine is called unweighted).

            All three are views into ``context.buffer``'s persistent slots — consume them
            before the next dispatch on the same context. ``tokens_per_expert`` is
            non-differentiable.
        """
        recv_tokens, dispatched_probs, tokens_per_expert = te_ep.ep_dispatch(
            context.handle, context.buffer, tokens, topk_idx, topk_weights
        )
        return recv_tokens, tokens_per_expert, dispatched_probs

    def nccl_ep_combine(context, expert_out, num_local_tokens=None):
        """Autograd-aware combine via TransformerEngine NCCL EP (no scatter step).

        Args:
            context (NcclEpContext): The per-layer EP routing context.
            expert_out (torch.Tensor): Expert outputs ``[recv_capacity_per_rank, hidden]``,
                already weighted.
            num_local_tokens (int): Rows of the result (local token count for this
                forward). When None, TE uses ``handle.max_tokens_per_rank``.

        Returns:
            torch.Tensor: ``[num_local_tokens, hidden]`` combined output, in local token
            order.
        """
        return te_ep.ep_combine(
            context.handle, context.buffer, expert_out, num_local_tokens=num_local_tokens
        )

else:
    nccl_ep_dispatch = None
    nccl_ep_combine = None
