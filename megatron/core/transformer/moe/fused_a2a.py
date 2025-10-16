# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE


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
    from hybrid_ep.controller import Controller
    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_controller = None

def init_controller(group: torch.distributed.ProcessGroup, hidden_dim: int, seq_len: int, num_local_experts: int, num_of_experts: int, num_sms_dispatch_api: int, num_sms_combine_api: int, fp8_dispatch: bool):
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _controller
    _controller = Controller(
        ep_group=group,
        hidden_dim=hidden_dim,  
        seq_len=seq_len,
        num_local_experts=num_local_experts,
        num_of_experts=num_of_experts,
        fp8_dispatch=fp8_dispatch,
        use_shared_buffer=True,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,

    )
    _controller.init_ep_config()
    _controller.init_underlying()

class HybridEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, routing_map, probs, group, num_local_experts, num_of_experts, num_sms_dispatch_api, num_sms_combine_api, pad_multiple):
        if _controller is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False # Currently, we do not support fp8 dispatch
            init_controller(group, hidden_dim, seq_len, num_local_experts, num_of_experts, num_sms_dispatch_api, num_sms_combine_api, fp8_dispatch)
        enable_permute = True # Curently, we only support fuse permute to hybrid-ep
        dispatched_hidden, dispatched_probs, dispatched_scaling_factor, tokens_per_expert, handle = _controller.dispatch(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            enable_permute=True, # if set to True, the local permute will be enbaled on the hybrid-ep
            pad_multiple=pad_multiple,
            use_host_meta=True,
        )

        tokens_per_expert = tokens_per_expert.tolist()

        ctx.handle = handle
        ctx.enable_permute = enable_permute
        ctx.pad_multiple = pad_multiple
        ctx.tokens_per_expert = tokens_per_expert
        # The last tensor in handle is row_id_map, which is of shape [dispatched_tokens, topk].
        num_dispatched_tokens = handle[-1].shape[0]
        num_permuted_tokens = sum(tokens_per_expert)
        ctx.num_dispatched_tokens = num_dispatched_tokens

        return dispatched_hidden, \
            dispatched_probs, \
            dispatched_scaling_factor, \
            tokens_per_expert, \
            handle, \
            num_dispatched_tokens, \
            num_permuted_tokens


    @staticmethod
    def backward(
        ctx, 
        grad_x, 
        grad_probs, 
        grad_scaling_factor, 
        grad_tokens_per_expert, 
        grad_handle, 
        grad_num_dispatched_tokens, 
        grad_num_permuted_tokens,
    ):
        handle = ctx.handle
        combined_hidden, combined_probs = _controller.combine(
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            enable_unpermute=ctx.enable_permute,
            pad_multiple=ctx.pad_multiple,
            num_dispatched_tokens=ctx.num_dispatched_tokens,
        )
        return combined_hidden, None, combined_probs, None, None, None, None, None, None, None, None, None

class HybridEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, handle, pad_multiple, num_dispatched_tokens, num_permuted_tokens):
        enable_unpermute = True
        combined_hidden, _ = _controller.combine(
            hidden=x,
            handle=handle,
            enable_unpermute=enable_unpermute,
            pad_multiple=pad_multiple,
            num_dispatched_tokens=num_dispatched_tokens,
        )
        ctx.handle = handle
        ctx.enable_unpermute = enable_unpermute
        ctx.num_dispatched_tokens = num_dispatched_tokens
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.pad_multiple = pad_multiple
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _controller.dispatch(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            enable_permute=ctx.enable_unpermute, # if set to True, the local permute will be enbaled on the hybrid-ep
            pad_multiple=ctx.pad_multiple,
            num_dispatched_tokens=ctx.num_dispatched_tokens,
            num_permuted_tokens=ctx.num_permuted_tokens,
            use_host_meta=False,
        )
        return dispatched_hidden, None, None, None, None

if HAVE_HYBRIDEP:
    def hybrid_ep_dispatch(x, routing_map, probs, group, num_local_experts, num_of_experts, num_sms_dispatch_api, num_sms_combine_api, pad_multiple):
        return HybridEPDispatch.apply(x, routing_map, probs, group, num_local_experts, num_of_experts, num_sms_dispatch_api, num_sms_combine_api, pad_multiple)

    def hybrid_ep_combine(x, handle, pad_multiple, num_dispatched_tokens, num_permuted_tokens):
        return HybridEPCombine.apply(x, handle, pad_multiple, num_dispatched_tokens, num_permuted_tokens)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None
