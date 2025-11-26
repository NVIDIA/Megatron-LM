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

from typing import Optional
import torch
from torch._subclasses.fake_tensor import DispatchCacheInfo

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
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
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
        seq_len (int):
            Maximum sequence length of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (int):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (int):
            Number of SMs used by the combine API.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
    '''
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
    )


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
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_dispatched_tokens=None,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        if _hybrid_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 token dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
            )
        # Defaultly, the output token_per_expert and num_dispatched_tokens_tensor
        # will be put on the CPU to avoid the potential sync in combine/backward pass,
        # but if we provide the num_dispatched_tokens and num_permuted_tokens on CPU,
        # we do not need to the D2H here.
        use_host_meta = num_dispatched_tokens is None or num_permuted_tokens is None
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
            num_dispatched_tokens=num_dispatched_tokens,
            num_permuted_tokens=num_permuted_tokens,
            use_host_meta=use_host_meta,
            use_fp8=False,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_dispatched_tokens = num_dispatched_tokens
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
            num_dispatched_tokens=ctx.num_dispatched_tokens,
        )
        return combined_hidden, None, combined_probs, None, None, None, None, None, None, None


class HybridEPCombine(torch.autograd.Function):
    '''
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(
        ctx, x, handle, num_dispatched_tokens=None, num_permuted_tokens=None, pad_multiple=None
    ):
        '''
        Forward pass of fused combine of the HybridEP backend
        '''
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=pad_multiple,
            num_dispatched_tokens=num_dispatched_tokens,
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_dispatched_tokens = num_dispatched_tokens
        ctx.num_permuted_tokens = num_permuted_tokens
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
            num_dispatched_tokens=ctx.num_dispatched_tokens,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        return dispatched_hidden, None, None, None, None



try:
    from transformer_engine.pytorch.tensor import QuantizedTensor
except ImportError:
    HAVE_TE_QUANTIZED_TENSOR = False
else:
    HAVE_TE_QUANTIZED_TENSOR = True

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor 
import transformer_engine_torch as tex

class HybridEPExpertDispatch(torch.autograd.Function):
    '''
    Fused dispatch operation for expert dispatch using the HybridEP backend
    '''
    expert_dispatch_buffer = None

    @staticmethod
    def forward(ctx, routing_map, group, num_local_echo_experts, num_sms_dispatch_api, num_sms_combine_api, num_dispatched_weights, weight_chunk_size, *expert_weights):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        # Extract raw weight and scales from 
        num_total_experts = routing_map.shape[1]
        num_local_home_experts = len(expert_weights)
        weight_list = []
        scale_list = []
        weight_shape = expert_weights[0].shape
        ctx.weight_shape = weight_shape
        fp8_dispatch = False
        quantized_tensor_class = None
        for weight in expert_weights:
            if HAVE_TE_QUANTIZED_TENSOR and isinstance(weight, QuantizedTensor):
                quantized_tensor_class = weight.__class__
                row_weight, col_weight = weight.get_data_tensors()
                metadata = weight.get_metadata()
                row_scale, col_scale = (
                    metadata['rowwise_scale_inv'].view(torch.float32),
                    metadata['columnwise_scale_inv'].view(torch.float32),
                )
                weight_list.extend([row_weight.ravel(), col_weight.ravel()])
                scale_list.extend([row_scale.ravel(), col_scale.ravel()])
                fp8_dispatch = True
            else:
                weight_list.append(weight.ravel())

        # Chunk the weight for hybridep to dispatch a small piece each time
        weight_tensor = torch.stack(weight_list, dim=0).reshape(num_local_home_experts, -1)
        num_chunks_per_weight = weight_tensor.shape[1] // weight_chunk_size
        ctx.num_chunks_per_weight = num_chunks_per_weight
        ctx.num_local_echo_experts = num_local_echo_experts
        ctx.num_local_home_experts = num_local_home_experts
        weight_tensor = weight_tensor.reshape(num_local_home_experts*num_chunks_per_weight, weight_chunk_size)

        if fp8_dispatch:
            scale_tensor = torch.stack(scale_list, dim=0)
            scale_tensor = scale_tensor.reshape(num_local_home_experts*num_chunks_per_weight, -1)
        else:
            scale_tensor = None
        routing_map = (
            routing_map.reshape(num_local_home_experts, 1, num_total_experts)
            .expand(-1, num_chunks_per_weight, -1)
            .reshape(num_local_home_experts * num_chunks_per_weight, num_total_experts)
        ).contiguous()

        # Dispatch the data and scales with hybridep
        ## Initialize the buffer for hybridep
        seq_len = routing_map.shape[0]
        if HybridEPExpertDispatch.expert_dispatch_buffer is None:
            seq_len, hidden_dim = weight_tensor.shape
            HybridEPExpertDispatch.expert_dispatch_buffer = HybridEPBuffer(
                group=group,
                hidden_dim=hidden_dim,
                max_num_of_tokens_per_rank=seq_len,
                num_local_experts=num_local_echo_experts,
                use_fp8=fp8_dispatch,
                num_sms_dispatch_api=num_sms_dispatch_api,
                num_sms_combine_api=num_sms_combine_api,
            )
        use_host_meta = num_dispatched_weights is None
        if fp8_dispatch:
            assert scale_tensor.dtype == torch.float32
            assert weight_tensor.shape[1] // scale_tensor.shape[1] == 128
        # Process the dispatch
        (
            dispatched_weight,
            _,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = HybridEPExpertDispatch.expert_dispatch_buffer.dispatch_with_permute(
            hidden=weight_tensor,
            routing_map=routing_map,
            probs=None,
            scaling_factor=scale_tensor,
            pad_multiple=None,
            num_dispatched_tokens=num_dispatched_weights * num_chunks_per_weight,
            num_permuted_tokens=num_dispatched_weights * num_chunks_per_weight,
            use_host_meta=use_host_meta,
        )

        ctx.handle = handle
        if use_host_meta:
            ctx.num_dispatched_tokens = tokens_per_expert.sum()
            ctx.num_permuted_tokens = ctx.num_dispatched_tokens
        else:
            ctx.num_dispatched_tokens = num_dispatched_weights * num_chunks_per_weight
            ctx.num_permuted_tokens = num_dispatched_weights * num_chunks_per_weight

        # Wrap the data into quantized tensor
        if fp8_dispatch:
            dispatched_raw_weight = dispatched_weight.chunk(num_dispatched_weights, dim=0)
            dispatched_raw_scale = dispatched_scaling_factor.chunk(num_dispatched_weights, dim=0)
            dispatched_weight_list = []
            for i in range(num_dispatched_weights):
                row_weight, col_weight = dispatched_raw_weight[i].chunk(2, dim=0)
                row_scale, col_scale = dispatched_raw_scale[i].chunk(2, dim=0)
                if quantized_tensor_class is MXFP8Tensor:
                    weight_tensor = MXFP8Tensor(
                        weight_shape,
                        torch.bfloat16,
                        rowwise_data=row_weight.reshape(weight_shape),
                        rowwise_scale_inv=row_scale.view(torch.uint8).reshape(weight_shape[0], -1),
                        columnwise_data=col_weight.reshape(weight_shape),
                        columnwise_scale_inv=col_scale.view(torch.uint8).reshape(-1, weight_shape[1]),
                        fp8_dtype=tex.DType.kFloat8E4M3,
                        quantizer=None,
                    )
                elif quantized_tensor_class is Float8BlockwiseQTensor:
                    weight_tensor = Float8BlockwiseQTensor(
                        weight_shape,
                        torch.bfloat16,
                        rowwise_data=row_weight.reshape(weight_shape),
                        rowwise_scale_inv=row_scale.reshape(weight_shape[0], -1),
                        columnwise_data=col_weight.reshape(weight_shape),
                        columnwise_scale_inv=col_scale.reshape(-1, weight_shape[1]),
                        fp8_dtype=tex.DType.kFloat8E4M3,
                        quantizer=None,
                        is_2D_scaled=False,
                    )
                dispatched_weight_list.append(weight_tensor)
        else:
            dispatched_weight_list = [weight.reshape(weight_shape) for weight in dispatched_weight.chunk(num_dispatched_weights, dim=0)]

        ctx.handle = handle
        ctx.fp8_dispatch = fp8_dispatch
        ctx.num_local_echo_experts = num_local_echo_experts
        ctx.num_local_home_experts = num_local_home_experts
        return tuple(dispatched_weight_list)
    
    @staticmethod
    def backward(ctx, *grad_expert_weights):
        '''
        Backward pass of fused dispatch of the HybridEP backend
        '''
        # TODO: dispatch and accmualte the gradient of the expert weights with fp32
        num_chunks_per_weight = ctx.num_chunks_per_weight
        weight_shape = ctx.weight_shape
        if ctx.fp8_dispatch:
            ctx.handle[-1].hidden_dim //= 2
        # chunk the grad_expert_weights into pieces
        expert_grad_tensor = torch.stack(grad_expert_weights, dim=0).reshape(ctx.num_local_echo_experts*num_chunks_per_weight, -1)


        combined_expert_grad, _ = HybridEPExpertDispatch.expert_dispatch_buffer.combine_with_unpermute(
            hidden=expert_grad_tensor,
            probs=None,
            handle=ctx.handle,
            pad_multiple=None,
            num_dispatched_tokens=ctx.num_dispatched_tokens,
        )
        # Extract grad for each expert
        weight_grad_list = [weight_grad.reshape(weight_shape) for weight_grad in combined_expert_grad.chunk(ctx.num_local_home_experts, dim=0)]

        return None, None, None, None, None, None, None, *weight_grad_list

if HAVE_HYBRIDEP:

    def hybrid_ep_expert_dispatch(
        expert_weights,
        routing_map,
        group, 
        num_local_experts,
        num_of_experts,
        num_sms_dispatch_api,
        num_dispatched_weights,
    ):
        """
        """
        pass

    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_dispatched_tokens=None,
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
            num_sms_dispatch_api (int):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (int):
                Number of SMs used by the combine API.
            num_dispatched_tokens (int):
                Number of tokens after dispatch but before permute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from
                a GPU tensor, which causes a D2H synchronization.
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
            num_dispatched_tokens,
            num_permuted_tokens,
            pad_multiple,
        )

    def hybrid_ep_combine(x, handle, num_dispatched_tokens, num_permuted_tokens, pad_multiple):
        '''
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_dispatched_tokens (int):
                The number of tokens after unpermute but before combine. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        '''
        return HybridEPCombine.apply(
            x, handle, num_dispatched_tokens, num_permuted_tokens, pad_multiple
        )

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None
