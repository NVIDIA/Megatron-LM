# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import inspect
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
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

try:
    from transformer_engine.pytorch import MXFP8Quantizer
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.triton.gather_quantize_mxfp8 import gather_quantize_mxfp8
except ImportError:
    gather_quantize_mxfp8 = None

_hybrid_ep_buffer = None


def _dense_rows_to_tokens(dense_to_expert_map, num_dense_tokens, probs, num_permuted_tokens):
    '''
    Invert HybridEP's dense_to_expert_map: the source token of every permuted row (-1 on
    padding rows) and, when probs are given, the probabilities in permuted order (0 on padding
    rows). Sync-free: entries without a row scatter into a spare slot that is sliced off.
    '''
    assert dense_to_expert_map.dtype == torch.int32
    num_rows, num_experts = dense_to_expert_map.shape
    device = dense_to_expert_map.device
    tokens = torch.arange(num_rows, dtype=torch.int32, device=device)
    valid = (dense_to_expert_map >= 0) & (tokens[:, None] < num_dense_tokens)
    rows = torch.where(valid, dense_to_expert_map, num_permuted_tokens).long()
    row_to_token = torch.full((num_permuted_tokens + 1,), -1, dtype=torch.int32, device=device)
    row_to_token.scatter_(
        0, rows.reshape(-1), tokens[:, None].expand(num_rows, num_experts).reshape(-1)
    )
    permuted_probs = None
    if probs is not None:
        permuted_probs = torch.zeros(num_permuted_tokens + 1, dtype=torch.float32, device=device)
        permuted_probs.scatter_(0, rows[: probs.shape[0]].reshape(-1), probs.reshape(-1))
        permuted_probs = permuted_probs[:num_permuted_tokens]
    return row_to_token[:num_permuted_tokens], permuted_probs


if HAVE_HYBRIDEP:

    class HybridEPQuantizePermuteBuffer(HybridEPBuffer):
        '''
        HybridEP buffer of a single-rank group whose permutes quantize the rows to MXFP8.

        On one rank the dispatch is a self-permutation: every token row is copied into its
        expert slots and the slots of each expert are padded. This buffer keeps HybridEP's
        metadata preprocessing and its combine, and replaces the dispatch copy and the permute
        of ``dispatch_with_permute`` with one gather that writes the GroupedTensor the
        Transformer Engine grouped MLP consumes (rowwise and columnwise MXFP8 data with
        GEMM-swizzled scales), so the grouped MLP does not quantize the permuted rows again.
        The backward permute of the expert-output gradient keeps the gathered BF16 rows as
        well, for the fc2 bias gradient.
        '''

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert self.group_size == 1, "the quantizing permute is a single-rank self-permutation"
            self.quantizer = MXFP8Quantizer(
                TE_DType[torch.float8_e4m3fn], rowwise=True, columnwise=True
            )
            self.quantizer.optimize_for_gemm = True

        def dispatch_with_permute(self, **kwargs):
            '''
            HybridEP's dispatch_with_permute with the permuted rows quantized: the dispatched
            tokens are a GroupedTensor (with the BF16 rows kept when a handle is given, i.e. in
            the backward pass). Falls back to HybridEP's own permute for what the gather does
            not cover (fused permute-dispatch, scaling factors, non-BF16 rows, no static
            number of permuted tokens, a pad multiple that is not a multiple of 128).
            '''
            hidden = kwargs["hidden"]
            handle = kwargs.get("handle")
            pad_multiple = kwargs.get("pad_multiple")
            num_permuted_tokens = kwargs.get("num_permuted_tokens")
            if (
                kwargs.get("fuse_permute_dispatch")
                or kwargs.get("scaling_factor") is not None
                or hidden.dtype != torch.bfloat16
                or not pad_multiple
                or pad_multiple % 128
                or num_permuted_tokens is None
                or num_permuted_tokens % 128
            ):
                return super().dispatch_with_permute(**kwargs)
            is_backward = handle is not None
            hidden = hidden.contiguous()
            num_tokens, hidden_dim = hidden.shape
            probs = None
            if not is_backward:
                # HybridEP's own prologue of dispatch_with_permute, up to the metadata.
                topk, routing_data, probs, _ = self._prepare_routing_data(
                    topk_idx=None,
                    topk_weights=None,
                    num_of_tokens=num_tokens,
                    num_of_experts=None,
                    probs=kwargs["probs"],
                    routing_map=kwargs["routing_map"],
                    num_of_experts_per_rank=kwargs["num_of_experts_per_rank"],
                )
                num_token_slots = self._resolve_token_slots(
                    kwargs.get("num_of_tokens_per_rank"), num_tokens
                )
                routing_data = self._pad_routing_rows(routing_data, topk, num_token_slots)
                config = self.update_template_config(
                    hidden_dim=hidden_dim,
                    num_of_tokens_per_rank=num_token_slots,
                    num_local_experts=kwargs["num_of_experts_per_rank"],
                    pad_multiple=pad_multiple,
                    use_fp8=kwargs.get("use_fp8"),
                    fuse_permute_dispatch=False,
                    topk=topk,
                )
                handle_impl = self.runtime.metadata_preprocessing(
                    config=config,
                    routing_map=routing_data,
                    num_of_tokens_per_rank=num_token_slots,
                    num_of_valid_tokens=num_tokens,
                    num_permuted_tokens=num_permuted_tokens,
                    pad_multiple=pad_multiple,
                    enable_permute=True,
                    fuse_permute_dispatch=False,
                    non_blocking=True,
                )
                handle = (
                    handle_impl.sparse_to_dense_map,
                    handle_impl.rdma_to_attn_map,
                    handle_impl.attn_to_rdma_map,
                    handle_impl.num_dispatched_tokens_tensor,
                    handle_impl.local_expert_routing_map,
                    handle_impl.dense_chunk_layout,
                    handle_impl.dense_to_expert_map,
                    handle_impl.tokens_per_expert,
                    handle_impl.num_of_tokens_per_rank,
                    handle_impl.config,
                    handle_impl.num_of_valid_tokens,
                    handle_impl.overflow_flag,
                )
            # handle[5] / handle[6] / handle[7]: dense_chunk_layout / dense_to_expert_map /
            # tokens_per_expert.
            row_to_token, permuted_probs = _dense_rows_to_tokens(
                handle[6], handle[5][-1], probs, num_permuted_tokens
            )
            tokens_per_expert = handle[7].to(torch.int64)
            padded_tokens_per_expert = (tokens_per_expert + pad_multiple - 1) // pad_multiple
            padded_tokens_per_expert *= pad_multiple
            dispatched = gather_quantize_mxfp8(
                hidden,
                row_to_token,
                padded_tokens_per_expert,
                self.quantizer,
                keep_high_precision=is_backward,
            )
            return dispatched, permuted_probs, None, padded_tokens_per_expert, handle


# HybridEP dispatch/combine kernels use 64-token chunks for their public APIs.
HYBRIDEP_TOKEN_ALIGNMENT = 64


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
    permute_quantize: bool = False,
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
        permute_quantize (bool):
            Whether the permutes of a single-rank group quantize the rows to MXFP8 for the
            Transformer Engine grouped MLP (``HybridEPQuantizePermuteBuffer``).
    '''
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    if permute_quantize and gather_quantize_mxfp8 is None:
        raise ImportError(
            "The quantizing HybridEP permute requires a Transformer Engine with "
            "gather_quantize_mxfp8."
        )
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
    buffer_cls = HybridEPQuantizePermuteBuffer if permute_quantize else HybridEPBuffer
    _hybrid_ep_buffer = buffer_cls(
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
        permute_quantize=False,
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
                permute_quantize,
            )
        if isinstance(_hybrid_ep_buffer, HybridEPQuantizePermuteBuffer) != permute_quantize:
            raise RuntimeError(
                "The HybridEP buffer was initialized with permute_quantize="
                f"{not permute_quantize}; every MoE layer of a process must agree "
                "(reset_hybrid_ep_buffer() between configurations)."
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
        permute_quantize=False,
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
            permute_quantize (bool):
                On a single-rank group, quantize the permuted rows to MXFP8 for the
                Transformer Engine grouped MLP instead of permuting them in BF16; the
                dispatched tokens are then a GroupedTensor. Requires num_permuted_tokens and a
                pad_multiple that is a multiple of 128.
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
            permute_quantize,
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


def ensure_nccl_ep_bootstrapped(
    ep_group,
    num_experts,
    max_tokens_per_rank,
    recv_capacity_per_rank,
    hidden_dim,
    num_topk,
    num_sms=0,
    zero_copy=False,
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
        recv_capacity_per_rank (int, optional): Per-rank receive-buffer capacity in tokens. Must
            be ``>= max_tokens_per_rank``. ``None`` selects eager mode, where TE sizes the
            receive buffer per step from the actual received-token count.
        hidden_dim (int): Token hidden size.
        num_topk (int): Per-token top-k over ``ep_group``; sizes NCCL EP's internal buffers.
            This is the same TP-scaled top-k used for the receive-capacity budget, not the
            raw ``moe_router_topk``.
        num_sms (int): SM cap passed to TE as ``max_num_sms`` (0 lets TE/NCCL choose).
    """
    if not HAVE_TE_EP:
        raise RuntimeError(
            "transformer_engine.pytorch.ep is unavailable. The 'ncclep' flex dispatcher backend "
            "requires a TransformerEngine build with NCCL EP support (NVTE_BUILD_WITH_NCCL_EP=1)."
        )
    if is_nccl_ep_bootstrapped():  # reuse TE's own one-time guard; no parallel state to drift
        return
    te_ep.ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        num_topk=num_topk,
        max_num_sms=num_sms,
        zero_copy=zero_copy,
        drop_on_overflow=recv_capacity_per_rank is not None,
    )


def is_nccl_ep_bootstrapped() -> bool:
    """Whether TE's process-wide NCCL EP context is live. is_ep_bootstrapped is TE 3321."""
    if not HAVE_TE_EP:
        return False
    if hasattr(te_ep, "is_ep_bootstrapped"):
        return te_ep.is_ep_bootstrapped()
    return te_ep._BOOTSTRAPPED


def nccl_ep_finalize():
    """Tear down the NCCL EP context. Idempotent; safe when never bootstrapped.

    Releases the borrowed NCCL communicator and must run before the process group is
    destroyed.
    """
    if HAVE_TE_EP:
        te_ep.ep_finalize()


if HAVE_TE_EP:

    def alloc_ep_symm_buffer(shape, dtype, ep_group):
        """Allocate one persistent NCCL symm-mem buffer (per-buffer collective rendezvous). mcore's
        zero-copy buffers are all persistent and non-pool; the symm mem-pool is used only by TE for
        the per-call recv buffers it recycles."""
        return te_ep.symm_mem_alloc(shape, dtype, ep_group)

    # Whether this TE build supports quantized EP dispatch/combine payloads.
    _TE_EP_BUFFER_HAS_QUANT_RECIPE = (
        "dispatch_fwd_quant_recipe" in inspect.signature(te_ep.EpBuffer).parameters
    )

    def new_nccl_ep_buffer(
        top_k,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        num_local_experts,
        alignment=0,
        dispatch_fwd_quant_recipe=None,
        combine_bwd_quant_recipe=None,
    ):
        """Build a fresh TE EpBuffer for one dispatch/combine pair.

        The buffer owns handle_mem (the routing table dispatch writes and combine reads); a new one
        is built per dispatch and dropped after combine. Payload symm buffers are not owned here —
        they are caller-supplied to dispatch/combine or allocated on the fly by TE.

        dispatch_fwd_quant_recipe / combine_bwd_quant_recipe (TE Recipe, optional): quantize the
        dispatch forward payload / combine backward gradient over the wire,
        the corresponding output is returned as a per-expert GroupedTensor.
        """
        quant_kwargs = {}
        if dispatch_fwd_quant_recipe is not None or combine_bwd_quant_recipe is not None:
            if not _TE_EP_BUFFER_HAS_QUANT_RECIPE:
                raise RuntimeError(
                    "Quantized NCCL EP dispatch/combine payloads require a TransformerEngine "
                    "build with EpBuffer quant-recipe support "
                    "(dispatch_fwd_quant_recipe / combine_bwd_quant_recipe)."
                )
            quant_kwargs = dict(
                dispatch_fwd_quant_recipe=dispatch_fwd_quant_recipe,
                combine_bwd_quant_recipe=combine_bwd_quant_recipe,
            )
        return te_ep.EpBuffer(
            top_k=top_k,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            alignment=alignment,
            **quant_kwargs,
        )

    def nccl_ep_dispatch(
        buffer, tokens, topk_idx, topk_weights, recv_tokens=None, recv_topk_weights=None
    ):
        """Autograd-aware prepare + dispatch via TransformerEngine NCCL EP.

        Args:
            buffer (te_ep.EpBuffer): The TE EP buffer for this dispatch.
            tokens (torch.Tensor): Local input tokens ``[num_local_tokens, hidden]``
                (leading dims flattened by TE), ``payload_dtype``.
            topk_idx (torch.Tensor): ``int64`` ``[num_local_tokens, top_k]`` global expert
                ids per token.
            topk_weights (torch.Tensor): ``float32`` ``[num_local_tokens, top_k]`` weights.
            recv_tokens, recv_topk_weights (torch.Tensor, optional): caller-owned symm dispatch
                recv buffers (fp8 zero-copy). Left None, TE allocates them (bf16 zero-copy: symm
                mem-pool; normal: plain).

        Returns:
            tuple: ``(recv_tokens, tokens_per_expert, dispatched_probs)``:
              * ``recv_tokens``: packed received tokens ``[recv_rows, hidden]``, grouped by local
                expert (no separate compaction step). ``recv_rows`` is
                ``recv_capacity_per_rank``, or this step's received-token count in eager mode.
              * ``tokens_per_expert``: ``int64`` ``[num_local_experts]`` device tensor of
                received counts per local expert (feeds grouped GEMM as group sizes;
                alignment-padded, == actual when ``alignment=0``).
              * ``dispatched_probs``: ``float32`` ``[recv_rows]`` per-slot
                weights; apply them in the expert MLP (combine is called unweighted).

            ``tokens_per_expert`` is non-differentiable.
        """
        recv_tokens, dispatched_probs, tokens_per_expert = te_ep.ep_dispatch(
            buffer,
            tokens,
            topk_idx,
            topk_weights,
            recv_tokens=recv_tokens,
            recv_topk_weights=recv_topk_weights,
        )
        return recv_tokens, tokens_per_expert, dispatched_probs

    def nccl_ep_combine(buffer, expert_out, num_local_tokens=None, grad_out=None):
        """Autograd-aware combine via TransformerEngine NCCL EP (no scatter step).

        Args:
            buffer (te_ep.EpBuffer): The TE EP buffer for this combine.
            expert_out (torch.Tensor): Expert outputs ``[recv_rows, hidden]`` (the row count
                ``nccl_ep_dispatch`` returned), already weighted.
            num_local_tokens (int): Rows of the result (local token count for this
                forward). When None, TE uses ``buffer.max_tokens_per_rank``.
            grad_out (torch.Tensor, optional): caller-owned symm buffer the backward scatters the
                expert_out grad into (zero-copy). Left None, TE allocates it (bf16: symm mem-pool;
                normal: plain).

        Returns:
            torch.Tensor: ``[num_local_tokens, hidden]`` combined output, in local token
            order.
        """
        return te_ep.ep_combine(
            buffer, expert_out, num_local_tokens=num_local_tokens, grad_out=grad_out
        )

else:
    alloc_ep_symm_buffer = None
    new_nccl_ep_buffer = None
    nccl_ep_dispatch = None
    nccl_ep_combine = None
