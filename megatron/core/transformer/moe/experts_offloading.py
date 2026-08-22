# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Autograd machinery for the MoE experts offloading path.

Contents:

1) ``OffloadingExpertsGroupedMLP``: an autograd function for the forward and backward pass
   of the grouped SwiGLU MLP in offloading experts. Both passes interleave GPU computation
   with CPU-GPU communication at chunk granularity to hide the transfer latency of the
   expert weights.
"""

from __future__ import annotations

import torch

from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
    from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    from grouped_gemm.backend import batched_h2d_async

    HAVE_BATCHED_H2D = True
except ImportError:
    HAVE_BATCHED_H2D = False

from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu, weighted_swiglu_back
from megatron.core.transformer.moe.experts_offloading_util import (
    ExpertsWgradScheduler,
    StreamManager,
    get_dummy_wgrad,
    release,
)


def split_per_expert(
    t: torch.Tensor, total_token_num_per_chunk: list[int], tokens_per_expert_chunks: list[list[int]]
) -> list[list[torch.Tensor]]:
    """Split a ``[sum(tokens), K]`` tensor into per-expert views.

    Returns ``[num_chunks][chunk_size]`` views: ``out[i]`` is the operand list for
    offloading chunk ``i``, and ``flatten_per_expert(out)`` is the operand list for
    a whole-layer wgrad gemm.
    """
    return [
        list(torch.split(chunk, tokens))
        for chunk, tokens in zip(
            torch.split(t, total_token_num_per_chunk), tokens_per_expert_chunks
        )
    ]


def flatten_per_expert(per_expert: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Flatten ``split_per_expert`` output into one view per local expert."""
    return [t for chunk in per_expert for t in chunk]


class OffloadingExpertsGroupedMLP(torch.autograd.Function):
    '''
    Autograd function for Offloading Experts Grouped SwiGLU MLP.

    The forward and backward pass are implemented with chunk-level interleaving of GPU
    computation and CPU-GPU communication to hide the data transfer latency of expert
    weights.
    '''

    @staticmethod
    def _grouped_gemm(
        a_chunks: list[torch.Tensor],
        b_chunks: list[torch.Tensor],
        m_splits: list[int],
        trans_a: bool,
        trans_b: bool,
        compute_streams: list[torch.cuda.Stream],
        c: torch.Tensor | list[torch.Tensor],
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        A grouped gemm wrapper over TE's ``general_grouped_gemm``.

        Computes ``c[i] = beta * c[i] + alpha * (op(a_chunks[i]) @ op(b_chunks[i]))``.

        Both operands arrive already split into one view per expert: ``a_chunks``
        always comes from ``split_per_expert``, ``b_chunks`` either from the GPU
        weight buffers (fprop/dgrad) or from ``split_per_expert`` too (wgrad).

        ``trans_a=False`` is the dgrad/fprop layout: ``c`` is a single token-major
        ``[sum(m_splits), N]`` tensor that TE splits internally via ``m_splits``.

        ``trans_a=True`` is the wgrad layout: ``c`` holds one ``[K, N]`` weight
        gradient per expert.

        TE runs the per-expert gemms on its own pool of cuBLAS streams, forking from
        and joining back into whichever stream is current at call time. We enter from
        ``compute_streams[0]`` so that the ordering ``StreamManager`` has already
        established for the compute streams (the H2D copy-done event before the call,
        the h2d/launch stream waits after it) also applies to TE's internal streams.
        """
        assert HAVE_TE, "Offloading experts grouped gemm requires TransformerEngine."
        assert alpha == 1.0 and beta in (0.0, 1.0)
        if trans_a:
            out = c
            single_output = False
        else:
            out = [c]
            single_output = True

        # TE evaluates ``D = op(B) @ op(A)`` in row-major terms, so ``A`` takes ``b``
        # and ``B`` takes ``a``; layout[0] transposes ``b`` and layout[1] transposes ``a``.
        layout = ('T' if trans_b else 'N') + ('T' if trans_a else 'N')

        with torch.cuda.stream(compute_streams[0]):
            general_grouped_gemm(
                A=b_chunks,
                B=a_chunks,
                out=out,
                out_dtype=out[0].dtype,
                workspaces=get_multi_stream_cublas_workspace(),
                layout=layout,
                m_splits=m_splits,
                single_output=single_output,
                accumulate=(beta == 1.0),
            )

        return c

    @classmethod
    def _prefetch_expert_weights(
        cls,
        chunk_idx: int,
        cpu_weights: list[torch.nn.Parameter],
        gpu_buffers: list[list[torch.Tensor]],
        stream_manager: StreamManager,
        config: TransformerConfig,
    ) -> tuple[int, int, int, int, torch.cuda.Event]:
        h2d_stream_idx = chunk_idx % config.moe_offloading_num_stages
        gpu_buffer_idx = h2d_stream_idx
        h2d_stream = stream_manager.get_h2d_stream(h2d_stream_idx)

        with torch.cuda.stream(h2d_stream):
            experts_idx_start = chunk_idx * config.moe_offloading_chunk_size
            experts_idx_end = (chunk_idx + 1) * config.moe_offloading_chunk_size
            cpu_weights_slice = cpu_weights[experts_idx_start:experts_idx_end]
            buf = gpu_buffers[gpu_buffer_idx]

            assert len(cpu_weights_slice) == len(buf), (
                f"Number of weights in CPU slice {len(cpu_weights_slice)} does not match "
                f"number of GPU buffers {len(buf)}"
            )
            # NOTE: batched H2D copy is used to reduce cpu overhead
            if HAVE_BATCHED_H2D:
                batched_h2d_async(cpu_weights_slice, buf, h2d_stream.cuda_stream)
            else:  # NOTE: fallback to non-batched copy if not pinned
                for idx in range(experts_idx_start, experts_idx_end):
                    buf[idx - experts_idx_start].copy_(cpu_weights[idx].data, non_blocking=True)

        copy_done_event = torch.cuda.Event()
        copy_done_event.record(h2d_stream)

        # NOTE: this is a temp solution to resolve a correctness issue encoutered
        # when VPP is enabled
        # h2d_stream.synchronize()

        return (gpu_buffer_idx, h2d_stream_idx, experts_idx_start, experts_idx_end, copy_done_event)

    @classmethod
    def call_forward_a(
        cls,
        cpu_w1: list[torch.nn.Parameter],
        gpu_w1_buffers: list[list[torch.Tensor]],
        permuted_local_hidden_states: torch.Tensor,
        x_per_expert: list[list[torch.Tensor]],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[list[int]],
        stream_manager: StreamManager,
        config: TransformerConfig,
    ) -> torch.Tensor:
        """Run the fc1 gemm for every local expert, one offloading chunk at a time.

        Returns:
            torch.Tensor: the fc1 output for all local tokens, token-major.
        """
        # allocate output buffer for the first linear layer
        fc1_output = torch.empty(
            permuted_local_hidden_states.shape[0],
            config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1),
            device=permuted_local_hidden_states.device,
            dtype=permuted_local_hidden_states.dtype,
        )
        fc1_output_per_chunk = list(torch.split(fc1_output, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls._prefetch_expert_weights(
            0, cpu_w1, gpu_w1_buffers, stream_manager, config
        )

        # fc1 chunk-level interleaving computation
        stream_manager.compute_streams_wait_launch_streams()
        next_buffer_metadata = None
        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls._prefetch_expert_weights(
                    chunk_idx + 1, cpu_w1, gpu_w1_buffers, stream_manager, config
                )

            # computation on the current GPU buffer
            experts_chunk = gpu_w1_buffers[curr_buffer_metadata[0]]
            hidden_states_chunk = x_per_expert[chunk_idx]
            fc1_output_chunk = fc1_output_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            OffloadingExpertsGroupedMLP._grouped_gemm(
                a_chunks=hidden_states_chunk,
                b_chunks=experts_chunk,
                m_splits=tokens_per_expert_chunk,
                trans_a=False,
                trans_b=False,
                compute_streams=stream_manager.get_compute_stream_objects(),
                c=fc1_output_chunk,
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])

            # update current buffer metadata
            curr_buffer_metadata = (
                next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None
            )
        stream_manager.launch_streams_wait_compute_streams()

        return fc1_output

    @classmethod
    def call_forward_y(
        cls,
        cpu_w2: list[torch.nn.Parameter],
        gpu_w2_buffers: list[list[torch.Tensor]],
        permuted_local_hidden_states: torch.Tensor,
        fc1_output: torch.Tensor,
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[list[int]],
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        config: TransformerConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the weighted SwiGLU activation and run the fc2 gemm, chunk by chunk.

        Returns:
            tuple: the layer output for all local tokens, and the activation output.
        """
        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls._prefetch_expert_weights(
            0, cpu_w2, gpu_w2_buffers, stream_manager, config
        )

        s = weighted_swiglu(fc1_output, permuted_probs.unsqueeze(-1))
        s_per_expert = split_per_expert(s, total_token_num_per_chunk, tokens_per_expert_chunks)

        # fc2 chunk-level interleaving computation
        fc2_output = torch.empty_like(permuted_local_hidden_states)
        fc2_output_per_chunk = list(torch.split(fc2_output, total_token_num_per_chunk))

        stream_manager.compute_streams_wait_launch_streams()
        next_buffer_metadata = None
        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls._prefetch_expert_weights(
                    chunk_idx + 1, cpu_w2, gpu_w2_buffers, stream_manager, config
                )

            # computation on the current GPU buffer
            experts_chunk = gpu_w2_buffers[curr_buffer_metadata[0]]
            s_chunk = s_per_expert[chunk_idx]
            fc2_output_chunk = fc2_output_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            cls._grouped_gemm(
                a_chunks=s_chunk,
                b_chunks=experts_chunk,
                m_splits=tokens_per_expert_chunk,
                trans_a=False,
                trans_b=False,
                compute_streams=stream_manager.get_compute_stream_objects(),
                c=fc2_output_chunk,
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])

            # update current buffer metadata
            curr_buffer_metadata = (
                next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None
            )
        stream_manager.launch_streams_wait_compute_streams()

        return fc2_output, s

    @classmethod
    def call_backward_grad_a(
        cls,
        grad_y: torch.Tensor,
        grad_y_per_expert: list[list[torch.Tensor]],
        a: torch.Tensor,
        cpu_w2: list[torch.nn.Parameter],
        gpu_w2_buffers: list[torch.Tensor],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[list[int]],
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        config: TransformerConfig,
    ) -> torch.Tensor:
        """Compute the dgrad through fc2 and the SwiGLU, chunk by chunk.

        Returns:
            torch.Tensor: gradient of the fc1 output (the activation input).
        """
        grad_s = torch.empty(
            grad_y.shape[0], config.moe_ffn_hidden_size, device=grad_y.device, dtype=grad_y.dtype
        )
        grad_s_per_chunk = list(torch.split(grad_s, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls._prefetch_expert_weights(
            0, cpu_w2, gpu_w2_buffers, stream_manager, config
        )

        stream_manager.compute_streams_wait_launch_streams()
        next_buffer_metadata = None
        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls._prefetch_expert_weights(
                    chunk_idx + 1, cpu_w2, gpu_w2_buffers, stream_manager, config
                )

            # computation on the current GPU buffer
            experts_chunk = gpu_w2_buffers[curr_buffer_metadata[0]]
            grad_y_chunk = grad_y_per_expert[chunk_idx]
            grad_s_chunk = grad_s_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            cls._grouped_gemm(
                a_chunks=grad_y_chunk,
                b_chunks=experts_chunk,
                m_splits=tokens_per_expert_chunk,
                trans_a=False,
                trans_b=True,
                compute_streams=stream_manager.get_compute_stream_objects(),
                c=grad_s_chunk,
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])

            # update current buffer metadata
            curr_buffer_metadata = (
                next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None
            )
        stream_manager.launch_streams_wait_compute_streams()

        grad_a, grad_probs = weighted_swiglu_back(grad_s, a, permuted_probs.unsqueeze(-1))
        return grad_a, grad_probs.squeeze(-1)

    @classmethod
    def call_backward_grad_x(
        cls,
        grad_a: torch.Tensor,
        grad_a_per_expert: list[list[torch.Tensor]],
        cpu_w1: list[torch.nn.Parameter],
        gpu_w1_buffers: list[torch.Tensor],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[list[int]],
        stream_manager: StreamManager,
        config: TransformerConfig,
    ) -> torch.Tensor:
        """Compute the dgrad through fc1, chunk by chunk.

        Returns:
            torch.Tensor: gradient of the permuted expert input.
        """
        grad_x = torch.empty(
            grad_a.shape[0], config.hidden_size, device=grad_a.device, dtype=grad_a.dtype
        )
        grad_x_per_chunk = list(torch.split(grad_x, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls._prefetch_expert_weights(
            0, cpu_w1, gpu_w1_buffers, stream_manager, config
        )

        stream_manager.compute_streams_wait_launch_streams()
        next_buffer_metadata = None
        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls._prefetch_expert_weights(
                    chunk_idx + 1, cpu_w1, gpu_w1_buffers, stream_manager, config
                )

            # computation on the current GPU buffer
            experts_chunk = gpu_w1_buffers[curr_buffer_metadata[0]]
            grad_a_chunk = grad_a_per_expert[chunk_idx]
            grad_x_chunk = grad_x_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            cls._grouped_gemm(
                a_chunks=grad_a_chunk,
                b_chunks=experts_chunk,
                m_splits=tokens_per_expert_chunk,
                trans_a=False,
                trans_b=True,
                compute_streams=stream_manager.get_compute_stream_objects(),
                c=grad_x_chunk,
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])

            # update current buffer metadata
            curr_buffer_metadata = (
                next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None
            )
        stream_manager.launch_streams_wait_compute_streams()

        return grad_x

    @staticmethod
    def _wgrad_post_process(
        w: list[torch.nn.Parameter],
        wgrad_output: list[torch.Tensor],
        fuse_gradient_accumulation: bool,
    ) -> None:
        # handle ddp
        assert (
            fuse_gradient_accumulation
        ), "Only support fuse_gradient_accumulation for offloading experts."
        for i in range(len(w)):
            if fuse_gradient_accumulation:
                w[i].grad_added_to_main_grad = True
                w[i].grad = get_dummy_wgrad(w[i].shape, w[i].dtype, w[i].device)

    @classmethod
    def call_backward_grad_w2(
        cls,
        grad_y: torch.Tensor,
        grad_y_per_expert: list[torch.Tensor],
        a: torch.Tensor,
        cpu_w2: list[torch.nn.Parameter],
        gpu_w2_buffers: list[torch.Tensor],
        tokens_per_expert: list[int],
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        config: TransformerConfig,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ) -> list[torch.Tensor]:
        """Calculate the weight gradient of the second linear layer in the backward pass.

        Returns:
            list[torch.Tensor]: one weight gradient per local expert.
        """
        s = weighted_swiglu(a, permuted_probs.unsqueeze(-1))
        s_per_expert = list(torch.split(s, tokens_per_expert))

        wgrad_output = None
        alpha = 1.0
        beta = 0.0
        if fuse_gradient_accumulation:
            wgrad_output = [w.main_grad for w in cpu_w2]
            beta = 1.0
        else:
            wgrad_output = [
                torch.empty(w.shape, device=grad_y.device, dtype=w.dtype) for w in cpu_w2
            ]
            beta = 0.0

        # compute wgrad immediately if not delay_wgrad_compute or wgrad_scheduler is None
        stream_manager.compute_streams_wait_launch_streams()
        grad_w2 = cls._grouped_gemm(
            a_chunks=s_per_expert,
            b_chunks=grad_y_per_expert,
            m_splits=tokens_per_expert,
            trans_a=True,
            trans_b=False,
            compute_streams=stream_manager.get_compute_stream_objects(),
            c=wgrad_output,
            alpha=alpha,
            beta=beta,
        )
        stream_manager.launch_streams_wait_compute_streams()

        OffloadingExpertsGroupedMLP._wgrad_post_process(
            cpu_w2, wgrad_output, fuse_gradient_accumulation
        )
        return grad_w2

    @classmethod
    def call_backward_grad_w1(
        cls,
        grad_a: torch.Tensor,
        grad_a_per_expert: list[torch.Tensor],
        x_per_expert: list[torch.Tensor],
        cpu_w1: list[torch.nn.Parameter],
        tokens_per_expert: list[int],
        stream_manager: StreamManager,
        wgrad_scheduler: ExpertsWgradScheduler | None = None,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ) -> list[torch.Tensor]:
        """Calculate the weight gradient of the first linear layer in the backward pass.

        Note: For now fuse_gradient_accumulation is not supported.

        Args:
            grad_a (torch.Tensor): gradient of the input to the activation function
            grad_a_per_expert (list[torch.Tensor]): ``grad_a`` as one view per local expert
            x_per_expert (list[torch.Tensor]): input to the first linear layer, one view
                per local expert
            w1 (torch.nn.Parameter): weight parameter for the first linear layer
            tokens_per_expert (list[int]): number of tokens assigned to each expert
            fuse_gradient_accumulation (bool, optional): Fuse gradient accumulation in
                gemm. Defaults to False.

        Returns:
            torch.Tensor: gradient of the weight parameter for the first linear layer
        """
        wgrad_output = None
        alpha = 1.0
        beta = 0.0
        if fuse_gradient_accumulation:
            wgrad_output = [w.main_grad for w in cpu_w1]
            beta = 1.0
        else:
            wgrad_output = [
                torch.empty(w.shape, device=grad_a.device, dtype=w.dtype) for w in cpu_w1
            ]
            beta = 0.0

        # compute wgrad immediately if not delay_wgrad_compute or wgrad_scheduler is None
        stream_manager.compute_streams_wait_launch_streams()
        grad_w1 = cls._grouped_gemm(
            a_chunks=x_per_expert,
            b_chunks=grad_a_per_expert,
            m_splits=tokens_per_expert,
            trans_a=True,
            trans_b=False,
            compute_streams=stream_manager.get_compute_stream_objects(),
            c=wgrad_output,
            alpha=alpha,
            beta=beta,
        )
        stream_manager.launch_streams_wait_compute_streams()

        # post process wgrad for ddp
        OffloadingExpertsGroupedMLP._wgrad_post_process(cpu_w1, grad_w1, fuse_gradient_accumulation)
        return grad_w1

    @staticmethod
    def forward(ctx, *args, **kwargs) -> tuple[torch.Tensor, None]:
        """Run the offloaded grouped SwiGLU MLP forward pass.

        Returns:
            tuple: the layer output, and ``None`` for the unused bias slot.
        """
        if len(args) < 9:
            raise ValueError(
                "Insufficient arguments for forward pass of GroupedSwiMLP. "
                f"Expected at least 9, got {len(args)}"
            )

        cpu_weights: list[torch.nn.Parameter] = list(args[:-10])
        cpu_w1: list[torch.nn.Parameter] = cpu_weights[: len(cpu_weights) // 2]
        cpu_w2: list[torch.nn.Parameter] = cpu_weights[len(cpu_weights) // 2 :]
        gpu_w1_buffers: list[torch.Tensor] = args[-10]
        gpu_w2_buffers: list[torch.Tensor] = args[-9]
        permuted_local_hidden_states: torch.Tensor = args[-8]
        tokens_per_expert: torch.Tensor = args[-7]
        num_local_experts: int = args[-6]
        permuted_probs: torch.Tensor = args[-5]
        expert_wgrad_scheduler: ExpertsWgradScheduler = args[-4]
        stream_manager: StreamManager = args[-3]
        config: TransformerConfig = args[-2]
        wgrad_accumulation_and_reduce_hooks: list = args[-1]

        # split hidden states, outputs and token_per_experts into chunks for each expert chunks
        chunk_size = config.moe_offloading_chunk_size
        tokens_per_expert = tokens_per_expert.tolist()
        tokens_per_expert_chunks = [
            tokens_per_expert[i : i + chunk_size]
            for i in range(0, len(tokens_per_expert), chunk_size)
        ]
        total_token_num_per_chunk = [sum(chunk) for chunk in tokens_per_expert_chunks]
        x_per_expert = split_per_expert(
            permuted_local_hidden_states, total_token_num_per_chunk, tokens_per_expert_chunks
        )

        # forward for the first linear layer
        fc1_output = OffloadingExpertsGroupedMLP.call_forward_a(
            cpu_w1,
            gpu_w1_buffers,
            permuted_local_hidden_states,
            x_per_expert,
            total_token_num_per_chunk,
            tokens_per_expert_chunks,
            stream_manager,
            config,
        )

        # activation and forward for the second linear layer
        y, _ = OffloadingExpertsGroupedMLP.call_forward_y(
            cpu_w2,
            gpu_w2_buffers,
            permuted_local_hidden_states,
            fc1_output,
            total_token_num_per_chunk,
            tokens_per_expert_chunks,
            permuted_probs,
            stream_manager,
            config,
        )

        # context saving for backward
        ctx.wgrad_accumulation_and_reduce_hooks = wgrad_accumulation_and_reduce_hooks
        ctx.tokens_per_expert = tokens_per_expert
        ctx.tokens_per_expert_chunks = tokens_per_expert_chunks
        ctx.total_token_num_per_chunk = total_token_num_per_chunk
        ctx.expert_wgrad_scheduler = expert_wgrad_scheduler
        ctx.cpu_w1 = cpu_w1
        ctx.cpu_w2 = cpu_w2
        ctx.gpu_w1_buffers = gpu_w1_buffers
        ctx.gpu_w2_buffers = gpu_w2_buffers
        ctx.stream_manager = stream_manager
        ctx.config = config

        activation_recompute = (
            config.recompute_granularity == 'selective' and "moe_act" in config.recompute_modules
        )
        ctx.activation_recompute = activation_recompute
        if activation_recompute:
            ctx.save_for_backward(permuted_local_hidden_states, None, permuted_probs)
            release(fc1_output)
        else:
            ctx.save_for_backward(permuted_local_hidden_states, fc1_output, permuted_probs)
        return y, None

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple:
        """Run the offloaded grouped SwiGLU MLP backward pass.

        Returns:
            tuple: the input gradient, followed by ``None`` for every non-tensor input.
        """
        config: TransformerConfig = ctx.config
        cpu_w1: list[torch.nn.Parameter] = ctx.cpu_w1
        cpu_w2: list[torch.nn.Parameter] = ctx.cpu_w2
        gpu_w1_buffers: list[torch.Tensor] = ctx.gpu_w1_buffers
        gpu_w2_buffers: list[torch.Tensor] = ctx.gpu_w2_buffers
        stream_manager: StreamManager = ctx.stream_manager
        expert_wgrad_scheduler: ExpertsWgradScheduler = ctx.expert_wgrad_scheduler
        total_token_num_per_chunk: list[int] = ctx.total_token_num_per_chunk
        tokens_per_expert_chunks: list[list[int]] = ctx.tokens_per_expert_chunks
        tokens_per_expert: list[int] = ctx.tokens_per_expert
        permuted_local_hidden_states, fc1_output, permuted_probs = ctx.saved_tensors

        # rematerialize activation if needed
        x_per_expert = split_per_expert(
            permuted_local_hidden_states, total_token_num_per_chunk, tokens_per_expert_chunks
        )
        if ctx.activation_recompute:
            fc1_output = OffloadingExpertsGroupedMLP.call_forward_a(
                cpu_w1,
                gpu_w1_buffers,
                permuted_local_hidden_states,
                x_per_expert,
                total_token_num_per_chunk,
                tokens_per_expert_chunks,
                stream_manager,
                config,
            )

        grad_y = grad_outputs[0].contiguous()
        # ``grad_y`` feeds both the dgrad and the wgrad gemm, ``grad_a`` likewise, so
        # each is split once here and the views are reused by both consumers.
        grad_y_per_expert = split_per_expert(
            grad_y, total_token_num_per_chunk, tokens_per_expert_chunks
        )

        # backward computation
        grad_a, grad_probs = OffloadingExpertsGroupedMLP.call_backward_grad_a(
            grad_y,
            grad_y_per_expert,
            fc1_output,
            cpu_w2,
            gpu_w2_buffers,
            total_token_num_per_chunk,
            tokens_per_expert_chunks,
            permuted_probs,
            stream_manager,
            config,
        )

        grad_a_per_expert = (
            None
            if grad_a is None
            else split_per_expert(grad_a, total_token_num_per_chunk, tokens_per_expert_chunks)
        )

        grad_x = (
            None
            if grad_a is None
            else OffloadingExpertsGroupedMLP.call_backward_grad_x(
                grad_a,
                grad_a_per_expert,
                cpu_w1,
                gpu_w1_buffers,
                total_token_num_per_chunk,
                tokens_per_expert_chunks,
                stream_manager,
                config,
            )
        )

        grad_w2 = OffloadingExpertsGroupedMLP.call_backward_grad_w2(
            grad_y,
            flatten_per_expert(grad_y_per_expert),
            fc1_output,
            cpu_w2,
            gpu_w2_buffers,
            tokens_per_expert,
            permuted_probs,
            stream_manager,
            config,
            config.delay_wgrad_compute,
            config.gradient_accumulation_fusion,
        )

        grad_w1 = (
            None
            if grad_a is None
            else OffloadingExpertsGroupedMLP.call_backward_grad_w1(
                grad_a,
                flatten_per_expert(grad_a_per_expert),
                flatten_per_expert(x_per_expert),
                cpu_w1,
                tokens_per_expert,
                stream_manager,
                expert_wgrad_scheduler,
                config.delay_wgrad_compute,
                config.gradient_accumulation_fusion,
            )
        )

        # NOTE: gradients have been attached in _wgrad_post_process,
        # so we can return None for grad_w1 and grad_w2
        grad_w1_ret = [None for _ in cpu_w1]
        grad_w2_ret = [None for _ in cpu_w2]

        # NOTE: manually trigger wgrad accumulation hook
        # this is needed as the hook may fail to be triggered if
        # the parameter is on CPU, and hence cause hanging when
        # overlap_grad_reduce is enabled
        for hook_fn in ctx.wgrad_accumulation_and_reduce_hooks:
            hook_fn()

        return (
            *grad_w1_ret,
            *grad_w2_ret,
            None,
            None,
            grad_x,
            None,
            None,
            grad_probs,
            None,
            None,
            None,
            None,
        )


def offloading_grouped_mlp(
    cpu_w1: list[torch.nn.Parameter],
    cpu_w2: list[torch.nn.Parameter],
    gpu_w1_buffers: list[torch.Tensor],
    gpu_w2_buffers: list[torch.Tensor],
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
    expert_wgrad_scheduler: ExpertsWgradScheduler,
    stream_manager: StreamManager,
    config: TransformerConfig,
    wgrad_accumulation_and_reduce_hooks: list,
) -> torch.Tensor:
    """Autograd function for Offloading Experts Grouped SwiGLU MLP.

    Args:
        cpu_w1 (list[torch.nn.Parameter]): CPU weight parameters for the first linear layer
        cpu_w2 (list[torch.nn.Parameter]): CPU weight parameters for the second linear layer
        gpu_w1_buffers (list[torch.Tensor]): GPU buffers for w1 weights
        gpu_w2_buffers (list[torch.Tensor]): GPU buffers for w2 weights
        permuted_local_hidden_states (torch.Tensor): input hidden states
        tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
        num_local_experts (int): number of local experts
        permuted_probs (torch.Tensor): probability derived from router
        expert_wgrad_scheduler (ExpertsWgradScheduler): scheduler for expert weight gradients
        stream_manager (StreamManager): manager for CUDA streams
        config (TransformerConfig): transformer configuration

    Returns:
        torch.Tensor: output of the MLP
    """
    output, _ = OffloadingExpertsGroupedMLP.apply(
        *cpu_w1,
        *cpu_w2,
        gpu_w1_buffers,
        gpu_w2_buffers,
        permuted_local_hidden_states,
        tokens_per_expert,
        num_local_experts,
        permuted_probs,
        expert_wgrad_scheduler,
        stream_manager,
        config,
        wgrad_accumulation_and_reduce_hooks,
    )

    return output
