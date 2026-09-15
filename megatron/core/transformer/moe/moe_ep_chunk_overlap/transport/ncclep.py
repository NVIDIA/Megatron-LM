# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ep_chunk_overlap import EXPERT_ALIGNMENT_TO_GROUPED_GEMM, _MoEEPChunkOverlapConfig
from . import DispatchedChunk, MoEEPTransport


@dataclass(frozen=True)
class NcclEPHandle:
    buffer: object
    num_source_tokens: int
    psum_num_recv_tokens_per_expert: torch.Tensor


_ep_ready = False


def _bootstrap_ep_group(group, config: _MoEEPChunkOverlapConfig) -> None:
    from transformer_engine.pytorch.ep import ep_bootstrap

    global _ep_ready
    if _ep_ready:
        return

    ep_bootstrap(
        group,
        num_experts=config.num_experts,
        max_tokens_per_rank=config.largest_token_chunk_size,
        recv_capacity_per_rank=config.max_num_expanded_tokens_per_chunk,
        hidden_dim=config.hidden_size,
        max_num_sms=config.num_comm_SMs,
        zero_copy=False,
        max_token_dtype=torch.bfloat16,
    )
    _ep_ready = True


def release_ep_resources() -> None:
    global _ep_ready
    if not _ep_ready:
        return

    from transformer_engine.pytorch.ep import ep_finalize

    ep_finalize()
    _ep_ready = False


class NcclEPTransport(MoEEPTransport):

    def __init__(self, config: _MoEEPChunkOverlapConfig, group) -> None:
        from transformer_engine.pytorch.ep import EpBuffer

        self.config = config
        _bootstrap_ep_group(group, config)

        device = torch.device("cuda", torch.cuda.current_device())

        self._buffers = tuple(
            EpBuffer(
                top_k=config.router_topk,
                max_tokens_per_rank=config.largest_token_chunk_size,
                recv_capacity_per_rank=config.max_num_expanded_tokens_per_chunk,
                hidden_dim=config.hidden_size,
                num_local_experts=config.num_local_experts,
                alignment=EXPERT_ALIGNMENT_TO_GROUPED_GEMM,
                payload_dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(config.num_token_chunks)
        )

        self._next_slot = 0

    def _fresh_receive(
        self, device: torch.device, *, weights: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        capacity = self.config.max_num_expanded_tokens_per_chunk
        recv_tokens = torch.zeros(
            capacity, self.config.hidden_size, dtype=torch.bfloat16, device=device
        )
        if not weights:
            return recv_tokens, None
        return recv_tokens, torch.zeros(capacity, dtype=torch.float32, device=device)

    def dispatch_forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> DispatchedChunk:
        slot = self._next_slot
        self._next_slot = (slot + 1) % self.config.num_token_chunks
        buffer = self._buffers[slot]

        topk_indices = topk_indices.to(torch.int64)
        with torch.cuda.stream(stream):
            recv_tokens, recv_weights = self._fresh_receive(hidden_states.device, weights=True)
            torch.ops.transformer_engine_ep.prepare(
                buffer.handle_mem, buffer.top_k, topk_indices, buffer.token_counts, buffer.alignment
            )
            torch.ops.transformer_engine_ep.dispatch(
                buffer.handle_mem,
                topk_indices,
                hidden_states,
                topk_weights,
                recv_tokens,
                recv_weights,
            )
            alignment = EXPERT_ALIGNMENT_TO_GROUPED_GEMM
            counts = buffer.token_counts.to(torch.int64)
            cumulative = torch.cumsum(((counts + (alignment - 1)) // alignment) * alignment, 0)
        handle = NcclEPHandle(
            buffer=buffer,
            num_source_tokens=hidden_states.shape[0],
            psum_num_recv_tokens_per_expert=cumulative,
        )
        return self._chunk_at_capacity(recv_tokens, recv_weights, handle)

    def dispatch_backward(
        self, grad_output: torch.Tensor, saved_handle: NcclEPHandle, stream: torch.cuda.Stream
    ) -> DispatchedChunk:
        with torch.cuda.stream(stream):
            grad = grad_output.contiguous()
            grad_recv, _weights = self._fresh_receive(grad_output.device, weights=False)
            torch.ops.transformer_engine_ep.combine_bwd(
                saved_handle.buffer.handle_mem, grad, grad_recv
            )

        grad_output.record_stream(stream)
        self.record_handle_stream(saved_handle, stream)
        return self._chunk_at_capacity(grad_recv, None, saved_handle)

    def combine_forward(
        self, permuted_hidden_states: torch.Tensor, handle: NcclEPHandle, stream: torch.cuda.Stream
    ) -> torch.Tensor:
        with torch.cuda.stream(stream):
            result = torch.empty(
                handle.num_source_tokens,
                self.config.hidden_size,
                dtype=torch.bfloat16,
                device=permuted_hidden_states.device,
            )
            torch.ops.transformer_engine_ep.combine(
                handle.buffer.handle_mem, permuted_hidden_states, result
            )
            return result

    def combine_backward(
        self,
        grad_permuted: torch.Tensor,
        grad_probs_permuted: torch.Tensor,
        handle: NcclEPHandle,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.stream(stream):
            grad_hidden = torch.empty(
                handle.num_source_tokens,
                self.config.hidden_size,
                dtype=torch.bfloat16,
                device=grad_permuted.device,
            )
            grad_probs = torch.empty(
                handle.num_source_tokens,
                self.config.router_topk,
                dtype=torch.float32,
                device=grad_permuted.device,
            )
            torch.ops.transformer_engine_ep.dispatch_bwd(
                handle.buffer.handle_mem,
                grad_permuted.contiguous(),
                grad_probs_permuted,
                grad_hidden,
                grad_probs,
            )
            return grad_hidden, grad_probs

    @staticmethod
    def record_handle_stream(handle: NcclEPHandle, stream: torch.cuda.Stream) -> None:
        handle.psum_num_recv_tokens_per_expert.record_stream(stream)
        handle.buffer.handle_mem.record_stream(stream)
        handle.buffer.token_counts.record_stream(stream)

    def destroy_buffers(self) -> None:
        self._buffers = ()
