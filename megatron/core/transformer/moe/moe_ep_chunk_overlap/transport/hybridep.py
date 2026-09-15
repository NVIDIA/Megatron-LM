# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The four collectives one MoE layer's overlap needs, over HybridEP. The default transport."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...fused_a2a import (
    HAVE_HYBRIDEP,
    hybrid_ep_combine_leg,
    hybrid_ep_dispatch_leg,
    make_hybrid_ep_buffer,
)
from ..ep_chunk_overlap import EXPERT_ALIGNMENT_TO_GROUPED_GEMM, _MoEEPChunkOverlapConfig
from . import DispatchedChunk, MoEEPTransport


@dataclass(frozen=True)
class HybridEPHandle:
    buffer: object
    #: HybridEP's ``dispatch_with_permute`` handle.
    routing_plan: tuple
    num_source_tokens: int
    topk_indices: torch.Tensor
    #: Cumulative alignment-padded receive counts, one per LOCAL expert.
    psum_num_recv_tokens_per_expert: torch.Tensor


class HybridEPTransport(MoEEPTransport):
    def __init__(self, config: _MoEEPChunkOverlapConfig, group) -> None:
        if not HAVE_HYBRIDEP:
            raise RuntimeError(
                "moe_ep_chunk_overlap's HYBRID_EP transport needs a deep_ep exporting "
                "HybridEPBuffer; run a HybridEP image, or MoEEPTransportType.NCCL_EP."
            )

        self.config = config
        self._buffers = tuple(
            make_hybrid_ep_buffer(
                group,
                config.hidden_size,
                config.largest_token_chunk_size,
                config.num_local_experts,
                num_sms_dispatch_api=config.num_comm_SMs,
                num_sms_combine_api=config.num_comm_SMs,
                # assume BF16 on the wire for now
                fp8_dispatch=False,
            )
            for _ in range(config.num_token_chunks)
        )
        self._next_slot = 0

    def dispatch_forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> DispatchedChunk:
        """Dispatch one BF16 source-token chunk into expanded expert-row order."""
        slot = self._next_slot
        self._next_slot = (slot + 1) % self.config.num_token_chunks
        buffer = self._buffers[slot]
        with torch.cuda.stream(stream):
            payload, weights, _scales, padded_counts, routing_plan = hybrid_ep_dispatch_leg(
                buffer,
                hidden_states,
                topk_idx=topk_indices,
                topk_weights=topk_weights,
                num_of_experts=self.config.num_experts,
                num_of_experts_per_rank=self.config.num_local_experts,
                pad_multiple=EXPERT_ALIGNMENT_TO_GROUPED_GEMM,
                num_permuted_tokens=self.config.max_num_expanded_tokens_per_chunk,
                non_blocking=True,
            )
            cumulative = torch.cumsum(padded_counts, 0)
        handle = HybridEPHandle(
            buffer=buffer,
            routing_plan=routing_plan,
            num_source_tokens=hidden_states.shape[0],
            topk_indices=topk_indices,
            psum_num_recv_tokens_per_expert=cumulative,
        )
        return self._chunk_at_capacity(payload, weights, handle)

    def dispatch_backward(
        self, grad_output: torch.Tensor, saved_handle: HybridEPHandle, stream: torch.cuda.Stream
    ) -> DispatchedChunk:
        with torch.cuda.stream(stream):
            payload, _weights, _scales, _counts, _inner = hybrid_ep_dispatch_leg(
                saved_handle.buffer,
                grad_output.contiguous(),
                handle=saved_handle.routing_plan,
                pad_multiple=EXPERT_ALIGNMENT_TO_GROUPED_GEMM,
                num_permuted_tokens=self.config.max_num_expanded_tokens_per_chunk,
                non_blocking=True,
            )

        grad_output.record_stream(stream)
        self.record_handle_stream(saved_handle, stream)
        return self._chunk_at_capacity(payload, None, saved_handle)

    def combine_forward(
        self,
        permuted_hidden_states: torch.Tensor,
        handle: HybridEPHandle,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        with torch.cuda.stream(stream):
            return hybrid_ep_combine_leg(
                handle.buffer,
                permuted_hidden_states,
                handle.routing_plan,
                pad_multiple=EXPERT_ALIGNMENT_TO_GROUPED_GEMM,
            )[0]

    def combine_backward(
        self,
        grad_permuted: torch.Tensor,
        grad_probs_permuted: torch.Tensor,
        handle: HybridEPHandle,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.stream(stream):
            grad_hidden, grad_probs_dense = hybrid_ep_combine_leg(
                handle.buffer,
                grad_permuted.contiguous(),
                handle.routing_plan,
                probs=grad_probs_permuted,
                pad_multiple=EXPERT_ALIGNMENT_TO_GROUPED_GEMM,
            )
            return grad_hidden, torch.gather(grad_probs_dense, 1, handle.topk_indices)

    @staticmethod
    def record_handle_stream(handle: HybridEPHandle, stream: torch.cuda.Stream) -> None:
        """Keep every tensor a HybridEP handle owns alive on ``stream``."""
        handle.psum_num_recv_tokens_per_expert.record_stream(stream)
        handle.topk_indices.record_stream(stream)
        for entry in handle.routing_plan:
            if torch.is_tensor(entry):
                entry.record_stream(stream)

    def destroy_buffers(self) -> None:
        self._buffers = ()
