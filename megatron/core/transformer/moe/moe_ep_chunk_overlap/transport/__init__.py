# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

__all__ = ["DispatchedChunk", "MoEEPTransport", "MoEEPTransportType", "make_ep_transport"]


@dataclass(frozen=True)
class DispatchedChunk:
    """What one dispatch hands back: the payload, its routing probabilities, and the handle.

    ``recv_x`` is always ``config.max_num_expanded_tokens_per_chunk`` rows of plain contiguous
    BF16 (rule one, enforced by :meth:`MoEEPTransport._chunk_at_capacity`).
    ``received_topk_weights`` is ``None`` on the backward dispatch, which never reads it.
    """

    recv_x: torch.Tensor
    received_topk_weights: torch.Tensor | None
    dispatch_handle: object


class MoEEPTransport(ABC):
    """One chunk's four collectives. Streams, chunk lifecycle and expert math are the runtime's."""

    @abstractmethod
    def dispatch_forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> DispatchedChunk:
        """Send one BF16 source-token chunk to its experts, in expanded expert-row order."""

    @abstractmethod
    def combine_forward(
        self, permuted_hidden_states: torch.Tensor, handle, stream: torch.cuda.Stream
    ) -> torch.Tensor:
        """Return one FC2 output chunk to source-token order, summing each token's experts."""

    @abstractmethod
    def dispatch_backward(
        self, grad_output: torch.Tensor, saved_handle, stream: torch.cuda.Stream
    ) -> DispatchedChunk:
        """Apply the combine adjoint: re-send one output-gradient chunk into the forward's layout.
        ``received_topk_weights`` is unused here and may be ``None``: the backward never reads."""

    @abstractmethod
    def combine_backward(
        self,
        grad_permuted: torch.Tensor,
        grad_probs_permuted: torch.Tensor,
        handle,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the dispatch adjoint, returning ``(dX, dprob)`` in source-token order.
        ``grad_probs_permuted`` goes in FLAT per expanded row, ``dprob`` out ``[tokens, topk]``."""

    @staticmethod
    @abstractmethod
    def record_handle_stream(handle, stream: torch.cuda.Stream) -> None:
        """Keep every tensor ``handle`` owns alive on ``stream``."""

    @abstractmethod
    def destroy_buffers(self) -> None:
        """Drop this layer's collective resources."""

    def _chunk_at_capacity(
        self, payload: torch.Tensor, topk_weights: torch.Tensor | None, handle: object
    ) -> DispatchedChunk:
        """Wrap one dispatch's receive as the seam's payload, at the width rule one fixes."""
        capacity = self.config.max_num_expanded_tokens_per_chunk
        if payload.shape[0] != capacity:
            raise RuntimeError(
                f"{type(self).__name__} returned {payload.shape[0]} expanded rows, not the "
                f"{capacity} the expert GEMMs are compiled for: its capacity formula has drifted"
            )
        return DispatchedChunk(
            recv_x=payload, received_topk_weights=topk_weights, dispatch_handle=handle
        )


class MoEEPTransportType(Enum):
    NCCL_EP = "nccl_ep"
    HYBRID_EP = "hybrid_ep"


def make_ep_transport(config, group, *, transport_type: MoEEPTransportType) -> MoEEPTransport:
    if transport_type is MoEEPTransportType.NCCL_EP:
        from .ncclep import NcclEPTransport

        return NcclEPTransport(config, group)

    if transport_type is MoEEPTransportType.HYBRID_EP:
        from .hybridep import HybridEPTransport

        return HybridEPTransport(config, group)

    raise ValueError(f"no transport is registered for {transport_type!r}")
