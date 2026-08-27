# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Differentiable explicit-group P2P transport for MLA latent CP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from .layout import PhaseSpec
from .utils import _require


@dataclass(frozen=True)
class PayloadLease:
    """A synchronously ready latent payload and its original CP owner."""

    owner: int
    tensor: Tensor


class LatentCPTransport(Protocol):
    """Extension seam for future A2A+P2P transports."""

    def iter_payloads(
        self, local_payload: Tensor, phase_plan: tuple[PhaseSpec, ...]
    ) -> Iterator[PayloadLease]:
        """Yield one ready payload lease for every phase."""
        ...


class _LatentRingExchange(torch.autograd.Function):
    """One explicit-group clockwise ring hop with the exact reverse backward hop."""

    @staticmethod
    def forward(
        ctx: Any,
        payload: Tensor,
        cp_group: dist.ProcessGroup,
        previous_peer: int,
        next_peer: int,
    ) -> Tensor:
        """Send one payload clockwise and receive the preceding owner's payload."""
        receive = torch.empty_like(payload)
        operations = [
            dist.P2POp(dist.isend, payload, next_peer, group=cp_group),
            dist.P2POp(dist.irecv, receive, previous_peer, group=cp_group),
        ]
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        ctx.cp_group = cp_group
        ctx.previous_peer = previous_peer
        ctx.next_peer = next_peer
        return receive

    @staticmethod
    def backward(ctx: Any, grad_receive: Tensor) -> tuple[Tensor, None, None, None]:
        """Route the received-payload gradient through the reverse ring hop."""
        grad_receive = grad_receive.contiguous()
        grad_payload = torch.empty_like(grad_receive)
        operations = [
            dist.P2POp(dist.isend, grad_receive, ctx.previous_peer, group=ctx.cp_group),
            dist.P2POp(dist.irecv, grad_payload, ctx.next_peer, group=ctx.cp_group),
        ]
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        return grad_payload, None, None, None


class P2PRingTransport:
    """Synchronous wait-at-each-hop v1 transport."""

    def __init__(self, cp_group: dist.ProcessGroup):
        self.cp_group = cp_group
        self.group_ranks = tuple(dist.get_process_group_ranks(cp_group))
        self.rank = dist.get_rank(cp_group)
        self.size = dist.get_world_size(cp_group)
        _require(len(self.group_ranks) == self.size, "invalid CP peer list")
        self.previous_peer = self.group_ranks[(self.rank - 1) % self.size]
        self.next_peer = self.group_ranks[(self.rank + 1) % self.size]

    def iter_payloads(
        self, local_payload: Tensor, phase_plan: tuple[PhaseSpec, ...]
    ) -> Iterator[PayloadLease]:
        """Yield the local payload followed by each synchronous clockwise hop."""
        _require(len(phase_plan) == self.size, "phase-plan length must equal CP size")
        for phase_index, phase in enumerate(phase_plan):
            expected_owner = (self.rank - phase_index) % self.size
            _require(
                phase.phase == phase_index, "phase-plan indices must be contiguous"
            )
            _require(
                phase.owner == expected_owner,
                "phase-plan owner order disagrees with the P2P ring",
            )

        payload = local_payload
        for phase_index, phase in enumerate(phase_plan):
            yield PayloadLease(owner=phase.owner, tensor=payload)
            if phase_index + 1 < self.size:
                payload = _LatentRingExchange.apply(
                    payload, self.cp_group, self.previous_peer, self.next_peer
                )
