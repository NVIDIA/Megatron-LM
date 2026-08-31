# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Differentiable explicit-group P2P transport for MLA latent CP."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from .layout import PhaseSpec
from .utils import _require


@dataclass(frozen=True)
class PayloadLease:
    """A consumer-stream-ordered latent payload and its original CP owner."""

    owner: int
    tensor: Tensor


class LatentCPTransport(Protocol):
    """Extension seam for future A2A+P2P transports."""

    def iter_payloads(
        self, local_payload: Tensor, phase_plan: tuple[PhaseSpec, ...]
    ) -> Iterator[PayloadLease]:
        """Yield one consumer-stream-ordered payload lease for every phase."""
        ...


_COMMUNICATION_STREAM_LOCK = threading.RLock()
_COMMUNICATION_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


def _communication_stream(payload: Tensor) -> torch.cuda.Stream | None:
    """Return one process/device-local stream shared by latent ring transports."""

    if not payload.is_cuda:
        return None
    device_index = payload.device.index
    _require(device_index is not None, "CUDA payload must have a concrete device")
    key = (os.getpid(), device_index)
    with _COMMUNICATION_STREAM_LOCK:
        stream = _COMMUNICATION_STREAMS.get(key)
        if stream is None:
            with torch.cuda.device(payload.device):
                stream = torch.cuda.Stream(device=payload.device)
            _COMMUNICATION_STREAMS[key] = stream
        return stream


@dataclass
class _PendingExchange:
    """CUDA readiness state for one prefetched receive."""

    ready_event: torch.cuda.Event | None = None
    send_tensor: Tensor | None = None
    waited: bool = False

    def wait_on_current_stream(self, receive: Tensor) -> None:
        """Order the consumer stream after the prefetched receive without a host wait."""

        _require(not self.waited, "a prefetched ring payload was consumed twice")
        if self.ready_event is not None:
            torch.cuda.current_stream(receive.device).wait_event(self.ready_event)
        self.waited = True
        self.send_tensor = None


def _launch_ring_exchange(
    payload: Tensor,
    cp_group: dist.ProcessGroup,
    send_peer: int,
    receive_peer: int,
    communication_stream: torch.cuda.Stream | None,
    pending: _PendingExchange,
    wait_for_compute_stream: bool,
) -> Tensor:
    """Launch one explicit-group exchange, isolated from the attention stream."""

    receive = torch.empty_like(payload)
    operations = [
        dist.P2POp(dist.isend, payload, send_peer, group=cp_group),
        dist.P2POp(dist.irecv, receive, receive_peer, group=cp_group),
    ]
    if communication_stream is None:
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        pending.send_tensor = payload
        return receive

    with torch.cuda.stream(communication_stream):
        if wait_for_compute_stream:
            communication_stream.wait_stream(torch.cuda.current_stream(payload.device))
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        ready_event = torch.cuda.Event()
        ready_event.record(communication_stream)
        payload.record_stream(communication_stream)
        receive.record_stream(communication_stream)
    pending.ready_event = ready_event
    pending.send_tensor = payload
    return receive


class _LatentRingExchange(torch.autograd.Function):
    """One explicit-group clockwise ring hop with the exact reverse backward hop."""

    @staticmethod
    def forward(
        ctx: Any,
        payload: Tensor,
        cp_group: dist.ProcessGroup,
        previous_peer: int,
        next_peer: int,
        communication_stream: torch.cuda.Stream | None,
        pending: _PendingExchange,
        wait_for_compute_stream: bool,
    ) -> Tensor:
        """Prefetch the preceding owner's payload on the communication stream."""
        ctx.cp_group = cp_group
        ctx.previous_peer = previous_peer
        ctx.next_peer = next_peer
        ctx.communication_stream = communication_stream
        return _launch_ring_exchange(
            payload,
            cp_group,
            next_peer,
            previous_peer,
            communication_stream,
            pending,
            wait_for_compute_stream,
        )

    @staticmethod
    def backward(
        ctx: Any, grad_receive: Tensor
    ) -> tuple[Tensor, None, None, None, None, None, None]:
        """Route the received-payload gradient through the reverse ring hop."""
        grad_receive = grad_receive.contiguous()
        pending = _PendingExchange()
        grad_payload = _launch_ring_exchange(
            grad_receive,
            ctx.cp_group,
            ctx.previous_peer,
            ctx.next_peer,
            ctx.communication_stream,
            pending,
            True,
        )
        pending.wait_on_current_stream(grad_payload)
        return grad_payload, None, None, None, None, None, None


class P2PRingTransport:
    """One-hop-prefetched P2P transport with an explicit reverse autograd ring."""

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
        """Yield each payload after ordering the consumer behind its prefetched receive."""
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
        pending: _PendingExchange | None = None
        communication_stream = (
            _communication_stream(local_payload) if self.size > 1 else None
        )
        for phase_index, phase in enumerate(phase_plan):
            if pending is not None:
                pending.wait_on_current_stream(payload)

            next_payload: Tensor | None = None
            next_pending: _PendingExchange | None = None
            if phase_index + 1 < self.size:
                next_pending = _PendingExchange()
                next_payload = _LatentRingExchange.apply(
                    payload,
                    self.cp_group,
                    self.previous_peer,
                    self.next_peer,
                    communication_stream,
                    next_pending,
                    phase_index == 0,
                )

            yield PayloadLease(owner=phase.owner, tensor=payload)
            if next_payload is not None and next_pending is not None:
                payload = next_payload
                pending = next_pending
