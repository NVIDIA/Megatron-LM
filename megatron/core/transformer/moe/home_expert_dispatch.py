# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deferred physical home exchange, independent of logical expert identity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import SchedulerContext
from megatron.core.transformer.moe.replica_weight_transport import HomeExpertPlacement

if TYPE_CHECKING:
    from megatron.core.transformer.moe.home_expert_state import ShardedHomeExpertState
    from megatron.core.transformer.moe.replica_weight_transport import ReplicaWeightTransport

logger = logging.getLogger(__name__)


class HomeExpertDispatch(torch.nn.Module):
    """Stage a permutation in forward; migrate current state at the optimizer boundary.

    The state adapter owns training-recipe details. The borrowed transport is
    also used by replica dispatch, with separate operation handles and staging.
    Neither component needs logical expert identities.
    """

    dispatcher_name = "home"

    def __init__(self, buffer_size_bytes: int = 256 * 1024 * 1024) -> None:
        super().__init__()
        self.transport: ReplicaWeightTransport | None = None
        self.state_adapter: ShardedHomeExpertState | None = None
        if buffer_size_bytes <= 0:
            raise ValueError("Home exchange buffer size must be positive.")
        self.buffer_size_bytes = buffer_size_bytes
        self._pending = None
        self._completed_version = 0
        self._failed = False
        self._layer_number = None

    def bind_transport(self, transport: ReplicaWeightTransport) -> None:
        """Borrow the existing per-layer transport; Peer-TMA is a reserved path."""
        if not transport.capabilities.home_exchange:
            raise NotImplementedError(
                f"{transport.transport_name} home exchange is not implemented."
            )
        self.transport = transport

    def bind_state_adapter(self, state_adapter: ShardedHomeExpertState) -> None:
        """Bind after optimizer construction, without registering its Parameters again."""
        self.state_adapter = state_adapter

    def assert_healthy(self) -> None:
        """Reject reuse or serialization after a partial installation."""
        if self._failed:
            raise RuntimeError(
                "Home state installation failed; restore from a checkpoint before retrying."
            )

    def assert_ready(self) -> None:
        """Require healthy, initialized transport and optimizer bindings."""
        self.assert_healthy()
        if self.transport is None or self.state_adapter is None:
            raise RuntimeError("Home optimizer bindings must be initialized before training.")

    def export_placement(self, placement: HomeExpertPlacement) -> None:
        """Execute an isolated terminal exchange without consuming a pending training plan."""
        self.assert_ready()
        exporter = HomeExpertDispatch(self.buffer_size_bytes)
        exporter.bind_transport(self.transport)
        exporter.bind_state_adapter(self.state_adapter)
        exporter.dispatch(None, placement, None)
        try:
            exporter.step()
        except Exception:
            # Export cannot be retried after an uncertain/partial installation.
            self._failed = True
            raise

    @property
    def has_pending(self) -> bool:
        """Whether this layer needs the post-update migration window."""
        return self._pending is not None

    def dispatch(
        self,
        experts: torch.nn.Module | None,
        placement: HomeExpertPlacement,
        context: SchedulerContext | None,
    ) -> None:
        """Snapshot only the physical plan; never snapshot pre-update parameter values."""
        del experts
        if self.transport is None:
            raise RuntimeError("Bind a home transport before dispatch.")
        if placement.version <= self._completed_version:
            raise ValueError("Home exchange received a stale proposal.")
        if self._pending is not None:
            raise RuntimeError("Home exchange already has a pending proposal.")
        self._pending = self.transport.prepare_home_exchange(placement)
        self._layer_number = getattr(context, "layer_number", None)

    @torch.no_grad()
    def step(self) -> int | None:
        """Install all received state in stable slots and return the completion version.

        Caller must finish backward/gradient communication and defer next-step
        parameter gathers. Exceptions are fatal once installation starts; no
        optimistic mapping commit or partial retry is permitted.
        """
        self.assert_healthy()
        if self._pending is None:
            return None
        if self.state_adapter is None:
            raise RuntimeError("Bind optimizer home state before calling scheduler.step().")
        components = self.state_adapter.prepare_components()
        signature = (
            self._pending.placement.version,
            self._pending.placement.source_slots.tolist(),
            self.buffer_size_bytes,
            [(c.name, len(c.shards), c.numel, c.dtype) for c in components],
        )
        signatures = [None] * self.transport.config.world_size
        dist.all_gather_object(signatures, signature, group=self.transport.config.group)
        if any(other != signature for other in signatures):
            raise ValueError("Home plans or state component schemas differ across EP ranks.")
        # Expert-DP replicas must reconstruct the same source slots in the same
        # tile order, even when their local optimizer shard boundaries differ.
        if self.state_adapter.dp_group is not None:
            signatures = [None] * dist.get_world_size(self.state_adapter.dp_group)
            dist.all_gather_object(signatures, signature, group=self.state_adapter.dp_group)
            if any(other != signature for other in signatures):
                raise ValueError("Home plans differ across expert DP ranks.")
        try:
            self.transport.exchange_home_chunks(
                state=self.state_adapter,
                components=components,
                plan=self._pending,
                buffer_size_bytes=self.buffer_size_bytes,
            )
        except Exception:
            # Earlier chunks may already have been installed. Never reuse this
            # state or publish a mapping after any uncertain partial exchange.
            self._failed = True
            raise
        version = self._pending.placement.version
        if dist.get_rank(self.transport.config.group) == 0:
            sources = self._pending.placement.source_slots.flatten()
            moved = (sources != torch.arange(sources.numel(), device=sources.device)).sum().item()
            logger.info(
                "Home expert exchange completed: layer=%s version=%d moved_slots=%d/%d",
                self._layer_number,
                version,
                moved,
                sources.numel(),
            )
        self._pending = None
        self._completed_version = version
        return version
