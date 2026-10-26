# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transport contracts for runtime expert replica weights and gradients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, NoReturn

import torch
import torch.distributed as dist

REPLICA_EXPERT_DISPATCHER_TYPES = ("replica_peer_tma", "replica_hybridep", "replica_nccl")


class ReplicaWeightLayout(Enum):
    """Storage interpretation; quantization and TE wrappers remain in the runtime."""

    PLAIN = "plain"
    ROWWISE = "rowwise"
    COLUMNWISE = "columnwise"


if TYPE_CHECKING:
    from megatron.core.transformer.moe.home_expert_state import (
        HomeStateComponent,
        ShardedHomeExpertState,
    )


@dataclass(frozen=True, slots=True)
class ReplicaOwnership:
    """Canonical source locations, independent of execution placement.

    With no tables, source slot ``s`` lives at ``divmod(s, home_experts)``.
    Explicit tables are an extension point for nonuniform ownership, not an
    optimizer-state migration protocol. Ranks always refer to the EP group.
    """

    home_experts: int
    owner_rank: torch.Tensor | None = None
    owner_local_index: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.home_experts <= 0:
            raise ValueError("Canonical home expert count must be positive.")
        if (self.owner_rank is None) != (self.owner_local_index is None):
            raise ValueError("Explicit ownership requires both rank and local-index tables.")


@dataclass(frozen=True, slots=True, eq=False)
class ReplicaPlacement:
    """Physical source-home slots in replica destinations, independent of logical identity.

    ``slot_to_expert`` is int32 ``[ep_size, local_replica_slots]``; -1 means
    unused. The caller retains the tensor without mutation through backward.
    ``version`` changes whenever a reusable slot receives a new placement.
    A version is scoped to its dispatcher, not a process-global cache key.
    """

    slot_to_expert: torch.Tensor
    ownership: ReplicaOwnership
    version: int


@dataclass(frozen=True, slots=True, eq=False)
class ReplicaPreparedPlan:
    """Backend-private scheduling metadata, separate from per-operation completion.

    Backends may cache pointer metadata, chunk routing or host send/recv lists
    in ``metadata``. They must not cache source addresses without invalidation.
    """

    placement: ReplicaPlacement
    transport: ReplicaWeightTransport
    metadata: Any = None


@dataclass(frozen=True, slots=True, eq=False)
class ReplicaTransferHandle:
    """One operation's completion and keepalive state.

    Completion includes unpack/reduction, not just network transfer. Repeated
    waits must order every calling stream, without consuming the handle.
    Backends must also track allocator streams or retain in-flight storage
    until GPU completion: a Python reference does not prevent overwrites.
    """

    transport: ReplicaWeightTransport
    completion: Any
    keepalive: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class HomeExpertPlacement:
    """Destination home slots to CURRENT physical sources; no logical identities.

    ``source_slots`` is an integer [EP, H] permutation of ``range(EP * H)``.
    ``version`` identifies a single deferred proposal within one layer.
    """

    source_slots: torch.Tensor
    version: int


@dataclass(frozen=True, slots=True, eq=False)
class HomePreparedPlan:
    """Immutable host schedule for a deferred home permutation."""

    placement: HomeExpertPlacement
    transport: ReplicaWeightTransport
    metadata: Any


@dataclass(frozen=True, slots=True, eq=False)
class HomeTransferHandle:
    """Completion and staging tensors; original slot storage is never overwritten."""

    transfer: ReplicaTransferHandle
    received: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class ReplicaTransportCapabilities:
    """Conservative promises of an implemented backend, not future aspirations."""

    weight_formats: tuple[str, ...] = ()
    grad_dtypes: tuple[torch.dtype, ...] = ()
    device_plan: bool = False
    cuda_graph: bool = False
    explicit_ownership: bool = False
    home_exchange: bool = False


@dataclass(frozen=True, slots=True)
class ReplicaTransportConfig:
    """Fixed storage and launch requirements shared by replica transports."""

    group: dist.ProcessGroup
    device: torch.device
    world_size: int
    num_local_home_experts: int
    num_local_replica_slots: int
    member_shapes: tuple[tuple[int, int], tuple[int, int]]
    weight_format: str
    rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    grad_dtype: torch.dtype
    num_sms: int | None
    replica_storage: str = "shared"
    replica_mxfp8_storage: str = "reuse"
    replica_exchange_buffer_bytes: int = 256 * 1024 * 1024
    """Combined NCCL send, receive and FP32 reduction scratch budget."""


@dataclass(frozen=True, slots=True)
class ReplicaWeightSource:
    """One projection's tensors in canonical local-expert order.

    Tensor dtype/shape plus ``layout`` describe the data and scale components.
    The backend chooses its wire encoding. Pointer tables are optional legacy
    peer-TMA fast paths, never a requirement for other backends.
    """

    data: tuple[torch.Tensor, ...]
    scales: tuple[torch.Tensor, ...] | None
    layout: ReplicaWeightLayout = ReplicaWeightLayout.PLAIN
    data_bases: torch.Tensor | None = None
    scale_bases: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class ReplicaGradDestination:
    """One projection's native gradient destinations and optional pointer table."""

    tensors: tuple[torch.Tensor, ...]
    bases: torch.Tensor | None = None


class ReplicaWeightTransport(ABC):
    """Move expert storage without owning TE or optimizer semantics.

    Views remain stable until destroy. Callers must order producers before
    start, consumers after wait, and finish consumers before reusing storage.
    Shared workspaces require serialization across layers. Gradient reduction
    adds replica contributions to existing native staging (never clears it),
    including local replicas, with FP32 accumulation and one final cast.
    Backend reduction orders need not be bitwise identical.
    """

    transport_name = "abstract"
    capabilities = ReplicaTransportCapabilities()

    def prepare_home_exchange(self, placement: HomeExpertPlacement) -> HomePreparedPlan:
        """Reserve the exchange interface for backends including Peer-TMA."""
        raise NotImplementedError(f"{self.transport_name} home expert exchange is not implemented.")

    def exchange_home_chunks(
        self,
        *,
        state: ShardedHomeExpertState,
        components: tuple[HomeStateComponent, ...],
        plan: HomePreparedPlan,
        buffer_size_bytes: int,
    ) -> None:
        """Migrate authoritative state in place with bounded staging at a drained boundary."""
        raise NotImplementedError(
            f"{self.transport_name} does not implement chunked home exchange."
        )

    def start_home_exchange(
        self, *, sources: tuple[torch.Tensor, ...], plan: HomePreparedPlan
    ) -> HomeTransferHandle:
        """Exchange typed [H, ...] components into separate staging storage."""
        raise NotImplementedError(f"{self.transport_name} home expert exchange is not implemented.")

    def wait_home_exchange(self, handle: HomeTransferHandle) -> tuple[torch.Tensor, ...]:
        """Order all received components before the caller installs them."""
        raise NotImplementedError(f"{self.transport_name} home expert exchange is not implemented.")

    def __init__(self, config: ReplicaTransportConfig) -> None:
        if config.replica_storage not in ("shared", "layerwise"):
            raise ValueError("replica_storage must be shared or layerwise.")
        if config.replica_mxfp8_storage not in ("reuse", "separate"):
            raise ValueError("replica_mxfp8_storage must be reuse or separate.")
        self.config = config
        if config.weight_format not in self.capabilities.weight_formats:
            raise ValueError(
                f"{self.transport_name} does not support {config.weight_format} weights."
            )
        if config.grad_dtype not in self.capabilities.grad_dtypes:
            raise ValueError(
                f"{self.transport_name} does not support {config.grad_dtype} gradients."
            )

    def prepare_plan(self, placement: ReplicaPlacement) -> ReplicaPreparedPlan:
        """Validate metadata without GPU-to-CPU reads; subclasses compile schedules.

        Host-scheduled backends must override this method and explicitly reject
        capture unless they implement a capture-safe scheduling path.
        """
        table = placement.slot_to_expert
        config = self.config
        if (
            table.dtype != torch.int32
            or table.device != config.device
            or tuple(table.shape) != (config.world_size, config.num_local_replica_slots)
            or not table.is_contiguous()
        ):
            raise ValueError("Replica placement must be a contiguous device int32 slot table.")
        if placement.ownership.home_experts != config.num_local_home_experts:
            raise ValueError("Replica ownership does not match the transport's home expert count.")
        if placement.ownership.owner_rank is not None and not self.capabilities.explicit_ownership:
            raise ValueError(f"{self.transport_name} requires fixed uniform canonical ownership.")
        for owners in (placement.ownership.owner_rank, placement.ownership.owner_local_index):
            if owners is not None and (
                owners.dtype not in (torch.int32, torch.int64)
                or owners.device != config.device
                or tuple(owners.shape) != (config.world_size * config.num_local_home_experts,)
                or not owners.is_contiguous()
            ):
                raise ValueError(
                    "Ownership tables must contain one device integer per logical expert."
                )
        if config.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            if not (self.capabilities.cuda_graph and self.capabilities.device_plan):
                raise ValueError(
                    f"{self.transport_name} cannot prepare a plan during CUDA capture."
                )
        return ReplicaPreparedPlan(placement=placement, transport=self)

    def validate_plan(self, plan: ReplicaPreparedPlan) -> None:
        """Reject scheduling metadata produced by another transport instance."""
        if plan.transport is not self:
            raise ValueError("Prepared replica plan belongs to a different transport.")

    @property
    @abstractmethod
    def grad_dtype(self) -> torch.dtype:
        """Return the dtype of transport-owned replica gradient storage."""

    @abstractmethod
    def projection_views(self, projection_index: int) -> tuple[tuple[Any, ...], torch.Tensor]:
        """Return local replica weight components and gradient storage for a projection."""

    @abstractmethod
    def start_weight_sync(
        self, *, sources: tuple[ReplicaWeightSource, ...], plan: ReplicaPreparedPlan
    ) -> ReplicaTransferHandle:
        """Start an asynchronous owner-to-replica weight transfer."""

    @abstractmethod
    def wait_weight_sync(self, handle: ReplicaTransferHandle) -> None:
        """Order the current stream after a weight-transfer handle."""

    @abstractmethod
    def start_grad_reduce(
        self,
        *,
        native_grads: tuple[ReplicaGradDestination, ...],
        plan: ReplicaPreparedPlan,
        projections: tuple[int, ...],
    ) -> ReplicaTransferHandle:
        """Start replica-to-owner gradient reduction for selected projections."""

    @abstractmethod
    def wait_grad_reduce(self, handle: ReplicaTransferHandle) -> None:
        """Order the current stream after a gradient-reduction handle."""

    def destroy(self) -> None:
        """Release layer-local transport resources."""


class UnimplementedReplicaWeightTransport(ReplicaWeightTransport):
    """Fail before allocation or collectives for reserved transport types."""

    def __init__(self, config: ReplicaTransportConfig) -> None:
        self.config = config
        self._unavailable()

    def _unavailable(self) -> NoReturn:
        raise NotImplementedError(
            f"{self.transport_name} weight transport is not implemented. "
            "Use replica_peer_tma for the existing NVLink implementation. "
            "The old replica_hybridep configuration was renamed to replica_peer_tma; "
            "replica_hybridep now reserves the actual HybridEP weight backend."
        )

    @property
    def grad_dtype(self) -> torch.dtype:
        self._unavailable()

    def projection_views(self, projection_index: int) -> tuple[tuple[Any, ...], torch.Tensor]:
        self._unavailable()

    def prepare_plan(self, placement: ReplicaPlacement) -> ReplicaPreparedPlan:
        self._unavailable()

    def start_weight_sync(
        self, *, sources: tuple[ReplicaWeightSource, ...], plan: ReplicaPreparedPlan
    ) -> ReplicaTransferHandle:
        self._unavailable()

    def wait_weight_sync(self, handle: ReplicaTransferHandle) -> None:
        self._unavailable()

    def start_grad_reduce(
        self,
        *,
        native_grads: tuple[ReplicaGradDestination, ...],
        plan: ReplicaPreparedPlan,
        projections: tuple[int, ...],
    ) -> ReplicaTransferHandle:
        self._unavailable()

    def wait_grad_reduce(self, handle: ReplicaTransferHandle) -> None:
        self._unavailable()


def create_replica_weight_transport(
    expert_dispatcher_type: str, config: ReplicaTransportConfig
) -> ReplicaWeightTransport:
    """Build the transport selected by the single expert-dispatch configuration axis."""
    if expert_dispatcher_type == "replica_peer_tma":
        from megatron.core.transformer.moe.replica_peer_tma_transport import PeerTmaTransport

        return PeerTmaTransport(config)
    if expert_dispatcher_type == "replica_hybridep":
        from megatron.core.transformer.moe.replica_hybridep_transport import HybridEPWeightTransport

        return HybridEPWeightTransport(config)
    if expert_dispatcher_type == "replica_nccl":
        from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport

        return NcclP2PTransport(config)
    raise ValueError(
        "Unsupported replica expert-dispatch transport: " f"{expert_dispatcher_type!r}."
    )


_transport_finalizers: dict[str, Callable[[], None]] = {}


def register_replica_transport_finalizer(name: str, finalizer: Callable[[], None]) -> None:
    """Register cleanup only after a backend has allocated shared resources."""
    _transport_finalizers[name] = finalizer


def finalize_replica_weight_transports() -> None:
    """Release initialized backends without importing unused optional dependencies."""
    for name, finalizer in list(_transport_finalizers.items()):
        finalizer()
        del _transport_finalizers[name]
