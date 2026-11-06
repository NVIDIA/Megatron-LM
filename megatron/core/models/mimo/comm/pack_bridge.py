# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Feature transport for packed decoders with runtime context parallelism.

The boundary is explicitly differentiated: ``forward`` returns a detached leaf;
``backward`` implements the transpose of the saved route and returns the local
encoder-output gradient. The caller accumulates all rounds before backpropagating
through its original encoder outputs. Plain distributed P2P is not autograd-aware.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class FeatureSlice:
    """A projected feature block in one producer's ``[rows, hidden]`` tensor."""

    producer_rank: int
    offset: int
    length: int


@dataclass(frozen=True)
class PackFeaturePlan:
    """One pack's feature order and actual CP ranks, with its leader first.

    ``features`` follows placeholder order, not producer order. Repeated source
    slices are allowed and their backward contributions are summed.
    """

    cp_ranks: Tuple[int, ...]
    features: Tuple[FeatureSlice, ...]

    @property
    def num_rows(self) -> int:
        """Number of projected rows in this decoder pack."""
        return sum(feature.length for feature in self.features)


@dataclass
class PackFeatureTransfer:
    """Saved per-round boundary; producer-only ranks have no features or local plan."""

    features: Optional[torch.Tensor]
    local_features: torch.Tensor
    plans: Tuple[PackFeaturePlan, ...]
    local_plan: Optional[int]
    cp_group: Optional[dist.ProcessGroup]
    backward_complete: bool = False
    forward_only: bool = False


def _routes(plans):
    """Yield a deterministic message per (pack, producer), preserving row order."""
    for plan in plans:
        producers = {}
        target_offset = 0
        for feature in plan.features:
            if feature.length:
                producers.setdefault(feature.producer_rank, []).append((feature, target_offset))
            target_offset += feature.length
        for producer, pieces in sorted(producers.items()):
            yield plan.cp_ranks[0], producer, pieces


def _wait_p2p(operations):
    if operations:
        for work in dist.batch_isend_irecv(operations):
            work.wait()


class PackFeatureBridge:
    """Gather each pack to its leader, then broadcast only within its CP group.

    This TP1/PP1 bridge accepts explicit groups and never reads global model
    parallel state. All transport ranks must call forward/backward in the same
    round order, with identical plans. Plans partition the decoder ranks into
    disjoint CP groups. Producer-only ranks have no decoder features or CP group;
    a rank with no images still participates.

    Every rank completes all producer-to-leader P2P before entering its own CP
    broadcast. Backward reduces within each CP group before reverse P2P. There
    is no global feature all-gather and no CP averaging.
    """

    def __init__(
        self,
        transport_group: dist.ProcessGroup,
        *,
        producer_ranks: Optional[Sequence[int]] = None,
        decoder_ranks: Optional[Sequence[int]] = None,
    ) -> None:
        """Declare global-rank domains; omitted domains default to colocated transport."""
        self.transport_group = transport_group
        self.rank = dist.get_rank()
        self.ranks = tuple(dist.get_process_group_ranks(transport_group))
        self.producer_ranks = tuple(self.ranks if producer_ranks is None else producer_ranks)
        self.decoder_ranks = tuple(self.ranks if decoder_ranks is None else decoder_ranks)
        if (
            not self.producer_ranks
            or not self.decoder_ranks
            or len(set(self.producer_ranks)) != len(self.producer_ranks)
            or len(set(self.decoder_ranks)) != len(self.decoder_ranks)
            or set(self.producer_ranks) | set(self.decoder_ranks) != set(self.ranks)
            or self.rank not in self.ranks
        ):
            raise ValueError("Unique producer and decoder domains must cover the transport group")
        self._transport_initialized = False

    def _validate(self, local_features, plans, cp_group):
        if local_features.ndim != 2 or not local_features.is_floating_point():
            raise ValueError("local_features must be a floating-point [rows, hidden] tensor")
        if self.rank not in self.producer_ranks and local_features.shape[0]:
            raise ValueError("Decoder-only ranks must supply an empty source feature tensor")
        ranks = [rank for plan in plans for rank in plan.cp_ranks]
        if sorted(ranks) != sorted(self.decoder_ranks) or any(not plan.cp_ranks for plan in plans):
            raise ValueError("Pack CP groups must partition the decoder ranks exactly once")
        local_plan = next((i for i, plan in enumerate(plans) if self.rank in plan.cp_ranks), None)
        if local_plan is None:
            if cp_group is not None:
                raise ValueError("Producer-only ranks must not supply a decoder CP group")
        else:
            cp_ranks = tuple(plans[local_plan].cp_ranks)
            if cp_group is None:
                if len(cp_ranks) != 1:
                    raise ValueError("A non-singleton pack requires its explicit CP process group")
            elif tuple(dist.get_process_group_ranks(cp_group)) != cp_ranks:
                raise ValueError("CP process group ranks must match the saved pack plan")
        for plan in plans:
            for feature in plan.features:
                if feature.producer_rank not in self.producer_ranks:
                    raise ValueError("Feature producer is outside the producer domain")
                if feature.offset < 0 or feature.length < 0:
                    raise ValueError("Feature offsets and lengths must be nonnegative")
                if (
                    feature.producer_rank == self.rank
                    and feature.offset + feature.length > local_features.shape[0]
                ):
                    raise ValueError("Feature slice exceeds its producer's local row count")
        return local_plan

    @torch.no_grad()
    def forward(
        self,
        local_features: torch.Tensor,
        plans: Sequence[PackFeaturePlan],
        cp_group: Optional[dist.ProcessGroup] = None,
        *,
        forward_only: bool = False,
    ) -> PackFeatureTransfer:
        """Collect projected rows into this rank's pack and return a boundary leaf.

        All producers use the same hidden size, device type and floating dtype.
        An empty producer supplies ``[0, hidden]`` on that same device. ``None``
        is accepted as the CP group for singleton packs and producer-only ranks.
        Producer-only transfers have ``features=None``. ``forward_only`` explicitly
        disables gradients on the received boundary for evaluation.
        """
        plans = tuple(plans)
        local_plan = self._validate(local_features, plans, cp_group)
        if not self._transport_initialized:
            # NCCL's first batched P2P requires all group ranks to participate.
            # Initialize it collectively even when some ranks have no messages.
            dist.all_reduce(local_features.new_zeros(1), group=self.transport_group)
            self._transport_initialized = True

        hidden = local_features.shape[1]
        plan = plans[local_plan] if local_plan is not None else None
        features = local_features.new_empty((plan.num_rows, hidden)) if plan is not None else None
        operations, buffers, received = [], [], []
        for leader, producer, pieces in _routes(plans):
            if self.rank == producer:
                payload = torch.cat(
                    [
                        local_features.narrow(0, feature.offset, feature.length)
                        for feature, _ in pieces
                    ]
                ).contiguous()
                buffers.append(payload)
                if leader == producer:
                    received.append((pieces, payload))
                else:
                    operations.append(
                        dist.P2POp(dist.isend, payload, leader, group=self.transport_group)
                    )
            elif self.rank == leader:
                rows = sum(feature.length for feature, _ in pieces)
                payload = local_features.new_empty((rows, hidden))
                buffers.append(payload)
                received.append((pieces, payload))
                operations.append(
                    dist.P2POp(dist.irecv, payload, producer, group=self.transport_group)
                )
        _wait_p2p(operations)
        for pieces, payload in received:
            assert features is not None
            offset = 0
            for feature, target_offset in pieces:
                features.narrow(0, target_offset, feature.length).copy_(
                    payload.narrow(0, offset, feature.length)
                )
                offset += feature.length
        if plan is not None and plan.num_rows and len(plan.cp_ranks) > 1:
            dist.broadcast(features, src=plan.cp_ranks[0], group=cp_group)
        return PackFeatureTransfer(
            features=features.requires_grad_(not forward_only) if features is not None else None,
            local_features=local_features,
            plans=plans,
            local_plan=local_plan,
            cp_group=cp_group,
            forward_only=forward_only,
        )

    @torch.no_grad()
    def backward(
        self,
        transfer: PackFeatureTransfer,
        feature_grad: Optional[torch.Tensor] = None,
        *,
        gradient_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the source gradient using the transpose of the saved route.

        Missing leaf gradients contribute zeros. Communication and accumulation
        use ``gradient_dtype`` (FP32 by default), which must agree on all ranks.
        The returned gradient has that dtype and the original producer shape.
        Accumulate rounds before calling encoder backward, casting to the source
        output dtype if needed. This method itself never runs encoder backward.
        """
        if transfer.forward_only:
            raise RuntimeError("A forward-only feature transfer has no backward")
        if transfer.backward_complete:
            raise RuntimeError("A pack feature transfer can only be returned once")
        if transfer.features is None:
            if feature_grad is not None:
                raise ValueError("Producer-only ranks have no decoder feature gradient")
            gradient = None
        else:
            if feature_grad is None:
                feature_grad = transfer.features.grad
            if feature_grad is None:
                gradient = torch.zeros_like(transfer.features, dtype=gradient_dtype)
            else:
                if feature_grad.shape != transfer.features.shape:
                    raise ValueError("Feature gradient shape does not match the saved boundary")
                if feature_grad.device != transfer.features.device:
                    raise ValueError("Feature gradient must be on the saved boundary's device")
                gradient = feature_grad.detach().to(dtype=gradient_dtype).contiguous().clone()
        plan = transfer.plans[transfer.local_plan] if transfer.local_plan is not None else None
        if plan is not None and plan.num_rows and len(plan.cp_ranks) > 1:
            dist.reduce(gradient, dst=plan.cp_ranks[0], group=transfer.cp_group)

        source_gradient = torch.zeros_like(transfer.local_features, dtype=gradient_dtype)
        hidden = source_gradient.shape[1]
        operations, buffers, received = [], [], []
        for leader, producer, pieces in _routes(transfer.plans):
            if self.rank == leader:
                assert gradient is not None
                payload = torch.cat(
                    [gradient.narrow(0, target, feature.length) for feature, target in pieces]
                ).contiguous()
                buffers.append(payload)
                if leader == producer:
                    received.append((pieces, payload))
                else:
                    operations.append(
                        dist.P2POp(dist.isend, payload, producer, group=self.transport_group)
                    )
            elif self.rank == producer:
                rows = sum(feature.length for feature, _ in pieces)
                payload = source_gradient.new_empty((rows, hidden))
                buffers.append(payload)
                received.append((pieces, payload))
                operations.append(
                    dist.P2POp(dist.irecv, payload, leader, group=self.transport_group)
                )
        _wait_p2p(operations)
        for pieces, payload in received:
            offset = 0
            for feature, _ in pieces:
                source_gradient.narrow(0, feature.offset, feature.length).add_(
                    payload.narrow(0, offset, feature.length)
                )
                offset += feature.length
        transfer.backward_complete = True
        return source_gradient
