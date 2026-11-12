# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared cost-model protocol, NVLink domain sizes and geometry-only smoke model."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ..runtime.planner import RuntimePlanGeometry

# GPUs per NVLink domain, by the ``hardware`` of the runtime file: an H100 node has eight
# GPUs, a GB200 rack one domain of 72. A cost model takes its domain size from this table and
# exposes it as ``nvlink_domain_size``; no option, constructor argument or runtime-file key
# sets one. The packaged attention predictor keeps an equal table for its own use, and a test
# compares the two. The models read DPxCP rank ``r`` as a GPU of domain ``r // size``, so the
# launch has to place ranks consecutively, domain by domain.
NVLINK_DOMAIN_SIZES = MappingProxyType({"h100": 8, "gb200": 72})


@dataclass(frozen=True)
class PackLimits:
    """What a cost model can price in one pack of a CP size, beyond the planner geometry.

    A model that cannot price every pack the geometry admits declares it through an
    optional ``pack_limits(cp)`` method returning this object. The planner then never
    builds a pack outside the declared limits, and the start-up probe prices the largest
    pack the planner may build. ``None`` means no limit of the model's own; a new kind of
    limit is one more field here, read where the planner or the worker enforces it.
    """

    # Largest number of sequences in one pack of this CP size.
    max_sequences: Optional[int] = None
    # Longest sample, in logical tokens before any padding, in a pack of this CP size. The
    # CPU worker rejects a global batch holding a sample that no allowed CP size prices.
    max_sample_length: Optional[int] = None


def largest_priced_sequences(model, geometry: "RuntimePlanGeometry", cp: int, cap: int):
    """Most equal sequences filling one CP-``cp`` pack that ``model`` prices, up to ``cap``.

    Returns None when the model prices the largest full pack the geometry admits (it has
    no limit of its own) and 0 when it prices none. A pack with fewer sequences is assumed
    not to be harder to price, so the count is found by bisection.
    """
    ranks = list(range(cp))

    def priced(count):
        try:
            lengths = geometry.full_pack(cp, count)
            if getattr(model, "program_costs", None) is not None:
                model.program_costs(lengths, cp, ranks=ranks)
            model.forward(lengths, cp, ranks=ranks)
            model.backward(lengths, cp, ranks=ranks)
        except ValueError:
            return False
        return True

    # Counts for which a full pack of aligned sequences exists at all.
    most = min(int(cap), geometry.capacity(cp) // geometry.pack_unit(cp))
    if most < 1 or priced(most):
        return None
    low, high = 0, most
    while high - low > 1:
        middle = (low + high) // 2
        if priced(middle):
            low = middle
        else:
            high = middle
    return low


@runtime_checkable
class DcpCostModel(Protocol):
    """Callable contract consumed by ``models.cache.PackCostModel`` and the planner.

    ``nvlink_domain_size`` is the number of consecutive DPxCP ranks the model takes to share
    one NVLink domain: the ``NVLINK_DOMAIN_SIZES`` entry of its hardware, or None when the
    model does not know the hardware its coefficients describe. Start-up prints it and the
    plan metadata records it.

    Optional: ``pack_limits(cp) -> PackLimits`` when the model cannot price every pack
    that the planner geometry admits at a CP size; the class attribute ``largest_cp_size``
    when it prices CP groups up to that size only (the scheduler then forms no larger
    one); ``required_global_batch_size`` when its step overhead describes one global
    batch size (start-up and the CPU worker then reject any other).
    """

    name: str
    source_hashes: Dict[str, str]
    nvlink_domain_size: Optional[int]
    program_costs: Optional[Callable[..., Dict[str, float]]]

    def forward(
        self, lengths: Tuple[int, ...], cp: int, *, ranks: Optional[Sequence[int]] = None
    ) -> float:
        """Forward attention ms per layer for one pack."""

    def backward(
        self, lengths: Tuple[int, ...], cp: int, *, ranks: Optional[Sequence[int]] = None
    ) -> float:
        """Backward attention ms per layer for one pack."""

    def timing(self) -> Optional[dict]:
        """Execution program for ``TimelineModel`` (program mode) or None (scalar mode)."""

    def clear(self) -> None:
        """Drop memoized pack costs."""

    def check_domain(self, geometry: "RuntimePlanGeometry", config) -> None:
        """Raise ValueError when the geometry/config is outside the calibrated domain."""


class QuadraticCostModel:
    """Geometry-agnostic smoke model: ms = sum((stored/cp)^2)/1e6 + cp_step_ms*(cp-1).

    Scalar timeline (no execution program); test/smoke only.
    """

    name = "quadratic"
    nvlink_domain_size = None
    program_costs = None
    layer_count = 1
    forward_non_attention_ms = 1.0
    backward_non_attention_ms = 2.0

    def __init__(
        self, geometry: "RuntimePlanGeometry", cp_step_ms: float = 0.05, **_ignored
    ) -> None:
        self.geometry = geometry
        self.cp_step_ms = float(cp_step_ms)
        self.source_hashes = {"quadratic": f"cp_step_ms={self.cp_step_ms}"}

    def forward(self, lengths, cp, *, ranks=None) -> float:
        """Predict per-layer forward time for the quadratic smoke model."""
        cp = int(cp)
        if not 1 <= cp <= self.geometry.world_size:
            raise ValueError("cp outside the world")
        total = 0.0
        for n in lengths:
            per_rank = self.geometry.stored_length(int(n), cp) / cp
            total += per_rank * per_rank
        return total / 1e6 + self.cp_step_ms * (cp - 1)

    def backward(self, lengths, cp, *, ranks=None) -> float:
        """Predict per-layer backward time for the quadratic smoke model."""
        return 2.0 * self.forward(lengths, cp, ranks=ranks)

    def timing(self) -> None:
        """Use the shared scalar timeline for this smoke model."""
        return None

    def clear(self) -> None:
        """Accept the cache-reset hook for this stateless smoke model."""
        return None

    def check_domain(self, geometry, config) -> None:
        """Accept any geometry supported by the shared smoke-test planner."""
        return None
