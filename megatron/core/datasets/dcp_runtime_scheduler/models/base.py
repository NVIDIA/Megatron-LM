# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared cost-model protocol, NVLink domain sizes and geometry-only smoke model."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from ..runtime.planner import RuntimePlanGeometry

# GPUs per NVLink domain, by the ``hardware`` of the runtime file; nothing else sets one. A
# cost model exposes its entry as ``nvlink_domain_size`` and reads DPxCP rank ``r`` as a GPU
# of domain ``r // size``, so the launch has to place ranks consecutively, domain by domain.
NVLINK_DOMAIN_SIZES = MappingProxyType({"h100": 8, "gb200": 72})


def uses_p2p(config) -> bool:
    """Whether ``cp_comm_type`` selects P2P communication, for the model or for every layer."""
    value = getattr(config, "cp_comm_type", None)
    return value == "p2p" or (isinstance(value, list) and value.count("p2p") == len(value) > 0)


@dataclass(frozen=True)
class PackLimits:
    """What a cost model can price in one pack of a CP size, beyond the planner geometry.

    A model declares it through an optional ``pack_limits(cp)`` method. The planner then
    builds no pack outside the limits, and the start-up probe prices the largest pack the
    planner may build. ``None`` means no limit of the model's own; a new kind of limit is
    one more field here, read where the planner or the worker enforces it.
    """

    # Largest number of sequences in one pack of this CP size.
    max_sequences: int | None = None
    # Longest sample, in logical tokens before any padding, in a pack of this CP size. The
    # CPU worker rejects a global batch holding a sample that no allowed CP size prices.
    max_sample_length: int | None = None


@runtime_checkable
class DcpCostModel(Protocol):
    """Callable contract consumed by ``models.cache.PackCostModel`` and the planner.

    ``nvlink_domain_size`` is the number of consecutive DPxCP ranks the model takes to share
    one NVLink domain (``NVLINK_DOMAIN_SIZES``), or None when it does not know the hardware.

    Optional members:

    - ``pack_limits(cp) -> PackLimits`` when the model cannot price every pack that the
      geometry admits; the class attribute ``largest_cp_size`` when it prices CP groups up
      to that size only; ``required_global_batch_size`` when its step overhead describes
      one global batch size (start-up and the CPU worker then reject any other).
    - For the DynamicUSP solver: ``sample_weight_coefficients``, the ``(A, B, C)`` of the
      Stage 1 sample weight (absent or None: the weight is derived from the step times),
      with ``sample_weight_is_default`` True when they are the published default, and
      ``tier_time_factors``, a ``{CP size: factor}`` table that switches Stage 1 to time
      balance; its keys are among ``supported_cp_sizes`` when the model declares them.
    - For more than one pipeline stage: ``pipeline_group_times(lengths, cp) ->
      (forward_ms, backward_ms)``, the whole-model times of one CP group, and
      ``pipeline_terms``, a dictionary recorded with every such plan whose
      ``per_mb_overhead_ms``, a finite nonnegative number, is added once per microbatch.

    Stage 2 of the DynamicUSP solver (``runtime.stage2``, which has the rules) reads these
    members, all of them optional as a set. None of the methods has a rank argument: a
    model whose group time depends on the ranks of the group cannot offer them.

    - ``stage2_enabled`` (True switches Stage 2 on; absent or False, nothing else is read)
      and ``stage2_keep_threshold`` (a finite positive number).
    - ``stage2_reference_cp_size``: CP groups of at most this size are the reference tier.
    - ``stage2_global_constant_ms``: a finite nonnegative time added to the slowest group
      of a microbatch before two layouts are compared.
    - ``stage2_sample_loads(length, cp_sizes) -> (work, {cp: load})``: the work of one
      sample of ``length`` logical tokens and the load it adds to a group of every size in
      ``cp_sizes``; all finite and positive.
    - ``stage2_empty_group_load(cp) -> float``: the load of a group before its first
      sample; finite and nonnegative.
    - ``stage2_group_time(lengths, cp) -> float``: the exact time in ms of one CP group for
      one microbatch; finite and positive.
    - Optional, the placement guard of groups above the reference size:
      ``stage2_guard_threshold``, a finite number (absent or None: no guard), and
      ``stage2_guard_margin(lengths, cp) -> float``, the finite margin of a group of ``cp``
      ranks that holds samples of the logical ``lengths``, possibly none.

    A cost model must not change what its Stage 2 members give after the planner was
    built. Only the guard threshold is checked; the others are read again for every
    microbatch, and the plan statistics name the reference size read at construction.
    """

    name: str
    source_hashes: dict[str, str]
    nvlink_domain_size: int | None
    program_costs: Callable[..., dict[str, float]] | None

    def forward(
        self, lengths: tuple[int, ...], cp: int, *, ranks: Sequence[int] | None = None
    ) -> float:
        """Forward attention ms per layer for one pack."""

    def backward(
        self, lengths: tuple[int, ...], cp: int, *, ranks: Sequence[int] | None = None
    ) -> float:
        """Backward attention ms per layer for one pack."""

    def timing(self) -> dict | None:
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

    def clear(self) -> None:
        """Accept the cache-reset hook for this stateless smoke model."""

    def check_domain(self, geometry, config) -> None:
        """Accept any geometry supported by the shared smoke-test planner."""
