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
# sets one. The packaged attention predictor reads the same table. The models read DPxCP
# rank ``r`` as a GPU of domain ``r // size``, so the launch has to place ranks
# consecutively, domain by domain.
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
    batch size (start-up and the CPU worker then reject any other). Read by the DynamicUSP
    solver only: ``sample_weight_coefficients``, the ``(A, B, C)`` of its Stage 1 sample
    weight (absent or None: the weight is derived from the model's step times), and
    ``tier_time_factors``, a ``{CP size: factor}`` table that switches Stage 1 to time
    balance; its keys are among ``supported_cp_sizes`` when the model declares them.
    ``sample_weight_is_default`` is True when the coefficients are the published default
    because the coefficient file gives none; the plan statistics and, for a user's file,
    the start-up line say so. A model that offers a step time for more than one pipeline
    stage has ``pipeline_group_times(lengths, cp) -> (forward_ms, backward_ms)``, the
    whole-model times of one CP group, and ``pipeline_terms``, a dictionary recorded with
    every such plan whose ``per_mb_overhead_ms``, a finite nonnegative number, is added to
    the step time once per microbatch; the solver rejects more than one pipeline stage for
    a model without both.

    Stage 2 of the DynamicUSP solver (``runtime.stage2``) re-packs the samples of one
    microbatch on its CP groups. A model offers it through these members, all of them
    optional as a set. None of the methods has a rank argument: a model whose group time
    depends on the ranks of the group cannot offer them.

    - ``stage2_enabled`` (True switches Stage 2 on; absent or False, nothing else is read)
      and ``stage2_keep_threshold`` (a finite positive number): both from the ``solver``
      section of the coefficient file, as ``tier_time_factors``.
    - ``stage2_reference_cp_size``: CP groups of at most this size are the reference tier.
      Stage 2 leaves a microbatch of such groups alone.
    - ``stage2_global_constant_ms``: a finite nonnegative time added to the slowest group
      of a microbatch before two layouts are compared.
    - ``stage2_sample_loads(length, cp_sizes) -> (work, {cp: load})``: the work of one
      sample of ``length`` logical tokens and the load it adds to a group of every size in
      ``cp_sizes``; all finite and positive.
    - ``stage2_empty_group_load(cp) -> float``: the load of a group before its first
      sample; finite and nonnegative.
    - ``stage2_group_time(lengths, cp) -> float``: the exact time in ms of one CP group for
      one microbatch; finite and positive.
    - Optional, the placement guard that DynamicUSP applies to groups above the reference
      size: ``stage2_guard_threshold``, a finite number (absent or None: the model has no
      guard, and the next member is not read), and ``stage2_guard_margin(lengths, cp)
      -> float``, the margin of a group of ``cp`` ranks that holds the samples whose
      logical lengths are in the tuple ``lengths``, possibly none; a finite number. A
      sample joins such a group only if the margin with it is at least the smaller of the
      margin without it and the threshold, and a re-packed microbatch has no such group
      with a margin below the threshold (``runtime.stage2`` has the rules). The margin is
      also taken for these groups of the Stage 1 layout, where it is not tested but must
      be a finite number. The threshold is read when the planner is built, where a
      property that raises AttributeError reads as no guard, and again for every
      microbatch of a plan, where it is checked: a model that then gives another
      threshold, or none, fails Stage 2 for that microbatch.

    A cost model must not change what its Stage 2 members give after the planner was
    built: the reference size, the global constant, the loads and times that the methods
    return for the same arguments, and the guard threshold belong to the run. Only the
    guard threshold is checked. The others are read again for every microbatch: a model
    that, say, raises its reference size is planned with the groups up to the new size as
    the reference tier, without a guard on them and without an error, while the plan
    statistics name the size that was read when the planner was built.
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
