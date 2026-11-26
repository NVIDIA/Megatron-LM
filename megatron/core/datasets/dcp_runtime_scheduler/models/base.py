# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared cost-model protocol and NVLink domain sizes."""

from __future__ import annotations

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


@runtime_checkable
class DcpCostModel(Protocol):
    """Callable contract consumed by ``models.cache.PackCostModel`` and the planner.

    ``nvlink_domain_size`` is the number of consecutive DPxCP ranks the model takes to share
    one NVLink domain (``NVLINK_DOMAIN_SIZES``), or None when it does not know the hardware.

    Optional members:

    - ``max_sample_length``, the longest sample in logical tokens that the model prices (the
      CPU worker rejects a longer one); the class attribute ``largest_cp_size`` when it
      prices CP groups up to that size only; ``required_global_batch_size`` when its step
      overhead describes one global batch size (start-up and the CPU worker then reject any other).
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
    built: they are read again for every microbatch, and the plan statistics name the
    reference size and the guard threshold read at construction.
    """

    name: str
    source_hashes: dict[str, str]
    nvlink_domain_size: int | None
    program_costs: Callable[..., dict[str, float]]

    def forward(
        self, lengths: tuple[int, ...], cp: int, *, ranks: Sequence[int] | None = None
    ) -> float:
        """Forward attention ms per layer for one pack."""

    def backward(
        self, lengths: tuple[int, ...], cp: int, *, ranks: Sequence[int] | None = None
    ) -> float:
        """Backward attention ms per layer for one pack."""

    def timing(self) -> dict:
        """Execution program for ``TimelineModel``."""

    def clear(self) -> None:
        """Drop memoized pack costs."""

    def check_domain(self, geometry: "RuntimePlanGeometry", config) -> None:
        """Raise ValueError when the geometry/config is outside the calibrated domain."""
