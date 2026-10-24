# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cost-model protocol and PP1 forward/backward timeline construction."""

from __future__ import annotations

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

from .sched.timeline import TimelineModel

if TYPE_CHECKING:
    from .planner import RuntimePlanGeometry


@runtime_checkable
class DcpCostModel(Protocol):
    """Callable contract consumed by ``sched.costs.PackCostModel`` and the planner."""

    name: str
    source_hashes: Dict[str, str]
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
        cp = int(cp)
        if not 1 <= cp <= self.geometry.world_size:
            raise ValueError("cp outside the world")
        total = 0.0
        for n in lengths:
            per_rank = self.geometry.stored_length(int(n), cp) / cp
            total += per_rank * per_rank
        return total / 1e6 + self.cp_step_ms * (cp - 1)

    def backward(self, lengths, cp, *, ranks=None) -> float:
        return 2.0 * self.forward(lengths, cp, ranks=ranks)

    def timing(self) -> None:
        return None

    def clear(self) -> None:
        return None

    def check_domain(self, geometry, config) -> None:
        return None


def _freeze_op(op):
    if "cost" in op:
        return ("cp", op["cost"])
    return (op["sync"], float(op["duration_ms"]))


def build_timeline(model, geometry: "RuntimePlanGeometry") -> TimelineModel:
    """Program-mode timeline from ``model.timing()`` or a scalar timeline when it is None."""
    timing = model.timing()
    if timing is None:
        return TimelineModel(
            geometry.world_size,
            geometry.ep_size,
            int(getattr(model, "layer_count", 1)),
            float(getattr(model, "forward_non_attention_ms", 0.0)),
            float(getattr(model, "backward_non_attention_ms", 0.0)),
            0.0,
            True,
            None,
        )
    programs = timing["execution_program"]
    if len(programs) != 1:
        raise ValueError("program mode supports one virtual stage (PP1/VPP1)")
    program = programs[0]
    ops = (
        tuple(_freeze_op(op) for op in program["forward"]),
        tuple(_freeze_op(op) for op in program["backward"]),
    )
    return TimelineModel(
        geometry.world_size,
        geometry.ep_size,
        int(program["layer_count"]),
        0.0,
        0.0,
        float(timing.get("iteration_tail_ms", 0.0)),
        True,
        ops,
    )
