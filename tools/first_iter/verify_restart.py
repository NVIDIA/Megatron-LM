# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verify first-iteration restart gains without accepting cost relocation."""

from __future__ import annotations

import argparse
import dataclasses
import re
from datetime import datetime
from pathlib import Path
from statistics import median

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
_MARKER_RE = re.compile(
    r"\[(?P<name>before library-setup|after megatron is initialized|"
    r"after model, optimizer, and learning rate scheduler are built|"
    r"after dataloaders are built|before the start of training step)\] "
    r"datetime:\s*(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
)
_INIT_RE = re.compile(r"time to initialize megatron \(seconds\):\s*(?P<seconds>[0-9.]+)")
_ITER_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\] "
    r"iteration\s+(?P<iteration>\d+)/.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<elapsed_ms>[0-9.]+).*?"
    r"lm loss:\s*(?P<loss>[0-9.Ee+-]+).*?"
    r"grad norm:\s*(?P<grad_norm>[0-9.Ee+-]+).*?"
    r"number of skipped iterations:\s*(?P<skipped>\d+).*?"
    r"number of nan iterations:\s*(?P<nan>\d+)"
)
_ABA_START_RE = re.compile(r"\[ITER1-ENV-ABA\] start leg=(?P<label>\S+)")
_SCALE_RE = re.compile(r"\[ULTRAHALF-ITER1CAP\] nodes=(?P<nodes>\d+) world=(?P<world>\d+)")
_RUN_DIR_RE = re.compile(r"run_dir=(?P<run_dir>\S+)")
_SRUN_WALL_RE = re.compile(r"\[ULTRAHALF-ITER1CAP\] srun wall:\s*(?P<seconds>[0-9.]+)s")


@dataclasses.dataclass(frozen=True)
class IterationMetrics:
    iteration: int
    elapsed_s: float
    loss: str
    grad_norm: str
    skipped: int
    nan: int
    timestamp: datetime


@dataclasses.dataclass(frozen=True)
class RunMetrics:
    path: Path
    initialization_s: float
    pre_training_s: float
    iteration_1: IterationMetrics
    iteration_20: IterationMetrics | None
    iterations: dict[int, IterationMetrics] = dataclasses.field(default_factory=dict)

    @property
    def startup_through_iteration_1_s(self) -> float:
        """Library setup through the end of iteration 1, excluding container launch."""

        return self.pre_training_s + self.iteration_1.elapsed_s

    def median_iteration_s(self, start: int, end: int) -> float:
        """Return the median elapsed time for an inclusive iteration range."""

        missing = [
            iteration for iteration in range(start, end + 1) if iteration not in self.iterations
        ]
        if missing:
            raise ValueError(
                f"{self.path}: missing iteration(s) {missing} for steady-state range {start}-{end}"
            )
        return median(self.iterations[iteration].elapsed_s for iteration in range(start, end + 1))


@dataclasses.dataclass
class AbaLeg:
    label: str
    nodes: int | None = None
    world: int | None = None
    run_dir: Path | None = None
    srun_wall_s: float | None = None
    metrics: RunMetrics | None = None


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, _TIMESTAMP_FORMAT)


def parse_run_log(path: Path) -> RunMetrics:
    """Parse Megatron startup markers and printed iteration metrics."""

    text = path.read_text(errors="replace")
    markers = {
        match.group("name"): _parse_timestamp(match.group("timestamp"))
        for match in _MARKER_RE.finditer(text)
    }
    required_markers = {"before library-setup", "before the start of training step"}
    missing = sorted(required_markers - markers.keys())
    if missing:
        raise ValueError(f"{path}: missing startup marker(s): {', '.join(missing)}")

    init_matches = list(_INIT_RE.finditer(text))
    if len(init_matches) != 1:
        raise ValueError(
            f"{path}: expected one Megatron initialization timing, found {len(init_matches)}"
        )

    iterations: dict[int, IterationMetrics] = {}
    for match in _ITER_RE.finditer(text):
        iteration = int(match.group("iteration"))
        iterations[iteration] = IterationMetrics(
            iteration=iteration,
            elapsed_s=float(match.group("elapsed_ms")) / 1000.0,
            loss=match.group("loss"),
            grad_norm=match.group("grad_norm"),
            skipped=int(match.group("skipped")),
            nan=int(match.group("nan")),
            timestamp=_parse_timestamp(match.group("timestamp")),
        )
    if 1 not in iterations:
        raise ValueError(f"{path}: iteration 1 was not logged")

    pre_training_s = (
        markers["before the start of training step"] - markers["before library-setup"]
    ).total_seconds()
    if pre_training_s < 0:
        raise ValueError(f"{path}: startup markers are out of order")

    return RunMetrics(
        path=path,
        initialization_s=float(init_matches[0].group("seconds")),
        pre_training_s=pre_training_s,
        iteration_1=iterations[1],
        iteration_20=iterations.get(20),
        iterations=iterations,
    )


def _find_run_log(run_dir: Path) -> Path:
    matches = sorted((run_dir / "logs").glob("*.log"))
    if len(matches) != 1:
        raise ValueError(f"{run_dir}: expected one run log, found {len(matches)}")
    return matches[0]


def parse_aba_log(path: Path) -> list[AbaLeg]:
    """Resolve each same-allocation ABA leg and its Megatron run log."""

    legs: list[AbaLeg] = []
    current: AbaLeg | None = None
    for line in path.read_text(errors="replace").splitlines():
        if match := _ABA_START_RE.search(line):
            current = AbaLeg(label=match.group("label"))
            legs.append(current)
            continue
        if current is None:
            continue
        if match := _SCALE_RE.search(line):
            current.nodes = int(match.group("nodes"))
            current.world = int(match.group("world"))
        if match := _RUN_DIR_RE.search(line):
            current.run_dir = Path(match.group("run_dir"))
        if match := _SRUN_WALL_RE.search(line):
            current.srun_wall_s = float(match.group("seconds"))

    if len(legs) != 3:
        raise ValueError(f"{path}: expected three ABA legs, found {len(legs)}")
    for leg in legs:
        if leg.nodes is None or leg.world is None or leg.run_dir is None or leg.srun_wall_s is None:
            raise ValueError(f"{path}: incomplete launcher metadata for leg {leg.label}")
        leg.metrics = parse_run_log(_find_run_log(leg.run_dir))
    return legs


def _mean(left: float, right: float) -> float:
    return (left + right) / 2.0


def _spread(left: float, right: float) -> float:
    return abs(left - right)


def verify_aba(
    legs: list[AbaLeg],
    treatment_label: str,
    expected_nodes: int,
    expected_world: int,
    phase_tolerance_s: float,
    wall_tolerance_s: float,
    require_iteration_20: bool,
    minimum_iteration_1_improvement_s: float = 0.0,
    minimum_phase_improvement_s: float = 0.0,
    minimum_wall_improvement_s: float = 0.0,
    max_control_iteration_1_spread_s: float = 1.0,
    max_control_phase_spread_s: float = 5.0,
    max_control_wall_spread_s: float = 10.0,
    maximum_treatment_iteration_1_s: float | None = None,
    steady_state_start_iteration: int | None = None,
    steady_state_end_iteration: int | None = None,
    maximum_steady_state_regression_percent: float | None = None,
) -> list[str]:
    """Return gate failures; an empty list means the ABA passes."""

    by_label = {leg.label: leg for leg in legs}
    expected_labels = {"control-a", treatment_label, "control-b"}
    failures: list[str] = []
    if set(by_label) != expected_labels:
        return [f"expected ABA labels {sorted(expected_labels)}, found {sorted(by_label)}"]

    control_a = by_label["control-a"]
    treatment = by_label[treatment_label]
    control_b = by_label["control-b"]
    for leg in legs:
        if leg.nodes != expected_nodes or leg.world != expected_world:
            failures.append(
                f"{leg.label}: expected {expected_nodes} nodes/{expected_world} ranks, "
                f"found {leg.nodes}/{leg.world}"
            )
        assert leg.metrics is not None
        if leg.metrics.iteration_1.skipped or leg.metrics.iteration_1.nan:
            failures.append(f"{leg.label}: iteration 1 has skipped or NaN iterations")
        if require_iteration_20:
            if leg.metrics.iteration_20 is None:
                failures.append(f"{leg.label}: iteration 20 was not logged")
            elif leg.metrics.iteration_20.skipped or leg.metrics.iteration_20.nan:
                failures.append(f"{leg.label}: iteration 20 has skipped or NaN iterations")

    assert (
        control_a.metrics is not None
        and treatment.metrics is not None
        and control_b.metrics is not None
    )
    control_iteration_1_spread = _spread(
        control_a.metrics.iteration_1.elapsed_s, control_b.metrics.iteration_1.elapsed_s
    )
    if control_iteration_1_spread > max_control_iteration_1_spread_s:
        failures.append(
            "control iteration-1 spread is too large for causal attribution: "
            f"spread={control_iteration_1_spread:.4f}s "
            f"maximum={max_control_iteration_1_spread_s:.4f}s"
        )

    control_phase_spread = _spread(
        control_a.metrics.startup_through_iteration_1_s,
        control_b.metrics.startup_through_iteration_1_s,
    )
    if control_phase_spread > max_control_phase_spread_s:
        failures.append(
            "control startup-through-iteration-1 spread is too large for causal attribution: "
            f"spread={control_phase_spread:.4f}s "
            f"maximum={max_control_phase_spread_s:.4f}s"
        )

    assert control_a.srun_wall_s is not None and control_b.srun_wall_s is not None
    control_wall_spread = _spread(control_a.srun_wall_s, control_b.srun_wall_s)
    if control_wall_spread > max_control_wall_spread_s:
        failures.append(
            "control srun-wall spread is too large for causal attribution: "
            f"spread={control_wall_spread:.1f}s "
            f"maximum={max_control_wall_spread_s:.1f}s"
        )

    numeric_fields = ("loss", "grad_norm")
    for field in numeric_fields:
        values = [
            getattr(leg.metrics.iteration_1, field) for leg in (control_a, treatment, control_b)
        ]
        if len(set(values)) != 1:
            failures.append(f"iteration 1 {field} mismatch: {values}")
        if require_iteration_20 and all(leg.metrics.iteration_20 is not None for leg in legs):
            values_20 = [
                getattr(leg.metrics.iteration_20, field)
                for leg in (control_a, treatment, control_b)
            ]
            if len(set(values_20)) != 1:
                failures.append(f"iteration 20 {field} mismatch: {values_20}")

    control_iteration_1_mean = _mean(
        control_a.metrics.iteration_1.elapsed_s, control_b.metrics.iteration_1.elapsed_s
    )
    iteration_1_improvement = control_iteration_1_mean - treatment.metrics.iteration_1.elapsed_s
    if iteration_1_improvement < minimum_iteration_1_improvement_s:
        failures.append(
            "iteration 1 did not improve enough: "
            f"treatment={treatment.metrics.iteration_1.elapsed_s:.4f}s "
            f"control_mean={control_iteration_1_mean:.4f}s "
            f"improvement={iteration_1_improvement:.4f}s "
            f"minimum={minimum_iteration_1_improvement_s:.4f}s"
        )
    if (
        maximum_treatment_iteration_1_s is not None
        and treatment.metrics.iteration_1.elapsed_s > maximum_treatment_iteration_1_s
    ):
        failures.append(
            "treatment iteration 1 exceeds target: "
            f"treatment={treatment.metrics.iteration_1.elapsed_s:.4f}s "
            f"maximum={maximum_treatment_iteration_1_s:.4f}s"
        )

    control_phase_mean = _mean(
        control_a.metrics.startup_through_iteration_1_s,
        control_b.metrics.startup_through_iteration_1_s,
    )
    phase_improvement = control_phase_mean - treatment.metrics.startup_through_iteration_1_s
    if phase_improvement < minimum_phase_improvement_s:
        failures.append(
            "startup-through-iteration-1 did not improve enough: "
            f"treatment={treatment.metrics.startup_through_iteration_1_s:.4f}s "
            f"control_mean={control_phase_mean:.4f}s "
            f"improvement={phase_improvement:.4f}s "
            f"minimum={minimum_phase_improvement_s:.4f}s"
        )
    if treatment.metrics.startup_through_iteration_1_s > control_phase_mean + phase_tolerance_s:
        failures.append(
            "startup-through-iteration-1 regressed: "
            f"treatment={treatment.metrics.startup_through_iteration_1_s:.4f}s "
            f"control_mean={control_phase_mean:.4f}s tolerance={phase_tolerance_s:.4f}s"
        )

    if maximum_steady_state_regression_percent is not None:
        if steady_state_start_iteration is None or steady_state_end_iteration is None:
            failures.append(
                "steady-state start and end iterations are required when a regression limit is set"
            )
        elif steady_state_start_iteration > steady_state_end_iteration:
            failures.append("steady-state start iteration must not exceed the end iteration")
        else:
            try:
                control_a_steady = control_a.metrics.median_iteration_s(
                    steady_state_start_iteration, steady_state_end_iteration
                )
                treatment_steady = treatment.metrics.median_iteration_s(
                    steady_state_start_iteration, steady_state_end_iteration
                )
                control_b_steady = control_b.metrics.median_iteration_s(
                    steady_state_start_iteration, steady_state_end_iteration
                )
            except ValueError as error:
                failures.append(str(error))
            else:
                control_steady_mean = _mean(control_a_steady, control_b_steady)
                steady_state_regression_percent = (
                    (treatment_steady - control_steady_mean) / control_steady_mean * 100.0
                )
                if steady_state_regression_percent > maximum_steady_state_regression_percent:
                    failures.append(
                        "steady-state iteration time regressed: "
                        f"treatment_median={treatment_steady:.4f}s "
                        f"control_mean_median={control_steady_mean:.4f}s "
                        f"regression={steady_state_regression_percent:.2f}% "
                        f"maximum={maximum_steady_state_regression_percent:.2f}%"
                    )

    assert (
        control_a.srun_wall_s is not None
        and treatment.srun_wall_s is not None
        and control_b.srun_wall_s is not None
    )
    control_wall_mean = _mean(control_a.srun_wall_s, control_b.srun_wall_s)
    wall_improvement = control_wall_mean - treatment.srun_wall_s
    if wall_improvement < minimum_wall_improvement_s:
        failures.append(
            "srun wall did not improve enough: "
            f"treatment={treatment.srun_wall_s:.1f}s "
            f"control_mean={control_wall_mean:.1f}s "
            f"improvement={wall_improvement:.1f}s "
            f"minimum={minimum_wall_improvement_s:.1f}s"
        )
    if treatment.srun_wall_s > control_wall_mean + wall_tolerance_s:
        failures.append(
            f"srun wall regressed: treatment={treatment.srun_wall_s:.1f}s "
            f"control_mean={control_wall_mean:.1f}s tolerance={wall_tolerance_s:.1f}s"
        )
    return failures


def _format_table(legs: list[AbaLeg]) -> str:
    header = "leg                 init(s)  pre-train(s)  iter1(s)  startup+iter1(s)  srun(s)  loss       grad"
    rows = [header]
    for leg in legs:
        assert leg.metrics is not None and leg.srun_wall_s is not None
        metrics = leg.metrics
        rows.append(
            f"{leg.label:<20} {metrics.initialization_s:>7.3f}  {metrics.pre_training_s:>12.3f}  "
            f"{metrics.iteration_1.elapsed_s:>8.4f}  "
            f"{metrics.startup_through_iteration_1_s:>16.4f}  {leg.srun_wall_s:>7.1f}  "
            f"{metrics.iteration_1.loss:<10} {metrics.iteration_1.grad_norm}"
        )
    return "\n".join(rows)


def _format_steady_state(legs: list[AbaLeg], start: int, end: int) -> str:
    values = []
    for leg in legs:
        assert leg.metrics is not None
        values.append(f"{leg.label}={leg.metrics.median_iteration_s(start, end):.4f}s")
    return f"steady-state median iterations {start}-{end}: " + " ".join(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aba-log", type=Path, required=True)
    parser.add_argument("--treatment-label", required=True)
    parser.add_argument("--expected-nodes", type=int, default=16)
    parser.add_argument("--expected-world", type=int, default=64)
    parser.add_argument("--phase-tolerance-s", type=float, default=0.0)
    parser.add_argument("--wall-tolerance-s", type=float, default=0.0)
    parser.add_argument("--minimum-iteration-1-improvement-s", type=float, default=0.0)
    parser.add_argument(
        "--minimum-startup-through-iteration-1-improvement-s", type=float, default=0.0
    )
    parser.add_argument("--minimum-srun-wall-improvement-s", type=float, default=0.0)
    parser.add_argument("--max-control-iteration-1-spread-s", type=float, default=1.0)
    parser.add_argument("--max-control-phase-spread-s", type=float, default=5.0)
    parser.add_argument("--max-control-wall-spread-s", type=float, default=10.0)
    parser.add_argument("--maximum-treatment-iteration-1-s", type=float)
    parser.add_argument("--steady-state-start-iteration", type=int)
    parser.add_argument("--steady-state-end-iteration", type=int)
    parser.add_argument("--maximum-steady-state-regression-percent", type=float)
    parser.add_argument("--require-iteration-20", action="store_true")
    args = parser.parse_args()

    legs = parse_aba_log(args.aba_log)
    print(_format_table(legs))
    if (
        args.maximum_steady_state_regression_percent is not None
        and args.steady_state_start_iteration is not None
        and args.steady_state_end_iteration is not None
    ):
        try:
            print(
                _format_steady_state(
                    legs, args.steady_state_start_iteration, args.steady_state_end_iteration
                )
            )
        except ValueError:
            # verify_aba reports the missing range as a normal gate failure.
            pass
    failures = verify_aba(
        legs=legs,
        treatment_label=args.treatment_label,
        expected_nodes=args.expected_nodes,
        expected_world=args.expected_world,
        phase_tolerance_s=args.phase_tolerance_s,
        wall_tolerance_s=args.wall_tolerance_s,
        require_iteration_20=args.require_iteration_20,
        minimum_iteration_1_improvement_s=args.minimum_iteration_1_improvement_s,
        minimum_phase_improvement_s=(args.minimum_startup_through_iteration_1_improvement_s),
        minimum_wall_improvement_s=args.minimum_srun_wall_improvement_s,
        max_control_iteration_1_spread_s=args.max_control_iteration_1_spread_s,
        max_control_phase_spread_s=args.max_control_phase_spread_s,
        max_control_wall_spread_s=args.max_control_wall_spread_s,
        maximum_treatment_iteration_1_s=args.maximum_treatment_iteration_1_s,
        steady_state_start_iteration=args.steady_state_start_iteration,
        steady_state_end_iteration=args.steady_state_end_iteration,
        maximum_steady_state_regression_percent=(args.maximum_steady_state_regression_percent),
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("PASS: exact numerics, scale, startup-through-iteration-1, and srun-wall gates")


if __name__ == "__main__":
    main()
