# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from datetime import datetime, timedelta
from pathlib import Path

from tools.first_iter.verify_restart import (
    AbaLeg,
    IterationMetrics,
    RunMetrics,
    parse_run_log,
    verify_aba,
)


def _run_metrics(
    path: Path,
    init: float,
    pre_training: float,
    iter_1: float,
    iteration_20_loss: str | None = None,
) -> RunMetrics:
    iteration = IterationMetrics(
        iteration=1,
        elapsed_s=iter_1,
        loss="1.218077E+01",
        grad_norm="9.282",
        skipped=0,
        nan=0,
        timestamp=datetime(2026, 6, 29),
    )
    return RunMetrics(
        path=path,
        initialization_s=init,
        pre_training_s=pre_training,
        iteration_1=iteration,
        iteration_20=(
            IterationMetrics(
                iteration=20,
                elapsed_s=0.71,
                loss=iteration_20_loss,
                grad_norm="9.284",
                skipped=0,
                nan=0,
                timestamp=datetime(2026, 6, 29),
            )
            if iteration_20_loss is not None
            else None
        ),
    )


def test_parse_run_log_includes_pre_training_phase(tmp_path: Path) -> None:
    before = datetime(2026, 6, 29, 17, 48, 0)
    training = before + timedelta(seconds=19.25)
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "0: time to initialize megatron (seconds): 10.500",
                f"0: [before library-setup] datetime: {before:%Y-%m-%d %H:%M:%S.%f}",
                f"0: [before the start of training step] datetime: {training:%Y-%m-%d %H:%M:%S.%f}",
                "63: [2026-06-29 17:48:28.073200] iteration        1/ 10 | "
                "elapsed time per iteration (ms): 8823.2 | lm loss: 1.218077E+01 | "
                "grad norm: 9.282 | number of skipped iterations: 0 | number of nan iterations: 0 |",
            ]
        )
    )

    metrics = parse_run_log(log)

    assert metrics.initialization_s == 10.5
    assert metrics.pre_training_s == 19.25
    assert metrics.iteration_1.elapsed_s == 8.8232
    assert metrics.startup_through_iteration_1_s == 28.0732


def test_verify_aba_rejects_relocation_into_pre_training(tmp_path: Path) -> None:
    control_a = AbaLeg(
        label="control-a",
        nodes=16,
        world=64,
        srun_wall_s=110,
        metrics=_run_metrics(tmp_path / "a.log", init=13, pre_training=18, iter_1=12),
    )
    treatment = AbaLeg(
        label="treatment",
        nodes=16,
        world=64,
        srun_wall_s=112,
        metrics=_run_metrics(tmp_path / "t.log", init=20, pre_training=27, iter_1=5),
    )
    control_b = AbaLeg(
        label="control-b",
        nodes=16,
        world=64,
        srun_wall_s=108,
        metrics=_run_metrics(tmp_path / "b.log", init=13, pre_training=18, iter_1=12),
    )

    failures = verify_aba(
        [control_a, treatment, control_b],
        treatment_label="treatment",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=False,
    )

    assert any("startup-through-iteration-1 regressed" in failure for failure in failures)
    assert any("srun wall regressed" in failure for failure in failures)


def test_verify_aba_rejects_plain_phase_relocation(tmp_path: Path) -> None:
    legs = [
        AbaLeg(
            label="control-a",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(tmp_path / "a.log", init=13, pre_training=18, iter_1=12),
        ),
        AbaLeg(
            label="treatment",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(tmp_path / "t.log", init=20, pre_training=25, iter_1=5),
        ),
        AbaLeg(
            label="control-b",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(tmp_path / "b.log", init=13, pre_training=18, iter_1=12),
        ),
    ]

    failures = verify_aba(
        legs,
        treatment_label="treatment",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=False,
        minimum_phase_improvement_s=0.5,
    )

    assert any(
        "startup-through-iteration-1 did not improve enough" in failure for failure in failures
    )


def test_verify_aba_accepts_real_end_to_end_win(tmp_path: Path) -> None:
    legs = [
        AbaLeg(
            label="control-a",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(
                tmp_path / "a.log", init=13.372, pre_training=18.1469, iter_1=12.8003
            ),
        ),
        AbaLeg(
            label="topology-cache",
            nodes=16,
            world=64,
            srun_wall_s=105,
            metrics=_run_metrics(
                tmp_path / "t.log", init=14.633, pre_training=19.5356, iter_1=8.8232
            ),
        ),
        AbaLeg(
            label="control-b",
            nodes=16,
            world=64,
            srun_wall_s=108,
            metrics=_run_metrics(
                tmp_path / "b.log", init=13.407, pre_training=19.2088, iter_1=12.3538
            ),
        ),
    ]

    failures = verify_aba(
        legs,
        treatment_label="topology-cache",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=False,
    )

    assert failures == []


def test_verify_aba_rejects_relocation_before_startup_marker(tmp_path: Path) -> None:
    legs = [
        AbaLeg(
            label="control-a",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(tmp_path / "a.log", init=13, pre_training=18, iter_1=12),
        ),
        AbaLeg(
            label="treatment",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(tmp_path / "t.log", init=12, pre_training=17, iter_1=10),
        ),
        AbaLeg(
            label="control-b",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(tmp_path / "b.log", init=13, pre_training=18, iter_1=12),
        ),
    ]

    failures = verify_aba(
        legs,
        treatment_label="treatment",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=False,
        minimum_phase_improvement_s=0.5,
        minimum_wall_improvement_s=0.5,
    )

    assert any("srun wall did not improve enough" in failure for failure in failures)


def test_verify_aba_rejects_late_numeric_drift(tmp_path: Path) -> None:
    legs = [
        AbaLeg(
            label="control-a",
            nodes=16,
            world=64,
            srun_wall_s=119,
            metrics=_run_metrics(
                tmp_path / "a.log",
                init=15.9,
                pre_training=19.1,
                iter_1=9.4,
                iteration_20_loss="1.218087E+01",
            ),
        ),
        AbaLeg(
            label="treatment",
            nodes=16,
            world=64,
            srun_wall_s=110,
            metrics=_run_metrics(
                tmp_path / "t.log",
                init=18.8,
                pre_training=18.0,
                iter_1=8.4,
                iteration_20_loss="1.218086E+01",
            ),
        ),
        AbaLeg(
            label="control-b",
            nodes=16,
            world=64,
            srun_wall_s=109,
            metrics=_run_metrics(
                tmp_path / "b.log",
                init=17.1,
                pre_training=17.9,
                iter_1=9.0,
                iteration_20_loss="1.218087E+01",
            ),
        ),
    ]

    failures = verify_aba(
        legs,
        treatment_label="treatment",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=True,
    )

    assert any("iteration 20 loss mismatch" in failure for failure in failures)


def test_verify_aba_rejects_no_first_iteration_gain(tmp_path: Path) -> None:
    legs = [
        AbaLeg(
            label="control-a",
            nodes=16,
            world=64,
            srun_wall_s=106,
            metrics=_run_metrics(tmp_path / "a.log", init=14, pre_training=18, iter_1=8.8051),
        ),
        AbaLeg(
            label="treatment",
            nodes=16,
            world=64,
            srun_wall_s=105,
            metrics=_run_metrics(tmp_path / "t.log", init=14, pre_training=17, iter_1=8.9773),
        ),
        AbaLeg(
            label="control-b",
            nodes=16,
            world=64,
            srun_wall_s=106,
            metrics=_run_metrics(tmp_path / "b.log", init=14, pre_training=18, iter_1=8.7804),
        ),
    ]

    failures = verify_aba(
        legs,
        treatment_label="treatment",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=False,
    )

    assert any("iteration 1 did not improve enough" in failure for failure in failures)


def test_verify_aba_rejects_treatment_above_absolute_target(tmp_path: Path) -> None:
    legs = [
        AbaLeg(
            label="control-a",
            nodes=16,
            world=64,
            srun_wall_s=106,
            metrics=_run_metrics(tmp_path / "a.log", init=14, pre_training=18, iter_1=8.7),
        ),
        AbaLeg(
            label="treatment",
            nodes=16,
            world=64,
            srun_wall_s=103,
            metrics=_run_metrics(tmp_path / "t.log", init=14, pre_training=17, iter_1=5.1),
        ),
        AbaLeg(
            label="control-b",
            nodes=16,
            world=64,
            srun_wall_s=106,
            metrics=_run_metrics(tmp_path / "b.log", init=14, pre_training=18, iter_1=8.6),
        ),
    ]

    failures = verify_aba(
        legs,
        treatment_label="treatment",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=False,
        maximum_treatment_iteration_1_s=5.0,
    )

    assert any("treatment iteration 1 exceeds target" in failure for failure in failures)


def test_verify_aba_rejects_control_order_bias(tmp_path: Path) -> None:
    legs = [
        AbaLeg(
            label="control-a",
            nodes=16,
            world=64,
            srun_wall_s=120,
            metrics=_run_metrics(tmp_path / "a.log", init=21, pre_training=27, iter_1=10.0),
        ),
        AbaLeg(
            label="treatment",
            nodes=16,
            world=64,
            srun_wall_s=109,
            metrics=_run_metrics(tmp_path / "t.log", init=13, pre_training=17, iter_1=8.4),
        ),
        AbaLeg(
            label="control-b",
            nodes=16,
            world=64,
            srun_wall_s=108,
            metrics=_run_metrics(tmp_path / "b.log", init=13, pre_training=18, iter_1=8.3),
        ),
    ]

    failures = verify_aba(
        legs,
        treatment_label="treatment",
        expected_nodes=16,
        expected_world=64,
        phase_tolerance_s=0,
        wall_tolerance_s=0,
        require_iteration_20=False,
    )

    assert any("control iteration-1 spread" in failure for failure in failures)
    assert any("control startup-through-iteration-1 spread" in failure for failure in failures)
    assert any("control srun-wall spread" in failure for failure in failures)
