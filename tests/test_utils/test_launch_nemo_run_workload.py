# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
from unittest.mock import Mock

import pytest

from tests.test_utils.python_scripts import launch_nemo_run_workload


@pytest.mark.parametrize("target", ["tests/unit_tests/bridge", "tests/unit_tests/**/*.py"])
def test_unit_test_launch_target(monkeypatch, target):
    # Capture the script the launcher builds from the actual H100 recipe.
    class ScriptReady(Exception):
        pass

    script = Mock(side_effect=ScriptReady)
    monkeypatch.setattr(launch_nemo_run_workload.run, "Script", script)
    with pytest.raises(ScriptReady):
        launch_nemo_run_workload.main.callback(
            scope="unit-tests",
            model="unit-tests",
            test_case=target,
            environment="dev",
            platform="dgx_h100",
            tag="latest",
            container_image="test",
        )

    assert f'BUCKET="{target}"' in script.call_args.kwargs["inline"].splitlines()


def test_nccl_watchdog_timeout_is_flaky():
    log = "Watchdog caught collective operation timeout: WorkNCCL(SeqNum=281)"

    assert launch_nemo_run_workload.is_flaky_failure(log)


def test_hang_prone_flaky_failure_cancels_active_attempt():
    experiment = Mock()
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    stop_event = threading.Event()
    failure_detected_event = threading.Event()
    monitor = threading.Thread(
        target=launch_nemo_run_workload._cancel_on_flaky_failure,
        args=(experiment, "task-1", log_buffer, stop_event, failure_detected_event, 0.01),
    )

    monitor.start()
    log_buffer.write("Watchdog caught collective operation timeout")
    monitor.join(timeout=1)

    assert not monitor.is_alive()
    assert failure_detected_event.is_set()
    experiment.cancel.assert_called_once_with("task-1")


def test_non_hanging_flaky_failure_does_not_cancel_active_attempt():
    experiment = Mock()
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    log_buffer.write("found NaN in local forward loss calculation")
    stop_event = threading.Event()
    failure_detected_event = threading.Event()
    monitor = threading.Thread(
        target=launch_nemo_run_workload._cancel_on_flaky_failure,
        args=(experiment, "task-1", log_buffer, stop_event, failure_detected_event, 0.01),
    )

    monitor.start()
    assert not failure_detected_event.wait(timeout=0.05)
    stop_event.set()
    monitor.join(timeout=1)

    assert not monitor.is_alive()
    assert launch_nemo_run_workload.is_flaky_failure(log_buffer.getvalue())
    experiment.cancel.assert_not_called()


def test_stopped_monitor_does_not_cancel_attempt():
    experiment = Mock()
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    log_buffer.write("Watchdog caught collective operation timeout")
    stop_event = threading.Event()
    stop_event.set()
    failure_detected_event = threading.Event()

    launch_nemo_run_workload._cancel_on_flaky_failure(
        experiment, "task-1", log_buffer, stop_event, failure_detected_event, poll_interval=0.01
    )

    assert not failure_detected_event.is_set()
    experiment.cancel.assert_not_called()
