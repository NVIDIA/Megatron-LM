# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
from unittest.mock import Mock

from tests.test_utils.python_scripts import launch_nemo_run_workload


def test_nccl_watchdog_timeout_is_flaky():
    log = "Watchdog caught collective operation timeout: WorkNCCL(SeqNum=281)"

    assert launch_nemo_run_workload.is_flaky_failure(log)


# Heartbeat-monitor output (PyTorch ProcessGroupNCCL.cpp) when the watchdog thread is wedged.
_NCCL_WATCHDOG_HANG_LOG = (
    "[rank7]:[E1111 17:58:31.000000000 ProcessGroupNCCL.cpp:1987] [PG ID 0 PG GUID 0 Rank 7] "
    "ProcessGroupNCCL's watchdog got stuck for 480 seconds without making progress in "
    "monitoring enqueued collectives.\n"
    "[rank7]:[F1111 18:07:31.000000000 ProcessGroupNCCL.cpp:1724] [PG ID 0 PG GUID 0 Rank 7] "
    "Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL "
    "watchdog hang.\n"
)


def test_nccl_watchdog_hang_is_flaky():
    assert launch_nemo_run_workload.is_flaky_failure(_NCCL_WATCHDOG_HANG_LOG)


def test_nccl_watchdog_hang_is_not_hang_prone():
    # The heartbeat monitor aborts the wedged rank itself, so the streamed monitor need not cancel.
    assert not launch_nemo_run_workload._is_hang_prone_flaky_failure(_NCCL_WATCHDOG_HANG_LOG)


def test_nccl_heartbeat_monitor_startup_log_is_not_flaky():
    log = (
        "[rank0]:[I1111 17:31:48.304211939 ProcessGroupNCCL.cpp:1791] [PG ID 0 PG GUID 0 Rank 0] "
        "HeartbeatMonitor environments: TORCH_NCCL_ENABLE_MONITORING (Whether to kill program "
        "when no watchdog heartbeat detected): 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, "
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480"
    )

    assert not launch_nemo_run_workload.is_flaky_failure(log)


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
