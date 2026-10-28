# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import builtins
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.test_utils.python_scripts import launch_nemo_run_workload

EXPECTED_FLAKY_FAILURE_SIGNATURES = (
    "The server socket has failed to listen on any local network address.",
    "Some NCCL operations have failed or timed out.",
    "Watchdog caught collective operation timeout",
    "uncorrectable ECC error encountered",
    "illegal memory access",
    "illegal instruction",
    "torch.distributed.DistNetworkError",
    "Segmentation fault",
    "found NaN in",
    "For debugging consider passing CUDA_LAUNCH_BLOCKING=1",
    "double free or corruption",
    "Call to CUDA function failed.",
    "Connection reset by peer",
    "invalid pointer",
    "malloc(): unaligned tcache chunk detected",
    "zmq.error.ZMQError: Address already in use",
    "We couldn't connect to 'https://huggingface.co'",
    "Unpack failed: incomplete input",
    "The read operation timed out",
    "Read timed out",
    "TimeoutError",
    "Connection broken",
    "Temporary failure in name resolution",
    "unspecified launch failure",
    "free(): corrupted unsorted chunks",
    "Segfault encountered",
    "The following metrics failed",
    "removal of container",
    "is already in progress",
    "Error deleting container",
)


def test_nccl_watchdog_timeout_is_flaky():
    log = "Watchdog caught collective operation timeout: WorkNCCL(SeqNum=281)"

    assert launch_nemo_run_workload.is_flaky_failure(log)


@pytest.mark.parametrize("signature", EXPECTED_FLAKY_FAILURE_SIGNATURES)
def test_every_flaky_failure_signature_is_recognized(signature):
    assert launch_nemo_run_workload.is_flaky_failure(signature)


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


def test_failure_signature_split_at_file_chunk_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_nemo_run_workload, "_FAILURE_LOG_SCAN_CHUNK_SIZE", 16)
    signature = "Connection reset by peer"
    log_path = tmp_path / "task" / "attempt_0" / "rank0" / "stdout.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("x" * 13 + signature + " trailing text")

    assert launch_nemo_run_workload._has_flaky_failure_in_logs(tmp_path)


def test_failure_signature_at_end_of_file_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_nemo_run_workload, "_FAILURE_LOG_SCAN_CHUNK_SIZE", 16)
    signature = "The following metrics failed"
    log_path = tmp_path / "task.log"
    log_path.write_text("x" * 7 + signature)

    assert launch_nemo_run_workload._has_flaky_failure_in_logs(tmp_path)


def test_nested_rank_and_harness_logs_are_scanned(tmp_path):
    rank_log = tmp_path / "task" / "attempt_0" / "rank0" / "stdout.log"
    rank_log.parent.mkdir(parents=True)
    rank_log.write_text("training output without retry markers")
    task_log = tmp_path / "task" / "nemo-run.log"
    task_log.write_text("The following metrics failed: loss")

    assert launch_nemo_run_workload._has_flaky_failure_in_logs(tmp_path)


def test_resolved_paths_from_overlapping_globs_are_read_once(tmp_path, monkeypatch):
    rank_log = tmp_path / "task" / "attempt_0" / "rank0" / "stdout.log"
    rank_log.parent.mkdir(parents=True)
    rank_log.write_text("ordinary output")
    alias_log = tmp_path / "task" / "alias.log"
    alias_log.symlink_to(rank_log)
    original_open = builtins.open
    opened_paths = []

    def counting_open(path, *args, **kwargs):
        if Path(path).resolve() == rank_log.resolve():
            opened_paths.append(Path(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    assert not launch_nemo_run_workload._has_flaky_failure_in_logs(tmp_path)
    assert len(opened_paths) == 1


def test_malformed_utf8_is_replaced_and_scanned(tmp_path):
    log_path = tmp_path / "rank.log"
    log_path.write_bytes(b"\xff\xfe" + b"illegal memory access")

    assert launch_nemo_run_workload._has_flaky_failure_in_logs(tmp_path)


def test_unreadable_log_is_ignored(tmp_path, monkeypatch, caplog):
    log_path = tmp_path / "rank.log"
    log_path.write_text("ordinary output")
    original_open = builtins.open

    def failing_open(path, *args, **kwargs):
        if Path(path) == log_path:
            raise OSError("simulated read failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)

    assert not launch_nemo_run_workload._has_flaky_failure_in_logs(tmp_path)
    assert "simulated read failure" in caplog.text


def test_all_nonflaky_logs_return_false(tmp_path):
    log_path = tmp_path / "task" / "attempt_0" / "rank0" / "stdout.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("normal training output\nfinished without known transient errors")

    assert not launch_nemo_run_workload._has_flaky_failure_in_logs(tmp_path)


def test_existing_monitor_signal_short_circuits_log_scan(tmp_path, monkeypatch):
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    failure_detected_event = threading.Event()
    failure_detected_event.set()

    def unexpected_scan(_workdir):
        pytest.fail("archived logs should not be scanned after a monitor hit")

    monkeypatch.setattr(launch_nemo_run_workload, "_has_flaky_failure_in_logs", unexpected_scan)

    assert launch_nemo_run_workload._should_retry_failed_attempt(
        tmp_path, log_buffer, failure_detected_event
    )


def test_existing_tee_signature_short_circuits_log_scan(tmp_path, monkeypatch):
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    log_buffer.write("The server socket has failed to listen on any local network address.")

    def unexpected_scan(_workdir):
        pytest.fail("archived logs should not be scanned after a tee hit")

    monkeypatch.setattr(launch_nemo_run_workload, "_has_flaky_failure_in_logs", unexpected_scan)

    assert launch_nemo_run_workload._should_retry_failed_attempt(
        tmp_path, log_buffer, threading.Event()
    )
