# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.test_utils.python_scripts import launch_nemo_run_workload


@pytest.mark.parametrize(
    "failure_log",
    [
        "requests.exceptions.ConnectionError: [Errno 101] Network is unreachable",
        "urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected())",
        "curl: (6) Could not resolve host: registry.example.com",
        "Temporary failure resolving 'archive.ubuntu.com'",
        "docker: net/http: TLS handshake timeout",
        "Client.Timeout exceeded while awaiting headers",
    ],
)
def test_network_interruptions_are_flaky(failure_log):
    assert launch_nemo_run_workload.is_network_interruption(failure_log)
    assert launch_nemo_run_workload.is_flaky_failure(failure_log)


@pytest.mark.parametrize(
    "failure_log",
    [
        "requests.exceptions.HTTPError: 401 Client Error: Unauthorized",
        "requests.exceptions.HTTPError: 404 Client Error: Not Found",
        "TimeoutError: model synchronization timed out",
        "AssertionError: model output did not match expected output",
    ],
)
def test_deterministic_failures_are_not_network_interruptions(failure_log):
    assert not launch_nemo_run_workload.is_network_interruption(failure_log)


def test_network_retries_use_bounded_exponential_backoff():
    failure_log = "Connection reset by peer"

    assert launch_nemo_run_workload.network_retry_delay_seconds(1, failure_log) == 10
    assert launch_nemo_run_workload.network_retry_delay_seconds(2, failure_log) == 20
    assert launch_nemo_run_workload.network_retry_delay_seconds(3, failure_log) == 0
    assert launch_nemo_run_workload.network_retry_delay_seconds(1, "AssertionError") == 0
