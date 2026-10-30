# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tests.test_utils.python_scripts import launch_nemo_run_workload


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


@pytest.fixture
def launch_workload(monkeypatch):
    """Exercise CLI routing through a successful executor without launching Docker."""
    load_workloads = Mock()
    monkeypatch.setattr(launch_nemo_run_workload.recipe_parser, "load_workloads", load_workloads)
    executor = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(launch_nemo_run_workload.run, "DockerExecutor", executor)
    monkeypatch.setattr(launch_nemo_run_workload.run, "Script", Mock())
    monkeypatch.setattr(launch_nemo_run_workload.run, "Packager", Mock())
    experiment = MagicMock()
    experiment.__enter__.return_value = experiment
    experiment.add.return_value = "task-1"
    experiment.status.return_value = {"task-1": {"status": "SUCCEEDED"}}
    monkeypatch.setattr(launch_nemo_run_workload.run, "Experiment", Mock(return_value=experiment))
    monkeypatch.setattr(launch_nemo_run_workload, "_cancel_on_flaky_failure", Mock())

    def launch(**overrides):
        kwargs = dict(
            scope="unit-tests",
            model="unit-tests",
            test_case="tests/unit_tests/ssm/test_gated_delta_net2.py",
            environment="dev",
            platform="dgx_gb200",
            container_image="unused:test",
            tag="latest",
        )
        kwargs.update(overrides)
        load_workloads.return_value = [
            SimpleNamespace(
                type="test",
                spec={"name": "portable-test", "test_case": kwargs["test_case"], "script": "true"},
            )
        ]
        with pytest.raises(SystemExit) as exit_info:
            launch_nemo_run_workload.main.callback(**kwargs)
        assert exit_info.value.code == 0
        experiment.run.assert_called_once_with(detach=False, tail_logs=True, sequential=False)
        return load_workloads.call_args.kwargs, executor.call_args.kwargs["env_vars"]

    return launch


@pytest.mark.parametrize(
    "enabled,overrides,expected_scope",
    [
        ("true", {}, "unit-tests-github-portable"),
        (None, {}, "unit-tests"),
        ("false", {}, "unit-tests"),
        ("true", {"platform": "dgx_h100"}, "unit-tests"),
        ("true", {"environment": "lts"}, "unit-tests"),
        ("true", {"tag": "legacy"}, "unit-tests"),
        ("true", {"model": "gpt"}, "unit-tests"),
        ("true", {"scope": "mr-github"}, "mr-github"),
        ("true", {"test_case": "tests/unit_tests/ssm/test_gated_delta_net.py"}, "unit-tests"),
        ("true", {"test_case": "tests/unit_tests/new_unreviewed_bucket.py"}, "unit-tests"),
    ],
)
def test_portable_recipe_scope_is_limited_to_enabled_gb200_candidates(
    monkeypatch, launch_workload, enabled, overrides, expected_scope
):
    if enabled is None:
        monkeypatch.delenv("UNIT_TEST_GB200_ENABLED", raising=False)
    else:
        monkeypatch.setenv("UNIT_TEST_GB200_ENABLED", enabled)

    recipe_args, _ = launch_workload(**overrides)

    assert recipe_args["scope"] == expected_scope
    assert recipe_args["platform"] == overrides.get("platform", "dgx_gb200")
    assert recipe_args["environment"] == overrides.get("environment", "dev")
    assert recipe_args["tag"] == overrides.get("tag", "latest")


@pytest.mark.parametrize(
    "enabled,environment,tag,platform,expected_enabled",
    [
        ("true", "dev", "latest", "dgx_gb200", True),
        ("true", "dev", "latest", "dgx_h100", True),
        ("false", "dev", "latest", "dgx_gb200", False),
        ("true", "lts", "latest", "dgx_gb200", False),
        ("true", "dev", "legacy", "dgx_h100", False),
    ],
)
def test_portable_placement_flag_reaches_only_dev_latest_executors(
    monkeypatch, launch_workload, enabled, environment, tag, platform, expected_enabled
):
    monkeypatch.setenv("UNIT_TEST_GB200_ENABLED", enabled)

    _, executor_env = launch_workload(environment=environment, tag=tag, platform=platform)

    assert (executor_env.get("UNIT_TEST_GB200_ENABLED") == "true") is expected_enabled


@pytest.mark.parametrize(
    "overrides,expected_nonblocking",
    [
        ({"platform": "dgx_h100"}, True),
        ({"platform": "dgx_gb200"}, True),
        ({"platform": "dgx_a100"}, False),
        ({"scope": "mr-github"}, False),
        ({"environment": "lts"}, False),
        ({"tag": "legacy"}, False),
        ({"test_case": "tests/unit_tests/ssm/test_gated_delta_net2.py"}, False),
    ],
)
def test_transformer_lanes_preserve_nonblocking_nccl_finalization(
    monkeypatch, launch_workload, overrides, expected_nonblocking
):
    monkeypatch.setenv("UNIT_TEST_GB200_ENABLED", "true")
    kwargs = {"test_case": "tests/unit_tests/transformer/**/*.py", **overrides}

    _, executor_env = launch_workload(**kwargs)

    assert (executor_env.get("TORCH_NCCL_USE_COMM_NONBLOCKING") == "1") is expected_nonblocking
    if expected_nonblocking:
        assert executor_env["PYTEST_ADDOPTS"] == "-o faulthandler_timeout=600"
        assert executor_env["TORCH_CPP_LOG_LEVEL"] == "INFO"
        assert executor_env["NCCL_DEBUG_SUBSYS"] == "INIT,BOOTSTRAP,ENV,DESTROY"
        assert executor_env["NCCL_DEBUG_FILE"].endswith("nccl_debug.%h.%p.log")
