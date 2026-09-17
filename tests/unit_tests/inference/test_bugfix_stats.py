# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reporter contracts, including dependency-free child processes and atomic files."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPORTER = ROOT / "megatron/core/inference/bugfix_stats.py"
SUMMARY = ROOT / "tools/summarize_inference_bugfix_stats.py"
ENV = "MEGATRON_INFERENCE_BUGFIX_STATS_DIR"


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wait(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Reporter did not reach the expected state")


def _snapshots(directory):
    return [json.loads(path.read_text()) for path in sorted(directory.glob("bugfix-*.json"))]


@pytest.fixture
def reporter(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    module = _load(REPORTER)
    monkeypatch.setattr(module, "_FLUSH_INTERVAL", 0.03)
    yield module, tmp_path
    module._shutdown()


@pytest.mark.parametrize("directory", [None, "", "relative-directory"])
def test_disabled_does_not_start_reporting(directory, tmp_path, monkeypatch):
    if directory is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, directory)
    module = _load(REPORTER)

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled reporting attempted collection or I/O")

    monkeypatch.setattr(module, "_Reporter", forbidden)
    monkeypatch.setattr(module.tempfile, "mkstemp", forbidden)
    monkeypatch.setattr(module.getpass, "getuser", forbidden)
    for _ in range(100):
        module.record_bugfix("prefix_cache.disabled")
    module._shutdown()
    assert module._reporter is None
    assert not list(tmp_path.iterdir())


def test_concurrent_counts_and_periodic_atomic_snapshots(reporter):
    module, directory = reporter
    module.record_bugfix("first")
    _wait(lambda: len(_snapshots(directory)) == 1)
    first = _snapshots(directory)[0]
    assert first["counters"] == {"first": 1}
    observed = []
    read_errors = []
    done = threading.Event()

    def read_snapshots():
        try:
            while not done.is_set():
                observed.extend(_snapshots(directory))
                time.sleep(0.002)
        except Exception as error:
            read_errors.append(error)

    reader = threading.Thread(target=read_snapshots)
    reader.start()
    writers = [
        threading.Thread(target=lambda: [module.record_bugfix("parallel") for _ in range(500)])
        for _ in range(4)
    ]
    try:
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join()
        _wait(lambda: _snapshots(directory)[0]["counters"].get("parallel") == 2000)
    finally:
        done.set()
        reader.join()
    latest = _snapshots(directory)[0]
    assert latest["sequence"] > first["sequence"]
    assert latest["counters"] == {"first": 1, "parallel": 2000}
    assert latest["process_id"] == first["process_id"]
    assert observed and all(row["schema_version"] == 1 for row in observed)
    assert not read_errors
    assert len(list(directory.glob("bugfix-*.json"))) == 1
    assert next(directory.glob("bugfix-*.json")).stat().st_mode & 0o777 == 0o600
    assert str(directory) not in json.dumps(latest)


@pytest.mark.parametrize("failure", ["missing", "not_directory", "permission"])
def test_filesystem_failure_is_silent_and_does_not_escape(reporter, monkeypatch, capsys, failure):
    module, directory = reporter
    if failure == "missing":
        monkeypatch.setattr(module, "_DIRECTORY", str(directory / "missing"))
    elif failure == "not_directory":
        path = directory / "file"
        path.write_text("not a directory")
        monkeypatch.setattr(module, "_DIRECTORY", str(path))
    else:

        def denied(*args, **kwargs):
            raise PermissionError(f"private directory: {directory}")

        monkeypatch.setattr(module.tempfile, "mkstemp", denied)
    module.record_bugfix("failure")
    _wait(lambda: module._reporter.failed)
    module.record_bugfix("another")
    module._shutdown()
    assert not list(directory.glob("bugfix-*.json"))
    assert not (directory / "missing").exists()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_failed_replace_preserves_last_snapshot(reporter, monkeypatch):
    module, directory = reporter
    module.record_bugfix("preserved")
    _wait(lambda: bool(_snapshots(directory)))
    previous = _snapshots(directory)[0]

    def failed_replace(*args):
        raise OSError("write failed")

    monkeypatch.setattr(module.os, "replace", failed_replace)
    module.record_bugfix("not_persisted")
    _wait(lambda: module._reporter.failed)
    assert _snapshots(directory) == [previous]
    assert not list(directory.glob(".bugfix-*.tmp"))


def test_stalled_writer_does_not_block_calls_or_shutdown(reporter, monkeypatch):
    module, _ = reporter
    entered, release = threading.Event(), threading.Event()

    def stalled_write(self):
        entered.set()
        release.wait(10)

    monkeypatch.setattr(module._Reporter, "_write_snapshot", stalled_write)
    monkeypatch.setattr(module, "_EXIT_TIMEOUT", 0.01)
    module.record_bugfix("stalled")
    assert entered.wait(5)
    try:
        started = time.monotonic()
        for _ in range(100):
            module.record_bugfix("stalled")
        module._shutdown()
        assert time.monotonic() - started < 1
        assert module._reporter.counts["stalled"] == 101
    finally:
        release.set()
        module._reporter.thread.join(timeout=5)


def test_thread_start_failure_is_contained(reporter, monkeypatch):
    module, _ = reporter

    def cannot_start(self):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(module.threading.Thread, "start", cannot_start)
    module.record_bugfix("first")
    module.record_bugfix("second")
    module._shutdown()
    assert module._reporter.failed


BOOTSTRAP = """
import importlib.util, json, os, pathlib, sys, time
spec = importlib.util.spec_from_file_location('bugfix_stats', sys.argv[1])
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
assert 'torch' not in sys.modules and 'opentelemetry' not in sys.modules
directory = pathlib.Path(os.environ['MEGATRON_INFERENCE_BUGFIX_STATS_DIR'])
"""


def _child_env(directory, rank="0"):
    return dict(
        os.environ,
        MEGATRON_INFERENCE_BUGFIX_STATS_DIR=str(directory),
        SLURM_CLUSTER_NAME="test-cluster",
        SLURM_JOB_ID="123",
        SLURM_STEP_ID="0",
        SLURM_JOB_USER="test-user",
        RANK=rank,
    )


def test_multiple_processes_and_dependency_free_summary(tmp_path):
    code = BOOTSTRAP + "\nfor _ in range(int(sys.argv[2])): stats.record_bugfix('shared')\n"
    children = [
        subprocess.Popen(
            [sys.executable, "-S", "-c", code, str(REPORTER), str(rank + 2)],
            env=_child_env(tmp_path, str(rank)),
        )
        for rank in range(3)
    ]
    for child in children:
        assert child.wait(timeout=20) == 0
    rows = _snapshots(tmp_path)
    assert len(rows) == 3
    assert {row["rank"] for row in rows} == {"0", "1", "2"}
    assert {row["username"] for row in rows} == {"test-user"}
    result = subprocess.run(
        [sys.executable, "-S", str(SUMMARY), str(tmp_path), "--json", "--details"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    summary = json.loads(result.stdout)
    assert summary["fixes"] == [
        {"name": "shared", "process_hits": 9, "affected_jobs": 1, "affected_users": 1}
    ]
    assert summary["observed_processes"] == len(summary["processes"]) == 3


def test_normal_exit_flushes_changes_after_initial_snapshot(tmp_path):
    code = BOOTSTRAP + """
stats.record_bugfix('exit')
deadline = time.monotonic() + 10
while not list(directory.glob('bugfix-*.json')):
    assert time.monotonic() < deadline
    time.sleep(0.01)
stats.record_bugfix('exit')
stats.record_bugfix('exit')
"""
    subprocess.run(
        [sys.executable, "-S", "-c", code, str(REPORTER)],
        env=_child_env(tmp_path),
        check=True,
        timeout=20,
    )
    assert _snapshots(tmp_path)[0]["counters"] == {"exit": 3}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_resets_inherited_counts_and_locked_mutexes(tmp_path):
    code = BOOTSTRAP + """
stats.record_bugfix('forked')
stats.record_bugfix('forked')
stats.record_bugfix('forked')
stats._init_lock.acquire()
stats._reporter.lock.acquire()
pid = os.fork()
if pid == 0:
    stats.record_bugfix('forked')
    stats.record_bugfix('forked')
    stats._shutdown()
    os._exit(0)
stats._reporter.lock.release()
stats._init_lock.release()
assert os.waitpid(pid, 0)[1] == 0
stats.record_bugfix('forked')
"""
    subprocess.run(
        [sys.executable, "-S", "-c", code, str(REPORTER)],
        env=_child_env(tmp_path),
        check=True,
        timeout=25,
    )
    assert sorted(row["counters"]["forked"] for row in _snapshots(tmp_path)) == [2, 4]


def test_summary_deduplicates_snapshots_and_separates_clusters(tmp_path):
    def write(filename, process, sequence, count, cluster):
        (tmp_path / filename).write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    process_id=process,
                    sequence=sequence,
                    counters={"fix": count},
                    cluster=cluster,
                    job_id="123",
                    username="same-user",
                )
            )
        )

    write("bugfix-a.json", "a", 1, 2, "first")
    write("bugfix-a-new.json", "a", 2, 5, "first")
    write("bugfix-b.json", "b", 1, 7, "second")
    (tmp_path / "bugfix-invalid.json").write_text("partial json")
    (tmp_path / ".bugfix-unpublished.tmp").write_text("partial json")
    summary = _load(SUMMARY).summarize(tmp_path)
    assert summary["fixes"] == [
        {"name": "fix", "process_hits": 12, "affected_jobs": 2, "affected_users": 1}
    ]
    assert summary["skipped_files"] == 1
    assert len(summary["processes"]) == 2
