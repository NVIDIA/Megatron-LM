# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in, process-local counters for inference bugfix conditions.

Set MEGATRON_INFERENCE_BUGFIX_STATS_DIR before importing this module to an
existing, writable directory. No configuration means no reporting. This module
uses only the standard library; storage configuration belongs to the launcher.
"""

import atexit
import getpass
import json
import os
import tempfile
import threading
import time
import uuid

_FLUSH_INTERVAL = 30.0
_EXIT_TIMEOUT = 2.0
_DIRECTORY = os.environ.get("MEGATRON_INFERENCE_BUGFIX_STATS_DIR", "")
# Do not resolve paths, probe mounts, or create directories in an inference call.
if not os.path.isabs(_DIRECTORY):
    _DIRECTORY = ""
_reporter = None
_init_lock = threading.Lock()


class _Reporter:
    """One writer and one cumulative snapshot per process."""

    def __init__(self, directory):
        self.directory = directory
        self.counts = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.failed = False
        self.thread = threading.Thread(target=self._run, name="inference-bugfix-stats", daemon=True)

    def record(self, name):
        """Increment a counter without waiting for the writer's I/O."""
        if not self.failed:
            with self.lock:
                self.counts[name] = self.counts.get(name, 0) + 1

    def _run(self):
        # Identity discovery and all filesystem operations run off the caller's
        # thread. A missing mount or a stalled shared filesystem must not stall
        # inference. Never include paths or the whole environment in a snapshot.
        try:
            self.metadata = {
                "schema_version": 1,
                "process_id": uuid.uuid4().hex,
                "pid": os.getpid(),
                "username": os.environ.get("SLURM_JOB_USER") or getpass.getuser(),
                "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "step_id": os.environ.get("SLURM_STEP_ID"),
                "rank": os.environ.get("RANK", os.environ.get("SLURM_PROCID")),
                "world_size": os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS")),
                "started_at_ns": time.time_ns(),
            }
            self.sequence = 0
            while True:
                self._write_snapshot()
                if self.stop.wait(_FLUSH_INTERVAL):
                    self._write_snapshot()
                    return
        except Exception:
            # Reporting is best effort. Do not leak a private path through an
            # exception message, logging handler, or background-thread traceback.
            self.failed = True

    def _write_snapshot(self):
        with self.lock:
            counts = self.counts.copy()
        self.sequence += 1
        snapshot = dict(
            self.metadata, sequence=self.sequence, updated_at_ns=time.time_ns(), counters=counts
        )
        destination = os.path.join(self.directory, f"bugfix-{self.metadata['process_id']}.json")
        # mkstemp creates mode-0600 files. Independent processes never append to
        # a shared file; readers see either the previous or the next full snapshot.
        fd, temporary = tempfile.mkstemp(prefix=".bugfix-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(snapshot, stream, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def record_bugfix(name: str) -> None:
    """Count one execution of a named condition in this process, when opted in.

    Use a stable literal name, never request data. Counts include replicated
    execution on TP/PP ranks; they are not counts of distinct requests. The
    environment must be configured before import. Reporting never raises an
    ordinary exception into inference and never performs I/O in this call.
    """
    if not _DIRECTORY:
        return
    global _reporter
    try:
        if not isinstance(name, str) or not name:
            return
        if _reporter is None:
            with _init_lock:
                if _reporter is None:
                    _reporter = _Reporter(_DIRECTORY)
                    _reporter.record(name)
                    try:
                        _reporter.thread.start()
                    except Exception:
                        _reporter.failed = True
                    return
        _reporter.record(name)
    except Exception:
        pass


def _shutdown():
    if _reporter is not None:
        try:
            _reporter.stop.set()
            _reporter.thread.join(timeout=_EXIT_TIMEOUT)
        except Exception:
            pass


def _after_fork():
    # Fork does not inherit worker threads, but it does inherit counters and
    # potentially locked mutexes. The child starts an independent reporter.
    global _reporter, _init_lock
    _reporter = None
    _init_lock = threading.Lock()


if _DIRECTORY:
    atexit.register(_shutdown)
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork)
