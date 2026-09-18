# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Trainer worker-start compatibility after retiring the explicit OTel bootstrap."""

import ast
import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo.lens.resources import publish_otel_resource_attributes

ASYNC_PATH = Path(__file__).resolve().parents[3] / "megatron/training/async_utils.py"


@pytest.mark.parametrize("cpu_shm", [False, True])
def test_worker_start_preserves_backend_arguments_and_inherits_carrier(cpu_shm):
    events = []
    missing = object()

    class Queue:
        def __init__(self, persistent, cpu_shm_mode=missing):
            events.append(("create", persistent, cpu_shm_mode))

        @staticmethod
        def warmup_persistent_caller(rank, cpu_priority, io_priority, cpu_shm_mode=missing):
            events.append(("warmup", rank, cpu_priority, io_priority, cpu_shm_mode))
            events.append(("carrier", os.environ["OTEL_RESOURCE_ATTRIBUTES"]))

    def results_queue(mp_mode):
        events.append(("results", mp_mode))

    args = SimpleNamespace(
        async_ckpt_use_cpu_shm=cpu_shm, async_ckpt_cpu_priority=7, async_ckpt_io_priority=4
    )
    namespace = {
        "get_args": lambda: args,
        "AsyncCallsQueue": Queue,
        "get_write_results_queue": results_queue,
        "inspect": inspect,
        "time": time,
    }
    tree = ast.parse(ASYNC_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_persistent_async_worker"
    )
    function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
    exec(compile(ast.Module(body=[function], type_ignores=[]), ASYNC_PATH, "exec"), namespace)

    with publish_otel_resource_attributes({"nv.dl.rank": 3, "nv.dl.role": "trainer"}):
        carrier = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
        namespace["init_persistent_async_worker"](3, mp_mode="spawn")

    assert events == [
        ("create", True, cpu_shm),
        ("warmup", 3, 7, 4, cpu_shm),
        ("carrier", carrier),
        ("results", "fork"),
    ]


@pytest.mark.parametrize("cpu_shm", [False, True])
def test_legacy_nvrx_cpu_shm_guard_is_preserved(cpu_shm):
    events = []

    class LegacyQueue:
        def __init__(self, persistent):
            events.append(("create", persistent))

        @staticmethod
        def warmup_persistent_caller(rank, cpu_priority, io_priority):
            events.append(("warmup", rank, cpu_priority, io_priority))

    def results_queue(mp_mode):
        events.append(("results", mp_mode))

    namespace = {
        "get_args": lambda: SimpleNamespace(
            async_ckpt_use_cpu_shm=cpu_shm, async_ckpt_cpu_priority=7, async_ckpt_io_priority=4
        ),
        "AsyncCallsQueue": LegacyQueue,
        "get_write_results_queue": results_queue,
        "inspect": inspect,
        "time": time,
    }
    tree = ast.parse(ASYNC_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_persistent_async_worker"
    )
    function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
    exec(compile(ast.Module(body=[function], type_ignores=[]), ASYNC_PATH, "exec"), namespace)
    if cpu_shm:
        with pytest.raises(AssertionError, match="does not support cpu_shm_mode"):
            namespace["init_persistent_async_worker"](3)
        assert events == [("create", True)]
    else:
        namespace["init_persistent_async_worker"](3)
        assert events == [("create", True), ("warmup", 3, 7, 4), ("results", "fork")]
