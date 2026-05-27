# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.training import inprocess_restart


def test_inprocess_restart_returns_train_when_extension_unavailable(monkeypatch):
    train = object()
    monkeypatch.setattr(inprocess_restart, "inprocess", None)

    with pytest.warns(UserWarning, match="In-process restart is not available"):
        wrapped = inprocess_restart.inprocess_restart(train, SimpleNamespace())

    assert wrapped is train


def test_destroy_state_resets_training_and_rerun_state(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inprocess_restart.rerun_state_machine,
        "destroy_rerun_state_machine",
        lambda: calls.append("rerun"),
    )
    from megatron.training import training

    monkeypatch.setattr(training, "destroy_global_state", lambda: calls.append("training"))

    inprocess_restart.destroy_state()

    assert calls == ["training", "rerun"]


def test_inprocess_restart_builds_wrapper_with_node_granularity(monkeypatch):
    calls = []

    class FakeInprocess:
        class rank_assignment:
            class LayerFlag:
                RESERVE = "reserve"

            class Layer:
                def __init__(self, **kwargs):
                    calls.append(("layer", kwargs))

            class Tree:
                def __init__(self, layers):
                    calls.append(("tree", len(layers)))

        class finalize:
            class ThreadedFinalize:
                def __init__(self, **kwargs):
                    calls.append(("finalize", kwargs["fn"]))

        class initialize:
            class RetryController:
                def __init__(self, min_world_size):
                    calls.append(("retry", min_world_size))

        class nested_restarter:
            class NestedRestarterHandlingCompleted:
                pass

            class NestedRestarterHandlingStarting:
                pass

            class NestedRestarterFinalized:
                pass

            class NestedRestarterAborted:
                pass

        class abort:
            class Abort:
                pass

            class AbortTransformerEngine:
                pass

            class AbortTorchDistributed:
                pass

        class state:
            class FrozenState:
                pass

        class health_check:
            class CudaHealthCheck:
                def __init__(self, timeout):
                    calls.append(("health", timeout.total_seconds()))

        class Compose:
            def __init__(self, *items):
                calls.append(("compose", len(items)))

        class Wrapper:
            def __init__(self, **kwargs):
                calls.append(("wrapper", kwargs["enabled"], kwargs["store_kwargs"]["port"]))

            def __call__(self, train):
                return ("wrapped", train)

    args = SimpleNamespace(
        inprocess_active_world_size=4,
        inprocess_granularity="node",
        inprocess_empty_cuda_cache=True,
        async_strategy="mcore",
        inprocess_heartbeat_interval=1,
        inprocess_heartbeat_timeout=2,
        inprocess_barrier_timeout=3,
        inprocess_completion_timeout=4,
        inprocess_monitor_process_interval=5,
        inprocess_monitor_thread_interval=6,
        inprocess_last_call_wait=7,
        inprocess_soft_timeout=8,
        inprocess_hard_timeout=9,
        inprocess_termination_grace_time=10,
    )
    monkeypatch.setenv("TORCH_CPP_LOG_LEVEL", "error")
    monkeypatch.setenv("MASTER_PORT", "12000")
    monkeypatch.setattr(inprocess_restart, "inprocess", FakeInprocess)
    monkeypatch.setattr(inprocess_restart.torch.cuda, "device_count", lambda: 2)

    wrapped = inprocess_restart.inprocess_restart("train", args)

    assert wrapped == ("wrapped", "train")
    assert ("retry", 4) in calls
    assert ("tree", 2) in calls
    assert ("wrapper", True, 12002) in calls


def test_maybe_wrap_for_inprocess_restart_skips_tcp_store_when_disabled(monkeypatch):
    pretrain = object()
    monkeypatch.setattr(
        inprocess_restart.arguments,
        "parse_args",
        lambda ignore_unknown_args=True: SimpleNamespace(inprocess_restart=False),
    )

    wrapped, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    assert wrapped is pretrain
    assert store is None


def test_maybe_force_nccl_backend_init_noops_without_inprocess_restart(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inprocess_restart,
        "get_args",
        lambda: SimpleNamespace(inprocess_restart=False),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: calls.append("all_reduce"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))

    inprocess_restart.maybe_force_nccl_backend_init(torch.device("cpu"))

    assert calls == []


def test_maybe_force_nccl_backend_init_reduces_tensor_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inprocess_restart,
        "get_args",
        lambda: SimpleNamespace(inprocess_restart=True),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: calls.append(("all_reduce", tensor.shape)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(("synchronize",)))

    inprocess_restart.maybe_force_nccl_backend_init(torch.device("cpu"))

    assert calls == [("all_reduce", torch.Size([128])), ("synchronize",)]
