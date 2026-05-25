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
