# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic model offload callback: ordering, selection, and failure propagation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core  # noqa: F401


@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize("move_model", [False, True])
def test_before_model_offload_is_optional_and_precedes_transfer(
    transformer_engine_import_stub, monkeypatch, callback, move_model
):
    transformer_engine_import_stub()
    from megatron.lite.runtime import megatron_utils
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    calls = []
    monkeypatch.setattr(megatron_utils, "offload_model_to_cpu", lambda chunks: calls.append("move"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    extras = {"model_chunks": [torch.nn.Linear(2, 2)]}
    if callback:
        extras["before_model_offload"] = lambda: calls.append("callback")
    handle = SimpleNamespace(_extras=extras, _model=extras["model_chunks"][0], _optimizer=None)
    runtime = object.__new__(MegatronLiteRuntime)
    runtime.to(handle, "cpu", model=move_model, optimizer=False, grad=False)
    assert calls == ((["callback"] if callback else []) + ["move"] if move_model else [])


def test_offload_callback_failure_prevents_weight_transfer(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.runtime import megatron_utils
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    transfer = Mock()
    monkeypatch.setattr(megatron_utils, "offload_model_to_cpu", transfer)
    handle = SimpleNamespace(
        _model=torch.nn.Linear(2, 2),
        _optimizer=None,
        _extras={"before_model_offload": Mock(side_effect=RuntimeError("live lease"))},
    )
    with pytest.raises(RuntimeError, match="live lease"):
        object.__new__(MegatronLiteRuntime).to(handle, "cpu", optimizer=False)
    transfer.assert_not_called()
