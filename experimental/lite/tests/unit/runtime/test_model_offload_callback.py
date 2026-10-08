# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core  # noqa: F401


@pytest.mark.parametrize("callback", ["absent", "success", "failure"])
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

    def before_offload():
        calls.append("callback")
        if callback == "failure":
            raise RuntimeError("live lease")

    if callback != "absent":
        extras["before_model_offload"] = before_offload
    handle = SimpleNamespace(_extras=extras, _model=extras["model_chunks"][0], _optimizer=None)
    runtime = object.__new__(MegatronLiteRuntime)
    if callback == "failure" and move_model:
        with pytest.raises(RuntimeError, match="live lease"):
            runtime.to(handle, "cpu", model=True, optimizer=False, grad=False)
        assert calls == ["callback"]
    else:
        runtime.to(handle, "cpu", model=move_model, optimizer=False, grad=False)
        expected = ["move"] if callback == "absent" else ["callback", "move"]
        assert calls == (expected if move_model else [])
