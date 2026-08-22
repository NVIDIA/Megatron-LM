# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.module import GraphableMegatronModule


@pytest.mark.parametrize(
    ("fp32_residual_connection", "params_dtype", "expected_dtype"),
    [
        (True, torch.bfloat16, torch.float32),
        (False, torch.bfloat16, torch.bfloat16),
        (False, torch.float16, torch.float16),
    ],
)
def test_static_hidden_states_match_residual_stream_dtype(
    monkeypatch: pytest.MonkeyPatch,
    fp32_residual_connection: bool,
    params_dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    captured = {}
    sentinel = object()

    def fake_ones(shape, **kwargs):
        captured["shape"] = shape
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(torch, "ones", fake_ones)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    module = SimpleNamespace(
        config=SimpleNamespace(
            context_parallel_size=2,
            sequence_parallel=True,
            tensor_model_parallel_size=4,
            hidden_size=128,
            fp32_residual_connection=fp32_residual_connection,
            params_dtype=params_dtype,
        )
    )

    static_inputs = GraphableMegatronModule.get_layer_static_inputs(
        module, seq_length=64, micro_batch_size=2
    )

    assert static_inputs == {"hidden_states": sentinel}
    assert captured == {
        "shape": (8, 2, 128),
        "dtype": expected_dtype,
        "requires_grad": True,
        "device": 3,
    }
