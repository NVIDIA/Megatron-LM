# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contract checks for the PR1 Megatron Lite skeleton."""

from __future__ import annotations

import torch

from megatron.lite.model.registry import list_models, resolve_runtime_model_name
from megatron.lite.runtime import RuntimeConfig, create_runtime
from megatron.lite.runtime.backends.mlite import MegatronLiteConfig
from megatron.lite.runtime.contracts import OptimizerConfig, TrainBatch


def test_toy_dense_registry_contract():
    assert "toy_dense" in list_models()
    assert resolve_runtime_model_name("toy_dense", "torch") == "toy_dense"


def test_toy_dense_runtime_one_cpu_step():
    torch.manual_seed(1234)

    runtime = create_runtime(
        RuntimeConfig(
            backend="mlite",
            backend_cfg=MegatronLiteConfig(
                model_name="toy_dense",
                impl="torch",
                device="cpu",
                optimizer=OptimizerConfig(lr=0.05),
                impl_cfg={"input_dim": 4, "hidden_dim": 8, "output_dim": 2},
            ),
        )
    )
    handle = runtime.build_model()
    runtime.train_mode(handle)

    batch = TrainBatch(
        inputs=torch.randn(3, 4),
        targets=torch.randn(3, 2),
    )

    runtime.zero_grad(handle)
    result = runtime.forward_backward(handle, batch)
    ok, grad_norm, zero_grad_count = runtime.optimizer_step(handle)
    lr = runtime.lr_scheduler_step(handle)

    assert ok is True
    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss)
    assert grad_norm > 0.0
    assert zero_grad_count is not None
    assert lr == 0.05
