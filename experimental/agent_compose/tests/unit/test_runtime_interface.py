# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contract tests for the public runtime interface."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest


def test_runtime_import_stays_lightweight() -> None:
    script = (
        "import sys; "
        "import megatron.lite.runtime as runtime; "
        "assert runtime.RuntimeConfig().backend == 'mlite'; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_runtime_contracts() -> None:
    import torch

    from megatron.lite.runtime import PackedBatch, ParallelConfig

    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([2, 1]),
    )

    assert ParallelConfig(tp=2).tp == 2
    assert len(batch) == 2
    assert batch.total_tokens == 3
    assert batch.cu_seqlens.tolist() == [0, 2, 3]
    assert batch.make_position_ids().tolist() == [0, 1, 0]


def test_unregistered_backend_fails_explicitly() -> None:
    from megatron.lite.runtime import RuntimeConfig, create_runtime

    with pytest.raises(ValueError, match="No runtime backend registered"):
        create_runtime(RuntimeConfig(backend="missing"))


def test_runtime_required_method_set() -> None:
    from megatron.lite.runtime import Runtime

    assert Runtime.__abstractmethods__ == {
        "build_model",
        "eval_mode",
        "forward_backward",
        "load_checkpoint",
        "lr_scheduler_step",
        "optimizer_step",
        "save_checkpoint",
        "train_mode",
        "zero_grad",
    }


def test_registered_runtime_factory(monkeypatch) -> None:
    from megatron.lite.runtime import RuntimeConfig, create_runtime, register_runtime

    module = types.ModuleType("agent_compose_test_runtime")
    module.create = lambda hf_path, cfg: (hf_path, cfg)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    register_runtime("test", module.__name__)

    cfg = RuntimeConfig(backend="test", hf_path="model", backend_cfg={"tp": 2})
    assert create_runtime(cfg) == ("model", {"tp": 2})
