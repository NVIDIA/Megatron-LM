# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core import parallel_state

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]


@pytest.mark.parametrize("layout", ["quantization_state", "AUTOCAST_DEPTH", "FP8_AUTOCAST_DEPTH"])
@pytest.mark.parametrize("depth", [-1, 0, 1])
def test_teardown_requires_closed_fp8_autocast(monkeypatch, layout, depth):
    reset = Mock()
    if layout == "quantization_state":
        state = SimpleNamespace(autocast_depth=depth, fp8_enabled=True)
        manager = SimpleNamespace(quantization_state=state, reset=reset)
    else:
        manager = SimpleNamespace(**{layout: depth}, FP8_ENABLED=True, reset=reset)
        state = manager

    te_module = ModuleType("transformer_engine.pytorch")
    quantization_module = ModuleType("transformer_engine.pytorch.quantization")
    fp8_module = ModuleType("transformer_engine.pytorch.fp8")
    if layout == "FP8_AUTOCAST_DEPTH":
        # Before TE 2.9 the manager was exported from fp8, not quantization.
        fp8_module.FP8GlobalStateManager = manager
    else:
        quantization_module.FP8GlobalStateManager = manager
    for module in (te_module, quantization_module, fp8_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    fused_a2a = ModuleType("megatron.core.transformer.moe.fused_a2a")
    fused_a2a.nccl_ep_finalize = Mock()
    fused_a2a.reset_fused_a2a_buffers = Mock()
    monkeypatch.setitem(sys.modules, fused_a2a.__name__, fused_a2a)
    group = object()
    groups = [None, group]
    destroy_groups = Mock()
    destroy_memory = Mock()
    monkeypatch.setattr(parallel_state, "_MODEL_PARALLEL_GROUP", group)
    monkeypatch.setattr(parallel_state, "_global_process_group_list", groups)
    monkeypatch.setattr(parallel_state, "_destroy_created_process_groups", destroy_groups)
    monkeypatch.setattr(parallel_state.SymmetricMemoryManager, "destroy", destroy_memory)

    if depth <= 0:
        parallel_state.destroy_model_parallel()
        reset.assert_called_once()
        destroy_groups.assert_called_once()
        assert parallel_state._MODEL_PARALLEL_GROUP is None
        return

    with pytest.raises(RuntimeError, match="Exit Transformer Engine autocast"):
        parallel_state.destroy_model_parallel()
    assert parallel_state._MODEL_PARALLEL_GROUP is group
    assert parallel_state._global_process_group_list is groups
    assert groups == [None, group]
    reset.assert_not_called()
    fused_a2a.nccl_ep_finalize.assert_not_called()
    fused_a2a.reset_fused_a2a_buffers.assert_not_called()
    destroy_memory.assert_not_called()
    destroy_groups.assert_not_called()


def test_teardown_rejects_unknown_fp8_depth(monkeypatch):
    manager = SimpleNamespace(reset=Mock())
    te_module = ModuleType("transformer_engine.pytorch")
    quantization_module = ModuleType("transformer_engine.pytorch.quantization")
    quantization_module.FP8GlobalStateManager = manager
    monkeypatch.setitem(sys.modules, te_module.__name__, te_module)
    monkeypatch.setitem(sys.modules, quantization_module.__name__, quantization_module)
    with pytest.raises(RuntimeError, match="Cannot determine Transformer Engine autocast depth"):
        parallel_state.destroy_model_parallel()
    manager.reset.assert_not_called()


def test_teardown_with_real_fp8_autocast():
    te = pytest.importorskip("transformer_engine.pytorch")
    # Disabling quantization still enters the real TE context and increments depth.
    with te.fp8_autocast(enabled=False):
        with pytest.raises(RuntimeError, match="Exit Transformer Engine autocast"):
            parallel_state.destroy_model_parallel()
    parallel_state.destroy_model_parallel()


def test_teardown_after_real_fp8_reset_during_unwind():
    te = pytest.importorskip("transformer_engine.pytorch")
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    except ImportError:
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    with te.fp8_autocast(enabled=False):
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(FP8GlobalStateManager.reset).result(timeout=10)
    parallel_state.destroy_model_parallel()
    # A new real context must again enter and leave with consistent depth.
    with te.fp8_autocast(enabled=False):
        with pytest.raises(RuntimeError, match="Exit Transformer Engine autocast"):
            parallel_state.destroy_model_parallel()
    parallel_state.destroy_model_parallel()
