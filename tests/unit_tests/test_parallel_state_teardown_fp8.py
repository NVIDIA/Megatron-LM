# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core import parallel_state


@pytest.mark.parametrize("layout", ["quantization_state", "AUTOCAST_DEPTH", "FP8_AUTOCAST_DEPTH"])
def test_teardown_requires_closed_fp8_autocast(monkeypatch, layout):
    reset = Mock()
    if layout == "quantization_state":
        state = SimpleNamespace(autocast_depth=1, fp8_enabled=True)
        manager = SimpleNamespace(quantization_state=state, reset=reset)
    else:
        manager = SimpleNamespace(**{layout: 1}, FP8_ENABLED=True, reset=reset)
        state = manager
    original_state = vars(state).copy()

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

    with pytest.raises(RuntimeError, match="Exit Transformer Engine autocast"):
        parallel_state.destroy_model_parallel()
    assert vars(state) == original_state
    assert parallel_state._MODEL_PARALLEL_GROUP is group
    assert parallel_state._global_process_group_list is groups
    assert groups == [None, group]
    reset.assert_not_called()
    fused_a2a.nccl_ep_finalize.assert_not_called()
    fused_a2a.reset_fused_a2a_buffers.assert_not_called()
    destroy_memory.assert_not_called()
    destroy_groups.assert_not_called()
