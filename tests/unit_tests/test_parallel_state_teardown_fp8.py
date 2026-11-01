# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version

import megatron.core.extensions.transformer_engine as te_extension
from megatron.core import parallel_state
from megatron.core.parallel_state import TeardownStage

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]


def _fake_state_manager(te_version, depth):
    """Mimic where each TE release range keeps the autocast depth."""
    reset = Mock()
    if Version(te_version) >= Version("2.15.0"):
        state = SimpleNamespace(autocast_depth=depth)
        return SimpleNamespace(quantization_state=state, reset=reset)
    if Version(te_version) >= Version("2.9.0"):
        return SimpleNamespace(AUTOCAST_DEPTH=depth, reset=reset)
    return SimpleNamespace(FP8_AUTOCAST_DEPTH=depth, reset=reset)


@pytest.mark.parametrize("te_version", ["2.8.0", "2.9.0", "2.15.0"])
@pytest.mark.parametrize("depth", [-1, 0, 1])
def test_teardown_requires_closed_te_autocast(monkeypatch, te_version, depth):
    manager = _fake_state_manager(te_version, depth)
    monkeypatch.setattr(te_extension, "FP8GlobalStateManager", manager)
    monkeypatch.setattr(
        te_extension, "is_te_min_version", lambda version: Version(te_version) >= Version(version)
    )
    release = Mock()
    callbacks = {stage: [] for stage in TeardownStage}
    callbacks[TeardownStage.VALIDATE].append(te_extension._check_te_autocast_exited)
    callbacks[TeardownStage.RELEASE_COMMUNICATION].append(release)
    callbacks[TeardownStage.RESET_STATE].append(te_extension._reset_te_global_state)
    monkeypatch.setattr(parallel_state, "_MODEL_PARALLEL_TEARDOWN_CALLBACKS", callbacks)

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
        release.assert_called_once()
        manager.reset.assert_called_once()
        destroy_groups.assert_called_once()
        assert parallel_state._MODEL_PARALLEL_GROUP is None
        return

    with pytest.raises(RuntimeError, match="Exit Transformer Engine autocast"):
        parallel_state.destroy_model_parallel()
    assert parallel_state._MODEL_PARALLEL_GROUP is group
    assert parallel_state._global_process_group_list is groups
    assert groups == [None, group]
    release.assert_not_called()
    manager.reset.assert_not_called()
    destroy_memory.assert_not_called()
    destroy_groups.assert_not_called()


def test_teardown_with_real_te_autocast():
    te = pytest.importorskip("transformer_engine.pytorch")
    # Disabling quantization still enters the real TE context and increments depth.
    with te.fp8_autocast(enabled=False):
        with pytest.raises(RuntimeError, match="Exit Transformer Engine autocast"):
            parallel_state.destroy_model_parallel()
    parallel_state.destroy_model_parallel()


def test_teardown_after_real_te_reset_during_unwind():
    te = pytest.importorskip("transformer_engine.pytorch")
    with te.fp8_autocast(enabled=False):
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(te_extension.FP8GlobalStateManager.reset).result(timeout=10)
    parallel_state.destroy_model_parallel()
    # A new real context must again enter and leave with consistent depth.
    with te.fp8_autocast(enabled=False):
        with pytest.raises(RuntimeError, match="Exit Transformer Engine autocast"):
            parallel_state.destroy_model_parallel()
    parallel_state.destroy_model_parallel()
