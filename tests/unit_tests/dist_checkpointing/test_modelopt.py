# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron/core/dist_checkpointing/strategies/modelopt.py."""

import importlib
import sys
import types
from unittest import mock

import pytest
import torch

# ---------------------------------------------------------------------------
# Install modelopt stubs before the strategy module is imported.
# _MTO and _DIST are MagicMocks so attribute access auto-creates child mocks;
# we configure return values in each test's setup_method.
# ---------------------------------------------------------------------------
_MTO = mock.MagicMock(name="modelopt.torch.opt")
_DIST = mock.MagicMock(name="modelopt.torch.utils.distributed")
_MODELOPT_ROOT = types.ModuleType("modelopt")
_MODELOPT_ROOT.__version__ = "0.0.0-test"

for _key, _val in {
    "modelopt": _MODELOPT_ROOT,
    "modelopt.torch": mock.MagicMock(name="modelopt.torch"),
    "modelopt.torch.opt": _MTO,
    "modelopt.torch.utils": mock.MagicMock(name="modelopt.torch.utils"),
    "modelopt.torch.utils.distributed": _DIST,
}.items():
    sys.modules.setdefault(_key, _val)

from megatron.core.dist_checkpointing.strategies.modelopt import (
    _load_extra_state_from_sharded_checkpoint,
    remove_per_module_state,
    restore_sharded_modelopt_state,
    save_modelopt_state,
    save_sharded_modelopt_state,
)


class TestModeloptLazyImport:
    def test_modelopt_not_imported_at_module_level(self):
        strategy_key = "megatron.core"
        modelopt_keys = [k for k in sys.modules if k == "modelopt" or k.startswith("modelopt.")]
        saved = {k: sys.modules.pop(k) for k in modelopt_keys + [strategy_key] if k in sys.modules}
        try:
            importlib.import_module(strategy_key)
            assert "modelopt" not in sys.modules
            assert "modelopt.torch.opt" not in sys.modules
            assert "modelopt.torch.utils.distributed" not in sys.modules
        finally:
            sys.modules.update(saved)


class TestRemovePerModuleState:
    def test_no_key_is_noop(self):
        state = {"other": 1}
        remove_per_module_state(state)
        assert state == {"other": 1}

    def test_removes_per_module_keys_keeps_others(self):
        state = {"modelopt_state_dict": [("mode", {"metadata": {
            "quantizer_state": "a", "subnet_config": "b",
            "real_quantizer_state": "c", "q_tensor_state": "d",
            "keep": True,
        }})]}
        remove_per_module_state(state)
        meta = state["modelopt_state_dict"][0][1]["metadata"]
        assert not any(k in meta for k in ("quantizer_state", "subnet_config", "real_quantizer_state", "q_tensor_state"))
        assert meta["keep"] is True

    def test_missing_metadata_filled_with_empty_dict(self):
        state = {"modelopt_state_dict": [("mode", {})]}
        remove_per_module_state(state)
        assert state["modelopt_state_dict"][0][1]["metadata"] == {}


class TestSaveModeloptState:
    def setup_method(self):
        _MTO.reset_mock()
        _MTO.ModeloptStateManager.is_converted.return_value = True
        _MTO.modelopt_state.return_value = {"modelopt_state_dict": []}

    def test_not_converted_is_noop(self):
        _MTO.ModeloptStateManager.is_converted.return_value = False
        state_dict = {}
        save_modelopt_state([mock.MagicMock()], state_dict)
        assert state_dict == {}


class TestRestoreShardedModeloptState:
    def setup_method(self):
        _MTO.reset_mock()
        _MTO.ModeloptStateManager.is_converted.return_value = False
        _MTO.restore_from_modelopt_state.side_effect = lambda m, _s: m

    def test_multiple_models_raises(self):
        with pytest.raises(ValueError, match="virtual pipeline"):
            restore_sharded_modelopt_state([mock.MagicMock(), mock.MagicMock()], "/ckpt")

    def test_missing_checkpoint_returns_early(self, tmp_path):
        restore_sharded_modelopt_state([mock.MagicMock()], str(tmp_path))
        _MTO.restore_from_modelopt_state.assert_not_called()


class TestLoadExtraStateFromShardedCheckpoint:
    def test_strips_prefix_and_filters_extra_state(self, tmp_path):
        prefix = "model."
        model = mock.MagicMock()
        model.sharded_state_dict.return_value = {
            f"{prefix}layer._extra_state": mock.MagicMock(),
            f"{prefix}layer.weight": mock.MagicMock(),
        }
        loaded = {f"{prefix}layer._extra_state": torch.tensor([1.0])}

        with mock.patch("megatron.core.dist_checkpointing.strategies.modelopt.load", return_value=loaded) as mock_load:
            _load_extra_state_from_sharded_checkpoint(model, str(tmp_path), prefix)

        passed = mock_load.call_args[0][0]
        assert f"{prefix}layer._extra_state" in passed
        assert f"{prefix}layer.weight" not in passed

        loaded_dict = model.load_state_dict.call_args[0][0]
        assert "layer._extra_state" in loaded_dict
        assert f"{prefix}layer._extra_state" not in loaded_dict
