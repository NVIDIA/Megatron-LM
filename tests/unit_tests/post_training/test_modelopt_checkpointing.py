# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron/core/dist_checkpointing/strategies/modelopt.py.

Skipped automatically when modelopt is not installed.
"""

import sys
from unittest import mock

import pytest
import torch

pytest.importorskip("modelopt", reason="modelopt is not installed")

import modelopt.torch.opt as mto
import modelopt.torch.utils.distributed as mdist

from megatron.core.post_training.modelopt.checkpointing import (
    _load_extra_state_from_sharded_checkpoint,
    remove_per_module_state,
    restore_sharded_modelopt_state,
    save_modelopt_state,
    save_sharded_modelopt_state,
)


class TestModelOptImports:
    def test_modelopt_imports(self):
        modelopt_keys = [k for k in sys.modules if k == "modelopt" or k.startswith("modelopt.")]
        saved = {k: sys.modules.pop(k) for k in modelopt_keys}
        try:
            import megatron.core  # noqa: F401
            assert "modelopt" not in sys.modules
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
    def test_not_converted_is_noop(self):
        state_dict = {}
        with mock.patch.object(mto.ModeloptStateManager, "is_converted", return_value=False):
            save_modelopt_state([mock.MagicMock()], state_dict)
        assert state_dict == {}

    def test_single_model_saved(self):
        fake_state = {"modelopt_state_dict": []}
        state_dict = {}
        with (
            mock.patch.object(mto.ModeloptStateManager, "is_converted", return_value=True),
            mock.patch.object(mto, "modelopt_state", return_value=fake_state),
        ):
            save_modelopt_state([mock.MagicMock()], state_dict)
        assert state_dict["modelopt_state"] is fake_state

    def test_multiple_models_use_indexed_keys(self):
        state_dict = {}
        with (
            mock.patch.object(mto.ModeloptStateManager, "is_converted", return_value=True),
            mock.patch.object(mto, "modelopt_state", side_effect=[{"i": i} for i in range(2)]),
            mock.patch("megatron.core.post_training.modelopt.checkpointing.mpu"),
        ):
            save_modelopt_state([mock.MagicMock(), mock.MagicMock()], state_dict)
        assert "modelopt_state_0" in state_dict
        assert "modelopt_state_1" in state_dict


class TestSaveShardedModeloptState:
    def test_multiple_models_raises(self):
        with (
            mock.patch.object(mto.ModeloptStateManager, "is_converted", return_value=True),
            mock.patch.object(mdist, "is_master", return_value=True),
            pytest.raises(ValueError, match="virtual pipeline"),
        ):
            save_sharded_modelopt_state([mock.MagicMock(), mock.MagicMock()], "/ckpt")

    def test_single_model_calls_save(self, tmp_path):
        with (
            mock.patch.object(mto.ModeloptStateManager, "is_converted", return_value=True),
            mock.patch.object(mto, "modelopt_state", return_value={"modelopt_state_dict": []}),
            mock.patch.object(mdist, "is_master", return_value=True),
            mock.patch("megatron.core.post_training.modelopt.checkpointing.save") as mock_save,
        ):
            save_sharded_modelopt_state([mock.MagicMock()], str(tmp_path))
        mock_save.assert_called_once()
        assert mock_save.call_args.args[1] == f"{tmp_path}/modelopt_state"


class TestRestoreShardedModeloptState:
    def test_multiple_models_raises(self):
        with pytest.raises(ValueError, match="virtual pipeline"):
            restore_sharded_modelopt_state([mock.MagicMock(), mock.MagicMock()], "/ckpt")

    def test_missing_checkpoint_returns_early(self, tmp_path):
        with (
            mock.patch.object(mto.ModeloptStateManager, "is_converted", return_value=False),
            mock.patch.object(mto, "restore_from_modelopt_state") as mock_restore,
        ):
            restore_sharded_modelopt_state([mock.MagicMock()], str(tmp_path))
        mock_restore.assert_not_called()

    def test_restores_model(self, tmp_path):
        (tmp_path / "modelopt_state").mkdir()
        with (
            mock.patch.object(mto.ModeloptStateManager, "is_converted", return_value=False),
            mock.patch.object(mto, "restore_from_modelopt_state", side_effect=lambda m, _s: m) as mock_restore,
            mock.patch("megatron.core.post_training.modelopt.checkpointing.load_common_state_dict", return_value={"modelopt_version": "1.0"}),
            mock.patch("megatron.core.post_training.modelopt.checkpointing._load_extra_state_from_sharded_checkpoint"),
            mock.patch("megatron.core.post_training.modelopt.checkpointing.logger", create=True),
        ):
            restore_sharded_modelopt_state([mock.MagicMock()], str(tmp_path))
        mock_restore.assert_called_once()


class TestLoadExtraStateFromShardedCheckpoint:
    def test_strips_prefix_and_filters_extra_state(self, tmp_path):
        prefix = "model."
        model = mock.MagicMock()
        model.sharded_state_dict.return_value = {
            f"{prefix}layer._extra_state": mock.MagicMock(),
            f"{prefix}layer.weight": mock.MagicMock(),
        }
        loaded = {f"{prefix}layer._extra_state": torch.tensor([1.0])}

        with mock.patch("megatron.core.post_training.modelopt.checkpointing.load", return_value=loaded) as mock_load:
            _load_extra_state_from_sharded_checkpoint(model, str(tmp_path), prefix)

        passed = mock_load.call_args.args[0]
        assert f"{prefix}layer._extra_state" in passed
        assert f"{prefix}layer.weight" not in passed

        loaded_dict = model.load_state_dict.call_args.args[0]
        assert "layer._extra_state" in loaded_dict
        assert f"{prefix}layer._extra_state" not in loaded_dict
