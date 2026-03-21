# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for megatron/training/tensor_inspect.py — NVIDIA DLFw Inspect integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

try:
    import nvdlfw_inspect.api as nvinspect_api
    from nvdlfw_inspect.logging import MetricLogger

    HAVE_NVINSPECT = True
except (ImportError, ModuleNotFoundError):
    HAVE_NVINSPECT = False

from megatron.training.tensor_inspect import (
    _clean_metric_name,
    _get_default_feature_dirs,
    _maybe_attach_metric_loggers,
    finalize_tensor_inspect_post_model,
    initialize_tensor_inspect_pre_model,
    tensor_inspect_end,
    tensor_inspect_step,
)

pytestmark = pytest.mark.skipif(not HAVE_NVINSPECT, reason="nvdlfw_inspect not available")


@pytest.fixture
def inspect_session(tmp_path):
    """Initialize nvdlfw_inspect and clean up after the test."""

    def _start(config_str="", feature_dirs=None):
        cfg_file = tmp_path / "config.yaml"
        if config_str:
            cfg_file.write_text(config_str)

        log_dir = str(tmp_path / "logs")
        Path(log_dir).mkdir(exist_ok=True)

        initialize_tensor_inspect_pre_model(
            enabled=True,
            features=str(cfg_file) if config_str else None,
            feature_dirs=feature_dirs if feature_dirs is not None else [],
            log_dir=log_dir,
        )
        return log_dir

    yield _start

    if nvinspect_api.DEBUG_MANAGER is not None:
        nvinspect_api.end_debug()
    MetricLogger.enabled_loggers.clear()


@pytest.mark.internal
class TestCleanMetricName:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("model.module.module.layer1.weight", "layer1.weight"),
            ("model.layer1.weight", "layer1.weight"),
        ],
    )
    def test_strips_wrapper_prefixes(self, name, expected):
        assert _clean_metric_name(name) == expected


@pytest.mark.internal
class TestMetricLoggers:
    def setup_method(self, method):
        MetricLogger.enabled_loggers.clear()

    def teardown_method(self, method):
        MetricLogger.enabled_loggers.clear()

    @pytest.mark.parametrize(
        "tb,wandb,expected_count", [(None, None, 0), (MagicMock(), MagicMock(log=MagicMock()), 2)]
    )
    def test_logger_attachment(self, tb, wandb, expected_count):
        _maybe_attach_metric_loggers(tb, wandb)
        assert len(MetricLogger.enabled_loggers) == expected_count

    def test_wandb_without_log_attr_skipped(self):
        _maybe_attach_metric_loggers(None, object())
        assert len(MetricLogger.enabled_loggers) == 0

    def test_wandb_cleans_metric_names(self):
        wandb_mock = MagicMock(log=MagicMock())
        _maybe_attach_metric_loggers(None, wandb_mock)
        MetricLogger.enabled_loggers[-1].log_scalar("model.module.module.decoder.weight", 1.0, 100)
        wandb_mock.log.assert_called_once_with({"decoder.weight": 1.0}, step=100)


class TestTensorInspectLifecycle:
    def setup_method(self, method):
        if nvinspect_api.DEBUG_MANAGER is not None:
            nvinspect_api.end_debug()
        MetricLogger.enabled_loggers.clear()

    def teardown_method(self, method):
        if nvinspect_api.DEBUG_MANAGER is not None:
            nvinspect_api.end_debug()
        MetricLogger.enabled_loggers.clear()

    def test_all_disabled_noop(self):
        """All public functions with enabled=False are no-ops."""
        initialize_tensor_inspect_pre_model(enabled=False)
        finalize_tensor_inspect_post_model(enabled=False, model=[])
        tensor_inspect_step(enabled=False)
        tensor_inspect_end(enabled=False)
        assert nvinspect_api.DEBUG_MANAGER is None

    @patch('megatron.training.tensor_inspect.HAVE_NVINSPECT', False)
    def test_missing_dep_raises_except_end(self):
        """init/finalize/step raise ImportError; end returns silently."""
        with pytest.raises(ImportError):
            initialize_tensor_inspect_pre_model(enabled=True)
        with pytest.raises(ImportError):
            finalize_tensor_inspect_post_model(enabled=True, model=[])
        with pytest.raises(ImportError):
            tensor_inspect_step(enabled=True)
        tensor_inspect_end(enabled=True)  # intentionally silent

    def test_creates_debug_manager(self, tmp_path):
        with patch(
            'megatron.training.tensor_inspect._get_default_feature_dirs', return_value=[]
        ) as mock_get_dirs:
            initialize_tensor_inspect_pre_model(
                enabled=True, feature_dirs=None, log_dir=str(tmp_path)
            )
        assert nvinspect_api.DEBUG_MANAGER is not None
        mock_get_dirs.assert_called_once()

    @pytest.mark.parametrize("step,expect_call", [(0, True), (None, False)])
    def test_training_step_initialization(self, step, expect_call, inspect_session):
        inspect_session()
        with (
            patch.object(nvinspect_api, 'initialize_training_step') as mock_init_step,
            patch(
                'megatron.core.parallel_state.get_tensor_and_data_parallel_group', return_value=None
            ),
        ):
            finalize_tensor_inspect_post_model(
                enabled=True, model=[nn.Linear(10, 5)], current_training_step=step
            )
        if expect_call:
            mock_init_step.assert_called_once_with(step)
        else:
            mock_init_step.assert_not_called()

    def test_full_lifecycle_with_te_config(self, tmp_path):
        """Full init→finalize→step→end with a real TE LogTensorStats config."""
        config = (
            "te_weight_stats:\n"
            "  enabled: True\n"
            "  layers:\n"
            "    layer_name_regex_pattern: \".*\"\n"
            "  transformer_engine:\n"
            "    LogTensorStats:\n"
            "      enabled: True\n"
            "      tensors: [weight]\n"
            "      stats: [mean, max]\n"
            "      freq: 1\n"
            "      start_step: 0\n"
            "      end_step: 10\n"
        )
        cfg_file = str(tmp_path / "config.yaml")
        Path(cfg_file).write_text(config)
        log_dir = str(tmp_path / "logs")
        Path(log_dir).mkdir(exist_ok=True)

        initialize_tensor_inspect_pre_model(
            enabled=True, features=cfg_file, feature_dirs=None, log_dir=log_dir
        )

        # Verify TE features were loaded
        from nvdlfw_inspect.registry import Registry

        assert "transformer_engine" in Registry.data
        te_features = list(Registry.data["transformer_engine"].features.keys())
        assert "LogTensorStats" in te_features
        assert "LogFp8TensorStats" in te_features
        assert "LogNvfp4TensorStats" in te_features

        model = nn.Linear(10, 5)
        with patch(
            'megatron.core.parallel_state.get_tensor_and_data_parallel_group', return_value=None
        ):
            finalize_tensor_inspect_post_model(enabled=True, model=[model], current_training_step=0)

        # Verify layer names were assigned
        assert hasattr(model, 'name')

        # Verify config was loaded — routing returns True for configured tensor
        result = nvinspect_api.transformer_engine.inspect_tensor_enabled(
            layer_name=model.name, tensor_name="weight", iteration=0
        )
        assert result[0] is True

        tensor_inspect_step(enabled=True)
        tensor_inspect_end(enabled=True)
        assert nvinspect_api.DEBUG_MANAGER is None
