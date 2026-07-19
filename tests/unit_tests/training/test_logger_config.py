# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.training.config.training_config import LoggerConfig


def test_per_param_norm_requires_global_param_norm_logging():
    with pytest.raises(ValueError, match="log_per_param_norm requires log_params_norm"):
        LoggerConfig(log_per_param_norm=True)


def test_per_param_norm_logging_accepts_complete_configuration():
    config = LoggerConfig(log_params_norm=True, log_per_param_norm=True)

    assert config.log_params_norm
    assert config.log_per_param_norm
