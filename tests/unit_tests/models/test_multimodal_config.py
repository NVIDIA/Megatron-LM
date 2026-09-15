# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from pathlib import Path
from types import SimpleNamespace

MULTIMODAL_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "multimodal"
sys.path.insert(0, str(MULTIMODAL_EXAMPLE_DIR))

import config as multimodal_config  # noqa: E402


def test_nemotron6_super_configs():
    language = multimodal_config.get_language_model_config(
        SimpleNamespace(language_model_type="nemotron6-super")
    )
    assert language.activation_func is multimodal_config.squared_relu
    assert language.bias_activation_fusion is False
    assert language.bias_dropout_fusion is False

    vision = multimodal_config.get_vision_model_config(
        SimpleNamespace(language_model_type="nemotron6-super", vision_model_type="radio")
    )
    assert vision.bias_dropout_fusion is False

    vision_projection = multimodal_config.get_vision_projection_config(
        SimpleNamespace(language_model_type="nemotron6-super"), hidden_size=4096
    )
    assert vision_projection.hidden_size == 4096
    assert vision_projection.ffn_hidden_size == 20480
    assert vision_projection.activation_func is multimodal_config.squared_relu
