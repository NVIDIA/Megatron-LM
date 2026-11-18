# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MULTIMODAL_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "multimodal"
sys.path.insert(0, str(MULTIMODAL_EXAMPLE_DIR))

import config as multimodal_config  # noqa: E402


@pytest.mark.parametrize("language_model_type", ["nemotron3-super", "nemotron6-super"])
def test_nemotron3_super_configs(language_model_type):
    language = multimodal_config.get_language_model_config(
        SimpleNamespace(language_model_type=language_model_type)
    )
    assert language.activation_func is multimodal_config.squared_relu
    assert language.bias_activation_fusion is False
    assert language.bias_dropout_fusion is False

    vision = multimodal_config.get_vision_model_config(
        SimpleNamespace(language_model_type=language_model_type, vision_model_type="radio")
    )
    assert vision.bias_dropout_fusion is False

    vision_projection = multimodal_config.get_vision_projection_config(
        SimpleNamespace(language_model_type=language_model_type), hidden_size=4096
    )
    assert vision_projection.hidden_size == 4096
    assert vision_projection.ffn_hidden_size == 20480
    assert vision_projection.activation_func is multimodal_config.squared_relu


# Keep these pairs explicit so a removed or incorrectly mapped alias fails the test.
MODEL_NAME_ALIASES = [
    ("nemotron5-8b", "nemotron2-8b", False),
    ("nemotron5-hybrid-8b", "nemotron2-hybrid-8b", True),
    ("nemotron5-hybrid-12b", "nemotron2-hybrid-12b", True),
    ("nemotron5-hybrid-56b", "nemotron2-hybrid-56b", True),
    ("nemotron6-moe", "nemotron3-moe", True),
    ("nemotron6-super", "nemotron3-super", True),
]


@pytest.mark.parametrize(("legacy", "preferred", "hybrid"), MODEL_NAME_ALIASES)
def test_model_name_aliases_select_same_model(legacy, preferred, hybrid):
    assert multimodal_config.canonical_language_model_type(legacy) == preferred
    assert multimodal_config.canonical_language_model_type(preferred) == preferred
    assert multimodal_config.is_hybrid_language_model_type(legacy) is hybrid
    assert multimodal_config.is_hybrid_language_model_type(preferred) is hybrid


@pytest.mark.parametrize(
    ("legacy", "preferred"), [(legacy, preferred) for legacy, preferred, _ in MODEL_NAME_ALIASES]
)
@pytest.mark.parametrize("enable_fusions", [False, True])
@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (multimodal_config.get_language_model_config, {}),
        (multimodal_config.get_vision_model_config, {}),
        (multimodal_config.get_vision_projection_config, {"hidden_size": 4096}),
    ],
)
def test_model_name_aliases_produce_identical_configs(
    legacy, preferred, enable_fusions, builder, kwargs
):
    configs = []
    for name in (legacy, preferred):
        config = builder(
            SimpleNamespace(language_model_type=name, vision_model_type="radio"),
            enable_fusions=enable_fusions,
            **kwargs,
        )
        # Normalization is for selection only; preserve the supplied/saved name.
        assert config.language_model_type == name
        values = vars(config).copy()
        del values["language_model_type"]
        configs.append(values)
    assert configs[0] == configs[1]


@pytest.mark.parametrize("enable_fusions", [False, True])
def test_moe_sound_projection_alias(enable_fusions):
    configs = []
    for name in ("nemotron6-moe", "nemotron3-moe"):
        config = multimodal_config.get_sound_projection_config(
            SimpleNamespace(language_model_type=name),
            hidden_size=4096,
            enable_fusions=enable_fusions,
        )
        assert config.language_model_type == name
        values = vars(config).copy()
        del values["language_model_type"]
        configs.append(values)
    assert configs[0] == configs[1]


@pytest.mark.parametrize("name", ["llama3_8b", "hf://org/nemotron6-super", "nemotron5-unknown"])
def test_unrelated_model_names_are_not_aliased(name):
    assert multimodal_config.canonical_language_model_type(name) == name
    assert not multimodal_config.is_hybrid_language_model_type(name)
