# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only validation of Liger configuration and norm provider selection."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_local_submodules,
)
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig


@pytest.mark.parametrize("config_cls", [TransformerConfig, MLATransformerConfig])
@pytest.mark.parametrize(
    "invalid_options",
    [
        {"normalization": "LayerNorm"},
        {"normalization": None},
        {"transformer_impl": "transformer_engine"},
        {"transformer_impl": "inference_optimized"},
        {"use_kitchen": True},
    ],
)
def test_liger_rejects_invalid_config_early(config_cls, invalid_options):
    options = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        normalization="RMSNorm",
        transformer_impl="local",
        use_liger=True,
    )
    with pytest.raises(ValueError, match="use_liger requires"):
        config_cls(**(options | invalid_options))


def test_liger_requires_explicit_local_rmsnorm_config():
    with pytest.raises(ValueError, match="use_liger requires"):
        TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, use_liger=True)


@pytest.mark.parametrize("use_liger", [False, True])
def test_valid_liger_config_does_not_load_optional_provider(use_liger):
    with patch("megatron.core.models.backends.get_backend") as backend:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            normalization="RMSNorm",
            transformer_impl="local",
            use_liger=use_liger,
        )
    assert config.use_liger is use_liger
    backend.assert_not_called()


@pytest.mark.parametrize("builder", [get_gpt_layer_local_spec, get_gpt_layer_local_submodules])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"normalization": None},
        {"normalization": "LayerNorm"},
        {"normalization": "RMSNorm", "use_kitchen": True},
    ],
)
def test_liger_rejects_invalid_low_level_spec_before_provider_load(builder, options):
    with patch("megatron.core.models.gpt.gpt_layer_specs.get_backend") as backend:
        with pytest.raises(ValueError, match="use_liger requires"):
            builder(use_liger=True, **options)
    backend.assert_not_called()


@pytest.mark.parametrize("normalization", [None, "LayerNorm", "RMSNorm"])
def test_disabled_liger_preserves_low_level_norm_selection(normalization):
    with patch("megatron.core.models.gpt.gpt_layer_specs.get_backend") as backend:
        submodules = get_gpt_layer_local_submodules(normalization=normalization, use_liger=False)
    backend.assert_called_once_with(
        "local",
        use_kitchen=False,
        use_kitchen_attention=False,
        kitchen_attention_backend="sdpa",
        use_liger=False,
    )
    backend.return_value.layer_norm.assert_any_call(
        rms_norm=normalization == "RMSNorm", for_qk=False, has_residual=True
    )
    assert submodules.input_layernorm is backend.return_value.layer_norm.return_value


@pytest.mark.parametrize(
    "transformer_impl,use_liger",
    [
        ("local", None),
        ("local", False),
        ("local", True),
        ("transformer_engine", False),
        ("inference_optimized", False),
    ],
)
def test_mla_resolver_only_forwards_norm_provider_option(transformer_impl, use_liger):
    config = SimpleNamespace(
        transformer_impl=transformer_impl,
        normalization="RMSNorm",
        q_lora_rank=16,
        experimental_attention_variant=None,
        use_kitchen=True,
        use_kitchen_attention=True,
        kitchen_attention_backend="flash",
        use_transformer_engine_op_fuser=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        cuda_graph_impl="local",
    )
    if use_liger is not None:
        config.use_liger = use_liger
    with patch("megatron.core.transformer.mla_qk_norm_config.get_backend") as backend:
        resolver = QKNormConfigResolver(config, SimpleNamespace())

    backend.assert_called_once_with(transformer_impl, use_liger=bool(use_liger))
    backend.return_value.layer_norm.assert_called_once_with(rms_norm=True, for_qk=True)
    assert resolver.qk_norm_impl is backend.return_value.layer_norm.return_value
    assert resolver.linear_impl is backend.return_value.column_parallel_linear.return_value
    assert (
        resolver.fused_norm_linear_impl
        is backend.return_value.column_parallel_layer_norm_linear.return_value
    )
