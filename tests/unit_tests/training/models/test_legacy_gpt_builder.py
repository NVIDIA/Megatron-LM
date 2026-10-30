# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only regression coverage for the legacy GPT norm dispatch."""

from unittest.mock import Mock, patch

import pytest

import gpt_builders
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    options = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        normalization="RMSNorm",
        transformer_impl="local",
    )
    return TransformerConfig(**(options | overrides))


def _args(**overrides):
    args = Mock(
        spec=None,
        yaml_cfg=None,
        transformer_impl="local",
        experimental_attention_variant=None,
        num_experts=None,
        heterogeneous_layers_config_path=None,
        normalization="RMSNorm",
        qk_l2_norm=False,
        mtp_num_layers=None,
    )
    args.spec = None
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


@pytest.mark.parametrize("config_source", ["explicit", "args", "yaml"])
@pytest.mark.parametrize("use_liger", [False, True])
def test_dense_builder_norm_routing(use_liger, config_source):
    config = _config(use_liger=use_liger)
    args = _args(yaml_cfg="config.yaml" if config_source == "yaml" else None)
    block = TransformerBlockSubmodules(layer_specs=[], layer_norm=object)
    with (
        patch.object(gpt_builders, "GPTModel") as model,
        patch.object(gpt_builders, "_get_transformer_layer_spec") as layer_spec,
        patch.object(gpt_builders, "get_gpt_decoder_block_spec", return_value=block) as block_spec,
        patch.object(gpt_builders, "core_transformer_config_from_args", return_value=config),
        patch.object(gpt_builders, "core_transformer_config_from_yaml", return_value=config),
    ):
        result = gpt_builders.gpt_builder(
            args,
            pre_process=True,
            post_process=True,
            vp_stage=1,
            config=config if config_source == "explicit" else None,
        )

    assert result is model.return_value
    if use_liger:
        layer_spec.assert_not_called()
        block_spec.assert_called_once_with(
            config,
            use_transformer_engine=False,
            normalization="RMSNorm",
            qk_l2_norm=False,
            vp_stage=1,
        )
        assert model.call_args.kwargs["transformer_layer_spec"] is block
    else:
        block_spec.assert_not_called()
        layer_spec.assert_called_once_with(False, config)
        assert model.call_args.kwargs["transformer_layer_spec"] is layer_spec.return_value


@pytest.mark.parametrize("use_liger", [False, True])
def test_local_layer_helper_forwards_liger(use_liger):
    config = _config(use_liger=use_liger)
    with patch.object(gpt_builders, "get_gpt_layer_local_spec") as local_spec:
        result = gpt_builders._get_transformer_layer_spec(False, config)

    assert result is local_spec.return_value
    local_spec.assert_called_once_with(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        config.experimental_attention_variant,
        normalization=config.normalization,
        use_kitchen=config.use_kitchen,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
        use_liger=use_liger,
    )


@pytest.mark.parametrize("use_liger", [False, True])
@pytest.mark.parametrize("unsupported", ["experimental", "heterogeneous"])
def test_default_spec_support_is_unchanged_without_liger(use_liger, unsupported):
    args = _args()
    config = _config(use_liger=use_liger)
    if unsupported == "experimental":
        args.experimental_attention_variant = "dsa"
        spec_name = "get_transformer_block_with_experimental_attention_variant_spec"
    else:
        args.heterogeneous_layers_config_path = "layers.json"
        spec_name = "get_gpt_heterogeneous_layer_spec"
    with (
        patch.object(gpt_builders, "GPTModel") as model,
        patch.object(gpt_builders, spec_name) as spec,
    ):
        if use_liger:
            with pytest.raises(ValueError, match="standard local GPT specs"):
                gpt_builders.gpt_builder(args, True, True, config=config)
            spec.assert_not_called()
            model.assert_not_called()
        else:
            gpt_builders.gpt_builder(args, True, True, config=config)
            spec.assert_called_once()
            assert model.call_args.kwargs["transformer_layer_spec"] is spec.return_value


def test_liger_rejects_heterogeneous_config_without_args_path():
    config = Mock(spec=HeterogeneousTransformerConfig, use_liger=True)
    with pytest.raises(ValueError, match="heterogeneous specs"):
        gpt_builders.gpt_builder(_args(), True, True, config=config)


@pytest.mark.parametrize("unsupported", ["experimental", "heterogeneous"])
def test_custom_spec_owns_normalization(unsupported):
    args = _args(spec="custom.spec")
    if unsupported == "experimental":
        args.experimental_attention_variant = "dsa"
    else:
        args.heterogeneous_layers_config_path = "layers.json"
    with (
        patch.object(gpt_builders, "GPTModel") as model,
        patch.object(gpt_builders, "import_module") as import_spec,
        patch.object(gpt_builders, "get_gpt_decoder_block_spec") as block_spec,
    ):
        gpt_builders.gpt_builder(args, True, True, config=_config(use_liger=True))

    import_spec.assert_called_once_with("custom.spec")
    block_spec.assert_not_called()
    assert model.call_args.kwargs["transformer_layer_spec"] is import_spec.return_value


def test_liger_empty_decoder_stage_forwards_norms_to_mtp():
    config = _config(use_liger=True, mtp_num_layers=1)
    args = _args(mtp_num_layers=1)
    block = TransformerBlockSubmodules(layer_specs=[], layer_norm=object)
    with (
        patch.object(gpt_builders, "GPTModel"),
        patch.object(gpt_builders, "get_gpt_decoder_block_spec", return_value=block),
        patch.object(gpt_builders, "get_gpt_layer_local_spec") as local_spec,
        patch.object(gpt_builders, "get_gpt_mtp_block_spec") as mtp_spec,
    ):
        gpt_builders.gpt_builder(args, True, True, config=config)

    assert local_spec.call_args.kwargs["use_liger"] is True
    assert mtp_spec.call_args.args[1] is local_spec.return_value
