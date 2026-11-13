# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only coverage of the legacy GPT builder's Liger norm routing."""

from unittest.mock import Mock, patch

import pytest

import gpt_builders
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    return TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        normalization="RMSNorm",
        transformer_impl="local",
        **overrides,
    )


def _args(**overrides):
    args = Mock(
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


@pytest.mark.parametrize("use_liger", [False, True])
def test_dense_builder_uses_block_spec_for_liger(use_liger, run_config):
    run_config.tokenizer.padded_vocab_size = 128
    with (
        patch.object(gpt_builders, "GPTModel") as model,
        patch.object(gpt_builders, "_get_transformer_layer_spec") as layer_spec,
        patch.object(gpt_builders, "get_gpt_decoder_block_spec") as block_spec,
    ):
        gpt_builders.gpt_builder(_args(), True, True, config=_config(use_liger=use_liger))
    expected = block_spec if use_liger else layer_spec
    assert model.call_args.kwargs["transformer_layer_spec"] is expected.return_value


def test_liger_empty_decoder_stage_forwards_norms_to_mtp(run_config):
    run_config.tokenizer.padded_vocab_size = 128
    block = TransformerBlockSubmodules(layer_specs=[], layer_norm=object)
    with (
        patch.object(gpt_builders, "GPTModel"),
        patch.object(gpt_builders, "get_gpt_decoder_block_spec", return_value=block),
        patch.object(gpt_builders, "get_gpt_layer_local_spec") as local_spec,
        patch.object(gpt_builders, "get_gpt_mtp_block_spec") as mtp_spec,
    ):
        config = _config(use_liger=True, mtp_num_layers=1)
        gpt_builders.gpt_builder(_args(mtp_num_layers=1), True, True, config=config)

    assert local_spec.call_args.kwargs["use_liger"] is True
    assert mtp_spec.call_args.args[1] is local_spec.return_value
