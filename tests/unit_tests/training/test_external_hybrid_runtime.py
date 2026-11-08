# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime integration for config-defined layers implemented outside Megatron Core."""

from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from typing import ClassVar
from unittest.mock import Mock

import pytest

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.flops import estimate_hybrid_config_list_flops
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.training import checkpointing
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.training import (
    _hybrid_config_list_moe_logging_metadata,
    num_floating_point_operations,
)
from tests.unit_tests.models.hybrid.external_layer import ExternalGainConfig, ExternalGainLayer


class ExternalMoEMetadataLayer(ExternalGainLayer):
    """Exercise the standard MoE logging contract without constructing a router."""

    is_moe_layer: ClassVar[bool] = True


@dataclass
class ExternalMoEMetadataConfig(ExternalGainConfig):
    """Select the test layer that participates in MoE logging."""

    layer_type: ClassVar[type[MegatronModule]] = ExternalMoEMetadataLayer


def _config(config_type=ExternalGainConfig, **kwargs):
    return config_type(num_layers=3, hidden_size=16, num_attention_heads=2, **kwargs)


def test_external_flops_count_all_pipeline_and_mtp_layers():
    config = _config()
    mlp = MLPLayerConfig.from_config(config)
    configs = [config, PipelineSplit, mlp, config, MTPSplit, config, MTPSplit, config]
    tokens = 7
    vocab_size = 32
    result = estimate_hybrid_config_list_flops(
        configs, None, config, vocab_size, tokens, seqlen_squared_sum_in_batch=25
    )

    # Four gain executions, one dense MLP, two MTP projections, and three logits heads.
    gain_flops = 4 * tokens * config.hidden_size
    mlp_flops = 4 * tokens * config.hidden_size * mlp.ffn_hidden_size
    mtp_flops = 2 * tokens * 2 * (3 * config.hidden_size + 2 * config.hidden_size**2)
    logits_flops = 2 * tokens * config.hidden_size * vocab_size * 3
    assert result == 3 * (gain_flops + mlp_flops + mtp_flops + logits_flops)


@pytest.mark.parametrize("layer_flops", [0, 123, 123.5])
def test_training_forwards_packed_sequence_counts_to_external_estimator(monkeypatch, layer_flops):
    config = _config()
    estimate_layer = Mock(return_value=layer_flops)
    monkeypatch.setattr(ExternalGainLayer, "estimate_flops", classmethod(estimate_layer))
    estimate_model = partial(
        estimate_hybrid_config_list_flops, [config], None, config, vocab_size=32
    )

    result = num_floating_point_operations(
        HybridModelConfig(transformer=config, vocab_size=32),
        batch_size=2,
        sequence_length=128,
        total_real_tokens_in_batch=7,
        seqlen_squared_sum_in_batch=25,
        model_flops_estimator=estimate_model,
    )

    estimate_layer.assert_called_once_with(
        ExternalGainLayer, config, total_real_tokens_in_batch=7, seqlen_squared_sum_in_batch=25
    )
    assert result == 3 * (layer_flops + 2 * 7 * config.hidden_size * 32)


@pytest.mark.parametrize("invalid_flops", [None, float("nan"), float("inf"), -1, True, "123"])
def test_invalid_external_flops_are_rejected_before_updating_counters(monkeypatch, invalid_flops):
    config = _config()
    monkeypatch.setattr(
        ExternalGainLayer, "estimate_flops", classmethod(Mock(return_value=invalid_flops))
    )
    with pytest.raises(ValueError, match="finite, nonnegative real number"):
        estimate_hybrid_config_list_flops(
            [config], None, config, 32, total_real_tokens_in_batch=7, seqlen_squared_sum_in_batch=25
        )


def test_missing_external_flops_does_not_select_the_legacy_estimator(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        ExternalGainLayer, "estimate_flops", classmethod(MegatronModule.estimate_flops.__func__)
    )
    estimate_model = partial(
        estimate_hybrid_config_list_flops, [config], None, config, vocab_size=32
    )
    with pytest.raises(NotImplementedError):
        num_floating_point_operations(
            HybridModelConfig(transformer=config, vocab_size=32),
            batch_size=2,
            sequence_length=128,
            model_flops_estimator=estimate_model,
        )


@pytest.mark.parametrize("repeated_mtp", [False, True])
def test_external_moe_metadata_uses_standard_tracker_conventions(repeated_mtp):
    dense = _config()
    moe = _config(ExternalMoEMetadataConfig, num_moe_experts=4, moe_z_loss_coeff=0.01)
    configs = [dense, moe, PipelineSplit, moe, MTPSplit, moe, MTPSplit, moe]

    metadata = _hybrid_config_list_moe_logging_metadata(
        configs, mtp_use_repeated_layer=repeated_mtp, moe_num_hash_layers=0
    )

    assert metadata == (["load_balancing_loss", "z_loss"], 5, 3 if repeated_mtp else 4, True, 0)


def test_builtin_hash_metadata_with_external_non_moe_layer():
    dense = _config()
    moe = MoELayerConfig.from_config(_config(num_moe_experts=4))
    assert _hybrid_config_list_moe_logging_metadata([dense, moe, moe], False, 9) == (
        ["load_balancing_loss"],
        3,
        2,
        True,
        2,
    )


def test_external_non_moe_config_does_not_enable_moe_metrics():
    config = _config()
    assert _hybrid_config_list_moe_logging_metadata([config], False, 0) == ([], 1, 0, False, 0)


def test_saved_hybrid_family_preserves_checkpoint_policy_and_runtime_args(run_config):
    config = _config()
    model = Mock(spec=HybridModel)
    model.hybrid_layer_config_list = [config]
    model.state_dict_for_save_checkpoint.return_value = {"weight": 1}
    args = Namespace(
        ckpt_format="torch_dist",
        save="/old",
        no_save_optim=False,
        padded_vocab_size=16,
        is_hybrid_model=False,
    )
    original_args = vars(args).copy()
    run_config.checkpoint.save = "/configured"
    run_config.checkpoint.save_optim = False
    run_config.tokenizer.padded_vocab_size = 32
    run_config.model = HybridModelConfig(
        transformer=config, vocab_size=32, hybrid_layer_config_list=[config]
    )

    state = checkpointing.generate_state_dict(
        args, [model], None, None, None, iteration=3, ckpt_format="torch", save="/actual"
    )

    saved_args = state["args"]
    assert saved_args.is_hybrid_model is True
    assert saved_args.ckpt_format == "torch"
    assert saved_args.save == "/actual"
    assert saved_args.no_save_optim is True
    assert saved_args.padded_vocab_size == 32
    assert vars(args) == original_args
    assert state["model"] == {"weight": 1}
    assert checkpointing._validate_checkpoint_model_family(
        args, Namespace(is_hybrid_model=False), [model], checkpoint_model=run_config.model.as_dict()
    ) == (True, True, None)


def test_external_layer_keeps_config_list_gpt_conversion_restriction():
    model = Mock(spec=HybridModel)
    model.hybrid_layer_config_list = [_config()]
    with pytest.raises(RuntimeError, match="config-list-defined hybrid model"):
        checkpointing._validate_checkpoint_model_family(
            Namespace(), Namespace(hybrid_layer_pattern=None), [model]
        )
