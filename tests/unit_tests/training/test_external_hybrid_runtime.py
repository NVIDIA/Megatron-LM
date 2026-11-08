# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime integration for config-defined layers implemented outside Megatron Core."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec, hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.utils import create_layer_config
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.training import checkpointing
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.training import (
    _hybrid_config_list_moe_logging_metadata,
    num_floating_point_operations,
)
from megatron.training.utils.checkpoint_utils import read_run_config
from tests.unit_tests.models.hybrid.external_layer import ExternalGainConfig, ExternalGainLayer


def _config(config_type=ExternalGainConfig, **kwargs):
    return config_type(num_layers=3, hidden_size=16, num_attention_heads=2, **kwargs)


def _model(config, layers, spec=None):
    return SimpleNamespace(
        config=config, hybrid_layer_config_list=tuple(layers), vocab_size=32, hybrid_stack_spec=spec
    )


def _flops(model):
    return num_floating_point_operations(
        HybridModelConfig(transformer=model.config, vocab_size=32),
        batch_size=2,
        sequence_length=128,
        total_real_tokens_in_batch=7,
        seqlen_squared_sum_in_batch=25,
        hybrid_model=model,
    )


@pytest.mark.parametrize("shared_weights", [False, True])
def test_external_flops_count_all_pipeline_and_mtp_layers(shared_weights):
    config = _config(mtp_num_layers=2, mtp_use_repeated_layer=shared_weights)
    mlp = MLPLayerConfig.from_config(config)
    configs = [config, PipelineSplit, mlp, config, MTPSplit, config, MTPSplit, config]
    tokens = 7
    vocab_size = 32
    result = _flops(_model(config, configs))

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
    model = _model(config, [config])
    stale_config = _config()
    stale_config.hidden_size = 32

    result = num_floating_point_operations(
        HybridModelConfig(transformer=stale_config, vocab_size=64),
        batch_size=2,
        sequence_length=128,
        total_real_tokens_in_batch=7,
        seqlen_squared_sum_in_batch=25,
        hybrid_model=model,
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
        _flops(_model(config, [config]))


def test_missing_external_flops_does_not_select_the_legacy_estimator(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        ExternalGainLayer, "estimate_flops", classmethod(MegatronModule.estimate_flops.__func__)
    )
    with pytest.raises(NotImplementedError):
        _flops(_model(config, [config]))


@pytest.mark.parametrize(
    "symbol,spec",
    [(symbol, hybrid_stack_spec) for symbol in ("*", "-", "M", "G", "E", "D", "C")]
    + [("M", gdp_stack_spec)],
)
def test_config_lists_preserve_existing_pattern_flops(symbol, spec):
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        ffn_hidden_size=32,
        multi_latent_attention=False,
        q_lora_rank=4,
        kv_lora_rank=6,
        qk_head_dim=4,
        qk_pos_emb_head_dim=2,
        v_head_dim=4,
        num_moe_experts=4 if symbol == "E" else None,
        moe_ffn_hidden_size=24,
        moe_router_topk=2,
        mamba_state_dim=4,
        mamba_head_dim=8,
        mamba_num_groups=1,
        mamba_num_heads=6,
        gdp_num_householder=2,
        linear_key_head_dim=4,
        linear_value_head_dim=6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=2,
        output_projection_groups=2,
        output_projection_lora_rank=4,
    )
    pattern_config = HybridModelConfig(
        transformer=config, vocab_size=32, hybrid_layer_pattern=symbol * 2, hybrid_stack_spec=spec
    )
    expected = num_floating_point_operations(
        pattern_config,
        batch_size=2,
        sequence_length=128,
        total_real_tokens_in_batch=7,
        seqlen_squared_sum_in_batch=25,
    )
    layers = [create_layer_config(config, symbol) for _ in range(2)]
    # Preserve the existing formulas and their final integer rounding, including
    # packed attention and the separate DSA/CSA backward factors.
    assert _flops(_model(config, layers, spec)) == pytest.approx(expected, rel=0, abs=1)


def test_mixed_ffn_widths_use_layer_configs_instead_of_stack_defaults():
    config = _config()
    narrow = MLPLayerConfig.from_config(config)
    wide = MLPLayerConfig.from_config(config)
    narrow.ffn_hidden_size, wide.ffn_hidden_size = 24, 48
    layers = [narrow, PipelineSplit, wide, config]
    expected_forward = 4 * 7 * config.hidden_size * (24 + 48)
    expected_forward += 7 * config.hidden_size  # external gain
    expected_forward += 2 * 7 * config.hidden_size * 32  # logits
    assert _flops(_model(config, layers)) == 3 * expected_forward


@pytest.mark.parametrize("repeated_mtp", [False, True])
def test_builtin_moe_metadata_with_external_layers_uses_standard_tracker_conventions(repeated_mtp):
    dense = _config()
    moe = MoELayerConfig.from_config(_config(num_moe_experts=4, moe_z_loss_coeff=0.01))
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


def test_external_layer_fields_are_recorded_in_run_config(run_config, tmp_path):
    root = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    custom = ExternalGainConfig.from_config(root)
    custom.gain_init = 0.5
    custom.notes.append("architecture-specific")
    run_config.model = HybridModelConfig(
        transformer=root, hybrid_layer_config_list=[custom, custom], vocab_size=64
    )
    path = tmp_path / "run_config.yaml"
    run_config.to_yaml(path)
    for saved in (run_config.to_dict(), read_run_config(str(path))):
        layers = saved["model"]["hybrid_layer_config_list"]
        assert layers[0]["_target_"] == (
            "tests.unit_tests.models.hybrid.external_layer.ExternalGainConfig"
        )
        assert layers[0]["gain_init"] == 0.5
        assert layers[0]["notes"] == ["external", "architecture-specific"]
        assert layers[0] == layers[1]
