# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training metrics use the complete constructed config-list architecture."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    CSALayerConfig,
)
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.training import (
    _find_hybrid_model_for_runtime_metrics,
    _hybrid_config_list_indexer_logging_metadata,
    _hybrid_config_list_moe_logging_metadata,
    num_floating_point_operations,
)


def _config(config_type=TransformerConfig, **overrides):
    config = config_type.from_config(
        TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _model(configs):
    # Exercise the real model-level estimator without allocating GPU parameters.
    model = object.__new__(HybridModel)
    torch.nn.Module.__init__(model)
    model.config = _config()
    model.hybrid_layer_config_list = configs
    model.hybrid_stack_spec = None
    model.vocab_size = 32
    return model


@pytest.mark.parametrize(
    "stats, tokens, squared_lengths",
    [({}, 24, 192), ({"total_real_tokens_in_batch": 7, "seqlen_squared_sum_in_batch": 25}, 7, 25)],
)
def test_constructed_model_flops_receive_resolved_bshd_and_thd_counts(
    stats, tokens, squared_lengths
):
    model = _model([_config(AttentionLayerConfig), _config(AttentionLayerConfig)])
    wrapped = SimpleNamespace(module=SimpleNamespace(language_model=model))
    actual_model = _find_hybrid_model_for_runtime_metrics(wrapped)
    estimator = Mock(wraps=actual_model.estimate_training_flops)

    actual = num_floating_point_operations(
        # The constructed model supplies dimensions and vocabulary; stale training
        # metadata must not be consulted when its estimator returns a value.
        SimpleNamespace(),
        batch_size=3,
        sequence_length=8,
        model_flops_estimator=estimator,
        **stats,
    )

    estimator.assert_called_once_with(
        total_real_tokens_in_batch=tokens, seqlen_squared_sum_in_batch=squared_lengths
    )
    # Two QKV/output projections (4096T), two causal attentions (64Q), logits (1024T).
    assert actual == 3 * (5120 * tokens + 64 * squared_lengths)


def test_pattern_model_retains_legacy_estimator():
    model = _model(None)
    config = HybridModelConfig(
        transformer=model.config, hybrid_layer_pattern="**", vocab_size=32, seq_length=64
    )
    expected = num_floating_point_operations(config, batch_size=3, sequence_length=8)
    estimator = Mock(wraps=model.estimate_training_flops)

    assert (
        num_floating_point_operations(
            config, batch_size=3, sequence_length=8, model_flops_estimator=estimator
        )
        == expected
    )
    estimator.assert_called_once_with(
        total_real_tokens_in_batch=24, seqlen_squared_sum_in_batch=192
    )


def test_zero_estimate_is_not_replaced_with_legacy_flops():
    assert (
        num_floating_point_operations(
            SimpleNamespace(), batch_size=1, sequence_length=8, model_flops_estimator=lambda **_: 0
        )
        == 0
    )


def test_model_discovery_ignores_unrelated_modules_and_handles_wrapper_cycles():
    unrelated = torch.nn.Module()
    unrelated.estimate_training_flops = Mock(side_effect=AssertionError("unrelated estimator"))
    wrapper = SimpleNamespace(language_model=unrelated)
    wrapper.module = wrapper
    assert _find_hybrid_model_for_runtime_metrics(wrapper) is None
    unrelated.estimate_training_flops.assert_not_called()


@pytest.mark.parametrize("shared", [False, True])
def test_moe_metadata_counts_hash_layers_and_shared_mtp_modules(shared):
    dense = _config(AttentionLayerConfig)
    main_moe = _config(MoELayerConfig, moe_router_load_balancing_type="aux_loss")
    other_moe = _config(MoELayerConfig, moe_router_load_balancing_type="seq_aux_loss")
    mtp_moe = _config(
        MoELayerConfig, moe_router_load_balancing_type="global_aux_loss", moe_z_loss_coeff=0.01
    )
    configs = [dense, main_moe, PipelineSplit, other_moe, MTPSplit, mtp_moe, MTPSplit, mtp_moe]

    assert _hybrid_config_list_moe_logging_metadata(configs, shared, 99) == (
        ["load_balancing_loss", "seq_load_balancing_loss", "global_load_balancing_loss", "z_loss"],
        5,
        3 if shared else 4,
        True,
        2,
    )


def test_dense_list_disables_moe_reporting_despite_root_expert_settings():
    dense = _config(AttentionLayerConfig, num_moe_experts=8)
    assert _hybrid_config_list_moe_logging_metadata([dense, PipelineSplit, dense], False, 2) == (
        [],
        2,
        0,
        False,
        0,
    )


@pytest.mark.parametrize("shared", [False, True])
def test_csa_metadata_uses_template_slots_and_physical_indexer_count(shared):
    active = _config(CSALayerConfig, compress_ratio=4, dsa_indexer_loss_coeff=0.1)
    window = _config(CSALayerConfig, compress_ratio=0, dsa_indexer_loss_coeff=0.1)
    heavy = _config(CSALayerConfig, compress_ratio=128, dsa_indexer_loss_coeff=0.1)
    dense = _config(
        CSALayerConfig, compress_ratio=4, csa_dense_mode=True, dsa_indexer_loss_coeff=0.1
    )
    no_loss = _config(CSALayerConfig, compress_ratio=4, dsa_indexer_loss_coeff=0.0)
    mlp = _config(MLPLayerConfig)
    # The three-entry MTP template has two indexers, reused at two depths.
    template = [active, mlp, active]
    configs = [
        active,
        window,
        PipelineSplit,
        heavy,
        dense,
        no_loss,
        MTPSplit,
        *template,
        MTPSplit,
        *template,
    ]

    # The root config need not set CSA ratios or an indexer loss coefficient.
    assert active.csa_compress_ratios is None
    assert _hybrid_config_list_indexer_logging_metadata(configs, shared) == (8, 3 if shared else 5)


@pytest.mark.parametrize("coeff", [None, 0.0])
def test_list_indexer_metadata_disables_unrecorded_losses(coeff):
    config = _config(CSALayerConfig, compress_ratio=4, dsa_indexer_loss_coeff=coeff)
    assert _hybrid_config_list_indexer_logging_metadata([config], False) == (1, 0)


@pytest.mark.parametrize("shared", [False, True])
def test_mixed_dsa_csa_indexer_counts_respect_local_mtp_numbering(shared):
    dsa = _config(DSALayerConfig, dsa_indexer_loss_coeff=0.1, dsa_indexer_topk_freq=2)
    csa = _config(CSALayerConfig, compress_ratio=4, dsa_indexer_loss_coeff=0.1)
    # DSA layer 1 computes top-k; layer 2 reuses it. The MTP template restarts
    # numbering at 1, independently of its enclosing prediction depth.
    template = [dsa, dsa, csa]
    configs = [dsa, dsa, PipelineSplit, csa, MTPSplit, *template, MTPSplit, *template]
    assert _hybrid_config_list_indexer_logging_metadata(configs, shared) == (6, 4 if shared else 6)


@pytest.mark.parametrize("shared", [False, True])
def test_csa_tracker_reserves_all_depth_slots_before_any_indexer_runs(shared):
    active = _config(CSALayerConfig, compress_ratio=4, dsa_indexer_loss_coeff=0.1)
    mlp = _config(MLPLayerConfig)
    configs = [active, mlp, MTPSplit, active, MTPSplit, active, MTPSplit, active]

    # The one-entry template records in slot 3, but save_loss_to_tracker allocates
    # five slots (two decoder layers + three depths). Ranks with no initial loss
    # must negotiate that size before later batches start contributing losses.
    assert _hybrid_config_list_indexer_logging_metadata(configs, shared) == (5, 2 if shared else 4)
