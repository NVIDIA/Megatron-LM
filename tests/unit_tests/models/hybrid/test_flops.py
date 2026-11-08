# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Global-batch FLOP accounting without constructing model modules or process groups."""

from types import SimpleNamespace

import pytest

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.dsa_flops import dsa_indexer_flops, dsa_layer_flops
from megatron.core.models.hybrid.layers.flops import estimate_hybrid_config_list_flops
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    CSALayerConfig,
)
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(config_type=TransformerConfig, **kwargs):
    return config_type(num_layers=1, hidden_size=16, num_attention_heads=2, **kwargs)


def _estimate(layers, tokens=7, squared_lengths=25, config=None, stack_spec=None):
    if config is None:
        config = _config()
        first_mtp = next((i for i, entry in enumerate(layers) if entry is MTPSplit), len(layers))
        config.num_layers = sum(entry is not PipelineSplit for entry in layers[:first_mtp])
        config.mtp_num_layers = sum(entry is MTPSplit for entry in layers) or None
    return estimate_hybrid_config_list_flops(
        layers, stack_spec, config, 32, tokens, squared_lengths
    )


def test_heterogeneous_ffn_and_moe_widths_use_each_layer_config():
    mlp = _config(MLPLayerConfig, ffn_hidden_size=32)
    moe = _config(
        MoELayerConfig,
        ffn_hidden_size=64,
        moe_ffn_hidden_size=48,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=24,
        gated_linear_unit=True,
    )
    # Dense two-matrix MLP; two selected three-matrix experts plus one shared expert.
    forward_per_token = 2 * (16 * 32 + 32 * 16)
    forward_per_token += 2 * 3 * 16 * (2 * 48 + 24)
    forward_per_token += 2 * 16 * 32  # logits
    assert _estimate([mlp, moe]) == 3 * 7 * forward_per_token


def test_packed_attention_uses_real_token_and_squared_length_totals():
    attention = _config(AttentionLayerConfig, num_query_groups=1)
    # Q/K/V/output projections: 16x16, 16x8, 16x8, 16x16.
    projection_per_token = 2 * (16 * 16 + 16 * 8 + 16 * 8 + 16 * 16)
    logits_per_token = 2 * 16 * 32
    packed = _estimate([attention], tokens=6, squared_lengths=2**2 + 4**2)
    assert packed == 3 * (6 * (projection_per_token + logits_per_token) + 2 * 20 * 16)
    unpacked = _estimate([attention], tokens=6, squared_lengths=6**2)
    assert unpacked - packed == 3 * 2 * (36 - 20) * 16


@pytest.mark.parametrize("pipeline_stages,virtual_stages", [(1, None), (2, None), (2, 2)])
def test_pipeline_markers_and_placement_do_not_change_global_count(pipeline_stages, virtual_stages):
    layers = [_config(AttentionLayerConfig), _config(MLPLayerConfig, ffn_hidden_size=40)] * 2
    root = _config()
    root.num_layers = len(layers)
    root.pipeline_model_parallel_size = pipeline_stages
    root.virtual_pipeline_model_parallel_size = virtual_stages
    segment_size = len(layers) // (pipeline_stages * (virtual_stages or 1))
    marked = []
    for index, layer in enumerate(layers):
        if index and index % segment_size == 0:
            marked.append(PipelineSplit)
        marked.append(layer)
    model = SimpleNamespace(
        hybrid_layer_config_list=marked, hybrid_stack_spec=None, config=root, vocab_size=32
    )
    assert HybridModel.estimate_training_flops(model, 7, 25) == _estimate(layers)


@pytest.mark.parametrize("shared_weights", [False, True])
def test_each_mtp_depth_counts_execution_and_logits_even_with_shared_weights(shared_weights):
    layer = _config(MLPLayerConfig, ffn_hidden_size=32)
    root = _config(mtp_num_layers=2, mtp_use_repeated_layer=shared_weights)
    configs = [layer, MTPSplit, layer, MTPSplit, layer]
    forward_per_token = 3 * 2 * (16 * 32 + 32 * 16)
    forward_per_token += 2 * 2 * (3 * 16 + 2 * 16**2)  # two MTP norms/projections
    forward_per_token += 3 * 2 * 16 * 32  # decoder and both MTP logits
    assert _estimate(configs, config=root) == 3 * 7 * forward_per_token


def test_pattern_model_keeps_legacy_estimator_selection():
    assert (
        HybridModel.estimate_training_flops(SimpleNamespace(hybrid_layer_config_list=None), 7, 25)
        is None
    )


def _dsa_config(**kwargs):
    return _config(
        DSALayerConfig,
        q_lora_rank=4,
        kv_lora_rank=6,
        qk_head_dim=4,
        qk_pos_emb_head_dim=2,
        v_head_dim=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=4,
        **kwargs,
    )


def _dsa_spec():
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
    )

    return ModuleSpec(
        module=object,
        submodules=SimpleNamespace(
            dsa_layer=ModuleSpec(
                module=object,
                submodules=SimpleNamespace(
                    self_attention=ModuleSpec(module=AbsorbedMLASelfAttention)
                ),
            )
        ),
    )


@pytest.mark.parametrize(
    "loss_coeff,sparse_loss,kpool",
    [(0.0, False, 1), (0.1, False, 1), (0.1, True, 1), (0.0, False, 2)],
)
def test_dsa_matches_established_loss_and_kpool_accounting(loss_coeff, sparse_loss, kpool):
    config = _dsa_config(
        dsa_indexer_loss_coeff=loss_coeff,
        dsa_indexer_use_sparse_loss=sparse_loss,
        dsa_indexer_kpool=kpool,
    )
    attention = dsa_layer_flops(
        16,
        256,
        16,
        2,
        4,
        6,
        4,
        2,
        4,
        4,
        dsa_indexer_kpool=kpool,
        dsa_indexer_kpool_always_select_tail=config.dsa_indexer_kpool_always_select_tail,
    )
    indexer = dsa_indexer_flops(
        16,
        256,
        16,
        4,
        2,
        4,
        4,
        1,
        dsa_indexer_loss_coeff=loss_coeff,
        dsa_indexer_use_sparse_loss=sparse_loss,
        dsa_indexer_kpool=kpool,
    )
    expected = 3 * (attention + 2 * 16 * 16 * 32) + indexer
    assert _estimate([config], 16, 256, stack_spec=_dsa_spec()) == pytest.approx(expected)


def test_dsa_reuse_counts_global_layer_positions_and_restarts_for_mtp():
    config = _dsa_config(dsa_indexer_topk_freq=2, dsa_indexer_loss_coeff=0.0)
    mlp = _config(MLPLayerConfig)
    layers = [config, config, mlp, config, MTPSplit, config, config]
    # Indexers execute at decoder position 1 and MTP position 1; positions 2/4 reuse.
    attention = dsa_layer_flops(16, 256, 16, 2, 4, 6, 4, 2, 4, 4)
    indexer = dsa_indexer_flops(16, 256, 16, 4, 2, 4, 4, 2)
    forward = 5 * attention + 4 * 16 * 16 * mlp.ffn_hidden_size
    forward += 2 * 16 * (3 * 16 + 2 * 16**2) + 2 * 2 * 16 * 16 * 32
    actual = _estimate(layers, 16, 256, stack_spec=_dsa_spec())
    assert actual == pytest.approx(3 * forward + indexer)
    marked = [config, PipelineSplit, config, mlp, PipelineSplit, config, MTPSplit, config, config]
    assert _estimate(marked, 16, 256, stack_spec=_dsa_spec()) == actual


def _csa_config(ratio, **kwargs):
    return _config(
        CSALayerConfig,
        compress_ratio=ratio,
        q_lora_rank=8,
        v_head_dim=4,
        qk_head_dim=2,
        qk_pos_emb_head_dim=2,
        output_projection_groups=2,
        output_projection_lora_rank=4,
        csa_window_size=2,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=2,
        **kwargs,
    )


@pytest.mark.parametrize(
    "ratio,dense,compressor,core",
    [(0, False, 0, 0), (4, True, 256, 65536), (128, False, 128, 2048)],
)
def test_csa_window_dense_and_high_compression_geometry(ratio, dense, compressor, core):
    config = _csa_config(ratio, csa_dense_mode=dense)
    # Q + KV + grouped output = 200 + 68 + 160; window term = 32 per token.
    forward = 2 * 128 * (428 + 32 + compressor) + core
    assert _estimate([config], 128, 128**2) == 3 * (forward + 2 * 128 * 16 * 32)


@pytest.mark.parametrize("sparse_loss,extra_backward", [(False, 12288), (True, 12032)])
def test_csa_indexer_backward_only_counts_trainable_paths(sparse_loss, extra_backward):
    without_loss = _csa_config(4, dsa_indexer_loss_coeff=0.0)
    with_loss = _csa_config(4, dsa_indexer_loss_coeff=0.1, dsa_indexer_use_sparse_loss=sparse_loss)
    baseline = _estimate([without_loss], 16, 256)
    # Indexer projection fwd=11264; causal score fwd=512, top-k pair fraction=3/4.
    # Detached projection inputs add only wgrad; scores add dq+dk (possibly sparse).
    assert baseline == 131968
    assert _estimate([with_loss], 16, 256) - baseline == extra_backward


@pytest.mark.parametrize("two_stage,forward_per_token", [(False, 2048), (True, 1920)])
def test_kda_direct_and_low_rank_gates(two_stage, forward_per_token):
    config = _config(
        KDALayerConfig,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        kda_two_stage_gates=two_stage,
    )
    assert _estimate([config]) == 3 * 7 * (forward_per_token + 2 * 16 * 32)


def test_gdn_recurrent_state_uses_key_and_value_dimensions():
    config = _config(
        GDNLayerConfig,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=6,
    )
    # 1152 projection + 160 convolution + 384 state operations + 384 output FMAs.
    assert _estimate([config]) == 3 * 7 * (2 * 2080 + 2 * 16 * 32)


def test_mamba_explicit_head_count_sets_projection_and_state_width():
    from megatron.core.ssm.mamba_mixer import MambaMixer

    config = _config(
        MambaLayerConfig, mamba_num_heads=3, mamba_head_dim=8, mamba_num_groups=1, mamba_state_dim=4
    )
    stack_spec = ModuleSpec(
        module=object,
        submodules=SimpleNamespace(
            mamba_layer=ModuleSpec(
                module=object, submodules=SimpleNamespace(mixer=ModuleSpec(module=MambaMixer))
            )
        ),
    )
    # Inner width is 3*8=24, overriding the default 2*hidden_size=32.
    forward_per_token = 2 * 16 * (2 * 24 + 2 * 4 + 3) + 7 * 24 * 4 + 2 * 24 * 16
    assert _estimate([config], stack_spec=stack_spec) == 3 * 7 * (forward_per_token + 2 * 16 * 32)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_token_statistics_fail_before_accounting(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _estimate([_config(MLPLayerConfig)], tokens=value)


def test_unknown_layer_config_does_not_silently_use_a_builtin_formula():
    with pytest.raises(ValueError, match="Unexpected hybrid layer config type"):
        _estimate([_config()])
