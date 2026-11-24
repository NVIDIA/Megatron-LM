# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FLOP estimates compose actual children without constructing CUDA model state."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.gated_delta_net.kda import KimiDeltaAttention
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.models.hybrid.external_layer import ExternalGainConfig, ExternalGainLayer

BATCH_STATS = dict(total_real_tokens_in_batch=7, seqlen_squared_sum_in_batch=25)


def _config(**kwargs):
    return ExternalGainConfig(num_layers=3, hidden_size=16, num_attention_heads=2, **kwargs)


def _bare(module_type, config, **attributes):
    module = module_type.__new__(module_type)
    MegatronModule.__init__(module, config)
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _stack(config, layers):
    return _bare(HybridStack, config, layers=torch.nn.ModuleList(layers))


def _model(config, layers, *, mtp=None, post_process=True):
    return _bare(
        HybridModel,
        config,
        decoder=_stack(config, layers),
        mtp=mtp,
        mtp_process=mtp is not None,
        post_process=post_process,
        vocab_size=32,
        share_embeddings_and_output_weights=False,
        output_layer=torch.nn.Linear(config.hidden_size, 32, bias=False, device="meta"),
    )


@pytest.mark.parametrize("shared_weights", [False, True])
@pytest.mark.parametrize("include_backward,expected", [(False, 66_304), (True, 199_296)])
def test_mtp_counts_executions_and_output_heads(shared_weights, include_backward, expected):
    config = _config(ffn_hidden_size=64, mtp_num_layers=2, mtp_use_repeated_layer=shared_weights)
    decoder = [
        _bare(ExternalGainLayer, config),
        _bare(MLP, config),
        _bare(ExternalGainLayer, config),
    ]
    mtp_layers = [
        _bare(
            MultiTokenPredictionLayer,
            config,
            mtp_model_layer=_stack(config, [_bare(ExternalGainLayer, config)]),
        )
        for _ in range(1 if shared_weights else 2)
    ]
    mtp = _bare(
        MultiTokenPredictionBlock,
        config,
        layers=torch.nn.ModuleList(mtp_layers),
        mtp_use_repeated_layer=shared_weights,
    )
    # T=7: four gain executions=448, FFN=28672, MTP=15680, three heads=21504.
    model = _model(config, decoder, mtp=mtp)
    assert model.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


@pytest.mark.parametrize(
    "ce,kl,training,compute_mtp_loss,detach_heads,forward_heads,training_matmuls",
    [
        (0, 0, True, True, False, 1, 3),
        (0, 1, True, True, False, 4, 10),
        (1, 1, True, True, True, 4, 8),
        (1, 1, True, False, False, 1, 3),
        (1, 1, False, True, False, 3, 9),
        (0, 1, False, True, False, 1, 3),
    ],
)
def test_mtp_counts_active_objectives(
    ce, kl, training, compute_mtp_loss, detach_heads, forward_heads, training_matmuls
):
    config = _config(
        ffn_hidden_size=32,
        mtp_num_layers=2,
        mtp_ce_loss_weight=ce,
        mtp_kl_loss_weight=kl,
        mtp_detach_heads=detach_heads,
    )
    model = _model(config, [], mtp=_stack(config, [_bare(MLP, config)]))
    model.train(training)
    # One FFN for the MTP block and one matmul per vocabulary projection/gradient.
    mtp_forward = 14_336 if compute_mtp_loss else 0
    for backward, matmuls in [(False, forward_heads), (True, training_matmuls)]:
        mtp_backward = backward and (training or ce > 0)
        expected = (3 if mtp_backward else 1) * mtp_forward + matmuls * 7168
        assert (
            model.estimate_flops(
                **BATCH_STATS, include_backward=backward, compute_mtp_loss=compute_mtp_loss
            )
            == expected
        )


@pytest.mark.parametrize("compute_mtp_loss", [False, True])
def test_frozen_backbone_counts_only_mtp_gradients(compute_mtp_loss):
    config = _config(ffn_hidden_size=32, mtp_num_layers=1, freeze_base_model_for_mtp=True)
    model = _model(config, [_bare(MLP, config)], mtp=_stack(config, [_bare(MLP, config)]))
    model.output_layer.requires_grad_(False)
    # Frozen decoder is forward-only. With MTP, both heads propagate activation gradients.
    expected = 14_336 + (3 * 14_336 + 4 * 7168 if compute_mtp_loss else 7168)
    assert (
        model.estimate_flops(
            **BATCH_STATS, include_backward=True, compute_mtp_loss=compute_mtp_loss
        )
        == expected
    )


@pytest.mark.parametrize("include_backward", [False, True])
def test_mtp_counts_each_mhc_hidden_stream(include_backward):
    config = _config(enable_mhc_connections=True, mhc_num_residual_streams=4)
    layer = _bare(MultiTokenPredictionLayer, config, mtp_model_layer=IdentityOp())
    # Four hidden streams plus one embedding projection. Norms also include the final output.
    forward = 2 * 7 * (5 * 16**2 + 6 * 16)
    assert (
        layer.estimate_flops(**BATCH_STATS, include_backward=include_backward)
        == (3 if include_backward else 1) * forward
    )


@pytest.mark.parametrize("include_backward,expected", [(False, 28_896), (True, 86_880)])
def test_nested_custom_children_and_wrappers_count_once(include_backward, expected):
    config = _config(ffn_hidden_size=64)
    transformer = _bare(
        TransformerLayer,
        config,
        self_attention=_bare(ExternalGainLayer, config),
        cross_attention=IdentityOp(),
        mlp=_bare(MLP, config),
    )
    wrapped = _bare(HyperConnectionHybridLayer, config, inner_layer=transformer)
    shortcut = _bare(
        ShortcutMoEBlock, config, compute_layer=wrapped, moe_layer=_bare(ExternalGainLayer, config)
    )
    model = _model(config, [shortcut], post_process=False)
    # Two gain executions plus one FFN, with no output head on this stage.
    assert model.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


@pytest.mark.parametrize("include_backward,expected", [(False, 39_536), (True, 118_704)])
def test_mixed_ffn_widths_use_actual_child_configs(include_backward, expected):
    config = _config()
    narrow = _bare(MLP, _config(ffn_hidden_size=24))
    wide = _bare(MLP, _config(ffn_hidden_size=48))
    model = _model(config, [narrow, wide, _bare(ExternalGainLayer, config)])
    assert model.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


@pytest.mark.parametrize(
    "two_stage_gates,include_backward,expected",
    [(False, False, 14_000), (False, True, 42_000), (True, False, 13_104), (True, True, 39_312)],
)
def test_kda_gate_projections_in_model_estimate(two_stage_gates, include_backward, expected):
    config = KDALayerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=3,
        kda_two_stage_gates=two_stage_gates,
    )
    layer = _bare(
        TransformerLayer,
        config,
        self_attention=_bare(KimiDeltaAttention, config),
        cross_attention=IdentityOp(),
        mlp=IdentityOp(),
    )
    model = _model(config, [layer], post_process=False)
    # MACs per token: QKV 384, beta 32, convolution 72, recurrence 128, output 128.
    # Direct gates add 256 MACs; the two 16 -> 4 -> 8 gates add 192.
    assert model.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


@pytest.mark.parametrize(
    "post_process,include_backward,expected",
    [(False, False, 0), (False, True, 0), (True, False, 7168), (True, True, 21_504)],
)
def test_empty_stages_count_only_local_heads(post_process, include_backward, expected):
    config = _config(mtp_num_layers=2)
    # Global MTP depth does not create work on chunks without an MTP block.
    model = _model(config, [], post_process=post_process)
    assert model.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


def test_missing_layer_hook_fails_explicitly():
    config = _config()
    model = _model(config, [_bare(MegatronModule, config)])
    with pytest.raises(NotImplementedError):
        model.estimate_flops(**BATCH_STATS)
