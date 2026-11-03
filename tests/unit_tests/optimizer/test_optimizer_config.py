# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import argparse

import pytest
import torch

from megatron.core.optimizer import _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey, ParamPredicate
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


@pytest.mark.parametrize("apply_wd_to_norm", [False, True])
@pytest.mark.parametrize("apply_wd_to_qk_layernorm", [False, True])
def test_normalization_weight_decay(monkeypatch, apply_wd_to_norm, apply_wd_to_qk_layernorm):
    """Norm weights decay independently of biases, scaling vectors and fused linear weights."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, **kwargs: output.__setitem__(0, value),
    )
    norm_names = (
        "input_layernorm",
        "pre_mlp_layernorm",
        "final_layernorm",
        "q_layernorm",
        "k_layernorm",
        "kv_layernorm",
        "k_norm",
        "enorm",
        "hnorm",
    )
    model = torch.nn.ModuleDict({name: torch.nn.LayerNorm(4) for name in norm_names})
    model["compressor"] = torch.nn.ModuleDict({"norm": torch.nn.RMSNorm(4)})
    model["linear_fc1"] = torch.nn.Linear(4, 4)
    model["linear_fc1"].register_parameter("layer_norm_weight", torch.nn.Parameter(torch.ones(4)))
    model["linear_fc1"].register_parameter("layer_norm_bias", torch.nn.Parameter(torch.ones(4)))
    model["controls"] = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(torch.ones(4)) for name in ("alpha", "norm_scale", "retention")}
    )
    model["controls"]["retention"].is_wide_residual_retention_parameter = True
    # A norm ancestor or a name containing "norm" must not opt unrelated weights into decay.
    model["compressor"]["norm"].register_parameter("scale", torch.nn.Parameter(torch.ones(4)))
    model["compressor"]["norm"].projection = torch.nn.ParameterDict(
        {"weight": torch.nn.Parameter(torch.ones(4))}
    )
    model["norm_projection"] = torch.nn.ParameterDict({"weight": torch.nn.Parameter(torch.ones(4))})
    model["fused_gate"] = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(torch.ones(4)) for name in ("q_weight", "k_weight")}
    )
    for param in model["fused_gate"].parameters():
        param.is_norm_weight = True
    expected_decay = {id(param): False for param in model.parameters()}
    for name in norm_names:
        is_qk = name in ("q_layernorm", "k_layernorm")
        expected_decay[id(model[name].weight)] = apply_wd_to_norm or (
            apply_wd_to_qk_layernorm and is_qk
        )
        # Preserve the old Q/K-only option, which also includes Q/K norm biases.
        expected_decay[id(model[name].bias)] = (
            apply_wd_to_qk_layernorm and is_qk and not apply_wd_to_norm
        )
    expected_decay[id(model["compressor"]["norm"].weight)] = apply_wd_to_norm
    expected_decay[id(model["linear_fc1"].layer_norm_weight)] = apply_wd_to_norm
    expected_decay[id(model["linear_fc1"].weight)] = True
    expected_decay[id(model["controls"]["retention"])] = True
    for param in model["fused_gate"].parameters():
        expected_decay[id(param)] = apply_wd_to_norm

    config = OptimizerConfig(
        lr=0.01,
        min_lr=0.0,
        weight_decay=0.1,
        apply_wd_to_norm=apply_wd_to_norm,
        apply_wd_to_qk_layernorm=apply_wd_to_qk_layernorm,
    )
    groups = _get_param_groups([model], config, get_standard_config_overrides(config))
    optimizer = torch.optim.AdamW(groups, lr=config.lr)
    OptimizerParamScheduler(
        optimizer,
        init_lr=0.01,
        max_lr=0.01,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=100,
        lr_decay_style="constant",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=100,
        wd_incr_style="constant",
    )
    for group in optimizer.param_groups:
        for param in group["params"]:
            assert group["wd_mult"] == float(expected_decay[id(param)])
            assert group["weight_decay"] == pytest.approx(0.1 if expected_decay[id(param)] else 0.0)
            param.data.fill_(1.0)
            param.grad = torch.zeros_like(param)
    optimizer.step()
    for name, param in model.named_parameters():
        expected = 0.999 if expected_decay[id(param)] else 1.0
        torch.testing.assert_close(param, torch.full_like(param, expected), msg=name)


@pytest.mark.parametrize("enabled", [False, True])
def test_normalization_weight_decay_training_args(enabled):
    """The CLI flag reaches OptimizerConfig and its standard no-decay overrides."""
    from megatron.training.arguments import _add_regularization_args
    from megatron.training.training import get_megatron_optimizer_config

    parser = _add_regularization_args(argparse.ArgumentParser())
    args = parser.parse_args(["--apply-wd-to-norm"] if enabled else [])
    config, overrides = get_megatron_optimizer_config(args)
    assert config.apply_wd_to_norm is enabled
    norm_weight = torch.nn.Parameter(torch.ones(4))
    no_decay = any(
        key.matches(norm_weight, "decoder.final_layernorm.weight") and value.get("wd_mult") == 0.0
        for key, value in overrides.items()
    )
    assert no_decay is not enabled


def test_layer_sharded_muon_tp_mode_requirements():
    """muon_tp_mode='layer_sharded' needs muon, the layer-wise path and no split-QKV."""
    ok = dict(optimizer='muon', use_layer_wise_distributed_optimizer=True, muon_split_qkv=False)
    OptimizerConfig(muon_tp_mode='layer_sharded', **ok)
    # dist_muon is the deprecated alias for muon + the layer-wise path.
    OptimizerConfig(muon_tp_mode='layer_sharded', optimizer='dist_muon', muon_split_qkv=False)
    with pytest.raises(ValueError, match="requires optimizer='muon'"):
        OptimizerConfig(muon_tp_mode='layer_sharded', **{**ok, 'optimizer': 'adaptive_muon'})
    with pytest.raises(ValueError, match="layer-wise"):
        OptimizerConfig(
            muon_tp_mode='layer_sharded', **{**ok, 'use_layer_wise_distributed_optimizer': False}
        )
    with pytest.raises(ValueError, match="split-QKV"):
        OptimizerConfig(muon_tp_mode='layer_sharded', **{**ok, 'muon_split_qkv': True})


@pytest.mark.parametrize('mode', [None, 'spectral', 'unit_rms_norm', 'shape_scaling', 'rms'])
def test_muon_rms_training_args(mode):
    """The existing scale-mode argument carries RMS and its target to OptimizerConfig."""
    from megatron.training.arguments import _add_regularization_args
    from megatron.training.training import get_megatron_optimizer_config

    parser = _add_regularization_args(argparse.ArgumentParser())
    cli = ['--muon-extra-scale-factor', '0.18']
    if mode is not None:
        cli += ['--muon-scale-mode', mode]
    args = parser.parse_args(cli)
    args.optimizer = 'muon'
    args.use_layer_wise_distributed_optimizer = True
    config, _ = get_megatron_optimizer_config(args)
    assert config.muon_scale_mode == (mode or 'spectral')
    assert config.muon_extra_scale_factor == 0.18
    assert config.use_layer_wise_distributed_optimizer


@pytest.mark.parametrize('optimizer', ['adam', 'adaptive_muon'])
def test_muon_rms_config_requires_muon(optimizer):
    with pytest.raises(ValueError, match="requires optimizer='muon'"):
        OptimizerConfig(optimizer=optimizer, muon_scale_mode='rms')


def test_muon_rms_config_rejects_layer_sharded():
    with pytest.raises(ValueError, match='does not support layer_sharded'):
        OptimizerConfig(
            optimizer='muon',
            muon_scale_mode='rms',
            muon_tp_mode='layer_sharded',
            muon_split_qkv=False,
            use_layer_wise_distributed_optimizer=True,
        )


@pytest.mark.parametrize('target', [0, -1, float('inf'), float('nan')])
def test_muon_rms_config_rejects_invalid_target(target):
    with pytest.raises(ValueError, match='finite positive extra_scale_factor'):
        OptimizerConfig(optimizer='muon', muon_scale_mode='rms', muon_extra_scale_factor=target)


def test_paramkey_matches():
    len_1_predicate = ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1)
    endswith_bias = ParamKey(name="*.bias")
    has_dotbias = ParamKey(name="*.bias*")
    len_1_param = ParamKey(predicate=len_1_predicate)
    has_bias_or_len1_param = ParamKey(name="*.bias", predicate=len_1_predicate)
    has_attr = ParamKey(attr="is_embedding_or_output_parameter")

    assert endswith_bias.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias")
    assert not endswith_bias.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "something.bias.other"
    )
    assert has_dotbias.matches(torch.nn.Parameter(torch.empty(10)), "random.biasstuff")
    assert not has_dotbias.matches(torch.nn.Parameter(torch.empty(10, 10)), "random_bias_name")
    assert len_1_param.matches(torch.nn.Parameter(torch.empty(10)), "interesting.bias")
    assert not len_1_param.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting_bias")
    assert has_bias_or_len1_param.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias"
    )
    assert has_bias_or_len1_param.matches(torch.nn.Parameter(torch.empty(10)), "interesting_bias")
    assert not has_bias_or_len1_param.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "random_bias_name"
    )
    p_with_attr = torch.nn.Parameter(torch.empty(10, 10))
    setattr(p_with_attr, "is_embedding_or_output_parameter", True)
    assert has_attr.matches(p_with_attr, "interesting.bias")
    assert not has_attr.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias")

    # We expect that if the return of the attribute is False, it should not match even if
    #  it has the attribute.
    setattr(p_with_attr, "is_embedding_or_output_parameter", False)
    assert not has_attr.matches(p_with_attr, "interesting.bias")
