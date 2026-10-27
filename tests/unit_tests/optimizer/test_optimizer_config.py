# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings

import pytest
import torch

from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey, ParamPredicate


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


def _skip_without_param_remainders_support():
    pytest.importorskip("transformer_engine")
    from megatron.core.utils import is_te_min_version

    if not is_te_min_version("2.1.0"):
        pytest.skip("store_param_remainders needs TE >= 2.1.0")


_PAO = dict(bf16=True, use_precision_aware_optimizer=True)
_NON_DELAYED_FP8_RECIPES = ["tensorwise", "mxfp8", "blockwise", "custom"]


def _config(fp8_param_gather=None, **kwargs) -> OptimizerConfig:
    """An OptimizerConfig, resolved like get_megatron_optimizer does unless fp8_param_gather is
    None (no DDP config to take it from)."""
    config = OptimizerConfig(**kwargs)
    if fp8_param_gather is None:
        return config
    return config.resolve_fp8_param_gather(fp8_param_gather)


def test_fp8_param_gather_is_not_user_settable():
    """Whether the params are quantized is a property of the model, taken from its DDP config."""
    with pytest.raises(TypeError):
        OptimizerConfig(fp8_param_gather=False)
    with pytest.raises(TypeError):
        OptimizerConfig(_fp8_param_gather=False)
    config = OptimizerConfig()
    with pytest.raises(AttributeError):
        config.fp8_param_gather = False
    assert config.fp8_param_gather is None


def test_optimizer_cuda_graph_disables_store_param_remainders():
    """TE FusedAdam capturable mode rejects param remainders, so the config must turn them off."""
    _skip_without_param_remainders_support()

    assert OptimizerConfig(**_PAO).resolved_store_param_remainders
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not OptimizerConfig(
            optimizer_cuda_graph=True, **_PAO
        ).resolved_store_param_remainders
    # An explicit request is overridden too, with a warning.
    with pytest.warns(UserWarning, match="optimizer-cuda-graph"):
        config = OptimizerConfig(optimizer_cuda_graph=True, store_param_remainders=True, **_PAO)
    assert not config.resolved_store_param_remainders


@pytest.mark.parametrize("fp8_param_gather", [None, True, False])
@pytest.mark.parametrize("optimizer_cpu_offload", [False, True])
@pytest.mark.parametrize("main_params_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fp8_recipe", [None, "delayed"] + _NON_DELAYED_FP8_RECIPES)
def test_precision_aware_gate_and_store_param_remainders_default(
    fp8_recipe, main_params_dtype, optimizer_cpu_offload, fp8_param_gather
):
    """With FP32 main params and a non-delayed FP8 recipe, TE owns the master weights only for BF16
    params, and param remainders default to off so the checkpoints keep FP32 master weights. In
    all other configurations TE owns the master weights and param remainders default to on, except
    with FP16 main params, where TE stores no remainders, and with CPU offload, whose optimizer
    keeps FP32 master weights."""
    _skip_without_param_remainders_support()

    config = _config(
        fp8_param_gather,
        fp8_recipe=fp8_recipe,
        main_params_dtype=main_params_dtype,
        optimizer_cpu_offload=optimizer_cpu_offload,
        **_PAO,
    )
    only_with_bf16_params = (
        fp8_recipe in _NON_DELAYED_FP8_RECIPES
        and main_params_dtype == torch.float32
        and not optimizer_cpu_offload
    )
    assert config._use_te_precision_aware_optimizer_legacy() == (not only_with_bf16_params)
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 == (
        not only_with_bf16_params or fp8_param_gather is False
    )
    assert config.store_param_remainders is None  # The field keeps the user's intent.
    # Only takes effect where TE owns the master weights.
    assert config.resolved_store_param_remainders == (
        not only_with_bf16_params
        and main_params_dtype == torch.float32
        and not optimizer_cpu_offload
    )


@pytest.mark.parametrize("store_param_remainders", [None, True])
@pytest.mark.parametrize("fp8_recipe", [None, "delayed", "tensorwise"])
def test_no_param_remainders_with_cpu_offload(fp8_recipe, store_param_remainders):
    """The CPU-offloaded HybridDeviceOptimizer keeps full FP32 master weights, so the config must
    not claim param remainders: the checkpoint's param remainders flag and its buffers follow this value.
    """
    _skip_without_param_remainders_support()

    config = _config(
        False,
        fp8_recipe=fp8_recipe,
        optimizer_cpu_offload=True,
        store_param_remainders=store_param_remainders,
        **_PAO,
    )
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not config._param_remainders_supported()
    assert not config.resolved_store_param_remainders


@pytest.mark.parametrize("store_param_remainders", [None, True])
@pytest.mark.parametrize("fp8_recipe", [None, "delayed", "tensorwise"])
def test_no_param_remainders_with_fp16_main_params(fp8_recipe, store_param_remainders):
    """TE FusedAdam ignores store_param_remainders for FP16 master weights, so the config must not
    claim them: the checkpoint's param remainders flag and its buffers follow this value."""
    _skip_without_param_remainders_support()

    config = _config(
        False,
        fp8_recipe=fp8_recipe,
        main_params_dtype=torch.float16,
        store_param_remainders=store_param_remainders,
        **_PAO,
    )
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not config._param_remainders_supported()
    assert not config.resolved_store_param_remainders


def test_no_param_remainders_without_bf16():
    _skip_without_param_remainders_support()

    config = _config(False, use_precision_aware_optimizer=True, store_param_remainders=True)
    assert not config.resolved_store_param_remainders


@pytest.mark.parametrize("fp8_recipe", _NON_DELAYED_FP8_RECIPES)
def test_gate_follows_fp8_param_gather_resolution(fp8_recipe):
    """The gate is derived on access, so it follows fp8_param_gather when the config is resolved
    after construction, and resolution does not modify the config it is called on."""
    config = OptimizerConfig(fp8_recipe=fp8_recipe, **_PAO)
    assert config.fp8_param_gather is None
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8  # Possibly quantized.

    bf16_params = config.resolve_fp8_param_gather(False)
    assert bf16_params.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not bf16_params.resolve_fp8_param_gather(
        True
    ).use_precision_aware_optimizer_no_fp8_or_ds_fp8

    assert config.fp8_param_gather is None
    assert bf16_params.fp8_param_gather is False


@pytest.mark.parametrize("fp8_recipe", [None, "delayed", "tensorwise"])
@pytest.mark.parametrize("store_param_remainders", [True, False])
def test_store_param_remainders_explicit_value_wins(fp8_recipe, store_param_remainders):
    _skip_without_param_remainders_support()

    config = _config(
        False, fp8_recipe=fp8_recipe, store_param_remainders=store_param_remainders, **_PAO
    )
    assert config.store_param_remainders is store_param_remainders
    assert config.resolved_store_param_remainders is store_param_remainders


def test_gate_without_precision_aware_optimizer():
    config = _config(False, fp8_recipe="tensorwise")
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8


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
