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


def test_optimizer_cuda_graph_disables_store_param_remainders():
    """TE FusedAdam capturable mode rejects param remainders, so the config must turn them off."""
    _skip_without_param_remainders_support()

    assert OptimizerConfig(**_PAO).resolved_store_param_remainders
    # Warns, and resolving the config again (which re-runs __post_init__) stays off.
    with pytest.warns(UserWarning, match="store_param_remainders"):
        config = OptimizerConfig(optimizer_cuda_graph=True, **_PAO)
    assert not config.resolved_store_param_remainders
    assert not config.resolve_fp8_param_gather(False).resolved_store_param_remainders
    # An explicit request is overridden as well.
    with pytest.warns(UserWarning, match="optimizer-cuda-graph"):
        config = OptimizerConfig(optimizer_cuda_graph=True, store_param_remainders=True, **_PAO)
    assert not config.resolved_store_param_remainders
    # Without the precision-aware optimizer the setting is irrelevant and left alone.
    assert OptimizerConfig(optimizer_cuda_graph=True).resolved_store_param_remainders


@pytest.mark.parametrize("fp8_param_gather", [None, True, False])
@pytest.mark.parametrize("optimizer_cpu_offload", [False, True])
@pytest.mark.parametrize("main_params_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fp8_recipe", [None, "delayed"] + _NON_DELAYED_FP8_RECIPES)
def test_store_param_remainders_default(
    fp8_recipe, main_params_dtype, optimizer_cpu_offload, fp8_param_gather
):
    """The default stores remainders, except when the precision-aware optimizer owns the masters
    only because the params are BF16 (FP32 main params and a non-delayed FP8 recipe): such runs
    used to keep FP32 masters in their checkpoints, which must stay loadable."""
    _skip_without_param_remainders_support()

    config = OptimizerConfig(
        fp8_recipe=fp8_recipe,
        main_params_dtype=main_params_dtype,
        optimizer_cpu_offload=optimizer_cpu_offload,
        fp8_param_gather=fp8_param_gather,
        **_PAO,
    )
    only_for_bf16_params = (
        fp8_recipe in _NON_DELAYED_FP8_RECIPES
        and main_params_dtype == torch.float32
        and not optimizer_cpu_offload
        and fp8_param_gather is False
    )
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 == (
        only_for_bf16_params
        or fp8_recipe not in _NON_DELAYED_FP8_RECIPES
        or main_params_dtype != torch.float32
        or optimizer_cpu_offload
    )
    assert config._use_precision_aware_optimizer_only_for_bf16_params() == only_for_bf16_params
    assert config.store_param_remainders is None  # The field keeps the user's intent.
    assert config.resolved_store_param_remainders == (not only_for_bf16_params)


@pytest.mark.parametrize("fp8_recipe", _NON_DELAYED_FP8_RECIPES)
def test_store_param_remainders_default_follows_fp8_param_gather_resolution(fp8_recipe):
    """The default must not be frozen when the config is created before fp8_param_gather is
    resolved from the DDP config (resolution copies the config with dataclasses.replace)."""
    _skip_without_param_remainders_support()

    config = OptimizerConfig(fp8_recipe=fp8_recipe, **_PAO)
    assert config.fp8_param_gather is None
    assert config.resolved_store_param_remainders  # Unresolved: same as before (MCore masters).

    bf16_params = config.resolve_fp8_param_gather(False)
    assert bf16_params.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not bf16_params.resolved_store_param_remainders
    assert bf16_params.resolve_fp8_param_gather(False).resolved_store_param_remainders is False

    fp8_params = config.resolve_fp8_param_gather(True)
    assert not fp8_params.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert fp8_params.resolved_store_param_remainders

    # The caller's config is not modified.
    assert config.fp8_param_gather is None and config.resolved_store_param_remainders


@pytest.mark.parametrize("fp8_recipe", [None, "delayed", "tensorwise"])
@pytest.mark.parametrize("store_param_remainders", [True, False])
def test_store_param_remainders_explicit_value_wins(fp8_recipe, store_param_remainders):
    _skip_without_param_remainders_support()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = OptimizerConfig(
            fp8_recipe=fp8_recipe,
            fp8_param_gather=False,
            store_param_remainders=store_param_remainders,
            **_PAO,
        )
    assert config.store_param_remainders is store_param_remainders
    assert config.resolved_store_param_remainders is store_param_remainders


def test_store_param_remainders_explicit_true_warns_only_for_bf16_params_case():
    """Opting in where the default is off for checkpoint compatibility warns about it."""
    _skip_without_param_remainders_support()

    # Unresolved fp8_param_gather: the BF16-params case is not known yet, so no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = OptimizerConfig(fp8_recipe="tensorwise", store_param_remainders=True, **_PAO)
        OptimizerConfig(fp8_recipe="delayed", store_param_remainders=True, **_PAO)
    with pytest.warns(UserWarning, match="not interchangeable"):
        resolved = config.resolve_fp8_param_gather(False)
    assert resolved.resolved_store_param_remainders


def test_store_param_remainders_without_precision_aware_optimizer():
    """Unused without the precision-aware optimizer; the resolved value mirrors the old default."""
    assert OptimizerConfig().resolved_store_param_remainders
    assert OptimizerConfig(store_param_remainders=True).resolved_store_param_remainders
    assert not OptimizerConfig(store_param_remainders=False).resolved_store_param_remainders
    # Not the precision-aware path, whatever the params are.
    config = OptimizerConfig(fp8_recipe="tensorwise", fp8_param_gather=False)
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not config._use_precision_aware_optimizer_only_for_bf16_params()


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
