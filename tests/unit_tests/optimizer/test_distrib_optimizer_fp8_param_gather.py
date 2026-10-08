# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

pytest.importorskip("transformer_engine")

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def _optimizer_config(fp8_recipe, fp8_param_gather, **kwargs) -> OptimizerConfig:
    return OptimizerConfig(
        bf16=True,
        use_precision_aware_optimizer=True,
        fp8_recipe=fp8_recipe,
        fp8_param_gather=fp8_param_gather,
        **kwargs,
    )


def _check(config, ddp_fp8_param_gather):
    ddp_config = DistributedDataParallelConfig(fp8_param_gather=ddp_fp8_param_gather)
    DistributedOptimizer._check_fp8_param_gather_consistency(config, ddp_config)


@pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
def test_ddp_fp8_params_with_optimizer_config_unset_raises(fp8_recipe):
    """FP8 params in the DDP config but not the optimizer config would silently take the
    precision-aware path, which cannot update quantized params."""
    config = _optimizer_config(fp8_recipe, fp8_param_gather=False)
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    with pytest.raises(ValueError, match="OptimizerConfig.fp8_param_gather=True"):
        _check(config, ddp_fp8_param_gather=True)


@pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
def test_matching_configs_do_not_raise(fp8_recipe):
    # FP8 params: MCore-managed masters, precision-aware path is off.
    config = _optimizer_config(fp8_recipe, fp8_param_gather=True)
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    _check(config, ddp_fp8_param_gather=True)
    # BF16 params with FP8 compute: the case this check must not block.
    config = _optimizer_config(fp8_recipe, fp8_param_gather=False)
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    _check(config, ddp_fp8_param_gather=False)


@pytest.mark.parametrize("fp8_recipe", [None, "delayed"])
def test_recipes_that_always_use_precision_aware_path_do_not_raise(fp8_recipe):
    """Not affected by fp8_param_gather, so a mismatch is not a new failure mode."""
    config = _optimizer_config(fp8_recipe, fp8_param_gather=False)
    _check(config, ddp_fp8_param_gather=True)


def test_non_fp32_main_params_do_not_raise():
    """main_params_dtype != fp32 already selected the precision-aware path before the flag."""
    config = _optimizer_config("mxfp8", fp8_param_gather=False, main_params_dtype=torch.float16)
    _check(config, ddp_fp8_param_gather=True)


def test_without_precision_aware_optimizer_does_not_raise():
    config = OptimizerConfig(bf16=True, fp8_recipe="mxfp8", fp8_param_gather=False)
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    _check(config, ddp_fp8_param_gather=True)
