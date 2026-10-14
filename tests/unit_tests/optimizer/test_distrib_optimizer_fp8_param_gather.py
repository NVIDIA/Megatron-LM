# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

pytest.importorskip("transformer_engine")

from megatron.core.optimizer.optimizer_config import OptimizerConfig


def _optimizer_config(fp8_recipe, fp8_param_gather=None, **kwargs) -> OptimizerConfig:
    return OptimizerConfig(
        bf16=True,
        use_precision_aware_optimizer=True,
        fp8_recipe=fp8_recipe,
        fp8_param_gather=fp8_param_gather,
        **kwargs,
    )


@pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
def test_unresolved_fp8_param_gather_is_conservative(fp8_recipe):
    """Unset means "possibly quantized": the precision-aware path stays off until resolved."""
    assert not _optimizer_config(fp8_recipe).use_precision_aware_optimizer_no_fp8_or_ds_fp8


@pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
def test_resolve_inherits_from_ddp_config(fp8_recipe):
    config = _optimizer_config(fp8_recipe)
    # BF16 params with FP8 compute: TE owns the masters.
    resolved = config.resolve_fp8_param_gather(False)
    assert resolved.fp8_param_gather is False
    assert resolved.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    # FP8 params: MCore-managed masters, precision-aware path is off.
    resolved = config.resolve_fp8_param_gather(True)
    assert resolved.fp8_param_gather is True
    assert not resolved.use_precision_aware_optimizer_no_fp8_or_ds_fp8


def test_resolve_does_not_mutate_caller_config():
    config = _optimizer_config("mxfp8")
    resolved = config.resolve_fp8_param_gather(False)
    assert resolved is not config
    assert config.fp8_param_gather is None
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8


@pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
def test_explicit_false_with_ddp_fp8_params_raises(fp8_recipe):
    """FP8 params in the DDP config but False here would silently take the precision-aware path,
    which cannot update quantized params."""
    config = _optimizer_config(fp8_recipe, fp8_param_gather=False)
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    with pytest.raises(ValueError, match="conflicts with"):
        config.check_fp8_param_gather_consistency(True)
    with pytest.raises(ValueError, match="conflicts with"):
        config.resolve_fp8_param_gather(True)


@pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
def test_matching_explicit_value_does_not_raise(fp8_recipe):
    for value in (True, False):
        config = _optimizer_config(fp8_recipe, fp8_param_gather=value)
        assert config.resolve_fp8_param_gather(value) is config


@pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
def test_explicit_true_with_bf16_ddp_params_does_not_raise(fp8_recipe):
    """Over-conservative but safe (MCore-managed masters), so not an error."""
    config = _optimizer_config(fp8_recipe, fp8_param_gather=True)
    config.check_fp8_param_gather_consistency(False)


@pytest.mark.parametrize("fp8_recipe", [None, "delayed"])
def test_recipes_that_always_use_precision_aware_path_do_not_raise(fp8_recipe):
    """Not affected by fp8_param_gather, so a mismatch is not a new failure mode."""
    config = _optimizer_config(fp8_recipe, fp8_param_gather=False)
    config.check_fp8_param_gather_consistency(True)


def test_non_fp32_main_params_do_not_raise():
    """main_params_dtype != fp32 already selected the precision-aware path before the flag."""
    config = _optimizer_config("mxfp8", fp8_param_gather=False, main_params_dtype=torch.float16)
    config.check_fp8_param_gather_consistency(True)


def test_without_precision_aware_optimizer_does_not_raise():
    config = OptimizerConfig(bf16=True, fp8_recipe="mxfp8", fp8_param_gather=False)
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    config.check_fp8_param_gather_consistency(True)
