# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings

import pytest
import torch

pytest.importorskip("transformer_engine")

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


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


# The tests below build the optimizer with get_megatron_optimizer, which resolves fp8_param_gather
# from the model's DDP config. The FP8 recipe only matters to the optimizer through the config, so
# the model runs in BF16.


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _ddp_model(ddp_fp8_param_gather=False, use_distributed_optimizer=True):
    model = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=use_distributed_optimizer),
        model,
    )
    # Only the optimizer's view of the DDP config matters here; the params stay BF16.
    model.ddp_config.fp8_param_gather = ddp_fp8_param_gather
    return model


def _get_optimizer(model, fp8_recipe="tensorwise", use_distributed_optimizer=True, **kwargs):
    config = OptimizerConfig(
        optimizer="adam",
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=use_distributed_optimizer,
        use_precision_aware_optimizer=True,
        fp8_recipe=fp8_recipe,
        **kwargs,
    )
    return config, get_megatron_optimizer(config, [model])


def _train_step(model, optimizer):
    optimizer.zero_grad()
    model.zero_grad_buffer()
    model(torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")).square().mean().backward()
    model.finish_grad_sync()
    update_successful, _, _ = optimizer.step()
    assert update_successful


def _master_params(optimizer):
    return [state["master_param"] for state in optimizer.optimizer.state.values() if state]


@pytest.mark.parametrize("fp8_recipe", ["tensorwise", "mxfp8", "blockwise"])
@pytest.mark.parametrize("ddp_fp8_param_gather", [False, True])
def test_get_megatron_optimizer_resolves_fp8_param_gather(
    model_parallel, fp8_recipe, ddp_fp8_param_gather
):
    """Library callers set fp8_param_gather only on the DDP config; the optimizer inherits it."""
    model = _ddp_model(ddp_fp8_param_gather)
    config, optimizer = _get_optimizer(model, fp8_recipe)

    assert optimizer.config.fp8_param_gather is ddp_fp8_param_gather
    bf16_params = not ddp_fp8_param_gather
    assert optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 == bf16_params
    # The default keeps FP32 master params when TE owns them only because the params are BF16.
    assert optimizer.config.resolved_store_param_remainders == (not bf16_params)
    # The caller's config is not modified.
    assert config.fp8_param_gather is None and config.resolved_store_param_remainders

    fused_adam = optimizer.optimizer
    assert bool(fused_adam.master_weights) == bf16_params
    if bf16_params:
        assert not fused_adam.store_param_remainders
        assert all(p.dtype == torch.bfloat16 for g in fused_adam.param_groups for p in g["params"])
    else:
        # MCore-managed FP32 main params, as before the BF16-params path existed.
        assert all(p.dtype == torch.float32 for g in fused_adam.param_groups for p in g["params"])


@pytest.mark.parametrize("store_param_remainders", [None, True])
def test_bf16_params_master_param_format(model_parallel, store_param_remainders):
    """With BF16 params and a non-delayed recipe the optimizer checkpoint's master params stay
    FP32 by default (as with the MCore-managed main params that were used before), and are int16
    remainders only when explicitly requested. Either way get_main_param_shard is exact."""
    model = _ddp_model()
    _, optimizer = _get_optimizer(model, store_param_remainders=store_param_remainders)
    assert optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8

    _train_step(model, optimizer)

    expected_dtype = torch.int16 if store_param_remainders else torch.float32
    masters = _master_params(optimizer)
    assert masters and all(master.dtype == expected_dtype for master in masters)

    shard = optimizer.get_main_param_shard(model.module.weight)
    if shard is not None:  # This rank may own no part of the weight.
        assert shard.dtype == torch.float32
        (distrib_optimizer,) = optimizer.chained_optimizers
        index_map = distrib_optimizer.model_param_group_index_map
        group_index, group_order = index_map[model.module.weight]
        bf16_shard = distrib_optimizer.optimizer.param_groups[group_index]["params"][group_order]
        # The BF16 param is the master rounded to nearest (within half a BF16 ulp), but the
        # master keeps more bits: the shard is not just the BF16 param upcast.
        torch.testing.assert_close(shard, bf16_shard.float(), rtol=2**-8, atol=0)
        assert not torch.equal(shard, bf16_shard.float())
        if not store_param_remainders:
            master = distrib_optimizer.optimizer.state[bf16_shard]["master_param"]
            assert torch.equal(shard, master)


@pytest.mark.parametrize("store_param_remainders", [None, True])
@pytest.mark.parametrize("fp8_recipe", ["tensorwise", "mxfp8", "blockwise"])
def test_optimizer_cuda_graph_with_bf16_params(model_parallel, fp8_recipe, store_param_remainders):
    """TE FusedAdam rejects param remainders in capturable mode, so building the optimizer for
    --optimizer-cuda-graph with BF16 params must turn them off rather than raise."""
    model = _ddp_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, optimizer = _get_optimizer(
            model,
            fp8_recipe,
            optimizer_cuda_graph=True,
            store_param_remainders=store_param_remainders,
        )

    assert optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not optimizer.config.resolved_store_param_remainders
    fused_adam = optimizer.optimizer
    assert fused_adam.capturable and fused_adam.master_weights
    assert not fused_adam.store_param_remainders


def test_precision_aware_optimizer_requires_distributed_optimizer(model_parallel):
    """Without the distributed optimizer nothing would write the decoupled grads that the
    BF16-params path consumes; this was, and must stay, rejected."""
    model = _ddp_model(use_distributed_optimizer=False)
    with pytest.raises(ValueError, match="only supported with distributed optimizer"):
        _get_optimizer(model, use_distributed_optimizer=False)
