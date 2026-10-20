# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings

import pytest
import torch

pytest.importorskip("transformer_engine")

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _optimizer_config(fp8_recipe, **kwargs) -> OptimizerConfig:
    return OptimizerConfig(
        bf16=True, use_precision_aware_optimizer=True, fp8_recipe=fp8_recipe, **kwargs
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


def test_resolving_again_takes_the_new_ddp_value():
    """The DDP config is the source of truth, so a config already resolved for one model follows
    the DDP config of the next model it is resolved for."""
    config = _optimizer_config("mxfp8").resolve_fp8_param_gather(False)
    assert config.resolve_fp8_param_gather(True).fp8_param_gather is True
    assert config.fp8_param_gather is False


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
    # The caller's config is not modified.
    assert config.fp8_param_gather is None

    fused_adam = optimizer.optimizer
    assert bool(fused_adam.master_weights) == bf16_params
    if bf16_params:
        assert not fused_adam.store_param_remainders
        assert all(p.dtype == torch.bfloat16 for g in fused_adam.param_groups for p in g["params"])
    else:
        # FP8 params: MCore-managed FP32 main params.
        assert all(p.dtype == torch.float32 for g in fused_adam.param_groups for p in g["params"])


@pytest.mark.parametrize("store_param_remainders", [None, True])
def test_bf16_params_master_param_format(model_parallel, store_param_remainders):
    """With BF16 params and a non-delayed recipe the optimizer checkpoint's master params are
    FP32 by default, and int16 remainders only when explicitly requested. Either way
    get_main_param_shard is exact."""
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


# (FP8 recipe, main params dtype, format saved with, format loaded with, whether the checkpoint
# records its format). The unrecorded ones stand for checkpoints written before the format was
# recorded, so they are saved the way those versions saved them: MCore-managed FP32 main params
# with a non-delayed recipe, param remainders without FP8 where TE owned FP32 master weights
# (unless store_param_remainders was False), and
# FP32 master values with FP16 main params, where TE stores no remainders.
_FP32, _FP16 = torch.float32, torch.float16
_CHECKPOINT_COMPATIBILITY_CASES = [
    ("tensorwise", _FP32, None, None, True),
    ("tensorwise", _FP32, True, True, True),
    ("tensorwise", _FP32, None, True, True),
    ("tensorwise", _FP32, True, None, True),
    ("tensorwise", _FP32, None, None, False),
    ("tensorwise", _FP32, None, True, False),
    (None, _FP32, None, None, True),
    (None, _FP32, False, False, True),
    (None, _FP32, None, False, True),
    (None, _FP32, None, None, False),
    # Old versions honored an explicit store_param_remainders=False without recording it, so the
    # setting of the loading run is taken to be the one the checkpoint was saved with.
    (None, _FP32, False, False, False),
    # TE ignores store_param_remainders for FP16 master weights, so it changes nothing here.
    (None, _FP16, None, None, True),
    (None, _FP16, None, None, False),
    (None, _FP16, True, False, False),
]


@pytest.mark.parametrize(
    ("fp8_recipe", "main_params_dtype", "save_remainders", "load_remainders", "record_format"),
    _CHECKPOINT_COMPATIBILITY_CASES,
)
def test_param_remainders_checkpoint_compatibility(
    model_parallel,
    tmp_path_dist_ckpt,
    fp8_recipe,
    main_params_dtype,
    save_remainders,
    load_remainders,
    record_format,
):
    """dist-checkpointing would load int16 param remainders as FP32 master params, or the
    reverse, with a silent value cast that corrupts the master weights, so a format mismatch must
    raise, and matching formats must load the exact master weights."""
    metadata = {"distrib_optim_sharding_type": "dp_reshardable", "checkpoint_version": 3.1}
    name = (
        f"remainders_{fp8_recipe}_{main_params_dtype}_{save_remainders}_{load_remainders}_"
        f"{record_format}"
    )
    with TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as ckpt_dir:
        model = _ddp_model()
        _, optimizer = _get_optimizer(
            model,
            fp8_recipe,
            main_params_dtype=main_params_dtype,
            store_param_remainders=save_remainders,
        )
        _train_step(model, optimizer)
        (saving,) = optimizer.chained_optimizers
        saved_remainders = saving._stores_param_remainders()
        if main_params_dtype != torch.float32:
            # The checkpoint format must match what TE holds: FP16 masters, never remainders.
            assert not saved_remainders
            assert all(master.dtype == torch.float16 for master in _master_params(optimizer))
        sharded_state_dict = saving.sharded_state_dict({}, metadata=metadata)
        assert sharded_state_dict["param_state_param_remainders"] == saved_remainders
        if not record_format:
            del sharded_state_dict["param_state_param_remainders"]
        save({"optimizer": sharded_state_dict}, ckpt_dir)
        saved_shard = saving.get_main_param_shard(model.module.weight)

        _, optimizer = _get_optimizer(
            model,
            fp8_recipe,
            main_params_dtype=main_params_dtype,
            store_param_remainders=load_remainders,
        )
        (loading,) = optimizer.chained_optimizers
        loaded = load(
            {"optimizer": loading.sharded_state_dict({}, metadata=metadata, is_loading=True)},
            ckpt_dir,
        )
        if saved_remainders != loading._stores_param_remainders():
            with pytest.raises(RuntimeError, match="silently corrupt"):
                loading.load_state_dict(loaded["optimizer"])
            return
        loading.load_state_dict(loaded["optimizer"])
        loaded_shard = loading.get_main_param_shard(model.module.weight)
        if saved_shard is not None:
            if main_params_dtype == torch.float32:
                assert torch.equal(loaded_shard, saved_shard)
            else:
                # TE rescales FP16 master weights from their absmax when setting them.
                torch.testing.assert_close(loaded_shard, saved_shard, rtol=2**-10, atol=0)


def _model_sharded_state_dict(model):
    weight = model.module.weight
    return {
        "weight": ShardedTensor.from_rank_offsets(
            "weight", weight, replica_id=torch.distributed.get_rank()
        )
    }


def _save_parameter_state(optimizer, model, sharding_type, ckpt_dir):
    if sharding_type == "torch":
        optimizer.save_parameter_state(str(ckpt_dir / "param_state.pt"))
        torch.distributed.barrier()
        return
    metadata = {"distrib_optim_sharding_type": sharding_type, "checkpoint_version": 3.1}
    sharded_state_dict = optimizer.sharded_state_dict(
        _model_sharded_state_dict(model), metadata=metadata
    )
    save({"optimizer": sharded_state_dict}, ckpt_dir)


def _load_parameter_state(optimizer, model, sharding_type, ckpt_dir):
    if sharding_type == "torch":
        optimizer.load_parameter_state(str(ckpt_dir / "param_state.pt"))
        return
    metadata = {"distrib_optim_sharding_type": sharding_type, "checkpoint_version": 3.1}
    sharded_state_dict = optimizer.sharded_state_dict(
        _model_sharded_state_dict(model), metadata=metadata, is_loading=True
    )
    optimizer.load_state_dict(load({"optimizer": sharded_state_dict}, ckpt_dir)["optimizer"])


@pytest.mark.parametrize("sharding_type", ["torch", "dp_zero_gather_scatter", "fully_reshardable"])
@pytest.mark.parametrize(
    ("fp8_recipe", "store_param_remainders"), [(None, None), ("tensorwise", True)]
)
def test_param_remainders_through_fp32_buffers(
    model_parallel, tmp_path_dist_ckpt, sharding_type, fp8_recipe, store_param_remainders
):
    """The DP-zero formats (--ckpt-format torch, dp_zero_gather_scatter) and the fully reshardable
    format gather the parameter state in FP32 buffers, but TE FusedAdam only accepts param
    remainders as int16. They must load the exact master weights."""
    name = f"remainders_fp32_buffers_{sharding_type}_{fp8_recipe}"
    with TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as ckpt_dir:
        model = _ddp_model()
        _, optimizer = _get_optimizer(
            model, fp8_recipe, store_param_remainders=store_param_remainders
        )
        _train_step(model, optimizer)
        (saving,) = optimizer.chained_optimizers
        assert saving._stores_param_remainders()
        saved_shard = saving.get_main_param_shard(model.module.weight)
        _save_parameter_state(saving, model, sharding_type, ckpt_dir)

        _, optimizer = _get_optimizer(
            model, fp8_recipe, store_param_remainders=store_param_remainders
        )
        (loading,) = optimizer.chained_optimizers
        _load_parameter_state(loading, model, sharding_type, ckpt_dir)
        if saved_shard is not None:
            assert torch.equal(loading.get_main_param_shard(model.module.weight), saved_shard)


def test_fp32_master_weights_are_not_loaded_as_param_remainders(model_parallel, tmp_path_dist_ckpt):
    """The --ckpt-format torch parameter state does not record its master weight format, so FP32
    master weights loaded into a run with param remainders must be rejected, not truncated."""
    with TempNamedDir(tmp_path_dist_ckpt / "fp32_masters_as_remainders", sync=True) as ckpt_dir:
        model = _ddp_model()
        _, optimizer = _get_optimizer(model, None, store_param_remainders=False)
        _train_step(model, optimizer)
        (saving,) = optimizer.chained_optimizers
        assert not saving._stores_param_remainders()
        _save_parameter_state(saving, model, "torch", ckpt_dir)

        _, optimizer = _get_optimizer(model, None)
        (loading,) = optimizer.chained_optimizers
        assert loading._stores_param_remainders()
        with pytest.raises(RuntimeError, match="stores full main params"):
            _load_parameter_state(loading, model, "torch", ckpt_dir)
