# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings

import pytest
import torch

pytest.importorskip("transformer_engine")

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
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
    assert resolved.resolved_fp8_param_gather is False
    assert resolved.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    # FP8 params: MCore-managed masters, precision-aware path is off.
    resolved = config.resolve_fp8_param_gather(True)
    assert resolved.resolved_fp8_param_gather is True
    assert not resolved.use_precision_aware_optimizer_no_fp8_or_ds_fp8


def test_resolve_does_not_mutate_caller_config():
    config = _optimizer_config("mxfp8")
    resolved = config.resolve_fp8_param_gather(False)
    assert resolved is not config
    assert config.resolved_fp8_param_gather is None
    assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8


def test_resolving_again_takes_the_new_ddp_value():
    """The DDP config is the source of truth, so a config already resolved for one model follows
    the DDP config of the next model it is resolved for."""
    config = _optimizer_config("mxfp8").resolve_fp8_param_gather(False)
    assert config.resolve_fp8_param_gather(True).resolved_fp8_param_gather is True
    assert config.resolved_fp8_param_gather is False


# The tests below build the optimizer with get_megatron_optimizer, which resolves fp8_param_gather
# from the model's DDP config. The FP8 recipe only matters to the optimizer through the config, so
# the model runs in BF16.


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


class _MixedPrecisionLinear(torch.nn.Linear):
    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight).float() + self.bias


def _ddp_model(ddp_fp8_param_gather=False, use_distributed_optimizer=True, fp32_bias_value=None):
    if fp32_bias_value is None:
        model = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    else:
        model = _MixedPrecisionLinear(128, 128, dtype=torch.bfloat16, device="cuda")
        model.bias.data = model.bias.data.float().fill_(fp32_bias_value)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        # Keep gradient dtypes uniform for fused clipping with mixed BF16/FP32 parameters.
        DistributedDataParallelConfig(
            use_distributed_optimizer=use_distributed_optimizer, grad_reduce_in_fp32=True
        ),
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
    update_successful, grad_norm, _ = optimizer.step()
    assert update_successful
    assert torch.isfinite(torch.as_tensor(grad_norm)).all()


def _master_state(distrib_optimizer, model_param):
    """The master weight state distrib_optimizer holds for model_param, or None if this rank
    holds no shard of it. Exact, for comparing master weights across checkpoint round trips:

    - MCore-managed: the FP32 main param shard.
    - TE-owned FP32 masters: TE's master_param itself (so tests can modify it in place).
    - TE-owned int16 param remainders: the BF16 param shard bits stacked with the remainders,
      which together make up the FP32 master.
    - TE-owned FP16 masters: the unscaled FP32 values.
    """
    if model_param not in distrib_optimizer.model_param_group_index_map:
        return None
    group_index, group_order = distrib_optimizer.model_param_group_index_map[model_param]
    inner_param = distrib_optimizer.optimizer.param_groups[group_index]["params"][group_order]
    if not distrib_optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
        return inner_param.detach()
    master = distrib_optimizer.optimizer.state[inner_param]["master_param"]
    if master.dtype == torch.int16:
        return torch.stack([inner_param.detach().view(torch.int16), master])
    if master.dtype == torch.float32:
        return master
    return distrib_optimizer.optimizer.get_unscaled_state(inner_param, "master_param")


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

    assert optimizer.config.resolved_fp8_param_gather is ddp_fp8_param_gather
    bf16_params = not ddp_fp8_param_gather
    assert optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 == bf16_params
    # The caller's config is not modified.
    assert config.resolved_fp8_param_gather is None

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
    FP32 by default, and int16 remainders only when explicitly requested."""
    model = _ddp_model()
    _, optimizer = _get_optimizer(model, store_param_remainders=store_param_remainders)
    assert optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8

    _train_step(model, optimizer)

    expected_dtype = torch.int16 if store_param_remainders else torch.float32
    masters = _master_params(optimizer)
    assert masters and all(master.dtype == expected_dtype for master in masters)



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


# (FP8 recipe, main params dtype, store_param_remainders saved with, loaded with, whether the
# checkpoint stores the param remainders flag). The ones without it stand for legacy checkpoints,
# so they are saved the way legacy versions saved them: MCore-managed FP32 main params
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
    ("fp8_recipe", "main_params_dtype", "save_remainders", "load_remainders", "with_flag"),
    _CHECKPOINT_COMPATIBILITY_CASES,
)
def test_param_remainders_checkpoint_compatibility(
    model_parallel,
    tmp_path_dist_ckpt,
    fp8_recipe,
    main_params_dtype,
    save_remainders,
    load_remainders,
    with_flag,
):
    """dist-checkpointing would load int16 param remainders as FP32 master params, or the
    reverse, with a silent value cast that corrupts the master weights, so a format mismatch must
    raise, and matching formats must load the exact master weights."""
    metadata = {"distrib_optim_sharding_type": "dp_reshardable", "checkpoint_version": 3.1}
    name = (
        f"remainders_{fp8_recipe}_{main_params_dtype}_{save_remainders}_{load_remainders}_"
        f"{with_flag}"
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
        if not with_flag:
            del sharded_state_dict["param_state_param_remainders"]
        save({"optimizer": sharded_state_dict}, ckpt_dir)
        saved_shard = _master_state(saving, model.module.weight)
        if saved_shard is not None:
            saved_shard = saved_shard.clone()

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
        loaded_shard = _master_state(loading, model.module.weight)
        if saved_shard is not None:
            if main_params_dtype == torch.float32:
                assert torch.equal(loaded_shard, saved_shard)
            else:
                # TE rescales FP16 master weights from their absmax when setting them.
                torch.testing.assert_close(loaded_shard, saved_shard, rtol=2**-10, atol=0)


def _model_sharded_state_dict(model):
    return {
        name: ShardedTensor.from_rank_offsets(name, param, replica_id=torch.distributed.get_rank())
        for name, param in model.module.named_parameters()
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


def _save_legacy_parameter_state(optimizer, model, sharding_type, ckpt_dir):
    """Like _save_parameter_state, but without the param remainders flag, as checkpoints saved
    before it existed."""
    if sharding_type == "torch":
        state_dict = optimizer.get_parameter_state_dp_zero()
        if state_dict is not None:
            del state_dict["param_state_param_remainders"]
        if optimizer.data_parallel_group.rank() == 0:
            torch.save(state_dict, str(ckpt_dir / "param_state.pt"))
        torch.distributed.barrier()
        return
    metadata = {"distrib_optim_sharding_type": sharding_type, "checkpoint_version": 3.1}
    sharded_state_dict = optimizer.sharded_state_dict(
        _model_sharded_state_dict(model), metadata=metadata
    )
    del sharded_state_dict["param_state_param_remainders"]
    # dp_zero_gather_scatter wraps the DP-rank-0 gathered state, which records the flag too.
    gathered_state = getattr(sharded_state_dict["param_state"], "data", None)
    if isinstance(gathered_state, dict):
        del gathered_state["param_state_param_remainders"]
    save({"optimizer": sharded_state_dict}, ckpt_dir)


@pytest.mark.parametrize(
    "sharding_type", ["torch", "dp_zero_gather_scatter", "fully_reshardable", "dp_reshardable"]
)
@pytest.mark.parametrize("fp8_recipe", ["tensorwise", "mxfp8"])
def test_resume_mcore_managed_checkpoint_with_te_owned_master_weights(
    model_parallel, tmp_path_dist_ckpt, monkeypatch, sharding_type, fp8_recipe
):
    """The upgrade path: before BF16 params could use TE-owned master weights, the
    precision-aware optimizer kept MCore-managed FP32 main params with any non-delayed FP8
    recipe, and its checkpoints have no param remainders flag. Resuming one with BF16 params must
    load the same FP32 master weights and Adam moments into TE FusedAdam, then keep training."""
    name = f"mcore_managed_to_te_owned_{sharding_type}_{fp8_recipe}"
    with TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as ckpt_dir:
        model = _ddp_model()
        with monkeypatch.context() as patch:
            # Left unresolved, the config gates TE ownership as earlier versions did.
            patch.setattr(OptimizerConfig, "resolve_fp8_param_gather", lambda self, _: self)
            _, optimizer = _get_optimizer(model, fp8_recipe)
        (saving,) = optimizer.chained_optimizers
        assert not saving.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        _train_step(model, optimizer)
        _train_step(model, optimizer)
        saved_states = {
            param_name: {
                key: value.detach().clone()
                for key, value in saving._get_main_param_and_optimizer_states(param).items()
            }
            for param_name, param in model.module.named_parameters()
            if param in saving.model_param_group_index_map
        }
        _save_legacy_parameter_state(saving, model, sharding_type, ckpt_dir)

        _, optimizer = _get_optimizer(model, fp8_recipe)
        (loading,) = optimizer.chained_optimizers
        assert loading.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        # The default keeps full FP32 master weights, so the legacy checkpoint loads as is.
        assert not loading._stores_param_remainders()
        _load_parameter_state(loading, model, sharding_type, ckpt_dir)
        for param_name, param in model.module.named_parameters():
            if param_name not in saved_states:
                continue
            loaded_state = loading._get_main_param_and_optimizer_states(param)
            assert set(loaded_state) == set(saved_states[param_name])
            for key, saved_value in saved_states[param_name].items():
                assert loaded_state[key].dtype == saved_value.dtype, key
                assert torch.equal(loaded_state[key], saved_value), key
        _train_step(model, optimizer)


def test_quantized_params_with_bf16_ddp_flags_raise_a_clear_error(model_parallel):
    """Whether TE FusedAdam owns the master weights follows ddp_config.fp8_param_gather. If the
    model holds quantized params while that flag says otherwise, building the optimizer must
    fail clearly instead of giving TE FusedAdam a missing (None) shard that crashes training."""
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8CurrentScaling
    from transformer_engine.pytorch import fp8 as te_fp8

    available, reason = te_fp8.check_fp8_support()
    if not available:
        pytest.skip(reason)
    model_init = getattr(te, "quantized_model_init", None) or te.fp8_model_init
    with model_init(enabled=True, recipe=Float8CurrentScaling()):
        linear = te.Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="cuda")
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True),
        linear,
    )
    assert not model.ddp_config.fp8_param_gather
    with pytest.raises(ValueError, match="needs an MCore-managed FP32 main param"):
        _get_optimizer(model, "tensorwise")


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
        saved_shard = _master_state(saving, model.module.weight)
        if saved_shard is not None:
            saved_shard = saved_shard.clone()
        _save_parameter_state(saving, model, sharding_type, ckpt_dir)

        _, optimizer = _get_optimizer(
            model, fp8_recipe, store_param_remainders=store_param_remainders
        )
        (loading,) = optimizer.chained_optimizers
        _load_parameter_state(loading, model, sharding_type, ckpt_dir)
        if saved_shard is not None:
            assert torch.equal(_master_state(loading, model.module.weight), saved_shard)


@pytest.mark.parametrize(
    "sharding_type", ["torch", "dp_zero_gather_scatter", "fully_reshardable", "dp_reshardable"]
)
@pytest.mark.parametrize("fp32_bias_value", [0.125, 1.0])
def test_param_remainders_with_fp32_model_params(
    model_parallel, tmp_path_dist_ckpt, sharding_type, fp32_bias_value
):
    """FP32 model params keep full master weights alongside BF16 params' int16 remainders."""
    name = f"mixed_dtype_remainders_{sharding_type}_{fp32_bias_value}"
    with TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as ckpt_dir:
        model = _ddp_model(fp32_bias_value=fp32_bias_value)
        _, optimizer = _get_optimizer(model, store_param_remainders=True)
        _train_step(model, optimizer)
        (saving,) = optimizer.chained_optimizers
        assert saving._stores_param_remainders()
        saved_shards = {}
        for name, param in model.module.named_parameters():
            shard = _master_state(saving, param)
            if shard is None:
                continue
            if param.dtype == torch.float32:
                # Cover fractional weights (which must not be rejected) and integer weights
                # (which must not be mistaken for remainders).
                shard.fill_(fp32_bias_value)
            saved_shards[name] = shard.clone()
            state = saving._get_main_param_and_optimizer_states(param)
            expected_dtype = torch.int16 if param.dtype == torch.bfloat16 else torch.float32
            assert state["param"].dtype == expected_dtype
        _save_parameter_state(saving, model, sharding_type, ckpt_dir)

        _, optimizer = _get_optimizer(model, store_param_remainders=True)
        (loading,) = optimizer.chained_optimizers
        _load_parameter_state(loading, model, sharding_type, ckpt_dir)
        for name, param in model.module.named_parameters():
            if name in saved_shards:
                assert torch.equal(_master_state(loading, param), saved_shards[name])


@pytest.mark.parametrize("save_remainders", [False, True])
def test_dp_zero_param_remainders_flag_survives_repeated_loads(model_parallel, save_remainders):
    """Reusing a checkpoint must preserve format checks after both failed and successful loads."""
    model = _ddp_model()
    _, optimizer = _get_optimizer(model, None, store_param_remainders=save_remainders)
    _train_step(model, optimizer)
    (saving,) = optimizer.chained_optimizers
    state_dict = saving.get_parameter_state_dp_zero()
    saved_shard = _master_state(saving, model.module.weight)
    if saved_shard is not None:
        saved_shard = saved_shard.clone()

    _, optimizer = _get_optimizer(model, None, store_param_remainders=not save_remainders)
    (loading,) = optimizer.chained_optimizers
    for _ in range(2):
        with pytest.raises(RuntimeError, match="silently corrupt"):
            loading.load_parameter_state_from_dp_zero(state_dict)
        if state_dict is not None:
            assert state_dict["param_state_param_remainders"] == save_remainders

    _, optimizer = _get_optimizer(model, None, store_param_remainders=save_remainders)
    (loading,) = optimizer.chained_optimizers
    loading.load_state_dict(loading.state_dict())
    for _ in range(2):
        loading.load_parameter_state_from_dp_zero(state_dict)
        if state_dict is not None:
            assert state_dict["param_state_param_remainders"] == save_remainders
        if saved_shard is not None:
            assert torch.equal(_master_state(loading, model.module.weight), saved_shard)


@pytest.mark.parametrize(("save_remainders", "load_remainders"), [(False, None), (None, False)])
def test_torch_format_rejects_mismatched_param_remainders(
    model_parallel, tmp_path_dist_ckpt, save_remainders, load_remainders
):
    """--ckpt-format torch stores the param remainders flag too: a mismatch must raise on every
    rank instead of silently corrupting the master weights."""
    name = f"torch_format_mismatch_{save_remainders}_{load_remainders}"
    with TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as ckpt_dir:
        model = _ddp_model()
        _, optimizer = _get_optimizer(model, None, store_param_remainders=save_remainders)
        _train_step(model, optimizer)
        (saving,) = optimizer.chained_optimizers
        _save_parameter_state(saving, model, "torch", ckpt_dir)

        _, optimizer = _get_optimizer(model, None, store_param_remainders=load_remainders)
        (loading,) = optimizer.chained_optimizers
        assert saving._stores_param_remainders() != loading._stores_param_remainders()
        with pytest.raises(RuntimeError, match="silently corrupt"):
            _load_parameter_state(loading, model, "torch", ckpt_dir)


def test_unrecorded_fp32_master_weights_are_not_loaded_as_param_remainders(
    model_parallel, tmp_path_dist_ckpt
):
    """A legacy --ckpt-format torch parameter state, without the param remainders flag, is assumed
    to match this run's store_param_remainders. FP32 master weights loaded into a run with param
    remainders must then still be rejected, not truncated to int16."""
    with TempNamedDir(tmp_path_dist_ckpt / "fp32_masters_as_remainders", sync=True) as ckpt_dir:
        model = _ddp_model()
        _, optimizer = _get_optimizer(model, None, store_param_remainders=False)
        _train_step(model, optimizer)
        (saving,) = optimizer.chained_optimizers
        assert not saving._stores_param_remainders()
        _save_parameter_state(saving, model, "torch", ckpt_dir)
        if saving.data_parallel_group.rank() == 0:
            state_dict = torch.load(ckpt_dir / "param_state.pt")
            del state_dict["param_state_param_remainders"]
            torch.save(state_dict, ckpt_dir / "param_state.pt")
        torch.distributed.barrier()

        _, optimizer = _get_optimizer(model, None)
        (loading,) = optimizer.chained_optimizers
        assert loading._stores_param_remainders()
        with pytest.raises(RuntimeError, match="stores full main params"):
            _load_parameter_state(loading, model, "torch", ckpt_dir)


@pytest.mark.parametrize(
    ("stored", "loading_config", "fix"),
    [
        (False, dict(fp8_recipe=None), "Set --no-store-param-remainders"),
        (True, dict(fp8_recipe="tensorwise"), "Set --store-param-remainders"),
        (True, dict(fp8_recipe=None, main_params_dtype=torch.float16), "cannot store"),
        (True, dict(fp8_recipe=None, optimizer_cpu_offload=True), "cannot store"),
        (True, dict(fp8_recipe=None, optimizer_cuda_graph=True), "cannot store"),
    ],
)
def test_param_remainders_mismatch_error_names_a_working_fix(stored, loading_config, fix):
    """The error must only suggest --store-param-remainders where this run can store them."""
    if not is_te_min_version("2.1.0"):
        pytest.skip("TE FusedAdam stores param remainders from TE 2.1")
    loading = object.__new__(DistributedOptimizer)
    loading.config = _optimizer_config(**loading_config).resolve_fp8_param_gather(False)
    assert loading._stores_param_remainders() != stored

    error = loading._param_remainders_mismatch(stored)
    assert "silently corrupt" in error and fix in error
