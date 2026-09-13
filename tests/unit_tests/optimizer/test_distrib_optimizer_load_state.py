# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.optimizer import (
    OptimizerConfig,
    _get_megatron_optimizer_based_on_param_groups,
    distrib_optimizer,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


def _wrap_inner_optimizer(optimizer, **config_overrides):
    """Supply only the shard metadata consumed by the real metadata loader."""
    wrapper = object.__new__(DistributedOptimizer)
    wrapper.optimizer = optimizer
    wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    wrapper.config = SimpleNamespace(
        optimizer="adam",
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        main_params_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        store_param_remainders=False,
        bf16=False,
        fp16=False,
    )
    for name, value in config_overrides.items():
        setattr(wrapper.config, name, value)
    wrapper.init_state_fn = None
    wrapper.grad_scaler = None
    wrapper.model_param_group_index_map = {}
    param_map = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for group_order, param in enumerate(group["params"]):
            wrapper.model_param_group_index_map[param] = (group_index, group_order)
            param_map[param] = {"gbuf_world": range(param.numel())}
    wrapper.gbuf_ranges = [{"buffer": [{"param_map": param_map}]}]
    return wrapper


def test_native_adam_restore_preserves_load_hooks_and_next_step(monkeypatch):
    """The TE fast path must not change the native optimizer's loader behavior."""
    monkeypatch.setattr(distrib_optimizer, "HAVE_APEX_OR_TE", False)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("optimizer placeholders must not allocate CUDA storage"),
    )
    param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    inner = torch.optim.Adam([param], lr=0.1)
    inner.state[param] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(param, 0.2),
        "exp_avg_sq": torch.full_like(param, 0.3),
    }
    wrapper = _wrap_inner_optimizer(inner)
    checkpoint_group = deepcopy(inner.state_dict()["param_groups"][0])
    checkpoint_group.pop("params")
    checkpoint_group.update(lr=0.025, step=3)
    observed = []

    def before_load(optimizer, state_dict):
        observed.append("pre")
        for state in state_dict["state"].values():
            assert state["exp_avg"].device.type == "cpu"
            assert state["exp_avg_sq"].device.type == "cpu"
            state["exp_avg"].fill_(0.2)
            state["exp_avg_sq"].fill_(0.3)

    def after_load(optimizer):
        observed.append("post")
        assert optimizer.param_groups[0]["lr"] == 0.025
        assert optimizer.state[param]["step"].item() == 3

    inner.register_load_state_dict_pre_hook(before_load)
    inner.register_load_state_dict_post_hook(after_load)
    wrapper.load_state_dict({"optimizer": {"param_groups": [checkpoint_group]}})
    assert observed == ["pre", "post"]

    reference_param = torch.nn.Parameter(param.detach().clone())
    reference = torch.optim.Adam([reference_param], lr=0.025)
    reference.state[reference_param] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(reference_param, 0.2),
        "exp_avg_sq": torch.full_like(reference_param, 0.3),
    }
    param.grad = torch.linspace(-0.4, 0.3, 8)
    reference_param.grad = param.grad.clone()
    inner.step()
    reference.step()
    torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            inner.state[param][key], reference.state[reference_param][key], rtol=0, atol=0
        )


def _make_te_wrapper(
    store_param_remainders, param_dtypes=(torch.bfloat16, torch.float32), **overrides
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and Transformer Engine")
    te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")
    params = [
        torch.nn.Parameter(torch.linspace(-0.5, 0.5, 128, device="cuda", dtype=dtype))
        for dtype in param_dtypes
    ]
    config = OptimizerConfig(
        optimizer="adam",
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        store_param_remainders=store_param_remainders,
        **overrides,
    )

    def wrap(inner, config, grad_scaler, initialize, **kwargs):
        wrapper = _wrap_inner_optimizer(inner)
        wrapper.config = config
        wrapper.init_state_fn = initialize
        return wrapper

    # Isolate the metadata loader from distributed-buffer setup. The round-trip test
    # below exercises the real DistributedOptimizer constructor and buffers as well.
    with patch("megatron.core.optimizer.DistributedOptimizer", side_effect=wrap):
        wrapper = _get_megatron_optimizer_based_on_param_groups(
            config,
            model_chunks=[torch.nn.Module()],
            param_groups=[
                {
                    "params": params,
                    "wd_mult": 1.0,
                    "param_names": [f"p{i}" for i in range(len(params))],
                },
                {"params": [], "wd_mult": 0.0, "param_names": []},
            ],
            pg_collection=SimpleNamespace(tp=None, expt_tp=None),
        )
    assert type(wrapper.optimizer) is te_optimizers.FusedAdam
    return wrapper, params


def _step_te(inner, params, step):
    for param in params:
        param.decoupled_grad = (
            torch.linspace(-0.2, 0.3, param.numel(), device=param.device) + step * 0.01
        )
    inner.step()


@pytest.mark.parametrize("store_param_remainders", [False, True])
def test_precision_aware_resume_reuses_live_state_and_preserves_next_update(store_param_remainders):
    """An initialized resume must not replace TE storage or alter the next Adam update."""
    resumed, params = _make_te_wrapper(store_param_remainders)
    reference, reference_params = _make_te_wrapper(store_param_remainders)
    for step in range(3):
        _step_te(resumed.optimizer, params, step)
        _step_te(reference.optimizer, reference_params, step)
    metadata = resumed.state_dict()
    metadata["optimizer"]["param_groups"].reverse()
    for group in metadata["optimizer"]["param_groups"]:
        group.pop("param_names")
    for group in resumed.optimizer.param_groups:
        group.update(lr=0.2, betas=(0.5, 0.9), step=99)
    before = {p: dict(resumed.optimizer.state[p]) for p in params}
    for _ in range(2):
        resumed.load_state_dict(deepcopy(metadata))
        for param in params:
            for key, tensor in before[param].items():
                assert (
                    resumed.optimizer.state[param][key] is tensor
                ), f"resume replaced {key} storage"
    assert resumed.optimizer.param_groups[0]["param_names"] == ["p0", "p1"]
    assert resumed.optimizer.param_groups[1]["params"] == []
    assert resumed.optimizer.param_groups[1]["step"] == 3
    _step_te(resumed.optimizer, params, 3)
    _step_te(reference.optimizer, reference_params, 3)
    for param, reference_param in zip(params, reference_params):
        torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
        for key, tensor in resumed.optimizer.state[param].items():
            torch.testing.assert_close(
                tensor, reference.optimizer.state[reference_param][key], rtol=0, atol=0
            )


@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("store_param_remainders", [False, True])
def test_precision_aware_initializer_uses_effective_remainder_dtype(
    param_dtype, store_param_remainders
):
    """The production initializer must match TE's parameter-specific master representation."""
    wrapper, (param,) = _make_te_wrapper(store_param_remainders, param_dtypes=(param_dtype,))
    wrapper.init_state_fn(wrapper.optimizer, wrapper.config)
    state = wrapper.optimizer.state[param]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    expected = (
        torch.int16 if store_param_remainders and param_dtype == torch.bfloat16 else torch.float32
    )
    assert state["master_param"].dtype == expected
    if expected == torch.float32:
        torch.testing.assert_close(state["master_param"], param.float(), rtol=0, atol=0)
    else:
        assert torch.count_nonzero(state["master_param"]).item() == 0


@pytest.mark.parametrize("hook_kind", ["pre", "post"])
def test_precision_aware_restore_honors_registered_load_hooks(hook_kind):
    """Registered public load hooks must still receive and influence the real restore."""
    wrapper, params = _make_te_wrapper(False)
    _step_te(wrapper.optimizer, params, 0)
    metadata = wrapper.state_dict()
    calls = []

    def pre_hook(optimizer, state_dict):
        calls.append("pre")
        replacement = deepcopy(state_dict)
        replacement["param_groups"][0]["lr"] = 0.0125
        return replacement

    def post_hook(optimizer):
        calls.append("post")
        optimizer.param_groups[0]["lr"] = 0.0125

    register = getattr(wrapper.optimizer, f"register_load_state_dict_{hook_kind}_hook")
    register(pre_hook if hook_kind == "pre" else post_hook)
    wrapper.load_state_dict(metadata)
    assert calls == [hook_kind]
    assert wrapper.optimizer.param_groups[0]["lr"] == 0.0125


@pytest.mark.parametrize(
    ("state_name", "state_dtype"),
    [
        (name, dtype)
        for name in ("exp_avg_dtype", "exp_avg_sq_dtype")
        for dtype in (torch.float16, torch.bfloat16, torch.uint8)
    ]
    + [("main_params_dtype", torch.float16)],
)
def test_reduced_precision_restore_matches_public_te_loader(state_name, state_dtype):
    """Scaled representations must retain the public loader's conversion behavior."""
    wrapper, params = _make_te_wrapper(False, **{state_name: state_dtype})
    reference, reference_params = _make_te_wrapper(False, **{state_name: state_dtype})
    for step in range(3):
        _step_te(wrapper.optimizer, params, step)
        _step_te(reference.optimizer, reference_params, step)
    metadata = wrapper.state_dict()
    reference.optimizer.load_state_dict(reference.optimizer.state_dict())
    with patch.object(
        wrapper.optimizer, "load_state_dict", wraps=wrapper.optimizer.load_state_dict
    ) as load:
        wrapper.load_state_dict(metadata)
        load.assert_called_once()
    _step_te(wrapper.optimizer, params, 3)
    _step_te(reference.optimizer, reference_params, 3)
    for param, ref_param in zip(params, reference_params):
        torch.testing.assert_close(param, ref_param, rtol=0, atol=0)
        for key in wrapper.optimizer.state[param]:
            torch.testing.assert_close(
                wrapper.optimizer.get_unscaled_state(param, key),
                reference.optimizer.get_unscaled_state(ref_param, key),
                rtol=0,
                atol=0,
            )


def test_optimizer_subclass_keeps_custom_load_behavior():
    wrapper, params = _make_te_wrapper(False)
    _step_te(wrapper.optimizer, params, 0)
    calls = []

    class CustomAdam(type(wrapper.optimizer)):
        def load_state_dict(self, state):
            calls.append("custom")
            super().load_state_dict(state)
            self.param_groups[0]["lr"] = 0.0125

    wrapper.optimizer.__class__ = CustomAdam
    wrapper.load_state_dict(wrapper.state_dict())
    assert calls == ["custom"]
    assert wrapper.optimizer.param_groups[0]["lr"] == 0.0125


def test_initializer_supports_legacy_te_one_argument_api(monkeypatch):
    """Exercise the production callback with TE's pre-2.1 call signature."""
    wrapper, params = _make_te_wrapper(False)
    modern_initialize = wrapper.optimizer.initialize_state
    calls = []

    def legacy_initialize(param):
        calls.append(param)
        modern_initialize(param, False)

    monkeypatch.setattr("megatron.core.optimizer.is_te_min_version", lambda version: False)
    monkeypatch.setattr(wrapper.optimizer, "initialize_state", legacy_initialize)
    wrapper.init_state_fn(wrapper.optimizer, wrapper.config)
    assert len(calls) == len(params)
    for param in params:
        torch.testing.assert_close(
            wrapper.optimizer.state[param]["master_param"], param.float(), rtol=0, atol=0
        )


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize(
    ("sharding_type", "store_param_remainders"),
    [("fully_reshardable", False), ("dp_reshardable", False), ("dp_reshardable", True)],
)
def test_precision_aware_dcp_round_trip(
    tmp_path_dist_ckpt, dp_size, store_param_remainders, sharding_type
):
    """A real sharded restore must preserve every Adam tensor and the next model update."""
    # fully_reshardable currently coalesces states into FP32 buffers and cannot
    # restore int16 master remainders, independently of the metadata loader.
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and Transformer Engine")
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing import ShardedTensor, load, save
    from megatron.core.dist_checkpointing.validation import StrictHandling
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import ChainedOptimizer, get_megatron_optimizer
    from megatron.core.transformer import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    world = Utils.world_size
    if world % dp_size:
        pytest.skip("test requires a world size divisible by the data parallel size")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=world // dp_size, pipeline_model_parallel_size=1
    )

    class TinyMixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TransformerConfig(
                num_layers=1, hidden_size=4, num_attention_heads=1, bf16=True
            )
            self.weight = torch.nn.Parameter(
                torch.full((32, 32), 0.5, device="cuda", dtype=torch.bfloat16)
            )
            self.bias = torch.nn.Parameter(
                torch.full((512,), 0.25, device="cuda", dtype=torch.float32)
            )

        def forward(self):
            return self.weight.float().sum() * 0.125 + self.bias.sum() * 0.25

        def sharded_state_dict(self):
            replica_id = (
                parallel_state.get_pipeline_model_parallel_rank(),
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_data_parallel_rank(with_context_parallel=True),
            )
            return {
                key: ShardedTensor.from_rank_offsets(key, value, replica_id=replica_id)
                for key, value in self.state_dict(keep_vars=True).items()
            }

    def make_optimizer():
        raw = TinyMixedModel()
        ddp = DistributedDataParallel(
            raw.config,
            DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True),
            raw,
        )
        config = OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            store_param_remainders=store_param_remainders,
        )
        outer = get_megatron_optimizer(config, [ddp])
        child = outer.chained_optimizers[0] if isinstance(outer, ChainedOptimizer) else outer
        assert isinstance(child, DistributedOptimizer)
        return raw, ddp, outer, child

    def take_step(ddp, optimizer):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        ddp().backward()
        ddp.finish_grad_sync()
        assert optimizer.step()[0]

    def assert_equal_states(left, right):
        for left_group, right_group in zip(
            left.optimizer.param_groups, right.optimizer.param_groups
        ):
            for key in ("lr", "step", "betas", "eps", "weight_decay"):
                assert left_group[key] == right_group[key]
            for left_param, right_param in zip(left_group["params"], right_group["params"]):
                for key in ("exp_avg", "exp_avg_sq", "master_param"):
                    a = left.optimizer.get_unscaled_state(left_param, key)
                    b = right.optimizer.get_unscaled_state(right_param, key)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    if key == "master_param":
                        expected = (
                            torch.int16
                            if store_param_remainders and right_param.dtype == torch.bfloat16
                            else torch.float32
                        )
                        assert b.dtype == expected

    try:
        raw_a, ddp_a, optim_a, inner_a = make_optimizer()
        take_step(ddp_a, optim_a)
        take_step(ddp_a, optim_a)
        checkpoint = (
            tmp_path_dist_ckpt
            / f"resume-{sharding_type}-dp{dp_size}-remainder{store_param_remainders}"
        )
        checkpoint.mkdir(exist_ok=True)
        metadata = {"distrib_optim_sharding_type": sharding_type}
        save(optim_a.sharded_state_dict(raw_a.sharded_state_dict(), metadata=metadata), checkpoint)

        raw_b, ddp_b, optim_b, inner_b = make_optimizer()
        assert not inner_b.optimizer.state
        with torch.no_grad():
            for a, b in zip(raw_a.parameters(), raw_b.parameters()):
                b.copy_(a)
        target = optim_b.sharded_state_dict(
            raw_b.sharded_state_dict(), metadata=metadata, is_loading=True
        )
        pointers = {
            p: {k: v.data_ptr() for k, v in state.items()}
            for p, state in inner_b.optimizer.state.items()
        }
        loaded, missing, unexpected = load(target, checkpoint, strict=StrictHandling.RETURN_ALL)
        assert not missing and not unexpected
        for param, state in inner_b.optimizer.state.items():
            assert {key: value.data_ptr() for key, value in state.items()} == pointers[param]
        optim_b.load_state_dict(loaded)
        for param, state in inner_b.optimizer.state.items():
            assert {key: value.data_ptr() for key, value in state.items()} == pointers[param]
        assert_equal_states(inner_a, inner_b)

        take_step(ddp_a, optim_a)
        take_step(ddp_b, optim_b)
        for a, b in zip(raw_a.parameters(), raw_b.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert_equal_states(inner_a, inner_b)
    finally:
        Utils.destroy_model_parallel()
