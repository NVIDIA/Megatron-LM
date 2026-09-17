# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU schema fixtures model device metadata; they are not GPU execution evidence."""

import importlib.util
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.determinism import hybrid_state as hybrid
from tools.determinism import training_state as states
from tools.determinism.megatron_state_worker import recipe_arguments


class StreamFixture:
    """Expose completion without permitting a synchronization workaround."""

    def __init__(self):
        self.complete = True

    def query(self):
        return self.complete

    def synchronize(self):
        pytest.fail("Capture must not repair an incomplete native transfer")


def native_hybrid_class():
    path = (
        Path(__file__).resolve().parents[3]
        / "megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py"
    )
    spec = importlib.util.spec_from_file_location("native_hybrid_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HybridDeviceOptimizer


class GPUAdamFixture(torch.optim.Optimizer):
    """Represent the audited TE FP32 layout without a CUDA dependency."""

    def __init__(self, groups):
        super().__init__(groups, {"lr": 0.01, "betas": (0.9, 0.999), "eps": 1e-8})
        for key, value in hybrid.GPU_POLICY.items():
            setattr(self, key, value)
        self.name_to_dtype_map = {
            k: torch.float32 for k in ("exp_avg", "exp_avg_sq", "master_param")
        }
        self.dtype_to_range_map = {
            torch.float16: torch.tensor([32752.0]),
            torch.uint8: torch.tensor([448.0]),
        }
        self._dummy_overflow_buf = torch.zeros(1, dtype=torch.int32)
        self._scales = {}
        self.adam_w_mode, self.set_grad_none = 1, None
        for key in hybrid.GPU_DISPATCH:
            setattr(self, key, torch.add)

    def state_dict(self):
        pytest.fail("Capture must use the raw base serializer")


def fixture(monkeypatch, precision_aware=False):
    model = torch.nn.Module()
    model.weights = torch.nn.ParameterList(
        [
            torch.nn.Parameter(
                torch.arange(4, dtype=torch.float32)
                .add(i)
                .to(torch.bfloat16 if i == 0 else torch.float32)
            )
            for i in range(4)
        ]
    )
    outer_by_model = {}
    for index, parameter in enumerate(model.weights):
        parameter.main_grad = torch.arange(4, dtype=torch.float32).add(index + 1)
        shard = parameter.detach().view(-1)
        if parameter.dtype == torch.bfloat16 and not precision_aware:
            parameter.main_param = shard.float().clone()
            parameter.main_param_sharded = True
            shard = parameter.main_param
        if precision_aware:
            shard.decoupled_grad = parameter.main_grad.view(-1)
        else:
            shard.grad = parameter.main_grad.view(-1)
        outer_by_model[parameter] = shard
    # Match the actual class of reorder: native FP32 shards precede low precision.
    parameters = [outer_by_model[model.weights[i]] for i in (3, 1, 2, 0)]
    cls = native_hybrid_class()
    optimizer = object.__new__(cls)
    options = dict(
        offload_fraction=0.5,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=GPUAdamFixture,
        param_update_in_fp32=True,
        pin_cpu_grads=True,
        pin_cpu_params=True,
        overlap_cpu_optimizer_d2h_h2d=True,
    )
    torch.optim.Optimizer.__init__(optimizer, parameters, options)
    for key, value in options.items():
        setattr(optimizer, key, value)
    optimizer.sub_optimizer_kwargs = {"lr": 0.01, "fused": True}
    optimizer.param_groups[0]["step"] = 3
    optimizer.param_to_inner_param = {}
    optimizer.param_to_fp32_param = {}
    optimizer.gpu_params_map_cpu_copy = {}
    optimizer.cpu_copy_map_grad = defaultdict(dict)
    cpu_storage, pinned_storage = set(), set()

    def mark_cpu(tensor, pinned=False):
        cpu_storage.add(tensor.untyped_storage().data_ptr())
        if pinned:
            pinned_storage.add(tensor.untyped_storage().data_ptr())
        return tensor

    for index, parameter in enumerate(parameters):
        is_cpu = index < 2
        inner = (
            parameter.float().clone() if is_cpu or parameter.dtype != torch.float32 else parameter
        )
        gradient = getattr(parameter, "decoupled_grad", parameter.grad)
        inner.grad = gradient.clone() if is_cpu else gradient
        optimizer.param_to_inner_param[parameter] = inner
        if parameter.dtype != torch.float32:
            optimizer.param_to_fp32_param[parameter] = inner
        if is_cpu:
            optimizer.gpu_params_map_cpu_copy[parameter] = inner
            optimizer.cpu_copy_map_grad[inner] = inner.grad
            mark_cpu(inner, True)
            mark_cpu(inner.grad, True)
    optimizer.inner_param_to_orig_param = {v: k for k, v in optimizer.param_to_inner_param.items()}
    optimizer.cpu_copys_map_gpu_param = {v: k for k, v in optimizer.gpu_params_map_cpu_copy.items()}
    optimizer.fp32_param_to_orig_param = {v: k for k, v in optimizer.param_to_fp32_param.items()}
    optimizer.cpu_param_groups = [{"params": list(optimizer.gpu_params_map_cpu_copy.values())}]
    optimizer.gpu_param_groups = [
        {"params": [optimizer.param_to_inner_param[p] for p in parameters[2:]], "step": 3}
    ]
    optimizer.cpu_optimizers = [
        torch.optim.AdamW([p]) for p in optimizer.cpu_param_groups[0]["params"]
    ]
    optimizer.gpu_optimizer = GPUAdamFixture(optimizer.gpu_param_groups)
    for child in optimizer.cpu_optimizers + [optimizer.gpu_optimizer]:
        cpu = child is not optimizer.gpu_optimizer
        for group in child.param_groups:
            for inner in group["params"]:
                values = {
                    "exp_avg": torch.full_like(inner, 0.5),
                    "exp_avg_sq": torch.full_like(inner, 0.25),
                    "master_param": inner,
                }
                if cpu:
                    values["step"] = torch.tensor(3.0)
                    for tensor in values.values():
                        mark_cpu(tensor, tensor is inner)
                child.state[inner] = optimizer.state[optimizer.inner_param_to_orig_param[inner]] = (
                    values
                )
    for tensor in optimizer.gpu_optimizer.dtype_to_range_map.values():
        mark_cpu(tensor)
    optimizer._d2h_stream, optimizer._h2d_stream = StreamFixture(), StreamFixture()
    optimizer._cpu_optimizer_map_data_event = {}
    optimizer._register_param_copy_back_gpu_hook()
    optimizer._register_load_state_dict_hooks()
    monkeypatch.setattr(torch.cuda, "current_stream", StreamFixture)
    monkeypatch.setattr(
        hybrid,
        "_device_type",
        lambda t: "cpu" if t.untyped_storage().data_ptr() in cpu_storage else "cuda",
    )
    monkeypatch.setattr(
        hybrid, "_is_pinned", lambda t: t.untyped_storage().data_ptr() in pinned_storage
    )
    config = SimpleNamespace(
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=0.5,
        overlap_cpu_optimizer_d2h_h2d=True,
        pin_cpu_params=True,
        pin_cpu_grads=True,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=precision_aware,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    wrapper = SimpleNamespace(
        optimizer=optimizer,
        config=config,
        grad_scaler=None,
        get_loss_scale=lambda: torch.tensor([1.0]),
        model_param_group_index_map={p: (0, i) for i, p in enumerate(model.weights)},
        _get_model_param_range_map=lambda p: {"param": SimpleNamespace(start=0, end=p.numel())},
    )
    return model, wrapper


@pytest.mark.parametrize("precision_aware", [False, True])
def test_complete_native_hybrid_state_and_actual_reordered_storage(monkeypatch, precision_aware):
    model, wrapper = fixture(monkeypatch, precision_aware)
    state, precision = hybrid.capture_hybrid_adam_state(wrapper, [model])
    assert state["owners"] == {0: "cpu:0", 1: "cpu:1", 2: "gpu", 3: "gpu"}
    assert [state["model_shards"][i]["parameter"] for i in range(4)] == [
        "weights.3",
        "weights.1",
        "weights.2",
        "weights.0",
    ]
    assert state["children"]["gpu"]["state"]["param_groups"][0]["step"] == 3
    for i, p in enumerate(wrapper.optimizer.param_groups[0]["params"]):
        assert state["state"]["state"][i]["exp_avg"] is wrapper.optimizer.state[p]["exp_avg"]
        assert (
            precision["parameter_shards"][i]["inner"] is wrapper.optimizer.param_to_inner_param[p]
        )
    assert "differentiable" not in wrapper.optimizer.defaults
    assert state["defaults"]["differentiable"] is False
    assert state["tensor_layout"]["tensors"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_moment",
        "missing_cpu_counter",
        "missing_gpu_counter",
        "wrong_outer_counter",
        "missing_owner",
        "wrong_reverse",
        "duplicate_inner",
        "missing_gradient_buffer",
        "stale_gradient",
        "stale_cpu_parameter",
        "extra_state",
        "copied_master",
        "wrong_moment_dtype",
        "missing_load_hook",
        "extra_hook",
        "spoofed_hook",
        "wrong_hook_owner",
        "pending_event",
        "pending_d2h",
        "pending_h2d",
        "same_stream",
        "missing_model",
        "wrong_shard_range",
        "copied_main_parameter",
        "copied_main_gradient",
        "unknown_attribute",
        "unknown_child_attribute",
        "true_differentiable",
        "numeric_differentiable",
        "wrong_fraction",
        "low_precision_moments",
        "missing_gpu_scale_contract",
        "constructor_membership",
    ],
)
def test_incomplete_or_unsupported_hybrid_state_is_unverified(monkeypatch, change):
    model, wrapper = fixture(monkeypatch)
    inner = wrapper.optimizer
    p = inner.param_groups[0]["params"][0]
    cpu = inner.param_to_inner_param[p]
    if change == "missing_moment":
        del inner.state[p]["exp_avg_sq"]
    elif change == "missing_cpu_counter":
        del inner.state[p]["step"]
    elif change == "missing_gpu_counter":
        del inner.gpu_optimizer.param_groups[0]["step"]
    elif change == "wrong_outer_counter":
        inner.param_groups[0]["step"] = 1
    elif change == "missing_owner":
        del inner.param_to_inner_param[p]
    elif change == "wrong_reverse":
        inner.inner_param_to_orig_param[cpu] = inner.param_groups[0]["params"][1]
    elif change == "duplicate_inner":
        inner.param_to_inner_param[inner.param_groups[0]["params"][1]] = cpu
    elif change == "missing_gradient_buffer":
        del inner.cpu_copy_map_grad[cpu]
    elif change == "stale_gradient":
        cpu.grad[0] += 1
    elif change == "stale_cpu_parameter":
        cpu[0] += 1
    elif change == "extra_state":
        inner.state[p]["unknown"] = torch.ones(1)
    elif change == "copied_master":
        inner.state[p]["master_param"] = cpu.clone()
    elif change == "wrong_moment_dtype":
        inner.state[p]["exp_avg"] = inner.state[p]["exp_avg"].half()
    elif change == "missing_load_hook":
        inner._optimizer_load_state_dict_post_hooks.clear()
    elif change == "extra_hook":
        inner.register_step_pre_hook(lambda *_: None)
    elif change == "spoofed_hook":
        hook = next(iter(inner._optimizer_load_state_dict_post_hooks.values()))
        replacement = lambda *_: None
        replacement.__qualname__ = hook.__qualname__
        inner._optimizer_load_state_dict_post_hooks[
            next(iter(inner._optimizer_load_state_dict_post_hooks))
        ] = replacement
    elif change == "wrong_hook_owner":
        other = object.__new__(type(inner))
        other.cpu_optimizers = [torch.optim.AdamW([cpu.clone()])]
        other.gpu_optimizer = None
        other.param_update_in_fp32 = True
        other._register_param_copy_back_gpu_hook()
        inner.cpu_optimizers[0]._optimizer_step_post_hooks = other.cpu_optimizers[
            0
        ]._optimizer_step_post_hooks
    elif change == "pending_event":
        inner._cpu_optimizer_map_data_event[inner.cpu_optimizers[0]] = object()
    elif change == "pending_d2h":
        inner._d2h_stream.complete = False
    elif change == "pending_h2d":
        inner._h2d_stream.complete = False
    elif change == "same_stream":
        inner._h2d_stream = inner._d2h_stream
    elif change == "missing_model":
        del wrapper.model_param_group_index_map[model.weights[0]]
    elif change == "wrong_shard_range":
        wrapper._get_model_param_range_map = lambda p: {
            "param": SimpleNamespace(start=1, end=p.numel())
        }
    elif change == "copied_main_parameter":
        model.weights[0].main_param = model.weights[0].main_param.clone()
    elif change == "copied_main_gradient":
        model.weights[0].main_grad = model.weights[0].main_grad.clone()
    elif change == "unknown_attribute":
        inner.unknown_state = torch.ones(1)
    elif change == "unknown_child_attribute":
        inner.gpu_optimizer.unknown_state = torch.ones(1)
    elif change == "true_differentiable":
        inner.defaults["differentiable"] = True
    elif change == "numeric_differentiable":
        inner.defaults["differentiable"] = 0
    elif change == "wrong_fraction":
        wrapper.config.optimizer_offload_fraction = 1.0
    elif change == "low_precision_moments":
        inner.gpu_optimizer.exp_avg_dtype = torch.float16
    elif change == "missing_gpu_scale_contract":
        inner.gpu_optimizer._scales[p] = {}
    elif change == "constructor_membership":
        inner.cpu_param_groups[0]["params"] = []
    with pytest.raises(states.UnverifiedState):
        hybrid.capture_hybrid_adam_state(wrapper, [model])


@pytest.mark.parametrize(
    "component", ["exp_avg", "exp_avg_sq", "cpu_step", "gpu_step", "gradient", "parameter"]
)
def test_hybrid_changes_break_complete_byte_comparison(monkeypatch, tmp_path, component):
    model, wrapper = fixture(monkeypatch)
    inner = wrapper.optimizer
    p = inner.param_groups[0]["params"][0]
    provenance = {
        "source_revision": "cpu-fixture",
        "source_dirty": False,
        "recipe": {"name": "hybrid-contract"},
        "hardware": {"device": "cpu"},
        "software": {"torch": str(torch.__version__)},
    }
    for index, name in enumerate(("reference", "changed")):
        if index:
            if component in ("exp_avg", "exp_avg_sq"):
                inner.state[p][component].view(torch.uint8)[0] ^= 1
            elif component == "cpu_step":
                inner.state[p]["step"].add_(1)
            elif component == "gpu_step":
                inner.param_groups[0]["step"] += 1
                inner.gpu_optimizer.param_groups[0]["step"] += 1
            elif component == "gradient":
                p.grad[0] += 1
                inner.param_to_inner_param[p].grad.copy_(p.grad)
            else:
                p[0] += 1
                inner.param_to_inner_param[p].copy_(p)
        optimizer, precision = hybrid.capture_hybrid_adam_state(wrapper, [model])
        snapshot = {
            "model": model.state_dict(),
            "gradients": {str(i): p.main_grad for i, p in enumerate(model.weights)},
            "optimizer": optimizer,
            "precision": precision,
            "rng": {"state": [1]},
            "scheduler": {"step": 3},
            "dataloader": {"cursor": 12},
        }
        monkeypatch.setattr(states, "PROCESS_ID", f"fixture-{index}")
        states.write_snapshot(
            tmp_path / name,
            snapshot,
            step=3,
            rank=0,
            world_size=1,
            run_id=name,
            provenance=provenance,
        )
        states.complete_rank(
            tmp_path / name, rank=0, world_size=1, run_id=name, steps=[3], provenance=provenance
        )
    result = states.compare_runs(
        tmp_path / "reference", tmp_path / "changed", steps=[3], world_size=1, comparison="fresh"
    )
    assert result["status"] == "different"


@pytest.mark.parametrize("mode", ["hybrid_fp32", "hybrid_precision_aware_fp32"])
def test_hybrid_recipe_preserves_horizon_native_backends_and_overlap(mode):
    command = recipe_arguments(8, 5, 3, optimizer_mode=mode)
    for option, value in (
        ("--train-iters", "5"),
        ("--lr-decay-iters", "5"),
        ("--exit-interval", "3"),
        ("--optimizer-offload-fraction", "0.5"),
        ("--exp-avg-dtype", "fp32"),
        ("--exp-avg-sq-dtype", "fp32"),
    ):
        assert command[command.index(option) + 1] == value
    assert "--optimizer-cpu-offload" in command and "--overlap-cpu-optimizer-d2h-h2d" in command
    assert ("--use-precision-aware-optimizer" in command) == (mode == "hybrid_precision_aware_fp32")
    assert "--use-torch-optimizer-for-cpu-offload" not in command


def test_restore_default_is_explicit_and_only_missing_false_is_equivalent(monkeypatch):
    model, wrapper = fixture(monkeypatch)
    reference, _ = hybrid.capture_hybrid_adam_state(wrapper, [model])
    wrapper.optimizer.defaults["differentiable"] = False
    restored, _ = hybrid.capture_hybrid_adam_state(wrapper, [model])
    assert reference["defaults"] == restored["defaults"]
    assert reference["schema"] == restored["schema"]
    assert reference["tensor_layout"] == restored["tensor_layout"]
    assert wrapper.optimizer.defaults["differentiable"] is False
