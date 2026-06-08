from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2 import (
    FSDP2Config,
    clip_grads_with_sharded_norm_,
    fsdp2_available,
)
from megatron.lite.primitive.optimizers.fsdp2.adamw import build_adamw_optimizer
from megatron.lite.primitive.optimizers.fsdp2.wrap import build_fsdp2_shard_placement_fn
from megatron.lite.primitive.parallel.state import ParallelState


pytestmark = pytest.mark.mlite

fsdp2_wrap = importlib.import_module("megatron.lite.primitive.optimizers.fsdp2.wrap")


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ToyBlock()
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(self.block(x))


class TwoBlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = ToyBlock()
        self.block1 = ToyBlock()
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(self.block1(self.block0(x)))


class NestedToyBlock(ToyBlock):
    def __init__(self):
        super().__init__()
        self.inner = ToyBlock()


def test_fsdp2_config_validates_empty_wrap_surface():
    with pytest.raises(ValueError, match="wrap_root=True"):
        FSDP2Config(wrap_root=False)


@pytest.mark.parametrize("field", ["mesh_dim_name", "device_type"])
def test_fsdp2_config_rejects_empty_names(field: str):
    with pytest.raises(ValueError, match=field):
        FSDP2Config(**{field: ""})


def test_fsdp2_config_normalizes_unit_and_leaf_modules():
    cfg = FSDP2Config(unit_modules=[nn.Linear], leaf_module_names=["embed"])

    assert cfg.unit_modules == (nn.Linear,)
    assert cfg.leaf_module_names == ("embed",)
    assert isinstance(fsdp2_available(), bool)


def test_fsdp2_shard_placement_prefers_first_divisible_dimension():
    placement_for_two = build_fsdp2_shard_placement_fn(2)
    placement_for_three = build_fsdp2_shard_placement_fn(3)

    assert placement_for_two(nn.Parameter(torch.empty(3, 4))).dim == 1
    assert placement_for_three(nn.Parameter(torch.empty(3, 4))).dim == 0


def test_fsdp2_shard_placement_rejects_invalid_group_size():
    with pytest.raises(ValueError, match="positive"):
        build_fsdp2_shard_placement_fn(0)


def test_fsdp2_rejects_invalid_unit_path():
    with pytest.raises(ValueError, match="Invalid FSDP2 unit module path"):
        fsdp2_wrap._resolve_unit_module_types(("Linear",))


def test_fsdp2_rejects_non_module_unit_path():
    with pytest.raises(TypeError, match="does not resolve"):
        fsdp2_wrap._resolve_unit_module_types(("math.sqrt",))


def test_wrap_fsdp2_requires_distributed_when_mesh_is_not_provided(monkeypatch):
    monkeypatch.setattr(
        fsdp2_wrap,
        "_load_fully_shard",
        lambda: lambda module, **kwargs: module,
    )

    with pytest.raises(RuntimeError, match="torch.distributed"):
        fsdp2_wrap.wrap_fsdp2(ToyModel(), ParallelState(), FSDP2Config())


def test_wrap_fsdp2_wraps_units_then_root_and_preserves_param_attrs(monkeypatch):
    model = ToyModel()
    model.block.proj.weight.tensor_model_parallel = True
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        for param in module.parameters():
            vars(param).clear()
        module._fake_fsdp2_kwargs = kwargs
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    result = fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(unit_modules=(ToyBlock,), reshard_after_forward=False),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert result is model
    assert calls == [model.block, model]
    assert model.block.proj.weight.tensor_model_parallel is True
    assert model._fake_fsdp2_kwargs["reshard_after_forward"] is False
    assert model._fake_fsdp2_kwargs["mesh"].name == "mesh"


def test_wrap_fsdp2_accepts_unit_module_import_paths(monkeypatch):
    model = ToyModel()
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(
            unit_modules=("torch.nn.modules.linear.Linear",),
            wrap_root=False,
        ),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert calls == [model.block.proj, model.out]


def test_wrap_fsdp2_uses_container_order_without_nested_unit_duplicates(monkeypatch):
    model = nn.Module()
    model.layers = nn.ModuleDict(
        {
            "10": NestedToyBlock(),
            "2": ToyBlock(),
            "11": ToyBlock(),
        }
    )
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        module._fake_fsdp2_kwargs = kwargs
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(unit_modules=(ToyBlock,), reshard_after_forward=True),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert calls == [
        model.layers["10"],
        model.layers["2"],
        model.layers["11"],
        model,
    ]
    assert model.layers["10"]._fake_fsdp2_kwargs["reshard_after_forward"] is True
    assert not hasattr(model.layers["10"].inner, "_fake_fsdp2_kwargs")
    assert model.layers["11"]._fake_fsdp2_kwargs["reshard_after_forward"] is False


def test_wrap_fsdp2_configures_default_forward_prefetch(monkeypatch):
    model = TwoBlockModel()
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        module._forward_prefetch = None
        module._backward_prefetch = None
        module._fake_fsdp2_kwargs = kwargs

        def set_forward_prefetch(modules, *, _module=module):
            _module._forward_prefetch = list(modules)

        def set_backward_prefetch(modules, *, _module=module):
            _module._backward_prefetch = list(modules)

        module.set_modules_to_forward_prefetch = set_forward_prefetch
        module.set_modules_to_backward_prefetch = set_backward_prefetch
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(
            unit_modules=(ToyBlock,),
            reshard_after_forward=True,
        ),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert calls == [model.block0, model.block1, model]
    assert model.block0._fake_fsdp2_kwargs["reshard_after_forward"] is True
    assert model.block1._fake_fsdp2_kwargs["reshard_after_forward"] is False
    assert model._fake_fsdp2_kwargs["reshard_after_forward"] is False
    assert model._forward_prefetch == [model.block0]
    assert model.block0._forward_prefetch == [model.block1]
    assert model.block1._backward_prefetch is None


def test_wrap_fsdp2_prefetch_depths(monkeypatch):
    model = nn.Sequential(ToyBlock(), ToyBlock(), ToyBlock())

    def fake_fully_shard(module, **kwargs):
        module._forward_prefetch = None
        module._backward_prefetch = None

        def set_forward_prefetch(modules, *, _module=module):
            _module._forward_prefetch = list(modules)

        def set_backward_prefetch(modules, *, _module=module):
            _module._backward_prefetch = list(modules)

        module.set_modules_to_forward_prefetch = set_forward_prefetch
        module.set_modules_to_backward_prefetch = set_backward_prefetch
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(
            unit_modules=(ToyBlock,),
            wrap_root=False,
            forward_prefetch_depth=2,
            backward_prefetch_depth=2,
        ),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert model[0]._forward_prefetch == [model[1], model[2]]
    assert model[1]._forward_prefetch == [model[2]]
    assert model[2]._backward_prefetch == [model[1], model[0]]


def test_clip_grads_with_sharded_norm_scales_cpu_grads_once():
    p0 = nn.Parameter(torch.ones(2))
    p1 = nn.Parameter(torch.ones(2))
    p0.grad = torch.tensor([3.0, 4.0])
    p1.grad = torch.tensor([0.0, 12.0])

    clip_grads_with_sharded_norm_([p0, p1], max_norm=6.5, total_norm=torch.tensor(13.0))

    scale = 6.5 / (13.0 + 1.0e-6)
    torch.testing.assert_close(p0.grad, torch.tensor([3.0, 4.0]) * scale)
    torch.testing.assert_close(p1.grad, torch.tensor([0.0, 12.0]) * scale)


def test_fp32_adamw_state_dict_roundtrip_cpu():
    param = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = build_adamw_optimizer(
        [{"params": [param], "weight_decay": 0.0}],
        all_params=[param],
        lr=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        foreach=False,
        use_fp32_master=True,
        cpu_update=False,
        model_param_dtypes={id(param): torch.bfloat16},
        opt=SimpleNamespace(),
    )
    param.grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    optimizer.step()
    state = optimizer.state_dict()

    loaded_param = nn.Parameter(torch.tensor([9.0, 9.0], dtype=torch.bfloat16))
    loaded_optimizer = build_adamw_optimizer(
        [{"params": [loaded_param], "weight_decay": 0.0}],
        all_params=[loaded_param],
        lr=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        foreach=False,
        use_fp32_master=True,
        cpu_update=False,
        model_param_dtypes={id(loaded_param): torch.bfloat16},
        opt=SimpleNamespace(),
    )
    loaded_optimizer.load_state_dict(state)
    loaded_state = loaded_optimizer.state_dict()

    assert loaded_state["step_count"] == state["step_count"]
    for key in ("master_params", "exp_avgs", "exp_avg_sqs", "steps"):
        assert len(loaded_state[key]) == len(state[key])
    torch.testing.assert_close(loaded_state["master_params"][0], state["master_params"][0])
    torch.testing.assert_close(loaded_state["exp_avgs"][0], state["exp_avgs"][0])
    torch.testing.assert_close(loaded_state["exp_avg_sqs"][0], state["exp_avg_sqs"][0])
    assert loaded_state["steps"] == state["steps"]


@pytest.mark.parametrize("cpu_update", [False, True])
def test_fp32_adamw_load_matches_uninterrupted_next_step_cpu(cpu_update: bool):
    def build(initial_value: torch.Tensor):
        param = nn.Parameter(initial_value.clone().to(dtype=torch.bfloat16))
        optimizer = build_adamw_optimizer(
            [{"params": [param], "weight_decay": 0.0}],
            all_params=[param],
            lr=0.1,
            weight_decay=0.0,
            betas=(0.9, 0.99),
            eps=1.0e-8,
            foreach=False,
            use_fp32_master=True,
            cpu_update=cpu_update,
            model_param_dtypes={id(param): torch.bfloat16},
            opt=SimpleNamespace(),
        )
        return param, optimizer

    initial = torch.tensor([1.0, -2.0], dtype=torch.float32)
    first_grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    second_grad = torch.tensor([-0.125, 0.375], dtype=torch.bfloat16)

    ckpt_param, ckpt_optimizer = build(initial)
    direct_param, direct_optimizer = build(initial)
    loaded_param, loaded_optimizer = build(initial)

    ckpt_param.grad = first_grad.clone()
    ckpt_optimizer.step()
    direct_param.grad = first_grad.clone()
    direct_optimizer.step()

    saved_param = ckpt_param.detach().clone()
    saved_state = copy.deepcopy(ckpt_optimizer.state_dict())

    with torch.no_grad():
        loaded_param.copy_(saved_param)
    loaded_optimizer.load_state_dict(saved_state)

    direct_param.grad = second_grad.clone()
    direct_optimizer.step()
    loaded_param.grad = second_grad.clone()
    loaded_optimizer.step()

    torch.testing.assert_close(loaded_param, direct_param, atol=0.0, rtol=0.0)
    direct_state = direct_optimizer.state_dict()
    loaded_state = loaded_optimizer.state_dict()
    assert loaded_state["step_count"] == direct_state["step_count"]
    for key in ("master_params", "exp_avgs", "exp_avg_sqs"):
        torch.testing.assert_close(loaded_state[key][0], direct_state[key][0], atol=0.0, rtol=0.0)
    assert loaded_state["steps"] == direct_state["steps"]
