from __future__ import annotations

import copy
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from megatron.lite.primitive.ckpt import save_training_checkpoint
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import ParallelConfig
from megatron.lite.runtime.contracts.handle import ModelHandle


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _clone_model_and_optimizer(model: nn.Module):
    clone = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(clone.parameters(), lr=1.0e-3, weight_decay=0.0)
    return clone, optimizer


def _assert_model_close(lhs: nn.Module, rhs: nn.Module):
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(
        lhs.named_parameters(),
        rhs.named_parameters(),
        strict=True,
    ):
        assert lhs_name == rhs_name
        torch.testing.assert_close(lhs_param, rhs_param, atol=0.0, rtol=0.0)


def test_runtime_local_checkpoint_load_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(2029)
    base = TinyMLP()
    ckpt_model, ckpt_optimizer = _clone_model_and_optimizer(base)
    direct_model, direct_optimizer = _clone_model_and_optimizer(base)
    loaded_model, loaded_optimizer = _clone_model_and_optimizer(base)
    x0, y0 = torch.randn(3, 4), torch.randn(3, 2)
    x1, y1 = torch.randn(3, 4), torch.randn(3, 2)

    _step(ckpt_model, ckpt_optimizer, x0, y0)
    _step(direct_model, direct_optimizer, x0, y0)

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(
        ModelHandle(model=ckpt_model, optimizer=ckpt_optimizer),
        str(tmp_path),
        step=1,
    )

    assert runtime.load_checkpoint(
        ModelHandle(model=loaded_model, optimizer=loaded_optimizer),
        str(tmp_path),
    ) == 1

    _step(direct_model, direct_optimizer, x1, y1)
    _step(loaded_model, loaded_optimizer, x1, y1)

    _assert_model_close(direct_model, loaded_model)


class DistOptLike:
    """Small optimizer wrapper with the same checkpoint contract as distopt."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.load_calls = 0
        self.parameter_save_calls = 0
        self.parameter_load_calls = 0
        self.update_legacy_format = None

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()
        return True, 0.0, 0

    def state_dict(self):
        state = self.optimizer.state_dict()
        state["distopt_like_marker"] = {"load_calls": self.load_calls}
        return state

    def load_state_dict(self, state):
        marker = state.pop("distopt_like_marker")
        self.load_calls = int(marker["load_calls"]) + 1
        self.optimizer.load_state_dict(state)

    def save_parameter_state(self, filename: str):
        self.parameter_save_calls += 1
        torch.save({"parameter_save_calls": self.parameter_save_calls}, filename)

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        state = torch.load(filename, weights_only=False)
        self.parameter_load_calls = int(state["parameter_save_calls"])
        self.update_legacy_format = update_legacy_format


def test_runtime_local_checkpoint_uses_optimizer_parameter_state_contract(tmp_path):
    torch.manual_seed(2030)
    model = TinyMLP()
    optimizer = DistOptLike(torch.optim.AdamW(model.parameters(), lr=1.0e-3))
    x, y = torch.randn(3, 4), torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(
        ModelHandle(model=model, optimizer=optimizer),
        str(tmp_path),
        step=7,
    )

    loaded_model = TinyMLP()
    loaded_optimizer = DistOptLike(torch.optim.AdamW(loaded_model.parameters(), lr=1.0e-3))

    assert runtime.load_checkpoint(
        ModelHandle(model=loaded_model, optimizer=loaded_optimizer),
        str(tmp_path),
        update_legacy_format=True,
    ) == 7
    assert loaded_optimizer.load_calls == 1
    assert loaded_optimizer.parameter_load_calls == 1
    assert loaded_optimizer.update_legacy_format is True
    assert (tmp_path / "training_state.optimizer_parameter_state.pt").exists()
    _assert_model_close(model, loaded_model)


def test_runtime_local_checkpoint_restores_rng_state(tmp_path):
    model = TinyMLP()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    random.seed(2031)
    np.random.seed(2031)
    torch.manual_seed(2031)

    runtime.save_checkpoint(
        ModelHandle(model=model, optimizer=None),
        str(tmp_path),
        step=9,
    )

    expected_python = random.random()
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)

    assert runtime.load_checkpoint(ModelHandle(model=model, optimizer=None), str(tmp_path)) == 9
    assert random.random() == expected_python
    np.testing.assert_allclose(np.random.random(4), expected_numpy, atol=0.0, rtol=0.0)
    torch.testing.assert_close(torch.rand(4), expected_torch, atol=0.0, rtol=0.0)


def test_runtime_local_checkpoint_uses_rank_specific_files_when_distributed(tmp_path):
    model = TinyMLP()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    with (
        patch("megatron.lite.primitive.ckpt.dcp.dist.is_available", return_value=True),
        patch("megatron.lite.primitive.ckpt.dcp.dist.is_initialized", return_value=True),
        patch("megatron.lite.primitive.ckpt.dcp.dist.get_rank", return_value=3),
    ):
        runtime.save_checkpoint(ModelHandle(model=model, optimizer=None), str(tmp_path), step=11)
        assert (tmp_path / "training_state_rank_00003.pt").exists()
        assert not (tmp_path / "training_state.pt").exists()
        assert runtime.load_checkpoint(ModelHandle(model=model, optimizer=None), str(tmp_path)) == 11


def test_primitive_auto_dcp_keeps_optimizer_checkpoints_local(tmp_path):
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    parallel = ParallelConfig(tp=1, ep=1, pp=1, cp=1)

    with patch("megatron.lite.primitive.ckpt.dcp.dcp.save") as dcp_save_mock:
        save_training_checkpoint(
            model,
            optimizer,
            12,
            str(tmp_path),
            parallel,
            object(),
        )

    dcp_save_mock.assert_not_called()
    assert (tmp_path / "training_state.pt").exists()


def test_primitive_explicit_dcp_rejects_optimizer_state(tmp_path):
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    parallel = ParallelConfig(tp=1, ep=1, pp=1, cp=1)

    try:
        save_training_checkpoint(
            model,
            optimizer,
            12,
            str(tmp_path),
            parallel,
            object(),
            use_dcp=True,
        )
    except NotImplementedError as exc:
        assert "optimizer state" in str(exc)
    else:
        raise AssertionError("explicit DCP optimizer checkpointing should be rejected")


def test_runtime_dcp_checkpoint_threads_parallel_config_and_protocol_hooks(tmp_path):
    model = TinyMLP()
    parallel = ParallelConfig(tp=2, ep=1, pp=1, cp=1)
    ps = object()

    def placement_fn(name: str):
        return ["placement", name]

    def expert_classifier(name: str):
        return name.endswith("expert")

    proto = SimpleNamespace(
        PLACEMENT_FN=placement_fn,
        EXPERT_CLASSIFIER=expert_classifier,
    )
    handle = ModelHandle(
        model=[model],
        optimizer=None,
        parallel_state=ps,
        config=SimpleNamespace(parallel=parallel),
        _extras={"model_chunks": [model], "protocol": proto},
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    with patch("megatron.lite.primitive.ckpt.save_training_checkpoint") as save_mock:
        runtime.save_checkpoint(handle, str(tmp_path), global_step=13, use_dcp=True)

    save_args = save_mock.call_args.args
    save_kwargs = save_mock.call_args.kwargs
    assert isinstance(save_args[0], nn.ModuleList)
    assert save_args[0][0] is model
    assert save_args[2] == 13
    assert save_args[3] == str(tmp_path)
    assert save_args[4] is parallel
    assert save_args[5] is ps
    assert save_kwargs["get_placements"] is placement_fn
    assert save_kwargs["is_expert"] is expert_classifier
    assert save_kwargs["use_dcp"] is True
    assert save_kwargs["save_rng"] is True

    with patch(
        "megatron.lite.primitive.ckpt.load_training_checkpoint",
        return_value=13,
    ) as load_mock:
        assert runtime.load_checkpoint(handle, str(tmp_path), use_dcp=True) == 13

    load_args = load_mock.call_args.args
    load_kwargs = load_mock.call_args.kwargs
    assert isinstance(load_args[0], nn.ModuleList)
    assert load_args[0][0] is model
    assert load_args[3] is parallel
    assert load_args[4] is ps
    assert load_kwargs["get_placements"] is placement_fn
    assert load_kwargs["is_expert"] is expert_classifier
    assert load_kwargs["use_dcp"] is True
    assert load_kwargs["load_rng"] is True
