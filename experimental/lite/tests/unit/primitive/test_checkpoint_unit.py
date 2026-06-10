from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = pytest.mark.mlite


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))

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
        lhs.named_parameters(), rhs.named_parameters(), strict=True
    ):
        assert lhs_name == rhs_name
        torch.testing.assert_close(lhs_param, rhs_param, atol=0.0, rtol=0.0)


def test_runtime_checkpoint_load_matches_uninterrupted_training(tmp_path):
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
    ckpt_handle = ModelHandle(
        model=ckpt_model, optimizer=ckpt_optimizer, _extras={"model_chunks": [ckpt_model]}
    )
    runtime.save_checkpoint(ckpt_handle, str(tmp_path), step=1, use_dcp=False)

    loaded_handle = ModelHandle(
        model=loaded_model, optimizer=loaded_optimizer, _extras={"model_chunks": [loaded_model]}
    )
    assert runtime.load_checkpoint(loaded_handle, str(tmp_path), use_dcp=False) == 1

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
        state = torch.load(filename)
        self.parameter_load_calls = int(state["parameter_save_calls"])


def test_runtime_checkpoint_uses_optimizer_state_dict_contract(tmp_path):
    torch.manual_seed(2030)
    model = TinyMLP()
    optimizer = DistOptLike(torch.optim.AdamW(model.parameters(), lr=1.0e-3))
    x, y = torch.randn(3, 4), torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(
        ModelHandle(model=model, optimizer=optimizer, _extras={"model_chunks": [model]}),
        str(tmp_path),
        step=7,
        use_dcp=False,
    )

    loaded_model = TinyMLP()
    loaded_optimizer = DistOptLike(torch.optim.AdamW(loaded_model.parameters(), lr=1.0e-3))
    loaded_handle = ModelHandle(
        model=loaded_model, optimizer=loaded_optimizer, _extras={"model_chunks": [loaded_model]}
    )

    assert runtime.load_checkpoint(loaded_handle, str(tmp_path), use_dcp=False) == 7
    assert loaded_optimizer.load_calls == 1
    assert loaded_optimizer.parameter_load_calls == 1
    assert (tmp_path / "training_state.optimizer_parameter_state.pt").exists()
    _assert_model_close(model, loaded_model)
