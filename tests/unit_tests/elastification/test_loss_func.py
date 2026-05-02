# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.elastification.loss_func.

Covers the non-distributed paths of ``_mask_loss`` and ``loss_func``:
no tensor-parallel reductions, no sequence parallel, no KD. The
distributed/KD branches require a multi-rank torch.distributed init and
are left for a follow-up.
"""

from argparse import Namespace
from unittest.mock import MagicMock

import pytest
import torch

from megatron.elastification import loss_func as loss_func_module
from megatron.elastification.loss_func import _mask_loss, loss_func


def _stub_args(**overrides):
    defaults = dict(
        router_beta=1.0,
        loss_alpha=1.0,
        freeze_router=False,
        tensor_model_parallel_size=1,
        export_kd_teacher_load=False,
        budget_list=[1.0, 0.5],
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.fixture
def patch_get_args(monkeypatch):
    """Replace get_args in the loss_func module with a stub returning the args we set."""
    holder = {"args": _stub_args()}

    def _set_args(**overrides):
        holder["args"] = _stub_args(**overrides)

    monkeypatch.setattr(loss_func_module, "get_args", lambda: holder["args"])
    return _set_args


def _flat_loss_tensor(values):
    """Build the (B, S) loss tensor that ``_mask_loss`` expects."""
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1)


class TestMaskLossPlainTensor:
    """When output_tensor is a plain Tensor, no param_loss is reported."""

    def test_returns_scalar_loss_tensor(self, patch_get_args):
        patch_get_args()
        out = torch.tensor([[1.0, 2.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        result = _mask_loss(out, mask)
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(3.0)  # 1*1 + 2*1 + 4*0


class TestMaskLossWithParamLossTuple:
    """output_tensor is (output, (param_loss, extra_dict))."""

    def test_positive_param_loss_added_to_lm(self, patch_get_args):
        patch_get_args(loss_alpha=1.0)
        out = torch.tensor([[1.0, 2.0]])
        mask = torch.tensor([[1.0, 1.0]])
        param_loss = torch.tensor([0.5])
        loss, param_item = _mask_loss((out, (param_loss, {})), mask)
        # lm = 1+2 = 3; param contribution = 0.5 * num_tokens(2) * alpha(1) = 1.0
        assert loss.item() == pytest.approx(4.0)
        assert param_item.item() == pytest.approx(1.0)

    def test_negative_param_loss_scaled_by_router_beta(self, patch_get_args):
        # router_beta flips and scales negative param losses.
        patch_get_args(router_beta=2.0, loss_alpha=1.0)
        out = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])
        param_loss = torch.tensor([-0.25])
        loss, param_item = _mask_loss((out, (param_loss, {})), mask)
        # param_loss negated and scaled: -2.0 * (-0.25) = 0.5
        # param_item = 0.5 * num_tokens(2) * alpha(1) = 1.0
        # lm contribution = 2; total = 3
        assert param_item.item() == pytest.approx(1.0)
        assert loss.item() == pytest.approx(3.0)

    def test_freeze_router_drops_param_contribution(self, patch_get_args):
        patch_get_args(freeze_router=True)
        out = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])
        param_loss = torch.tensor([0.5])
        result = _mask_loss((out, (param_loss, {})), mask)
        # When router is frozen, param_loss isn't added — bare scalar returned.
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(2.0)

    def test_loss_alpha_scales_param_contribution(self, patch_get_args):
        patch_get_args(loss_alpha=10.0)
        out = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])
        param_loss = torch.tensor([0.5])
        loss, param_item = _mask_loss((out, (param_loss, {})), mask)
        # param_item = 0.5 * 2 tokens * 10 = 10.0
        assert param_item.item() == pytest.approx(10.0)
        assert loss.item() == pytest.approx(12.0)  # 2 (lm) + 10 (param)


class TestLossFuncReportingNoKD:
    """Top-level loss_func paths that don't enter the KD branch."""

    def _model(self, training=True):
        m = MagicMock()
        m.training = training
        return m

    def test_full_model_step_routes_to_lm_loss_full(self, patch_get_args):
        patch_get_args()
        out = torch.tensor([[1.0, 2.0]])
        mask = torch.tensor([[1.0, 1.0]])
        # param_loss = 0 → recognized as full-model step
        zero_param = torch.tensor([0.0])
        loss, num_tokens, report = loss_func(
            mask, (out, (zero_param, {})), self._model(training=True)
        )
        # The report dict must contain both keys, but only "(full)" carries data.
        assert "lm loss (full)" in report and "lm loss (budget)" in report
        full_val, full_den = report["lm loss (full)"][0], report["lm loss (full)"][1]
        budget_val, budget_den = report["lm loss (budget)"][0], report["lm loss (budget)"][1]
        assert budget_val.item() == 0.0 and budget_den.item() == 0.0
        # full_val gets lm loss minus param contribution. param_loss=0 → just lm.
        assert full_val.item() == pytest.approx(3.0)
        assert num_tokens.item() == 2

    def test_sub_budget_step_routes_to_lm_loss_budget(self, patch_get_args):
        patch_get_args()
        out = torch.tensor([[1.0, 2.0]])
        mask = torch.tensor([[1.0, 1.0]])
        nonzero_param = torch.tensor([0.5])  # signals sub-budget step
        loss, num_tokens, report = loss_func(
            mask, (out, (nonzero_param, {})), self._model(training=True)
        )
        full_val = report["lm loss (full)"][0]
        budget_val = report["lm loss (budget)"][0]
        assert full_val.item() == 0.0
        # budget side carries the lm loss (3.0 = 1+2)
        assert budget_val.item() == pytest.approx(3.0)

    def test_num_tokens_clamped_when_all_masked(self, patch_get_args):
        patch_get_args()
        out = torch.tensor([[5.0, 5.0]])
        mask = torch.tensor([[0.0, 0.0]])
        zero_param = torch.tensor([0.0])
        _, num_tokens, _ = loss_func(mask, (out, (zero_param, {})), self._model(training=True))
        # Guard at line 94 clamps to min=1 to avoid divide-by-zero downstream.
        assert num_tokens.item() == 1

    def test_report_values_are_packed_pairs(self, patch_get_args):
        """Every report entry is converted to a (value, num_tokens) tensor pair."""
        patch_get_args()
        out = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])
        zero_param = torch.tensor([0.0])
        _, _, report = loss_func(mask, (out, (zero_param, {})), self._model(training=True))
        for key, val in report.items():
            assert isinstance(val, torch.Tensor), f"report[{key}] not packed"
            assert val.shape == (2,), f"report[{key}] not (value, num_tokens)"
