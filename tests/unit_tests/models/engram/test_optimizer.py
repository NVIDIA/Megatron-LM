# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    _validate_row_sparse_optimizer_config,
)
from megatron.core.optimizer.sparse_adam import RowSparseAdam
from megatron.core.transformer.engram.memory import (
    EngramTableParameterMetadata,
    mark_engram_table_parameter,
)

ROW_WIDTH = 80


class _OptimizerFixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table_with_no_embedding_in_name = nn.Parameter(torch.randn(7, ROW_WIDTH))
        mark_engram_table_parameter(
            self.table_with_no_embedding_in_name, EngramTableParameterMetadata(row_parallel=True)
        )
        self.fusion = nn.Linear(ROW_WIDTH, ROW_WIDTH, bias=False)
        self.ddp_config = SimpleNamespace(use_custom_fsdp=False, use_megatron_fsdp=False)


def _groups(model, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(0, value),
    )
    config = OptimizerConfig(optimizer="adam", lr=2.0e-4, min_lr=2.0e-5)
    return _get_param_groups([model], config, {})


def test_row_sparse_table_has_explicit_optimizer_policy(monkeypatch):
    model = _OptimizerFixtureModel()
    groups = _groups(model, monkeypatch)
    table_group = next(group for group in groups if group.get("is_engram_row_parallel"))
    fusion_group = next(
        group for group in groups if any(param is model.fusion.weight for param in group["params"])
    )

    assert table_group["params"] == [model.table_with_no_embedding_in_name]
    assert table_group["lr_mult"] == 5.0
    assert table_group["max_lr"] == 1.0e-3
    assert table_group["min_lr"] == 1.0e-4
    assert table_group["wd_mult"] == 0.0
    assert not fusion_group.get("is_engram_row_parallel", False)


def test_row_sparse_adam_updates_only_touched_rows():
    model = _OptimizerFixtureModel()
    table = model.table_with_no_embedding_in_name
    optimizer = RowSparseAdam([table], lr=1.0e-3, betas=(0.9, 0.95), eps=1.0e-8)

    before = table.detach().clone()
    rows = torch.tensor([[1, 4]])
    values = torch.full((2, table.shape[1]), 0.25)
    table.grad = torch.sparse_coo_tensor(rows, values, table.shape).coalesce()
    optimizer.step()

    assert torch.equal(before[[0, 2, 3, 5, 6]], table[[0, 2, 3, 5, 6]])
    assert not torch.equal(before[[1, 4]], table[[1, 4]])
    assert optimizer.state[table]["rows"].tolist() == [1, 4]


def _set_sparse_grad(param, rows, values):
    param.grad = torch.sparse_coo_tensor(
        torch.tensor([rows], dtype=torch.long), values.to(dtype=param.dtype), param.shape
    )


def _oracle_step(master, moments, steps, rows, values, *, lr, betas, eps):
    beta1, beta2 = betas
    coalesced = {}
    for row, value in zip(rows, values.float()):
        if row not in coalesced:
            coalesced[row] = torch.zeros_like(value)
        coalesced[row].add_(value)
    for row, grad in coalesced.items():
        exp_avg, exp_avg_sq = moments.setdefault(
            row, (torch.zeros_like(grad), torch.zeros_like(grad))
        )
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        steps[row] = steps.get(row, 0) + 1
        bias_correction1 = 1.0 - beta1 ** steps[row]
        bias_correction2 = 1.0 - beta2 ** steps[row]
        denom = exp_avg_sq.sqrt() / bias_correction2**0.5 + eps
        master[row].addcdiv_(exp_avg / bias_correction1, denom, value=-lr)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_sparse_adam_matches_float32_row_oracle_across_steps(dtype):
    lr, betas, eps = 3.0e-3, (0.8, 0.95), 1.0e-8
    initial = torch.arange(32 * ROW_WIDTH, dtype=torch.float32).view(32, ROW_WIDTH) / 97
    param = nn.Parameter(initial.to(dtype))
    optimizer = RowSparseAdam([param], lr=lr, betas=betas, eps=eps)
    master = param.detach().float().clone()
    moments, steps = {}, {}
    untouched = set(range(param.shape[0]))

    updates = [
        (
            [3, 3, 11],
            torch.stack(
                (
                    torch.full((ROW_WIDTH,), 0.25),
                    torch.full((ROW_WIDTH,), 0.50),
                    torch.arange(ROW_WIDTH, dtype=torch.float32) / 80 + 0.125,
                )
            ),
        ),
        (
            [11, 19, 11],
            torch.stack(
                (
                    torch.full((ROW_WIDTH,), -0.375),
                    torch.full((ROW_WIDTH,), 0.625),
                    torch.full((ROW_WIDTH,), 0.125),
                )
            ),
        ),
        (
            [2, 3],
            torch.stack(
                (
                    torch.arange(ROW_WIDTH, dtype=torch.float32) / 160 - 0.25,
                    torch.full((ROW_WIDTH,), -0.75),
                )
            ),
        ),
    ]
    for rows, values in updates:
        typed_values = values.to(dtype).float()
        _set_sparse_grad(param, rows, values)
        optimizer.step()
        _oracle_step(master, moments, steps, rows, typed_values, lr=lr, betas=betas, eps=eps)
        untouched.difference_update(rows)

    state = optimizer.state[param]
    assert state["rows"].tolist() == [3, 11, 19, 2]
    assert state["step"].tolist() == [2, 2, 1, 1]
    torch.testing.assert_close(param[list(untouched)], initial.to(dtype)[list(untouched)])
    torch.testing.assert_close(param, master.to(dtype), rtol=0, atol=0)
    for slot, row in enumerate(state["rows"].tolist()):
        torch.testing.assert_close(state["exp_avg"][slot], moments[row][0], rtol=0, atol=1.0e-7)
        torch.testing.assert_close(state["exp_avg_sq"][slot], moments[row][1], rtol=0, atol=1.0e-7)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(state["master_param"], master[state["rows"]])
    else:
        assert "master_param" not in state


def test_row_sparse_adam_state_is_amortized_and_compact():
    param = nn.Parameter(torch.zeros(4096, ROW_WIDTH))
    optimizer = RowSparseAdam([param], lr=1.0e-3, betas=(0.9, 0.95), eps=1.0e-8)
    pointers = []
    capacities = []
    for row in range(33):
        _set_sparse_grad(param, [row], torch.full((1, ROW_WIDTH), 0.25))
        optimizer.step()
        state = optimizer.state[param]
        pointers.append(state["_exp_avg_storage"].data_ptr())
        capacities.append(state["_capacity"])

    assert capacities == [16] * 16 + [32] * 16 + [64]
    assert len(set(pointers)) == 3
    state = optimizer.state[param]
    assert state["_exp_avg_storage"].shape == (64, ROW_WIDTH)
    assert all(
        not (torch.is_tensor(value) and value.shape == param.shape) for value in state.values()
    )

    compact = next(iter(optimizer.state_dict()["state"].values()))
    assert compact["rows"].shape == (33,)
    assert compact["exp_avg"].shape == (33, ROW_WIDTH)
    assert compact["exp_avg_sq"].shape == (33, ROW_WIDTH)
    assert compact["step"].shape == (33,)
    assert not any(key.startswith("_") for key in compact)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sorted_rows", [False, True])
def test_row_sparse_adam_state_dict_load_preserves_exact_next_update(dtype, sorted_rows):
    param = nn.Parameter(
        (torch.arange(24 * ROW_WIDTH, dtype=torch.float32).view(24, ROW_WIDTH) / 1024).to(dtype)
    )
    optimizer = RowSparseAdam([param], lr=2.0e-3, betas=(0.9, 0.95), eps=1.0e-8)
    _set_sparse_grad(
        param,
        [9, 2, 9],
        torch.stack(
            (
                torch.full((ROW_WIDTH,), 0.25),
                torch.full((ROW_WIDTH,), -0.5),
                torch.full((ROW_WIDTH,), 0.75),
            )
        ),
    )
    optimizer.step()
    _set_sparse_grad(param, [1], torch.full((1, ROW_WIDTH), -0.125))
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    if sorted_rows:
        compact = next(iter(saved["state"].values()))
        order = torch.argsort(compact["rows"])
        for key in ("rows", "exp_avg", "exp_avg_sq", "step", "master_param"):
            if key not in compact:
                continue
            compact[key] = compact[key][order]

    restored_param = nn.Parameter(param.detach().clone())
    restored = RowSparseAdam([restored_param], lr=2.0e-3, betas=(0.9, 0.95), eps=1.0e-8)
    restored.load_state_dict(saved)
    next_rows = [2, 17, 2]
    next_values = torch.stack(
        (
            torch.full((ROW_WIDTH,), 0.125),
            torch.full((ROW_WIDTH,), -0.25),
            torch.full((ROW_WIDTH,), 0.375),
        )
    )
    for target, target_optimizer in ((param, optimizer), (restored_param, restored)):
        _set_sparse_grad(target, next_rows, next_values)
        target_optimizer.step()
    assert torch.equal(restored_param, param)
    original_state = optimizer.compact_state(param)
    restored_state = restored.compact_state(restored_param)
    original_by_row = {int(row): slot for slot, row in enumerate(original_state["rows"].tolist())}
    restored_by_row = {int(row): slot for slot, row in enumerate(restored_state["rows"].tolist())}
    assert original_by_row.keys() == restored_by_row.keys()
    for row in original_by_row:
        for key in ("exp_avg", "exp_avg_sq", "step", "master_param"):
            if key not in original_state:
                continue
            assert torch.equal(
                original_state[key][original_by_row[row]], restored_state[key][restored_by_row[row]]
            )


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"optimizer": "sgd"}, "RowSparseAdam"),
        ({"loss_scale": 128.0}, "unity loss scaling"),
        ({"fp16": True}, "FP32 and BF16"),
    ],
)
def test_row_sparse_optimizer_rejects_unsafe_config(overrides, error):
    model = _OptimizerFixtureModel()
    values = {"optimizer": "adam", "lr": 2.0e-4, "min_lr": 2.0e-5}
    values.update(overrides)
    config = OptimizerConfig(**values)
    with pytest.raises(ValueError, match=error):
        _validate_row_sparse_optimizer_config(config, [model])
