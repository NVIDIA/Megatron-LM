# SPDX-License-Identifier: BSD-3-Clause
"""Exercise public DTensor state invalidation with an isolated CPU process group."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW, to_local_tensor


class _SavedState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trigger, state):
        ctx.save_for_backward(state)
        return trigger.clone()

    @staticmethod
    def backward(ctx, grad):
        _ = ctx.saved_tensors
        return grad, None


class _ScalarAdamW(FP32AdamW):
    def _foreach_candidate(self, param, grad):
        return None


@pytest.fixture(scope="module")
def cpu_mesh(tmp_path_factory):
    if dist.is_initialized():
        pytest.skip("CPU mutation checks require an isolated process group")
    init_path = tmp_path_factory.mktemp("adamw-mutation") / "rendezvous"
    dist.init_process_group("gloo", init_method=init_path.as_uri(), rank=0, world_size=1)
    try:
        yield DeviceMesh("cpu", [0])
    finally:
        dist.destroy_process_group()


def _build(mesh, cls, weight_decay):
    params = [
        torch.nn.Parameter(
            DTensor.from_local(
                torch.linspace(-0.5, 0.5, 17) + index / 8, mesh, (Replicate(),), run_check=False
            )
        )
        for index in range(2)
    ]
    optimizer = cls(
        params,
        lr=5e-5,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        model_param_dtypes={id(p): torch.bfloat16 for p in params},
    )
    for param in params:
        param.grad = DTensor.from_local(torch.ones(17), mesh, (Replicate(),), run_check=False)
    return optimizer


def _cohort_step(optimizer):
    # Only the math/mutation helper is exercised on CPU, not CUDA eligibility.
    cohort = []
    for param in optimizer.params:
        state = optimizer.state[param]
        cohort.append(
            (
                param,
                to_local_tensor(param),
                to_local_tensor(param.grad),
                to_local_tensor(state["master_param"]),
                to_local_tensor(state["exp_avg"]),
                to_local_tensor(state["exp_avg_sq"]),
            )
        )
    optimizer.step_count += 1
    with torch.no_grad():
        optimizer._step_cohort(cohort, optimizer.lr, optimizer.weight_decay, *optimizer.betas)


def _same_bytes(left, right):
    left, right = to_local_tensor(left), to_local_tensor(right)
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_dtensor_wrapper_versions_and_numeric_state(cpu_mesh, weight_decay):
    actual = _build(cpu_mesh, FP32AdamW, weight_decay)
    reference = _build(cpu_mesh, _ScalarAdamW, weight_decay)
    keys = ("master_param", "exp_avg", "exp_avg_sq")
    before = [
        [o.state[p][key]._version for p in o.params for key in keys] for o in (actual, reference)
    ]
    _cohort_step(actual)
    reference.step()
    after = [
        [o.state[p][key]._version for p in o.params for key in keys] for o in (actual, reference)
    ]
    assert [a - b for a, b in zip(after[0], before[0], strict=True)] == [
        a - b for a, b in zip(after[1], before[1], strict=True)
    ]
    assert actual.step_count == reference.step_count == 1
    for left, right in zip(actual.params, reference.params, strict=True):
        _same_bytes(left, right)
        for key in keys:
            _same_bytes(actual.state[left][key], reference.state[right][key])


@pytest.mark.parametrize("key", ["master_param", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("save_view", [False, True])
def test_dtensor_saved_state_is_invalidated(cpu_mesh, key, save_view):
    optimizer = _build(cpu_mesh, FP32AdamW, 0.1)
    state = optimizer.state[optimizer.params[0]][key]
    saved = state.view_as(state) if save_view else state
    stale = _SavedState.apply(torch.ones((), requires_grad=True), saved)
    _cohort_step(optimizer)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        stale.backward()
    trigger = torch.ones((), requires_grad=True)
    fresh = _SavedState.apply(trigger, saved)
    fresh.backward()
    assert trigger.grad.item() == 1
