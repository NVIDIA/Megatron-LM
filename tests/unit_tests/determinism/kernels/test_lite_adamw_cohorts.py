# SPDX-License-Identifier: BSD-3-Clause
"""Compare real DTensor local-view cohorts with unchanged scalar AdamW updates."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

# Lite is a source-tree package, separate from the root Core distribution.
_LITE_ROOT = Path(__file__).resolve().parents[4] / "experimental" / "lite"
if str(_LITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LITE_ROOT))

from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW, to_local_tensor  # noqa: E402
from tests.unit_tests.determinism.kernels.harness import (  # noqa: E402
    assert_replays_bit_exact,
    bytes_equal,
)


class _ScalarAdamW(FP32AdamW):
    def _foreach_candidate(self, param, grad):
        return None


class _ObservedAdamW(FP32AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batched_calls = 0
        self.cohort_sizes = []
        self.scalar_param_ids = set()

    def _step_cohort(self, cohort, *args):
        if cohort:
            self.cohort_sizes.append(len(cohort))
            assert len(cohort) <= 128
            assert sum(item[2].numel() for item in cohort) <= 1 << 22
        if len(cohort) > 1:
            self.batched_calls += 1
        return super()._step_cohort(cohort, *args)

    def _step_one_param(self, param, *args):
        self.scalar_param_ids.add(id(param))
        return super()._step_one_param(param, *args)


@pytest.fixture(scope="module")
def mesh():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if "RANK" not in os.environ and not dist.is_initialized():
        pytest.skip("run through the distributed unit-test launcher")
    owns_group = not dist.is_initialized()
    if owns_group:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    try:
        yield DeviceMesh("cuda", list(range(dist.get_world_size())))
    finally:
        if owns_group:
            dist.destroy_process_group()


def _snapshot(optimizer):
    result = [torch.tensor(optimizer.step_count)]
    result.extend(to_local_tensor(p).detach().clone() for p in optimizer.params)
    for param in optimizer.params:
        result.extend(
            to_local_tensor(optimizer.state[param][key]).detach().clone()
            for key in ("master_param", "exp_avg", "exp_avg_sq")
        )
        result.append(torch.tensor(optimizer.state[param]["step"]))
    return result


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_lite_adamw_cohorts_state_and_replay(mesh, weight_decay):
    generator = torch.Generator(device="cuda").manual_seed(79 + dist.get_rank())
    initial = [torch.randn(64, 257, device="cuda", generator=generator) for _ in range(4)]
    gradients = [
        [torch.randn(64, 257, device="cuda", generator=generator) for _ in initial]
        for _ in range(3)
    ]

    def wrap(local, device_mesh=None):
        return DTensor.from_local(
            local,
            mesh if device_mesh is None else device_mesh,
            (Shard(0),),
            run_check=False,
            shape=(64 * dist.get_world_size(), 257),
            stride=(257, 1),
        )

    def build(cls):
        params = [torch.nn.Parameter(wrap(value.clone())) for value in initial]
        return cls(
            params,
            lr=5e-5,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            model_param_dtypes={id(p): torch.bfloat16 for p in params},
        )

    def execute(cls):
        optimizer = build(cls)
        for step, grads in enumerate(gradients):
            for index, (param, gradient) in enumerate(zip(optimizer.params, grads, strict=True)):
                # A missing gradient splits per-parameter steps on the next update.
                param.grad = (
                    None
                    if (step == 1 and index == 2)
                    else wrap(gradient.clone(), param.device_mesh)
                )
            for param in optimizer.params:
                if param.grad is not None:
                    assert param.grad.device_mesh is param.device_mesh
                    for key in ("master_param", "exp_avg", "exp_avg_sq"):
                        assert optimizer.state[param][key].device_mesh is param.device_mesh
            optimizer.step()
        if cls is _ObservedAdamW:
            assert optimizer.batched_calls > 0
            restored = build(_ObservedAdamW)
            restored.load_state_dict(optimizer.state_dict())
            for left, right in zip(_snapshot(restored), _snapshot(optimizer), strict=True):
                assert bytes_equal(left, right)
            # Restore into a separate optimizer and compare the next update too.
            for left, right, gradient in zip(
                optimizer.params, restored.params, gradients[0], strict=True
            ):
                left.grad = wrap(gradient.clone(), left.device_mesh)
                right.grad = wrap(gradient.clone(), right.device_mesh)
            optimizer.step()
            restored.step()
            for left, right in zip(_snapshot(restored), _snapshot(optimizer), strict=True):
                assert bytes_equal(left, right)
        else:
            for param, gradient in zip(optimizer.params, gradients[0], strict=True):
                param.grad = wrap(gradient.clone(), param.device_mesh)
            optimizer.step()
        return _snapshot(optimizer)

    actual, expected = execute(_ObservedAdamW), execute(_ScalarAdamW)
    for left, right in zip(actual, expected, strict=True):
        assert bytes_equal(left, right)
    assert_replays_bit_exact(
        lambda: execute(_ObservedAdamW),
        (),
        replays=3,
        backward=False,
        contention=True,
        what="Lite local-view AdamW state",
    )


def _eligibility_case(mesh, case, cls):
    shape = (0, 3) if case == "empty" else (17, 3)
    if case == "oversized":
        shape = (1025, 1024)
    dtype = torch.bfloat16 if case == "bf16" else torch.float32
    local = torch.ones(shape, device="cuda", dtype=dtype)
    if case == "noncontiguous":
        local = local.T
    placement = Partial() if case == "partial" else Shard(0)
    global_shape = (local.shape[0] * dist.get_world_size(), local.shape[1])
    if case == "partial":
        global_shape = tuple(local.shape)
    param = torch.nn.Parameter(
        DTensor.from_local(
            local,
            mesh,
            (placement,),
            run_check=False,
            shape=global_shape,
            stride=(global_shape[1], 1),
        )
    )
    grad_mesh = (
        DeviceMesh("cuda", list(range(dist.get_world_size())))
        if case == "different_mesh"
        else param.device_mesh
    )
    grad_placement = Replicate() if case == "different_placement" else placement
    grad_shape = global_shape if case == "different_placement" else tuple(local.shape)
    grad = DTensor.from_local(
        torch.ones(grad_shape, device="cuda", dtype=dtype),
        grad_mesh,
        (grad_placement,),
        run_check=False,
        shape=global_shape,
        stride=(global_shape[1], 1),
    )
    optimizer = cls(
        [param],
        lr=5e-5,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        cpu_update=case == "cpu_update",
        model_param_dtypes={} if case == "missing_model_dtype" else {id(param): torch.bfloat16},
    )
    param.grad = grad
    if case == "different_mesh":
        assert grad.device_mesh == param.device_mesh
        assert grad.device_mesh is not param.device_mesh
    if case == "internal_alias":
        optimizer.state[param]["exp_avg"] = optimizer.state[param]["master_param"]
    return optimizer, param, grad


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "bf16",
        "noncontiguous",
        "empty",
        "oversized",
        "cpu_update",
        "missing_model_dtype",
        "partial",
        "internal_alias",
        "different_placement",
        "different_mesh",
    ],
)
def test_lite_adamw_real_dtensor_eligibility(mesh, case):
    optimizer, param, grad = _eligibility_case(mesh, case, _ObservedAdamW)
    reference, _, _ = _eligibility_case(mesh, case, _ScalarAdamW)
    assert (optimizer._foreach_candidate(param, grad) is not None) == (case == "ordinary")
    outcomes = []
    for selected in (optimizer, reference):
        try:
            selected.step()
        except (RuntimeError, ValueError, NotImplementedError) as error:
            outcomes.append(("rejected", type(error)))
        else:
            outcomes.append(("updated", _snapshot(selected)))
    assert id(param) in optimizer.scalar_param_ids
    assert outcomes[0][0] == outcomes[1][0]
    if outcomes[0][0] == "rejected":
        assert case in {"partial", "different_placement", "different_mesh"}
        assert outcomes[0][1] is outcomes[1][1]
    else:
        for actual, expected in zip(outcomes[0][1], outcomes[1][1], strict=True):
            assert bytes_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        "singleton",
        "count_limit",
        "numel_limit",
        "different_steps",
        "different_groups",
        "cross_parameter_alias",
        "uneven_shard",
        "ineligible_between_cohorts",
        "mutable_cross_alias",
        "shifted_cross_alias",
    ],
)
def test_lite_adamw_cohort_boundaries_match_scalar(mesh, case):
    count = (
        129
        if case == "count_limit"
        else 5 if case == "numel_limit" else 1 if case == "singleton" else 4
    )
    if case == "ineligible_between_cohorts":
        count = 5
    local_rows = 1 << 20 if case == "numel_limit" else 17
    global_rows = local_rows * dist.get_world_size()
    if case == "uneven_shard":
        global_rows = 2 * dist.get_world_size() + 1
        chunk = (global_rows + dist.get_world_size() - 1) // dist.get_world_size()
        local_rows = max(0, min(chunk, global_rows - dist.get_rank() * chunk))
    values = [torch.full((local_rows, 1), index / 16, device="cuda") for index in range(count)]
    gradients = [torch.full_like(value, (index + 1) / 32) for index, value in enumerate(values)]

    def wrap(local, device_mesh=None):
        return DTensor.from_local(
            local,
            mesh if device_mesh is None else device_mesh,
            (Shard(0),),
            run_check=False,
            shape=(global_rows, 1),
            stride=(1, 1),
        )

    def execute(cls):
        params = [torch.nn.Parameter(wrap(value.clone())) for value in values]
        if case == "ineligible_between_cohorts":
            params[2] = torch.nn.Parameter(wrap(values[2].bfloat16()))
        groups = (
            [{"params": [param], "lr": 5e-5 * (index + 1)} for index, param in enumerate(params)]
            if case == "different_groups"
            else params
        )
        optimizer = cls(
            groups,
            lr=5e-5,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
            model_param_dtypes={id(p): torch.bfloat16 for p in params},
        )
        wrapped_grads = [
            wrap(grad.clone(), param.device_mesh)
            for param, grad in zip(params, gradients, strict=True)
        ]
        if case == "cross_parameter_alias":
            wrapped_grads = [wrapped_grads[0]] * count
        if case == "mutable_cross_alias":
            wrapped_grads[1] = optimizer.state[params[0]]["master_param"]
        if case == "shifted_cross_alias":
            storage = torch.zeros(local_rows + 1, device="cuda")
            storage[-1] = 0.125
            first = wrap(storage[:-1].view(local_rows, 1), params[0].device_mesh)
            later = wrap(storage[1:].view(local_rows, 1), params[1].device_mesh)
            optimizer.state[params[0]]["master_param"] = first
            optimizer._master_for_param[params[0]] = first
            wrapped_grads[1] = later
            assert to_local_tensor(first).data_ptr() != to_local_tensor(later).data_ptr()
            assert (
                to_local_tensor(first).untyped_storage().data_ptr()
                == to_local_tensor(later).untyped_storage().data_ptr()
            )
        for index, (param, grad) in enumerate(zip(params, wrapped_grads, strict=True)):
            if case == "ineligible_between_cohorts" and index == 2:
                grad = grad.to(dtype=torch.bfloat16)
            param.grad = grad
        if case == "different_steps":
            optimizer.state[params[1]]["step"] = 1
        optimizer.step()
        if cls is _ObservedAdamW:
            if case in {"singleton", "different_groups", "cross_parameter_alias"} or not local_rows:
                assert optimizer.batched_calls == 0
            else:
                assert optimizer.batched_calls > 0
            if case in {"mutable_cross_alias", "shifted_cross_alias"}:
                assert id(params[0]) in optimizer.scalar_param_ids
                assert optimizer.cohort_sizes == [1, 3]
            if case == "ineligible_between_cohorts":
                assert id(params[2]) in optimizer.scalar_param_ids
                assert optimizer.cohort_sizes == [2, 2]
            if case == "count_limit":
                assert optimizer.cohort_sizes == [128, 1]
            if case == "numel_limit":
                assert optimizer.cohort_sizes == [4, 1]
        return _snapshot(optimizer)

    for actual, expected in zip(execute(_ObservedAdamW), execute(_ScalarAdamW), strict=True):
        assert bytes_equal(actual, expected)
