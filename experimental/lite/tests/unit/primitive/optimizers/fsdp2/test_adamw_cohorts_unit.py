# SPDX-License-Identifier: BSD-3-Clause
"""Check cohort arithmetic and fallback control flow independently of CUDA eligibility."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW, to_local_tensor


class _ScalarAdamW(FP32AdamW):
    def _foreach_candidate(self, param, grad):
        return None


def _assert_same_state(actual, expected):
    assert actual.step_count == expected.step_count
    for left, right in zip(actual.params, expected.params, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert actual.state[left]["step"] == expected.state[right]["step"]
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                to_local_tensor(actual.state[left][key]),
                to_local_tensor(expected.state[right][key]),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
def test_local_cohort_arithmetic_matches_scalar_updates(weight_decay, model_dtype):
    # Exercise cohort arithmetic on CPU; production eligibility is tested separately.
    def build(cls):
        params = [torch.nn.Parameter(torch.linspace(-0.5, 0.5, 17) + i / 8) for i in range(4)]
        return cls(
            params,
            lr=5e-5,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            model_param_dtypes={id(p): model_dtype for p in params},
        )

    actual, expected = build(FP32AdamW), build(_ScalarAdamW)

    def local_candidate(param, grad):
        state = actual.state[param]
        return (
            param,
            param.detach(),
            grad,
            state["master_param"],
            state["exp_avg"],
            state["exp_avg_sq"],
        )

    # Exercise the CPU math helper directly: mesh checks belong to CUDA DTensor tests.
    for step in range(4):
        cohort = []
        for i, (left, right) in enumerate(zip(actual.params, expected.params, strict=True)):
            grad = torch.linspace(-0.25, 0.25, 17) + (i + step) / 16
            left.grad, right.grad = grad.clone(), grad.clone()
            cohort.append(local_candidate(left, left.grad))
        actual.step_count += 1
        with torch.no_grad():
            actual._step_cohort(cohort, actual.lr, weight_decay, *actual.betas)
            expected.step()
        _assert_same_state(actual, expected)


@pytest.mark.parametrize("cpu_update", [False, True])
def test_cpu_parameters_use_original_fallback(cpu_update):
    param = torch.nn.Parameter(torch.ones(7))
    optimizer = FP32AdamW(
        [param],
        lr=1e-3,
        weight_decay=0,
        betas=(0.9, 0.999),
        eps=1e-8,
        cpu_update=cpu_update,
        model_param_dtypes={id(param): torch.bfloat16},
    )
    param.grad = torch.ones_like(param)
    assert optimizer._foreach_candidate(param, param.grad) is None
    optimizer.step()
    assert optimizer.state[param]["step"] == 1
    assert torch.isfinite(param).all()


@pytest.mark.parametrize(
    "ranges,expected",
    [
        (((0, 10),), False),
        (((10, 20),), True),
        (((19, 21),), True),
        (((20, 30),), False),
        (((29, 31),), True),
        (((40, 50),), False),
        (((0, 10), (39, 41)), True),
    ],
)
def test_storage_overlap_checks_complete_byte_intervals(ranges, expected):
    assert FP32AdamW._ranges_overlap(ranges, [(10, 20), (30, 40)]) is expected


def test_storage_overlap_detects_shifted_views():
    storage = torch.empty(33)
    first, second = storage[:17], storage[16:]
    assert first.data_ptr() != second.data_ptr()
    assert FP32AdamW._ranges_overlap(
        FP32AdamW._local_storage_ranges((second,)), list(FP32AdamW._local_storage_ranges((first,)))
    )
