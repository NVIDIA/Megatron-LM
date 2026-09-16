# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the actual model/kernel byte comparator using CPU tensors."""

import pytest
import torch

from tests.unit_tests.determinism.comparison import assert_bit_exact, bytes_equal
from tests.unit_tests.determinism.configs import (
    PARALLELISM_CONFIGS,
    gb200_compatible_configs,
    required_world_size,
)
from tools.determinism.coverage import (
    DETERMINISTIC,
    NONDETERMINISTIC,
    UNVERIFIED,
    ReplayMismatch,
    collect_observations,
)


def f32_bits(bits):
    return torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)


@pytest.mark.parametrize(
    "a,b",
    [
        (torch.tensor(0.0), torch.tensor(-0.0)),
        (f32_bits([0x7FC00000]), f32_bits([0x7FC00001])),
        (f32_bits([0x7FC00000]), f32_bits([0xFFC00000])),
        (torch.ones(2), torch.ones(1, 2)),
        (torch.ones(2), torch.ones(2, dtype=torch.float64)),
        (torch.ones(2), torch.tensor([1.0, 1.1])),
    ],
)
def test_model_output_and_gradient_bit_changes_are_rejected(a, b):
    observations = []
    with collect_observations(observations.append):
        with pytest.raises(ReplayMismatch, match="Output bytes"):
            assert_bit_exact(a, {"weight": torch.ones(1)}, b, {"weight": torch.ones(1)})
        with pytest.raises(ReplayMismatch, match="Gradient bytes.*weight"):
            assert_bit_exact(torch.ones(1), {"weight": a}, torch.ones(1), {"weight": b})
    assert [obs["status"] for obs in observations] == [NONDETERMINISTIC] * 2


def test_identical_nan_payloads_and_noncontiguous_layouts_pass():
    a = f32_bits([0x7FC00001, 0x80000000, 0x3F800000, 0x7F800000]).reshape(2, 2)
    b = a.t().contiguous().t()
    assert not b.is_contiguous()
    assert bytes_equal(a, b)
    observations = []
    with collect_observations(observations.append):
        assert_bit_exact(a, {"weight": a}, b, {"weight": b})
    assert observations[0]["status"] == DETERMINISTIC
    assert observations[0]["compared_gradients"] == 1


@pytest.mark.parametrize(
    "output,grads",
    [
        (torch.empty(0), {"w": torch.ones(1)}),
        (torch.ones(1), {}),
        (torch.ones(1), {"w": torch.empty(0)}),
    ],
)
def test_empty_model_comparisons_are_unverified(output, grads):
    observations = []
    with collect_observations(observations.append), pytest.raises(ValueError, match="nonempty"):
        assert_bit_exact(output, grads, output, grads)
    assert observations[0]["status"] == UNVERIFIED


def test_missing_parameter_gradient_is_a_replay_mismatch():
    with pytest.raises(ReplayMismatch, match="Grad keys"):
        assert_bit_exact(torch.ones(1), {"w": torch.ones(1)}, torch.ones(1), {})


def test_gb200_selection_preserves_larger_configs_without_marking_them():
    marked = gb200_compatible_configs(PARALLELISM_CONFIGS)
    assert [p.id for p in marked] == [p.id for p in PARALLELISM_CONFIGS]
    selected = [p for p in marked if any(m.name == "launch_on_gb200" for m in p.marks)]
    assert {p.id for p in selected} == {
        "tp4",
        "ep2",
        "tp2-ep2",
        "fsdp4",
        "pp2",
        "pp4",
        "tp2-pp2",
        "pp2-vpp2",
    }
    assert all(4 % required_world_size(p.values[0]) == 0 for p in selected)
    assert all(not p.marks for p in PARALLELISM_CONFIGS)
    # A small configuration must still divide four; merely fitting is insufficient.
    assert not gb200_compatible_configs([pytest.param({"TP": 3}, id="tp3")])[0].marks
