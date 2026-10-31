# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for the GDN profiling tool's snapshot comparison."""

import pytest
import torch

from tools.ssm.gdn_activation_offload import _snapshot_tensors_equal


@pytest.mark.parametrize(
    "expected,actual,equal",
    [
        (torch.tensor([0.0]), torch.tensor([-0.0]), False),
        (
            torch.tensor([0.0], dtype=torch.bfloat16),
            torch.tensor([-0.0], dtype=torch.bfloat16),
            False,
        ),
        (torch.tensor(0.0), torch.tensor(-0.0), False),
        (torch.tensor([1.0]), torch.tensor([1.0000001192092896]), False),
        (torch.arange(6.0).view(2, 3).T, torch.arange(6.0).view(2, 3).T.contiguous(), True),
        (torch.arange(4.0), torch.arange(4.0).view(2, 2), False),
        (torch.tensor([0.0]), torch.tensor([0], dtype=torch.int32), False),
        (torch.tensor([float("nan")]), torch.tensor([float("nan")]), False),
        (torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([-1.0, 0.0, 1.0]), True),
    ],
    ids=[
        "fp32-signed-zero",
        "bf16-signed-zero",
        "scalar-signed-zero",
        "one-ulp",
        "strided-equal",
        "different-shapes",
        "different-dtypes",
        "nan-rejected",
        "finite-equal",
    ],
)
def test_snapshot_tensor_equality(
    expected: torch.Tensor, actual: torch.Tensor, equal: bool
) -> None:
    """Exact-value checks also detect differing bit patterns without accepting NaNs."""
    assert _snapshot_tensors_equal(expected, actual) is equal
