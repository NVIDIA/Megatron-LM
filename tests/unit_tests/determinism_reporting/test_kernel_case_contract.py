# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the shared input and signature code on CPU; no GPU evidence claim."""

import sys
from types import ModuleType

import pytest
import torch

from tests.performance_tests.shell_test_utils.determinism.kernel_case import (
    case_signature,
    kernel_policy,
    make_case,
)


@pytest.mark.parametrize("case", ["bias_swiglu", "weighted_swiglu", "weighted_squared_relu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shared_factory_preserves_complete_arguments_native_precision_and_seed(
    monkeypatch, case, dtype
):
    # Replace only GPU-dependent production imports. Input generation/signatures
    # below execute their real implementations, including Torch RNG and hashing.
    swiglu = ModuleType("megatron.core.fusions.fused_bias_swiglu")
    swiglu.bias_swiglu_impl = lambda *args: args
    swiglu.weighted_bias_swiglu_impl = lambda *args: args
    squared = ModuleType("megatron.core.fusions.fused_weighted_squared_relu")
    squared.weighted_squared_relu_impl = lambda *args: args
    monkeypatch.setitem(sys.modules, swiglu.__name__, swiglu)
    monkeypatch.setitem(sys.modules, squared.__name__, squared)
    function, arguments, op_id = make_case(torch, case, 4, 8, dtype, device="cpu")
    assert function(*arguments) == arguments
    assert arguments[0].shape == (4, 8 if case == "weighted_squared_relu" else 16)
    assert arguments[0].dtype == dtype
    assert all(value.requires_grad for value in arguments if value is not None)
    if case == "weighted_swiglu":
        assert len(arguments) == 3 and arguments[1] is None
    if case != "bias_swiglu":
        assert arguments[-1].shape == (4, 1) and arguments[-1].dtype == torch.float32
    signature = case_signature(torch, case, arguments)
    torch.rand(100)
    _, repeated, repeated_id = make_case(torch, case, 4, 8, dtype, device="cpu")
    assert op_id == repeated_id
    assert signature == case_signature(torch, case, repeated)
    # FP32 is generated natively, rather than promoting a BF16 random sample.
    if dtype == torch.float32:
        assert not torch.equal(arguments[0], arguments[0].bfloat16().float())


def test_fingerprints_include_bit_patterns_stride_and_gradient_requirements():
    value = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    signature = case_signature(torch, "weighted_squared_relu", (value, torch.ones(2, 1)))
    changed = value.detach().clone()
    changed[0, 0] = -0.0
    changed.requires_grad_()
    other = case_signature(torch, "weighted_squared_relu", (changed, torch.ones(2, 1)))
    assert signature["inputs"][0]["sha256"] != other["inputs"][0]["sha256"]
    assert signature["inputs"][0]["requires_grad"] is True
    transposed = case_signature(torch, "weighted_squared_relu", (value.t(), torch.ones(2, 1)))
    assert signature["inputs"][0]["stride"] != transposed["inputs"][0]["stride"]
    assert signature["upstream_gradient"] == "ones_like_output"


def test_policy_restores_settings_after_a_failed_reference():
    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
    )
    with pytest.raises(RuntimeError, match="reference"):
        with kernel_policy(torch, True):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert not torch.backends.cudnn.benchmark
            raise RuntimeError("reference unavailable")
    assert original == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
    )
