# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Scoped GPU repeatability smoke tests, not backend-wide certification."""

import pytest
import torch

from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unpacked_reference_forward_and_backward_repeat(variant, dtype, monkeypatch):
    """Repeat identical small inputs on one GPU with fixed deterministic settings."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.warns(UserWarning, match="unknown"):
            kernel = select_gated_delta_rule(variant, deterministic=True)
        generator = torch.Generator(device="cuda").manual_seed(1729)
        q, k, v = [
            torch.randn(1, 16, 2, 8, device="cuda", dtype=dtype, generator=generator) * 0.1
            for _ in range(3)
        ]
        if variant == "gdn":
            gates = {
                "g": -torch.rand(1, 16, 2, device="cuda", generator=generator),
                "beta": torch.rand(1, 16, 2, device="cuda", generator=generator),
            }
        else:
            gates = {
                "g": -torch.rand(q.shape, device="cuda", generator=generator),
                "b": torch.rand(q.shape, device="cuda", generator=generator),
                "w": torch.rand(v.shape, device="cuda", generator=generator),
            }
        inputs = dict(q=q, k=k, v=v, **gates)
        results = []
        for _ in range(2):
            arguments = {
                name: value.detach().clone().requires_grad_() for name, value in inputs.items()
            }
            output, state = kernel(**arguments, output_final_state=True, chunk_size=16)
            (output.float().square().sum() + state.float().square().sum()).backward()
            tensors = [output, state, *(value.grad for value in arguments.values())]
            assert all(tensor is not None and torch.isfinite(tensor).all() for tensor in tensors)
            results.append([tensor.detach().clone() for tensor in tensors])
        assert all(torch.equal(first, second) for first, second in zip(*results))
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
