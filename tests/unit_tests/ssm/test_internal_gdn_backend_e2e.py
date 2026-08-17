# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GB200 end-to-end correctness and performance coverage for internal GDR."""

import os
from statistics import median

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.launch_on_gb200

_BATCH_SIZE = 2
_SEQUENCE_LENGTH = 8192
_HEADS = 64
_HEAD_DIM = 128
_SAMPLES = 5


def _make_inputs(device):
    torch.manual_seed(1234)
    shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _HEADS, _HEAD_DIM)
    gate_shape = shape[:-1]
    q = (0.1 * torch.randn(shape, device=device, dtype=torch.bfloat16)).requires_grad_()
    k = (0.1 * torch.randn(shape, device=device, dtype=torch.bfloat16)).requires_grad_()
    v = (0.1 * torch.randn(shape, device=device, dtype=torch.bfloat16)).requires_grad_()
    g = F.logsigmoid(torch.randn(gate_shape, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(gate_shape, device=device, dtype=torch.float32))
    inputs = (
        q,
        k,
        v,
        g.to(torch.bfloat16).requires_grad_(),
        beta.to(torch.bfloat16).requires_grad_(),
    )
    return inputs, torch.randn(shape, device=device, dtype=torch.bfloat16)


def _forward_backward(implementation, inputs, grad_output):
    output, _ = implementation.chunk_gated_delta_rule(
        q=inputs[0], k=inputs[1], v=inputs[2], g=inputs[3], beta=inputs[4], scale=_HEAD_DIM**-0.5
    )
    gradients = torch.autograd.grad(output, inputs, grad_outputs=grad_output)
    return output, gradients


def _median_gpu_ms(implementation, inputs, grad_output):
    for _ in range(2):
        _forward_backward(implementation, inputs, grad_output)
    torch.cuda.synchronize()

    samples = []
    for _ in range(_SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _forward_backward(implementation, inputs, grad_output)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)


def test_internal_gdr_cute_matches_and_outperforms_fla(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("the fused CuTe DSL kernels require SM100")
    if not torch.version.cuda or int(torch.version.cuda.split(".", 1)[0]) < 13:
        pytest.skip("the fused CuTe DSL kernels require CUDA 13+")
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    pytest.importorskip("fla")

    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation

    inputs, grad_output = _make_inputs(device)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    reference_output, reference_gradients = _forward_backward(implementation, inputs, grad_output)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    cute_output, cute_gradients = _forward_backward(implementation, inputs, grad_output)

    torch.testing.assert_close(cute_output, reference_output, atol=1e-2, rtol=1e-2)
    for name, actual, expected in zip(
        ("q", "k", "v", "g", "beta"), cute_gradients, reference_gradients
    ):
        tolerance = 1e-1 if name in ("g", "beta") else 5e-2
        torch.testing.assert_close(
            actual,
            expected,
            atol=tolerance,
            rtol=tolerance,
            msg=lambda message: f"{name}: {message}",
        )

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    fla_ms = _median_gpu_ms(implementation, inputs, grad_output)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    cute_ms = _median_gpu_ms(implementation, inputs, grad_output)

    speedup = fla_ms / cute_ms
    print(
        f"internal GDR E2E B={_BATCH_SIZE} T={_SEQUENCE_LENGTH}: "
        f"CuTe={cute_ms:.3f} ms, FLA={fla_ms:.3f} ms, speedup={speedup:.2f}x"
    )
    assert cute_ms <= fla_ms * 1.10, (
        f"CuTe path regressed: {cute_ms:.3f} ms vs FLA {fla_ms:.3f} ms " f"({speedup:.2f}x)"
    )
