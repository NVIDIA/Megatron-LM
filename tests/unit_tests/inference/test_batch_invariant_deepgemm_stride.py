# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

_SUPPORTED = bik.HAVE_DEEPGEMM_BF16 and torch.cuda.is_available()
_HOPPER = _SUPPORTED and torch.cuda.get_device_capability()[0] >= 9
pytestmark = pytest.mark.skipif(not _HOPPER, reason="Requires BF16 DeepGEMM on Hopper or newer.")

# DeepGEMM's own dense kernels are sm90_bf16_gemm_impl and sm100_bf16_gemm_impl.
_DEEPGEMM_DENSE_KERNEL = "bf16_gemm_impl"


@pytest.mark.parametrize("k, n", [(128, 60), (60, 128), (128, 128)])
def test_dense_deepgemm_stride_boundary_is_batch_invariant(monkeypatch, k, n):
    target = torch.randint(-1, 2, (8, k), device="cuda").to(torch.bfloat16)
    weight = torch.randint(-1, 2, (k, n), device="cuda").to(torch.bfloat16)
    small = torch.cat([target, target.new_zeros((56, k))])
    large = torch.cat([target.new_ones((47, k)), target, -target.new_ones((73, k))])
    calls = []
    native = bik.deep_gemm.bf16_gemm_nn

    def witness(a, b, d):
        calls.append((a.stride(0), b.stride(0), d.stride(0), a.data_ptr(), b.data_ptr()))
        return native(a, b, d)

    monkeypatch.setattr(bik.deep_gemm, "bf16_gemm_nn", witness)
    bik.enable_batch_invariant_mode(backend="deepgemm", collective="ordered")
    try:
        full64, full128 = torch.mm(small, weight), torch.mm(large, weight)
    finally:
        bik.disable_batch_invariant_mode()
    reference = (target.float().unsqueeze(2) * weight.float()).sum(1)
    assert torch.equal(full64[:8], full128[47:55]) and torch.equal(full64[:8].float(), reference)
    for output, rows in ((full64, 64), (full128, 128)):
        assert output.shape == (rows, n) and output.is_contiguous()
        assert output.dtype == target.dtype and output.device == target.device
    assert len(calls) == 2
    for strides in (call[:3] for call in calls):
        assert strides == ((k + 7) // 8 * 8, (n + 7) // 8 * 8, (n + 7) // 8 * 8)
    assert k != n or calls[0][3:] == (small.data_ptr(), weight.data_ptr())


def test_dense_deepgemm_runs_deepgemm_kernel(monkeypatch):
    """The backend runs DeepGEMM's fixed-K-order kernel, not DeepGEMM's cuBLASLt dispatch."""
    switch = []
    native_switch = bik.deep_gemm.use_deterministic_algorithms

    def record_switch(enabled):
        switch.append(enabled)
        return native_switch(enabled)

    monkeypatch.setattr(bik.deep_gemm, "use_deterministic_algorithms", record_switch)
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    bik.enable_batch_invariant_mode(backend="deepgemm", collective="ordered")
    try:
        F.linear(x, weight)  # JIT-compile outside the trace.
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            F.linear(x, weight)
            torch.cuda.synchronize()
    finally:
        bik.disable_batch_invariant_mode()
    names = sorted({event.name for event in prof.events()})
    assert any(_DEEPGEMM_DENSE_KERNEL in name for name in names), names
    assert switch == [True, False]


@pytest.mark.parametrize("k, n", [(4096, 4096), (16384, 64)])
def test_dense_deepgemm_rows_are_invariant_across_batch_size(k, n):
    """A token's output row has the same bits whatever the number of rows beside it."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = (torch.randn(n, k, device="cuda", generator=generator) * 0.02).to(torch.bfloat16)
    target = torch.randn(k, device="cuda", generator=generator).to(torch.bfloat16)
    rows = {}
    bik.enable_batch_invariant_mode(backend="deepgemm", collective="ordered")
    try:
        for m in (1, 16, 256, 4096):
            x = torch.randn(m, k, device="cuda", generator=generator).to(torch.bfloat16)
            x[0] = target
            rows[m] = F.linear(x, weight)[0]
    finally:
        bik.disable_batch_invariant_mode()
    reference = target.double() @ weight.double().t()
    torch.testing.assert_close(rows[1].double(), reference, atol=0.05, rtol=0.02)
    for m, row in rows.items():
        assert torch.equal(row, rows[1]), f"row 0 at M={m} differs from M=1"


@pytest.mark.parametrize("n, k", [(4096, 4096), (128, 60)])
def test_transposed_weight_reaches_deepgemm_uncopied(monkeypatch, n, k):
    """Linear layers hand DeepGEMM their weight K-major, copied only for an unaligned K."""
    weight = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(64, k, device="cuda").to(torch.bfloat16)
    reference = x.double() @ weight.double().t()
    operands = []
    native = bik.deep_gemm.bf16_gemm_nn

    def witness(a, b, d):
        operands.append((b.untyped_storage().data_ptr(), b.stride()))
        return native(a, b, d)

    monkeypatch.setattr(bik.deep_gemm, "bf16_gemm_nn", witness)
    linears = {
        "F.linear": lambda: F.linear(x, weight),
        "matmul": lambda: torch.matmul(x, weight.t()),
        "TE TN GEMM": lambda: bik.BatchInvariantTEGemmFn.apply(
            weight, x, None, torch.bfloat16, "TN"
        ),
    }
    # DeepGEMM reads b = weight.t() K-major; only an unaligned K needs a padded copy.
    expected_strides = (1, (k + 7) // 8 * 8)
    bik.enable_batch_invariant_mode(backend="deepgemm", collective="ordered")
    try:
        for name, linear in linears.items():
            operands.clear()
            output = linear()
            assert len(operands) == 1, name
            storage, strides = operands[0]
            assert strides == expected_strides, (name, strides)
            assert (storage == weight.untyped_storage().data_ptr()) == (k % 8 == 0), name
            torch.testing.assert_close(output.double(), reference, atol=0.05, rtol=0.02)
    finally:
        bik.disable_batch_invariant_mode()
