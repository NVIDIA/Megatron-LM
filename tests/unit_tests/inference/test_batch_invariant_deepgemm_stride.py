# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

_SUPPORTED = bik.HAVE_DEEPGEMM_BF16 and torch.cuda.is_available()
_HOPPER = _SUPPORTED and torch.cuda.get_device_capability()[0] >= 9
pytestmark = pytest.mark.skipif(not _HOPPER, reason="Requires BF16 DeepGEMM on Hopper or newer.")


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
