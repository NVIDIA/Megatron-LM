# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

_IS_BLACKWELL = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).major >= 10)

try:
    from flashinfer import mxfp8_quantize

    _HAVE_FLASHINFER = True
except ImportError:
    _HAVE_FLASHINFER = False

pytestmark = [
    pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 tests require Blackwell GPU (SM >= 10)"),
    pytest.mark.skipif(not _HAVE_FLASHINFER, reason="MXFP8 tests require FlashInfer"),
]


# ===========================================================================
# MXFP8ReshardTransform
# ===========================================================================


class TestMXFP8ReshardTransform:
    """Tests for the core MXFP8 reshard transform (transforms.py).

    These test the receiver-side BF16→MXFP8 conversion paths that run on
    every refit iteration, including the critical 1D-scale accumulation
    logic that avoids corrupting swizzled scales from partial updates.
    """

    def _make_persistent_buffers(self, shapes):
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        buffers = {}
        for name, (M, K) in shapes.items():
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            buffers[name] = MXFP8Tensor.from_bf16(x)
        return buffers

    def test_finalize_recv_bf16_2d_scale(self):
        """Receiver-side conversion with 2D scale: immediate per-slice quantization."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        M, K = 64, 128
        buf = MXFP8Tensor.from_bf16(torch.randn(M, K, dtype=torch.bfloat16, device="cuda"))

        if buf.scale.ndim != 2:
            pytest.skip("FlashInfer produced 1D swizzled scale; 2D-scale test not applicable")

        t = MXFP8ReshardTransform(
            convertible_params={"decoder.weight"},
            persistent_buffers={"weight": buf},
            buffer_key_prefix="decoder.",
            convert_on_send=False,
        )

        new_data = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        t.finalize_recv("decoder.weight", (slice(None), slice(None)), [new_data])

        expected = MXFP8Tensor.from_bf16(new_data)
        assert torch.equal(buf.data, expected.data)
        assert torch.equal(buf.scale, expected.scale)

    def test_finalize_recv_bf16_1d_scale_accumulation(self):
        """Receiver-side conversion with 1D scale: accumulate slices then quantize."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        M, K = 64, 128
        buf = MXFP8Tensor.from_bf16(torch.randn(M, K, dtype=torch.bfloat16, device="cuda"))

        if buf.scale.ndim != 1:
            pytest.skip("FlashInfer produced 2D scale; 1D-scale accumulation test not applicable")

        t = MXFP8ReshardTransform(
            convertible_params={"decoder.weight"},
            persistent_buffers={"weight": buf},
            buffer_key_prefix="decoder.",
            convert_on_send=False,
        )

        full_data = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        half = M // 2

        # First slice: should accumulate (not finalize yet)
        t.finalize_recv("decoder.weight", (slice(0, half), slice(None)), [full_data[:half]])
        assert "weight" in t._pending_1d, "Should be pending after partial slice"

        # Second slice: should trigger final quantization
        t.finalize_recv("decoder.weight", (slice(half, M), slice(None)), [full_data[half:]])
        assert "weight" not in t._pending_1d, "Should be finalized after all slices"

        expected = MXFP8Tensor.from_bf16(full_data)
        assert torch.equal(buf.data, expected.data)
        assert torch.equal(buf.scale, expected.scale)

    def test_finalize_recv_1d_scale_wrong_element_count(self):
        """1D accumulation should raise if total elements don't match (duplicate slices)."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        M, K = 64, 128
        buf = MXFP8Tensor.from_bf16(torch.randn(M, K, dtype=torch.bfloat16, device="cuda"))
        if buf.scale.ndim != 1:
            pytest.skip("Need 1D scale for this test")

        t = MXFP8ReshardTransform(
            convertible_params={"decoder.weight"},
            persistent_buffers={"weight": buf},
            buffer_key_prefix="decoder.",
            convert_on_send=False,
        )

        half_data = torch.randn(M // 2, K, dtype=torch.bfloat16, device="cuda")
        t.finalize_recv("decoder.weight", (slice(0, M // 2), slice(None)), [half_data])

        with pytest.raises(AssertionError, match="duplicate or missing"):
            overlap = torch.randn(M // 2 + 1, K, dtype=torch.bfloat16, device="cuda")
            t.finalize_recv("decoder.weight", (slice(M // 2 - 1, M), slice(None)), [overlap])


# ===========================================================================
# quantize_params_to_mxfp8
# ===========================================================================


class TestQuantizeParamsToMXFP8:
    """Tests for persistent buffer quantization (quantization/utils.py).

    The persistent buffer address stability is critical for CUDA graph
    compatibility — if addresses change, captured graphs segfault.
    """

    def test_basic_quantization_replaces_param(self):
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8

        model = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        buffers = quantize_params_to_mxfp8(model)

        assert "weight" in buffers
        assert isinstance(buffers["weight"], MXFP8Tensor)
        assert buffers["weight"].data.shape == (64, 128)
        assert "weight" not in model._parameters

    def test_persistent_buffer_reuse_preserves_addresses(self):
        """Second call must copy into existing buffers (CUDA graph address stability)."""
        from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8

        model = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        buffers = quantize_params_to_mxfp8(model)
        data_ptr = buffers["weight"].data.data_ptr()
        scale_ptr = buffers["weight"].scale.data_ptr()

        model2 = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        quantize_params_to_mxfp8(model2, persistent_buffers=buffers)

        assert buffers["weight"].data.data_ptr() == data_ptr
        assert buffers["weight"].scale.data_ptr() == scale_ptr

    def test_nested_module_fqn(self):
        """Recursive quantization should produce correct fully-qualified names."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64, bias=False), torch.nn.Linear(64, 32, bias=False)
        ).to(dtype=torch.bfloat16, device="cuda")
        buffers = quantize_params_to_mxfp8(model)

        assert "0.weight" in buffers and "1.weight" in buffers
        assert isinstance(buffers["0.weight"], MXFP8Tensor)


# ===========================================================================
# End-to-end MXFP8 refit integration (single-GPU)
# ===========================================================================


class TestMXFP8RefitIntegration:
    """Integration tests simulating the full send→recv→finalize refit flow."""

    def test_full_transform_roundtrip_bf16_wire(self):
        """Simulate sender sending BF16, receiver converting to MXFP8."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        M, K = 64, 128
        src_weight = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        src_param = torch.nn.Parameter(src_weight.clone())

        dst_buf = MXFP8Tensor.from_bf16(torch.randn(M, K, dtype=torch.bfloat16, device="cuda"))
        t = MXFP8ReshardTransform(
            convertible_params={"decoder.weight"},
            persistent_buffers={"weight": dst_buf},
            buffer_key_prefix="decoder.",
            convert_on_send=False,
        )

        # Simulate: prepare_send → wire → prepare_recv → finalize_recv
        sent = t.prepare_send("decoder.weight", (slice(None), slice(None)), src_param)
        recv_bufs = t.prepare_recv("decoder.weight", (slice(None), slice(None)))
        recv_bufs[0].copy_(sent[0])
        t.finalize_recv("decoder.weight", (slice(None), slice(None)), recv_bufs)

        expected = MXFP8Tensor.from_bf16(src_weight)
        assert torch.equal(dst_buf.data, expected.data)
        assert torch.equal(dst_buf.scale, expected.scale)

    def test_multi_slice_assembly(self):
        """Multiple row slices should correctly assemble the full quantized weight."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        M, K = 128, 256
        full_weight = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        dst_buf = MXFP8Tensor.from_bf16(torch.zeros(M, K, dtype=torch.bfloat16, device="cuda"))

        t = MXFP8ReshardTransform(
            convertible_params={"decoder.weight"},
            persistent_buffers={"weight": dst_buf},
            buffer_key_prefix="decoder.",
            convert_on_send=False,
        )

        # Send in 4 row-slices (simulates TP=4 refit)
        chunk = M // 4
        for i in range(4):
            row_slice = (slice(i * chunk, (i + 1) * chunk), slice(None))
            src_param = torch.nn.Parameter(full_weight.clone())
            sent = t.prepare_send("decoder.weight", row_slice, src_param)
            recv = t.prepare_recv("decoder.weight", row_slice)
            recv[0].copy_(sent[0])
            t.finalize_recv("decoder.weight", row_slice, recv)

        expected = MXFP8Tensor.from_bf16(full_weight)
        assert torch.equal(dst_buf.data, expected.data)
        assert torch.equal(dst_buf.scale, expected.scale)
