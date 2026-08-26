# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

_IS_BLACKWELL = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).major >= 10)

try:
    from flashinfer import mxfp8_quantize

    _HAVE_FLASHINFER = True
except ImportError:
    _HAVE_FLASHINFER = False

from megatron.core.inference.moe.flashinfer_mxfp8 import HAVE_FLASHINFER_ROUTED_MXFP8

pytestmark = pytest.mark.skipif(
    not _IS_BLACKWELL, reason="MXFP8 tests require Blackwell GPU (SM >= 10)"
)


# ===========================================================================
# MXFP8ReshardTransform
# ===========================================================================


class TestMXFP8ReshardTransform:
    """Tests for the core MXFP8 reshard transform (transforms.py).

    These test the receiver-side BF16→MXFP8 conversion paths that run on
    every refit iteration, including the critical 1D-scale accumulation
    logic that avoids corrupting swizzled scales from partial updates.
    """

    @pytest.mark.skipif(not _HAVE_FLASHINFER, reason="test requires FlashInfer")
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

    @pytest.mark.skipif(not _HAVE_FLASHINFER, reason="test requires FlashInfer")
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

    @pytest.mark.skipif(not _HAVE_FLASHINFER, reason="test requires FlashInfer")
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

    def test_repeated_finalize_updates_persistent_buffer_outside_inference_mode(self):
        """Persistent buffers created in inference mode remain mutable across refits."""
        from megatron.core.inference.quantization import utils as quantization_utils
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        M, K = 64, 128
        with torch.inference_mode():
            model = torch.nn.Linear(K, M, bias=False).to(dtype=torch.bfloat16, device="cuda")
            _pre_quantize_linear(model)
            buffers = quantization_utils.quantize_params_to_mxfp8(model, backend="triton")
        buf = buffers["weight"]
        assert not torch.is_inference(buf.data)
        assert not torch.is_inference(buf.scale)

        transform = MXFP8ReshardTransform(
            convertible_params={"decoder.weight"},
            persistent_buffers=buffers,
            buffer_key_prefix="decoder.",
        )
        data_ptr = buf.data.data_ptr()
        scale_ptr = buf.scale.data_ptr()

        for _ in range(2):
            assert torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
            new_data = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            transform.finalize_recv("decoder.weight", (slice(None), slice(None)), [new_data])
            expected = MXFP8Tensor.from_bf16(new_data, backend="triton")
            assert torch.equal(buf.data, expected.data)
            assert torch.equal(buf.scale.view(torch.uint8), expected.scale.view(torch.uint8))
            assert buf.data.data_ptr() == data_ptr
            assert buf.scale.data_ptr() == scale_ptr
            assert buf.backend == "triton"
            assert buf.scale.dtype == torch.float8_e8m0fnu

    def test_sender_side_conversion_supports_explicit_triton_backend(self):
        """A sender without persistent buffers can select the Triton wire format."""
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        transform = MXFP8ReshardTransform(
            convertible_params={"decoder.weight"},
            persistent_buffers={},
            convert_on_send=True,
            backend="triton",
        )
        source = torch.nn.Parameter(
            torch.randn(64, 128, dtype=torch.bfloat16, device="cuda"), requires_grad=False
        )
        data, scale = transform.prepare_send("decoder.weight", (slice(None), slice(None)), source)

        assert data.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float8_e8m0fnu

    def test_sender_side_conversion_requires_explicit_backend(self):
        """A sender without persistent buffers cannot infer its wire format."""
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        with pytest.raises(AssertionError, match="backend is required for sender-side conversion"):
            MXFP8ReshardTransform(
                convertible_params={"decoder.weight"}, persistent_buffers={}, convert_on_send=True
            )

    def test_mixed_persistent_buffer_backends_are_rejected(self):
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform

        triton_buffer = MXFP8Tensor.from_bf16(
            torch.randn(64, 128, dtype=torch.bfloat16, device="cuda"), backend="triton"
        )
        flashinfer_buffer = MXFP8Tensor(
            data=triton_buffer.data.clone(),
            scale=triton_buffer.scale.view(torch.uint8).clone(),
            dtype=torch.bfloat16,
            backend="flashinfer",
        )

        with pytest.raises(ValueError, match="backend is 'flashinfer'; expected 'triton'"):
            MXFP8ReshardTransform(
                convertible_params={"first", "second"},
                persistent_buffers={"first": triton_buffer, "second": flashinfer_buffer},
            )

    def test_concatenated_moe_buffers_remain_refittable(self):
        """Lazy MoE stacking must not replace persistent storage with inference tensors."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform
        from megatron.core.transformer.moe.experts import InferenceGroupedMLP

        class Namespace:
            _stack_mxfp8_linear_weight = InferenceGroupedMLP._stack_mxfp8_linear_weight

        num_experts, M, K = 2, 64, 128
        grouped_mlp = Namespace()
        grouped_mlp.num_local_experts = num_experts
        grouped_mlp.inference_grouped_gemm_backend = "torch"
        buffers = {}
        for linear_name in ("linear_fc1", "linear_fc2"):
            linear = Namespace()
            setattr(grouped_mlp, linear_name, linear)
            for expert_idx in range(num_experts):
                tensor = MXFP8Tensor.from_bf16(
                    torch.randn(M, K, dtype=torch.bfloat16, device="cuda"), backend="triton"
                )
                if expert_idx == 0:
                    tensor.dtype = None  # Simulate a legacy direct constructor.
                setattr(linear, f"weight{expert_idx}", tensor)
                buffers[f"{linear_name}.weight{expert_idx}"] = tensor

        with torch.inference_mode():
            InferenceGroupedMLP._build_concatenated_mxfp8_weights(grouped_mlp)

        assert not torch.is_inference(grouped_mlp._fc1_weight.data)
        assert not torch.is_inference(grouped_mlp._fc1_weight.scale)
        assert not torch.is_inference(grouped_mlp._fc2_weight.data)
        assert not torch.is_inference(grouped_mlp._fc2_weight.scale)
        assert grouped_mlp._fc1_weight.dtype == torch.bfloat16
        assert grouped_mlp._fc2_weight.dtype == torch.bfloat16

        transform = MXFP8ReshardTransform(
            convertible_params=set(buffers), persistent_buffers=buffers
        )
        for name, buf in buffers.items():
            data_ptr = buf.data.data_ptr()
            scale_ptr = buf.scale.data_ptr()
            new_data = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            transform.finalize_recv(name, (slice(None), slice(None)), [new_data])
            expected = MXFP8Tensor.from_bf16(new_data, backend="triton")

            assert torch.equal(buf.data, expected.data)
            assert torch.equal(buf.scale.view(torch.uint8), expected.scale.view(torch.uint8))
            assert buf.data.data_ptr() == data_ptr
            assert buf.scale.data_ptr() == scale_ptr
            assert buf.backend == "triton"

    @pytest.mark.skipif(
        not HAVE_FLASHINFER_ROUTED_MXFP8, reason="test requires FlashInfer routed MXFP8"
    )
    def test_flashinfer_routed_moe_buffers_refresh_in_place(self):
        """Refit refreshes derived Major-K weights without changing graph addresses."""
        from megatron.core.inference.moe import InferenceGroupedGemmBackend
        from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.resharding.transforms import MXFP8ReshardTransform
        from megatron.core.transformer.moe.experts import InferenceGroupedMLP

        class Namespace:
            pass

        num_experts, rows, cols = 2, 96, 128
        grouped_mlp = Namespace()
        grouped_mlp.num_local_experts = num_experts
        grouped_mlp.inference_grouped_gemm_backend = InferenceGroupedGemmBackend.FLASHINFER
        grouped_mlp._concatenated_weights_built = False
        buffers = {}
        for linear_name in ("linear_fc1", "linear_fc2"):
            linear = Namespace()
            setattr(grouped_mlp, linear_name, linear)
            for expert_idx in range(num_experts):
                tensor = MXFP8Tensor.from_bf16(
                    torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda"), backend="triton"
                )
                setattr(linear, f"weight{expert_idx}", tensor)
                buffers[f"{linear_name}.weight{expert_idx}"] = tensor

        InferenceGroupedMLP._build_concatenated_mxfp8_weights(grouped_mlp)
        grouped_mlp._concatenated_weights_built = True
        data_ptrs = (
            grouped_mlp._fc1_weight.data.data_ptr(),
            grouped_mlp._fc2_weight.data.data_ptr(),
        )
        scale_ptrs = (
            grouped_mlp._fc1_weight.scale.data_ptr(),
            grouped_mlp._fc2_weight.scale.data_ptr(),
        )

        transform = MXFP8ReshardTransform(
            convertible_params=set(buffers), persistent_buffers=buffers, backend="triton"
        )
        for name in buffers:
            new_data = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
            transform.finalize_recv(name, (slice(None), slice(None)), [new_data])

        InferenceGroupedMLP.refresh_flashinfer_mxfp8_weights(grouped_mlp)

        for linear_name, routed_weight in (
            ("linear_fc1", grouped_mlp._fc1_weight),
            ("linear_fc2", grouped_mlp._fc2_weight),
        ):
            canonical = InferenceGroupedMLP._stack_mxfp8_linear_weight(
                grouped_mlp, linear_name, "triton"
            )
            expected = prepare_routed_mxfp8_weights(canonical)
            assert torch.equal(routed_weight.data, expected.data)
            assert torch.equal(routed_weight.scale, expected.scale)

        assert data_ptrs == (
            grouped_mlp._fc1_weight.data.data_ptr(),
            grouped_mlp._fc2_weight.data.data_ptr(),
        )
        assert scale_ptrs == (
            grouped_mlp._fc1_weight.scale.data_ptr(),
            grouped_mlp._fc2_weight.scale.data_ptr(),
        )


# ===========================================================================
# quantize_params_to_mxfp8
# ===========================================================================


def _pre_quantize_linear(model: torch.nn.Module) -> None:
    """Replace every Linear's BF16 weight with an ``nn.Parameter`` wrapping a
    Transformer-Engine MXFP8 tensor.  ``quantize_params_to_mxfp8`` accepts
    inputs whose ``.data`` is a TEMXFP8Tensor; it does not accept plain BF16
    ``nn.Parameter`` (production callers wrap weights via TE's ``fp8_param``
    machinery before calling this function).
    """
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    for submodule in model.modules():
        if isinstance(submodule, torch.nn.Linear):
            te_mxfp8 = quantizer(submodule.weight.data)
            submodule.weight = torch.nn.Parameter(te_mxfp8, requires_grad=False)


@pytest.mark.skipif(not _HAVE_FLASHINFER, reason="tests require FlashInfer")
class TestQuantizeParamsToMXFP8:
    """Tests for persistent buffer quantization (quantization/utils.py).

    The persistent buffer address stability is critical for CUDA graph
    compatibility — if addresses change, captured graphs segfault.
    """

    def test_basic_quantization_replaces_param(self):
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8

        model = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        _pre_quantize_linear(model)
        buffers = quantize_params_to_mxfp8(model)

        assert "weight" in buffers
        assert isinstance(buffers["weight"], MXFP8Tensor)
        assert buffers["weight"].data.shape == (64, 128)
        assert "weight" not in model._parameters

    def test_persistent_buffer_reuse_preserves_addresses(self):
        """Second call must copy into existing buffers (CUDA graph address stability)."""
        from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8

        model = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        _pre_quantize_linear(model)
        buffers = quantize_params_to_mxfp8(model)
        data_ptr = buffers["weight"].data.data_ptr()
        scale_ptr = buffers["weight"].scale.data_ptr()

        model2 = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        _pre_quantize_linear(model2)
        quantize_params_to_mxfp8(model2, persistent_buffers=buffers)

        assert buffers["weight"].data.data_ptr() == data_ptr
        assert buffers["weight"].scale.data_ptr() == scale_ptr

    def test_persistent_buffer_reuse_rejects_backend_change(self):
        """A refresh cannot reinterpret persistent scale bytes under a new backend."""
        from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8

        model = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        _pre_quantize_linear(model)
        buffers = quantize_params_to_mxfp8(model, backend="flashinfer")

        model2 = torch.nn.Linear(128, 64, bias=False).to(dtype=torch.bfloat16, device="cuda")
        _pre_quantize_linear(model2)
        with pytest.raises(ValueError, match="expected 'triton'"):
            quantize_params_to_mxfp8(model2, persistent_buffers=buffers, backend="triton")

    def test_nested_module_fqn(self):
        """Recursive quantization should produce correct fully-qualified names."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64, bias=False), torch.nn.Linear(64, 32, bias=False)
        ).to(dtype=torch.bfloat16, device="cuda")
        _pre_quantize_linear(model)
        buffers = quantize_params_to_mxfp8(model)

        assert "0.weight" in buffers and "1.weight" in buffers
        assert isinstance(buffers["0.weight"], MXFP8Tensor)


# ===========================================================================
# End-to-end MXFP8 refit integration (single-GPU)
# ===========================================================================


@pytest.mark.skipif(not _HAVE_FLASHINFER, reason="tests require FlashInfer")
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
