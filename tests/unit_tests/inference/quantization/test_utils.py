# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.inference.quantization import utils as qutils
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor


class _FakeTEMXFP8Tensor:
    """Stand-in for transformer_engine.pytorch.tensor.mxfp8_tensor.MXFP8Tensor."""

    def __init__(self, dequantized: torch.Tensor):
        self._dq = dequantized
        self.shape = dequantized.shape
        self.dtype = dequantized.dtype
        self.device = dequantized.device

    def dequantize(self):
        return self._dq


class TestShouldQuantizeParam:

    def test_cpu_param_is_not_quantized(self):
        """CPU parameters return False regardless of type."""
        cpu_param = torch.zeros(4, 4)
        assert qutils._should_quantize_param(cpu_param) is False

    def test_te_mxfp8_tensor_is_quantized(self):
        """A TEMXFP8Tensor instance returns True (when HAVE_TE)."""
        # Force HAVE_TE=True and TEMXFP8Tensor=our fake.
        with (
            patch.object(qutils, "HAVE_TE", True),
            patch.object(qutils, "TEMXFP8Tensor", _FakeTEMXFP8Tensor, create=True),
        ):
            te_param = _FakeTEMXFP8Tensor(
                torch.zeros(2, 2, device="cuda" if torch.cuda.is_available() else "cpu")
            )
            # Tag .is_cuda for the CPU fallback.
            te_param.is_cuda = True
            assert qutils._should_quantize_param(te_param) is True

    def test_mxfp8_tensor_is_quantized(self):
        """A direct MXFP8Tensor instance is recognized."""
        t = MXFP8Tensor(
            data=torch.zeros(4, 32, dtype=torch.uint8), scale=torch.zeros(16, dtype=torch.uint8)
        )
        # Mark fake .is_cuda (function checks before isinstance).
        t.is_cuda = True
        assert qutils._should_quantize_param(t) is True

    def test_param_with_data_attr_pointing_to_mxfp8(self):
        """A param wrapping an MXFP8Tensor in `.data` is recognized."""
        wrapped = MagicMock()
        wrapped.is_cuda = True
        wrapped.data = MXFP8Tensor(
            data=torch.zeros(4, 32, dtype=torch.uint8), scale=torch.zeros(16, dtype=torch.uint8)
        )
        assert qutils._should_quantize_param(wrapped) is True

    def test_plain_cuda_bf16_is_not_quantized(self):
        """A plain BF16 CUDA tensor that is not an MXFP8 type returns False."""
        # We can't easily build a real CUDA tensor without GPU; use a MagicMock.
        wrapped = MagicMock()
        wrapped.is_cuda = True
        wrapped.data = torch.zeros(4, 32)  # not MXFP8Tensor
        with (
            patch.object(qutils, "HAVE_TE", True),
            patch.object(qutils, "TEMXFP8Tensor", _FakeTEMXFP8Tensor, create=True),
        ):
            # `wrapped` is a MagicMock, not a TEMXFP8Tensor / MXFP8Tensor / wrapper of either.
            assert qutils._should_quantize_param(wrapped) is False


class TestToBf16:

    def test_te_tensor_is_dequantized(self):
        """A TEMXFP8Tensor goes through .dequantize()."""
        bf16 = torch.zeros(4, 4)
        te = _FakeTEMXFP8Tensor(bf16)
        with (
            patch.object(qutils, "HAVE_TE", True),
            patch.object(qutils, "TEMXFP8Tensor", _FakeTEMXFP8Tensor, create=True),
        ):
            out = qutils._to_bf16(te)
        assert out is bf16

    def test_wrapped_te_tensor_is_dequantized(self):
        """A wrapper whose .data is a TEMXFP8Tensor goes through .data.dequantize()."""
        bf16 = torch.zeros(4, 4)
        te = _FakeTEMXFP8Tensor(bf16)
        wrapper = MagicMock()
        wrapper.data = te
        with (
            patch.object(qutils, "HAVE_TE", True),
            patch.object(qutils, "TEMXFP8Tensor", _FakeTEMXFP8Tensor, create=True),
        ):
            out = qutils._to_bf16(wrapper)
        assert out is bf16

    def test_plain_param_is_cast_to_bf16(self):
        """A plain non-TE parameter is cast to bfloat16 via .data.to(bfloat16)."""
        param = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
        with patch.object(qutils, "HAVE_TE", False):
            out = qutils._to_bf16(param)
        assert out.dtype == torch.bfloat16


class TestCollectMxfp8ParamMetadata:

    def test_returns_empty_when_no_quantizable_params(self):
        """A model with no quantizable parameters yields an empty metadata dict."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        with patch.object(qutils, "_should_quantize_param", return_value=False):
            assert qutils.collect_mxfp8_param_metadata(model) == {}

    def test_records_shape_dtype_device_for_plain_params(self):
        """For plain (non-TE) params, metadata records (shape, dtype, device) directly."""
        model = torch.nn.Linear(4, 8, bias=False)
        with (
            patch.object(qutils, "_should_quantize_param", return_value=True),
            patch.object(qutils, "HAVE_TE", False),
        ):
            md = qutils.collect_mxfp8_param_metadata(model)
        assert "weight" in md
        shape, dtype, device = md["weight"]
        assert shape == model.weight.shape
        assert dtype == model.weight.dtype

    def test_records_dequantized_metadata_for_te_params(self):
        """For TE-quantized params, metadata records dequantized shape/dtype/device."""
        model = torch.nn.Linear(4, 4, bias=False)

        # Replace the parameter with a TE-typed object.
        bf16_dq = torch.zeros(8, 4, dtype=torch.bfloat16)
        te_param = _FakeTEMXFP8Tensor(bf16_dq)
        # collect_mxfp8_param_metadata iterates named_parameters; we inject by overwriting
        # _parameters['weight'] in-place. A MagicMock that exposes .dequantize() is enough.
        model._parameters["weight"] = te_param  # type: ignore[assignment]

        with (
            patch.object(qutils, "HAVE_TE", True),
            patch.object(qutils, "TEMXFP8Tensor", _FakeTEMXFP8Tensor, create=True),
            patch.object(qutils, "_should_quantize_param", return_value=True),
        ):
            md = qutils.collect_mxfp8_param_metadata(model)

        assert md["weight"][0] == bf16_dq.shape
        assert md["weight"][1] == bf16_dq.dtype


class TestQuantizeModelToMxfp8:

    def test_assert_te_required(self):
        """quantize_model_to_mxfp8 asserts HAVE_TE."""
        model = torch.nn.Linear(4, 4)
        with patch.object(qutils, "HAVE_TE", False):
            with pytest.raises(AssertionError):
                qutils.quantize_model_to_mxfp8(model)
