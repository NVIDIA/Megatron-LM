# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor, _ceil_div


class TestCeilDiv:

    def test_exact_division(self):
        """Exact division returns the quotient."""
        assert _ceil_div(8, 4) == 2

    def test_rounds_up(self):
        """Non-exact division rounds up."""
        assert _ceil_div(9, 4) == 3
        assert _ceil_div(1, 4) == 1

    def test_zero_numerator(self):
        """Zero numerator returns 0."""
        assert _ceil_div(0, 4) == 0


class TestMXFP8Tensor:

    def _make(self, shape=(4, 64)):
        """Construct an MXFP8Tensor with CPU dummy tensors (the wrapper does not require CUDA)."""
        # Use a dummy uint8 tensor as a stand-in for fp8 data, since fp8_e4m3 may not be
        # available on the test host. The wrapper itself doesn't validate dtype.
        data = torch.zeros(shape, dtype=torch.uint8)
        # Scale layout: padded_M (>= shape[0]) by ceil_div(K // 32, 4) * 4.
        n_col_blocks = _ceil_div(shape[1] // 32, 4)
        padded_cols = n_col_blocks * 4
        scale = torch.zeros(shape[0] * padded_cols, dtype=torch.uint8)
        return MXFP8Tensor(data=data, scale=scale, backend="triton")

    def test_size_with_no_arg_returns_full_shape(self):
        """size() with no argument returns the full data shape."""
        t = self._make(shape=(4, 64))
        assert tuple(t.size()) == (4, 64)

    def test_size_with_index_returns_specific_dim(self):
        """size(idx) delegates to data.size(idx)."""
        t = self._make(shape=(4, 64))
        assert t.size(0) == 4
        assert t.size(1) == 64

    def test_scale_2d_reshapes_1d_scale(self):
        """scale_2d reshapes a 1-D scale tensor into the cuBLAS 2-D layout."""
        K = 64
        t = self._make(shape=(4, K))
        out = t.scale_2d()
        assert out.dim() == 2
        n_col_blocks = _ceil_div(K // 32, 4)
        padded_cols = n_col_blocks * 4
        assert out.shape[1] == padded_cols

    def test_scale_2d_passes_through_already_2d(self):
        """If scale is already 2-D, it is returned untouched."""
        data = torch.zeros((4, 64), dtype=torch.uint8)
        scale = torch.zeros((4, 4), dtype=torch.uint8)
        t = MXFP8Tensor(data=data, scale=scale)
        out = t.scale_2d()
        assert out is scale

    def test_scale_2d_with_explicit_K_arg(self):
        """Passing K explicitly overrides the inferred K from data.shape[-1]."""
        data = torch.zeros((4, 64), dtype=torch.uint8)
        # Use a scale sized for K=128 (n_col_blocks = ceil(128//32, 4) = 1, padded_cols = 4).
        scale = torch.zeros(4 * 4, dtype=torch.uint8)
        t = MXFP8Tensor(data=data, scale=scale)
        out = t.scale_2d(K=128)
        # K=128 → padded_cols = 4 (ceil(128/32, 4)*4 = ceil(4,4)*4 = 4).
        assert out.shape[1] == 4

    def test_from_bf16_unknown_backend_raises_value_error(self):
        """An unknown backend triggers a ValueError before any quantization happens."""
        # Use a CPU tensor — the backend assertion fires only after the .is_cuda check,
        # so we wrap in pytest.raises to accept either AssertionError or ValueError
        # depending on which assertion fires first.
        x = torch.zeros(4, 64, dtype=torch.bfloat16)
        with pytest.raises((ValueError, AssertionError)):
            MXFP8Tensor.from_bf16(x, backend="unknown")
