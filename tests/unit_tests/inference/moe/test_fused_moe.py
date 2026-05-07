# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.moe.fused_moe import ActivationType, _get_activation_func


class TestActivationType:

    def test_squared_relu_value(self):
        """SQUARED_RELU is the canonical string value for the enum."""
        assert ActivationType.SQUARED_RELU.value == "squared_relu"

    def test_lookup_by_value(self):
        """ActivationType('squared_relu') returns the SQUARED_RELU member."""
        assert ActivationType("squared_relu") is ActivationType.SQUARED_RELU


class TestGetActivationFunc:

    def test_squared_relu_unfused_returns_padded(self):
        """fused_quant=False returns the padded_squared_relu kernel."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        fn = _get_activation_func(ActivationType.SQUARED_RELU, fused_quant=False)
        assert fn is padded_squared_relu

    def test_squared_relu_fused_returns_fused(self):
        """fused_quant=True returns the fused activation+quantize kernel."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        fn = _get_activation_func(ActivationType.SQUARED_RELU, fused_quant=True)
        assert fn is squared_relu_and_quantize_mxfp8

    def test_unsupported_activation_raises(self):
        """A bogus activation type triggers ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation type"):
            _get_activation_func("not-an-activation")
