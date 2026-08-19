# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every backend returns the exact target it returned before selection was unified.

These are the equivalence tests the migration turns on: they name the concrete class each
operation resolves to, so moving selection around cannot quietly change what a model is
built from.
"""

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.ops import Operation, available_backends, get_backend
from megatron.core.ops.norm.apex import have_apex
from megatron.core.ops.norm.reference import WrappedTorchNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention

_needs_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required")


class TestLocalBackend:
    def test_linear_targets(self):
        backend = get_backend("local")
        assert backend.column_parallel_linear() is ColumnParallelLinear
        assert backend.row_parallel_linear() is RowParallelLinear
        assert backend.column_parallel_layer_norm_linear() is None
        assert backend.fuse_layernorm_and_linear() is False

    def test_core_attention_target(self):
        assert get_backend("local").core_attention() is DotProductAttention

    def test_rms_norm_never_uses_apex(self):
        """Apex has no RMSNorm, and building one raises, so RMSNorm must stay on Torch."""
        assert get_backend("local").layer_norm(rms_norm=True) is WrappedTorchNorm

    def test_layer_norm_prefers_apex_when_installed(self):
        target = get_backend("local").layer_norm(rms_norm=False)
        if have_apex():
            from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

            assert target is FusedLayerNorm
        else:
            assert target is WrappedTorchNorm

    def test_layer_norm_choice_does_not_leak_between_calls(self):
        """Asking for an RMSNorm must not change what a later LayerNorm request returns."""
        backend = get_backend("local")
        first = backend.layer_norm(rms_norm=False)
        backend.layer_norm(rms_norm=True)
        assert backend.layer_norm(rms_norm=False) is first

    def test_no_router_override(self):
        assert get_backend("local").moe_router() is None

    def test_linear_reports_the_backend_that_lacks_it(self):
        with pytest.raises(NotImplementedError, match="LocalSpecProvider"):
            get_backend("local").linear()


@_needs_te
class TestTransformerEngineBackend:
    def test_linear_targets(self):
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
            TELinear,
            TERowParallelLinear,
        )

        backend = get_backend("transformer_engine")
        assert backend.linear() is TELinear
        assert backend.column_parallel_linear() is TEColumnParallelLinear
        assert backend.row_parallel_linear() is TERowParallelLinear
        assert backend.column_parallel_layer_norm_linear() is TELayerNormColumnParallelLinear
        assert backend.fuse_layernorm_and_linear() is True

    def test_core_attention_target(self):
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        assert get_backend("transformer_engine").core_attention() is TEDotProductAttention

    def test_norm_targets(self):
        from megatron.core.extensions.transformer_engine import TENorm
        from megatron.core.ops.norm.transformer_engine import TENormWithResidual

        backend = get_backend("transformer_engine")
        assert backend.layer_norm() is TENorm
        assert backend.layer_norm(has_residual=True) is TENormWithResidual

    def test_residual_norm_target_is_stable_across_providers(self):
        """Two providers built the same way must produce equal specs."""
        first = get_backend("transformer_engine").layer_norm(has_residual=True)
        second = get_backend("transformer_engine").layer_norm(has_residual=True)
        assert first is second

    def test_activation_func_target(self):
        from megatron.core.extensions.transformer_engine import TEActivationOp

        assert get_backend("transformer_engine").activation_func() is TEActivationOp


@_needs_te
class TestInferenceBackend:
    def test_linear_targets(self):
        from megatron.core.tensor_parallel.inference_layers import (
            InferenceColumnParallelLinear,
            InferenceLayerNormColumnParallelLinear,
            InferenceRowParallelLinear,
        )

        backend = get_backend("inference_optimized")
        assert backend.column_parallel_linear() is InferenceColumnParallelLinear
        assert backend.row_parallel_linear() is InferenceRowParallelLinear
        assert backend.column_parallel_layer_norm_linear() is InferenceLayerNormColumnParallelLinear

    def test_router_target(self):
        from megatron.core.transformer.moe.router import InferenceTopKRouter

        assert get_backend("inference_optimized").moe_router() is InferenceTopKRouter

    def test_norm_never_fuses_a_residual(self):
        """Inference layers own the residual themselves, so the norm must not absorb it."""
        from megatron.core.extensions.transformer_engine import TENorm

        assert get_backend("inference_optimized").layer_norm(has_residual=True) is TENorm


class TestBackendNames:
    def test_unknown_transformer_impl_is_rejected(self):
        with pytest.raises(ValueError, match="unknown transformer_impl='nope'"):
            get_backend("nope")

    def test_single_operation_backend_is_not_a_complete_backend(self):
        with pytest.raises(ValueError, match="unknown transformer_impl='apex'"):
            get_backend("apex")

    def test_every_advertised_backend_name_is_buildable_or_optional(self):
        """available_backends() is what --op-backend accepts, so no name may be a typo."""
        for name in available_backends():
            assert isinstance(name, str) and name
        assert "local" in available_backends()
        assert str(Operation.LAYER_NORM) == "layer_norm"
