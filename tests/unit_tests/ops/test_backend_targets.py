# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every backend returns the exact target it returned before selection was unified.

These are the equivalence tests the migration turns on: they name the concrete class each
operation resolves to, so moving selection around cannot quietly change what a model is
built from.
"""

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.ops import BackendOptions, build_spec_provider, get_backend
from megatron.core.ops._availability import is_installed
from megatron.core.ops.norm import WrappedTorchNorm
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

    def test_local_is_torch_regardless_of_what_is_installed(self):
        """Installing an optional package must not change which kernel a local run gets."""
        assert get_backend("local").layer_norm(rms_norm=False) is WrappedTorchNorm
        assert get_backend("local").layer_norm(rms_norm=True) is WrappedTorchNorm

    def test_apex_is_selected_explicitly(self):
        if not is_installed("apex"):
            pytest.skip("Apex is not installed")
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

        provider = build_spec_provider(
            BackendOptions(transformer_impl="local", operation_backends={"layer_norm": "apex"})
        )
        assert provider.layer_norm(rms_norm=False) is FusedLayerNorm

    def test_a_config_torch_norm_cannot_serve_is_refused_with_the_fix(self):
        """Better than WrappedTorchNorm asserting deep inside construction."""
        with pytest.raises(ValueError, match="sequence parallelism.*--op-backend layer_norm=apex"):
            build_spec_provider(BackendOptions(transformer_impl="local", sequence_parallel=True))

    def test_layer_norm_choice_does_not_leak_between_calls(self):
        """Asking for an RMSNorm must not change what a later LayerNorm request returns."""
        backend = get_backend("local")
        first = backend.layer_norm(rms_norm=False)
        backend.layer_norm(rms_norm=True)
        assert backend.layer_norm(rms_norm=False) is first

    def test_no_router_override(self):
        assert get_backend("local").moe_router() is None

    def test_unowned_linear_says_how_to_select_one(self):
        """Megatron Core has no non-parallel linear, and the slot says so."""
        with pytest.raises(NotImplementedError, match="linear.linear"):
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
        from megatron.core.ops.norm.backends import TENormWithResidual

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
