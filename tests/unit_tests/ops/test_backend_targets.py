# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every backend returns the exact target it returned before the implementations moved.

These are the equivalence tests the migration turns on: they name the concrete class each
backend resolves to, so relocating an implementation cannot quietly change what a model is
built from. Selection is not exercised here -- these classes are what a provider picks
between, and the providers are tested where they live.
"""

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.ops import is_installed
from megatron.core.ops.attention import AttentionLocal, AttentionTE
from megatron.core.ops.linear import LinearLocal, LinearTE
from megatron.core.ops.mlp import MlpMegatron
from megatron.core.ops.moe import MoeLocal, MoeTE
from megatron.core.ops.norm import NormApex, NormTE, NormTorch, WrappedTorchNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention

_needs_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required")


class TestLocalBackends:
    def test_linear_targets(self):
        backend = LinearLocal()
        assert backend.column_parallel_linear() is ColumnParallelLinear
        assert backend.row_parallel_linear() is RowParallelLinear
        assert backend.column_parallel_layer_norm_linear() is None

    def test_core_attention_target(self):
        assert AttentionLocal().core_attention() is DotProductAttention

    def test_torch_norm_is_torch_regardless_of_what_is_installed(self):
        """Installing an optional package must not change which kernel this backend gives."""
        assert NormTorch().layer_norm(rms_norm=False) is WrappedTorchNorm
        assert NormTorch().layer_norm(rms_norm=True) is WrappedTorchNorm

    def test_apex_is_a_separate_backend(self):
        """Apex is its own class, so choosing it is a decision rather than an accident."""
        if not is_installed("apex"):
            pytest.skip("Apex is not installed")
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

        assert NormApex().layer_norm(rms_norm=False) is FusedLayerNorm

    def test_layer_norm_choice_does_not_leak_between_calls(self):
        """Asking for an RMSNorm must not change what a later LayerNorm request returns.

        The implementation this replaces mutated a module-level global to do this.
        """
        backend = NormTorch()
        first = backend.layer_norm(rms_norm=False)
        backend.layer_norm(rms_norm=True)
        assert backend.layer_norm(rms_norm=False) is first

    def test_no_router_override(self):
        assert MoeLocal().moe_router() is None

    def test_megatron_mlp_target(self):
        from megatron.core.transformer.mlp import MLP

        assert MlpMegatron().mlp_module() is MLP


@_needs_te
class TestTransformerEngineBackends:
    def test_linear_targets(self):
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
            TELinear,
            TERowParallelLinear,
        )

        backend = LinearTE()
        assert backend.linear() is TELinear
        assert backend.column_parallel_linear() is TEColumnParallelLinear
        assert backend.row_parallel_linear() is TERowParallelLinear
        assert backend.column_parallel_layer_norm_linear() is TELayerNormColumnParallelLinear

    def test_core_attention_target(self):
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        assert AttentionTE().core_attention() is TEDotProductAttention

    def test_norm_targets(self):
        from megatron.core.extensions.transformer_engine import TENorm

        assert NormTE().layer_norm() is TENorm

    def test_residual_fusion_target_is_stable(self):
        """The residual-fused norm must be one class, not a new one per call, or a spec
        built twice stops comparing equal."""
        assert NormTE().layer_norm(has_residual=True) is NormTE().layer_norm(has_residual=True)

    def test_activation_target(self):
        from megatron.core.extensions.transformer_engine import TEActivationOp

        assert MoeTE().activation_func() is TEActivationOp
