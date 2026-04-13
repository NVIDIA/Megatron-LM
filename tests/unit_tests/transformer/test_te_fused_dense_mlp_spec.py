# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for TEFusedDenseMLP spec selection in get_mlp_module_spec_for_backend.

Covers:
  - Baseline (no flags) → MLP
  - use_te_op_fuser alone → TEFusedMLP
  - use_grouped_gemm_for_dense=True + use_te_op_fuser=True → TEFusedDenseMLP
  - USE_GROUPED_GEMM_FOR_DENSE=1 env-var + use_te_op_fuser=True → TEFusedDenseMLP
  - use_grouped_gemm_for_dense=True without use_te_op_fuser → MLP (flag has no effect)
  - USE_GROUPED_GEMM_FOR_DENSE=1 without use_te_op_fuser → MLP (env-var has no effect)
"""

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.transformer.mlp import MLP
from megatron.core.utils import is_te_min_version

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEFusedDenseMLP, TEFusedMLP
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

# TEFusedMLP / TEFusedDenseMLP require TE >= 1.13.0
_HAVE_TE_FUSED = HAVE_TE and is_te_min_version("1.13.0")
_skip_no_te_fused = pytest.mark.skipif(
    not _HAVE_TE_FUSED, reason="Requires Transformer Engine >= 1.13.0"
)


class TestGetMlpModuleSpecForBackend:
    """Tests for the module-selection logic in get_mlp_module_spec_for_backend."""

    def test_no_flags_selects_mlp(self):
        """With no special flags the local backend returns plain MLP."""
        spec = get_mlp_module_spec_for_backend(
            backend=LocalSpecProvider(), use_te_op_fuser=False, use_grouped_gemm_for_dense=False
        )
        assert spec.module is MLP

    @_skip_no_te_fused
    def test_te_op_fuser_only_selects_te_fused_mlp(self):
        """use_te_op_fuser=True without grouped-GEMM flag selects TEFusedMLP."""
        spec = get_mlp_module_spec_for_backend(
            backend=TESpecProvider(), use_te_op_fuser=True, use_grouped_gemm_for_dense=False
        )
        assert spec.module is TEFusedMLP

    @_skip_no_te_fused
    def test_grouped_gemm_for_dense_flag_selects_te_fused_dense_mlp(self):
        """use_grouped_gemm_for_dense=True + use_te_op_fuser=True selects TEFusedDenseMLP."""
        spec = get_mlp_module_spec_for_backend(
            backend=TESpecProvider(), use_te_op_fuser=True, use_grouped_gemm_for_dense=True
        )
        assert spec.module is TEFusedDenseMLP

    @_skip_no_te_fused
    def test_env_var_selects_te_fused_dense_mlp(self, monkeypatch):
        """USE_GROUPED_GEMM_FOR_DENSE=1 acts identically to use_grouped_gemm_for_dense=True."""
        monkeypatch.setenv("USE_GROUPED_GEMM_FOR_DENSE", "1")
        spec = get_mlp_module_spec_for_backend(
            backend=TESpecProvider(),
            use_te_op_fuser=True,
            use_grouped_gemm_for_dense=False,  # param is False; env var must drive selection
        )
        assert spec.module is TEFusedDenseMLP

    def test_grouped_gemm_for_dense_without_te_op_fuser_selects_mlp(self):
        """use_grouped_gemm_for_dense=True without use_te_op_fuser has no effect; MLP is used."""
        spec = get_mlp_module_spec_for_backend(
            backend=LocalSpecProvider(), use_te_op_fuser=False, use_grouped_gemm_for_dense=True
        )
        assert spec.module is MLP

    def test_env_var_without_te_op_fuser_has_no_effect(self, monkeypatch):
        """USE_GROUPED_GEMM_FOR_DENSE=1 without use_te_op_fuser does not change the module."""
        monkeypatch.setenv("USE_GROUPED_GEMM_FOR_DENSE", "1")
        spec = get_mlp_module_spec_for_backend(backend=LocalSpecProvider(), use_te_op_fuser=False)
        assert spec.module is MLP
