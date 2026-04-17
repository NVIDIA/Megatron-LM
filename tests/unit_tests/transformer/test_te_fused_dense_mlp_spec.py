# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch.nn.functional as F
import pytest

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEFusedDenseMLP,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

_SKIP_REASON = "TEFusedDenseMLP requires Transformer Engine >= 2.14.0"
_SKIP = not HAVE_TE or not is_te_min_version("2.14.0")


def _make_submodules():
    return MLPSubmodules(
        linear_fc1=TELayerNormColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )


def _make_config(**overrides):
    defaults = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        use_cpu_initialization=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestTEFusedDenseMLPSpec:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_instantiation(self):
        config = _make_config()
        mlp = TEFusedDenseMLP(config, _make_submodules())
        assert isinstance(mlp, TEFusedDenseMLP)

    def test_wrong_activation_raises(self):
        config = _make_config(activation_func=F.gelu, gated_linear_unit=False)
        with pytest.raises(ValueError, match="SwiGLU activation"):
            TEFusedDenseMLP(config, _make_submodules())

    def test_gated_linear_unit_false_raises(self):
        config = _make_config(gated_linear_unit=False)
        with pytest.raises(ValueError, match="SwiGLU activation"):
            TEFusedDenseMLP(config, _make_submodules())

    def test_add_bias_linear_raises(self):
        config = _make_config(add_bias_linear=True)
        with pytest.raises(ValueError, match="add_bias_linear"):
            TEFusedDenseMLP(config, _make_submodules())
