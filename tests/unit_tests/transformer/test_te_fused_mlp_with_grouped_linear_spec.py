# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEFusedMLPWithGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

_SKIP_REASON = "TEFusedMLPWithGroupedLinear requires Transformer Engine >= 2.14.0"
_SKIP = not HAVE_TE or not is_te_min_version("2.14.0")


def _make_submodules():
    return MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear)


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
class TestTEFusedMLPWithGroupedLinearSpec:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_instantiation(self):
        config = _make_config()
        mlp = TEFusedMLPWithGroupedLinear(config, _make_submodules())
        assert isinstance(mlp, TEFusedMLPWithGroupedLinear)

    def test_wrong_activation_raises(self):
        config = _make_config(activation_func=F.gelu, gated_linear_unit=False)
        with pytest.raises(ValueError, match="SwiGLU activation"):
            TEFusedMLPWithGroupedLinear(config, _make_submodules())

    def test_gated_linear_unit_false_raises(self):
        config = _make_config(gated_linear_unit=False)
        with pytest.raises(ValueError, match="SwiGLU activation"):
            TEFusedMLPWithGroupedLinear(config, _make_submodules())

    def test_add_bias_linear_raises(self):
        config = _make_config(add_bias_linear=True)
        with pytest.raises(ValueError, match="add_bias_linear"):
            TEFusedMLPWithGroupedLinear(config, _make_submodules())

    def test_tensor_parallel_falls_back_to_base_fused_mlp(self, monkeypatch):
        import megatron.core.extensions.transformer_engine as te_ext

        calls = {}

        def fake_forward(self, hidden_states, **kwargs):
            calls["hidden_states"] = hidden_states
            calls["kwargs"] = kwargs
            return "output", "bias"

        monkeypatch.setattr(te_ext, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(te_ext.TEFusedMLP, "forward", fake_forward)

        config = _make_config()
        mlp = TEFusedMLPWithGroupedLinear(config, _make_submodules())
        hidden_states = object()

        assert mlp.forward(hidden_states, test_kwarg=True) == ("output", "bias")
        assert calls == {"hidden_states": hidden_states, "kwargs": {"test_kwarg": True}}

    def test_norm_seq_not_registered_as_submodule(self):
        # _norm_seq must be stored in a tuple (not directly as nn.Module) to avoid
        # PyTorch registering it as a submodule, which would duplicate norm weights
        # in state_dict/parameters. Verify it starts as None and is never a bare Module.
        import torch.nn as nn

        config = _make_config()
        mlp = TEFusedMLPWithGroupedLinear(config, _make_submodules())
        assert mlp._norm_seq is None
        assert '_norm_seq' not in dict(mlp.named_children())

        # Simulate what _make_fused_impl does and confirm the tuple-wrap holds.
        import transformer_engine.pytorch.ops as te_ops

        fake_seq = te_ops.Sequential()
        mlp._norm_seq = (fake_seq,)
        assert not isinstance(mlp._norm_seq, nn.Module)
        assert '_norm_seq' not in dict(mlp.named_children())
