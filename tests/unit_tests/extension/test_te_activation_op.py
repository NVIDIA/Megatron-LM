# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the TEActivationOp activation-to-Transformer-Engine-op mapping."""

import pytest
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE, TEActivationOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

if HAVE_TE:
    import transformer_engine as te

pytestmark = pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.13.0")),
    reason="TEActivationOp requires Transformer Engine 1.13.0+",
)


def _config(activation_func, gated_linear_unit):
    """Minimal TransformerConfig carrying the activation fields TEActivationOp reads."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        activation_func=activation_func,
        gated_linear_unit=gated_linear_unit,
        use_te_activation_func=True,
        add_bias_linear=False,
    )


class TestTEActivationOpMapping:
    """Each supported (activation_func, gated_linear_unit) pair maps to its TE op."""

    @pytest.mark.parametrize(
        "activation_func,gated_linear_unit,expected_op_name",
        [
            (F.silu, True, "SwiGLU"),
            (F.gelu, True, "GEGLU"),
            (F.relu, True, "ReGLU"),
            (F.gelu, False, "GELU"),
            (F.relu, False, "ReLU"),
        ],
    )
    def test_supported_activations_map_to_expected_op(
        self, activation_func, gated_linear_unit, expected_op_name
    ):
        layer = TEActivationOp(_config(activation_func, gated_linear_unit))
        assert isinstance(layer, getattr(te.pytorch.ops, expected_op_name))

    @pytest.mark.skipif(
        not is_te_min_version("2.8.0"), reason="Non-gated SiLU requires Transformer Engine 2.8.0+"
    )
    def test_non_gated_silu_maps_to_silu(self):
        layer = TEActivationOp(_config(F.silu, False))
        assert isinstance(layer, te.pytorch.ops.SiLU)

    def test_unsupported_activation_raises(self):
        # squared_relu has no Transformer Engine activation op counterpart.
        from megatron.core.activations import squared_relu

        config = _config(F.gelu, False)
        config.activation_func = squared_relu
        with pytest.raises(Exception, match="are supported by"):
            TEActivationOp(config)
