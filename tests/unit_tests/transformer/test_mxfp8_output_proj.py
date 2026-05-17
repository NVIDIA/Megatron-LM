# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for megatron/core/transformer/mxfp8_output_proj.py."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mxfp8_output_proj import (
    TELinearCrossEntropyModule,
    is_te_mxfp8_output_proj_active,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_IS_BLACKWELL = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).major >= 10)


def _fake_config(*, fp8_output_proj=True, fp8=True, fp8_recipe="mxfp8"):
    """SimpleNamespace stub of TransformerConfig used by is_te_mxfp8_output_proj_active."""
    return SimpleNamespace(fp8_output_proj=fp8_output_proj, fp8=fp8, fp8_recipe=fp8_recipe)


class TestIsTEMxfp8OutputProjActive:
    """Pure-Python tests for the gating helper."""

    def test_returns_true_for_complete_mxfp8_config(self):
        if not HAVE_TE:
            pytest.skip("TE not installed; helper always returns False")
        assert is_te_mxfp8_output_proj_active(_fake_config()) is True

    def test_returns_false_when_flag_off(self):
        assert is_te_mxfp8_output_proj_active(_fake_config(fp8_output_proj=False)) is False

    def test_returns_false_when_fp8_off(self):
        assert is_te_mxfp8_output_proj_active(_fake_config(fp8=False)) is False

    def test_returns_false_for_non_mxfp8_recipe(self):
        assert is_te_mxfp8_output_proj_active(_fake_config(fp8_recipe="tensorwise")) is False
        assert is_te_mxfp8_output_proj_active(_fake_config(fp8_recipe="delayed")) is False
        assert is_te_mxfp8_output_proj_active(_fake_config(fp8_recipe="blockwise")) is False

    def test_accepts_enum_style_recipe(self):
        if not HAVE_TE:
            pytest.skip("TE not installed; helper always returns False")
        enum_like = SimpleNamespace(value="mxfp8")
        assert is_te_mxfp8_output_proj_active(_fake_config(fp8_recipe=enum_like)) is True

    def test_returns_false_when_attributes_missing(self):
        assert is_te_mxfp8_output_proj_active(SimpleNamespace()) is False


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestTELinearCrossEntropyModuleRejections:
    """Constructor validation checks.

    Each invalid kwarg triggers a raise before super().__init__() runs, so these
    tests do not require GPU initialization of the TE parent.
    """

    def _kwargs(self, **overrides):
        kwargs = dict(
            input_size=8,
            output_size=16,
            config=_fake_config(),
            init_method=lambda w: None,
        )
        kwargs.update(overrides)
        return kwargs

    def test_rejects_inactive_config(self):
        with pytest.raises(RuntimeError, match="fp8_output_proj=True"):
            TELinearCrossEntropyModule(**self._kwargs(config=_fake_config(fp8=False)))

    def test_rejects_keep_master_weight_for_test(self):
        with pytest.raises(ValueError, match="keep_master_weight_for_test"):
            TELinearCrossEntropyModule(**self._kwargs(keep_master_weight_for_test=True))

    def test_rejects_skip_weight_param_allocation(self):
        with pytest.raises(ValueError, match="skip_weight_param_allocation"):
            TELinearCrossEntropyModule(**self._kwargs(skip_weight_param_allocation=True))

    def test_rejects_embedding_activation_buffer(self):
        with pytest.raises(ValueError, match="defer_embedding_wgrad_compute"):
            TELinearCrossEntropyModule(**self._kwargs(embedding_activation_buffer=[]))

    def test_rejects_grad_output_buffer(self):
        with pytest.raises(ValueError, match="defer_embedding_wgrad_compute"):
            TELinearCrossEntropyModule(**self._kwargs(grad_output_buffer=[]))

    def test_rejects_disable_grad_reduce(self):
        with pytest.raises(ValueError, match="disable_grad_reduce"):
            TELinearCrossEntropyModule(**self._kwargs(disable_grad_reduce=True))


class TestGPTModelOutputLayerSelection:
    """Verify GPTModel picks the right output-layer class based on config."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_default_uses_column_parallel_linear(self):
        """Without fp8_output_proj, the LM head is the plain ColumnParallelLinear."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )
        assert isinstance(model.output_layer, tensor_parallel.ColumnParallelLinear)
        assert not isinstance(model.output_layer, TELinearCrossEntropyModule)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not _IS_BLACKWELL, reason="MXFP8 output projection requires Blackwell (SM >= 10)"
    )
    def test_mxfp8_active_uses_te_lm_head(self):
        """With fp8_output_proj=True under mxfp8, the LM head is TELinearCrossEntropyModule."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid",
            fp8_recipe="mxfp8",
            fp8_output_proj=True,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=8,
        )
        assert isinstance(model.output_layer, TELinearCrossEntropyModule)
