# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests that JIT warmup skips only RNG-free fused kernels when disabled.

Uses mocks only (no CUDA required). Patches use
``megatron.training.initialize.<name>`` because ``_warmup_jit_function`` closes
over that module's globals (imported ``get_args``, ``torch``, fusion ops).

Determinism contract under test: the ``torch.rand`` warmup draws and the fused
bias+dropout+add warmup consume the default CUDA RNG stream before training
starts, and downstream numerics (functional-test golden values) encode exactly
those consumptions. They must therefore run unconditionally with their
historical shapes and order regardless of arguments. Only the RNG-free
bias_swiglu/bias_gelu/bias_geglu kernel calls may be skipped.
"""

from unittest import mock

import pytest
import torch

from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer import TransformerConfig
from megatron.training.initialize import _warmup_jit_function


def _config(**overrides):
    base = dict(
        bf16=False,
        fp16=False,
        use_te_activation_func=False,
        gated_linear_unit=False,
        activation_func=torch.nn.functional.gelu,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        sequence_parallel=False,
        ffn_hidden_size=64,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        num_layers=2,
        num_attention_heads=4,
        hidden_size=32,
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _fake_rand(*size, **kwargs):
    """Mirror ``torch.rand`` shapes used in ``_warmup_jit_function`` but on CPU."""
    dtype = kwargs.get("dtype", torch.float32)
    if len(size) == 1 and isinstance(size[0], tuple):
        shape = size[0]
    elif size:
        shape = size if len(size) > 1 else (size[0],)
    else:
        shape = ()
    return torch.zeros(shape, dtype=dtype, device="cpu")


@mock.patch("megatron.training.initialize.torch.cuda.empty_cache")
@mock.patch("megatron.training.initialize.bias_dropout_add_fused_train")
@mock.patch("megatron.training.initialize.torch.rand", side_effect=_fake_rand)
@mock.patch("megatron.training.initialize.get_args")
class TestJitWarmupSkippedWhenFusionDisabled:
    @mock.patch("megatron.training.initialize.bias_swiglu")
    def test_skips_bias_swiglu(self, mock_swiglu, mock_get_args, mock_rand, mock_dropout, _cache):
        config = _config(
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=False,
        )
        _warmup_jit_function(config, sequence_length=8, micro_batch_size=1)
        mock_get_args.assert_not_called()
        mock_swiglu.assert_not_called()
        # RNG-consuming draws and dropout warmup stay unconditional for
        # golden-value parity; mock dropout call count: zip(...) × range(5).
        assert mock_dropout.call_count == 10
        assert mock_rand.call_count == 5

    @mock.patch("megatron.training.initialize.bias_gelu")
    def test_skips_bias_gelu(self, mock_gelu, mock_get_args, mock_rand, mock_dropout, _cache):
        config = _config(bias_activation_fusion=False, gated_linear_unit=False)
        _warmup_jit_function(config, sequence_length=8, micro_batch_size=1)
        mock_gelu.assert_not_called()
        assert mock_dropout.call_count == 10
        assert mock_rand.call_count == 5

    @mock.patch("megatron.training.initialize.bias_geglu")
    def test_skips_bias_geglu(self, mock_geglu, mock_get_args, mock_rand, mock_dropout, _cache):
        config = _config(bias_activation_fusion=False, gated_linear_unit=True)
        _warmup_jit_function(config, sequence_length=8, micro_batch_size=1)
        mock_geglu.assert_not_called()
        assert mock_dropout.call_count == 10
        assert mock_rand.call_count == 5

    @mock.patch("megatron.training.initialize.bias_geglu")
    @mock.patch("megatron.training.initialize.bias_gelu")
    def test_skips_gelu_and_geglu_when_quick_geglu(
        self, mock_gelu, mock_geglu, mock_get_args, mock_rand, mock_dropout, _cache
    ):
        # Even with bias_gelu_fusion True, quick_geglu skips gelu and geglu warmups.
        config = _config(
            activation_func=quick_gelu, bias_activation_fusion=True, gated_linear_unit=True
        )
        _warmup_jit_function(config, sequence_length=8, micro_batch_size=1)
        mock_gelu.assert_not_called()
        mock_geglu.assert_not_called()
        # Draws and dropout warmup are still unconditional.
        assert mock_dropout.call_count == 10
        assert mock_rand.call_count == 5

    @mock.patch("megatron.training.initialize.bias_geglu")
    @mock.patch("megatron.training.initialize.bias_gelu")
    @mock.patch("megatron.training.initialize.bias_swiglu")
    def test_te_activation_skips_activation_warmups(
        self, mock_swiglu, mock_gelu, mock_geglu, mock_get_args, mock_rand, mock_dropout, _cache
    ):
        # TE activation skips only the RNG-free activation kernels; the
        # RNG-consuming dropout warmup still runs for golden-value parity.
        config = _config(
            use_te_activation_func=True,
            activation_func=torch.nn.functional.silu,
            bias_activation_fusion=False,
            gated_linear_unit=True,
        )
        _warmup_jit_function(config, sequence_length=8, micro_batch_size=1)
        mock_swiglu.assert_not_called()
        mock_gelu.assert_not_called()
        mock_geglu.assert_not_called()
        assert mock_dropout.call_count == 10
        assert mock_rand.call_count == 5


@mock.patch("megatron.training.initialize.bias_dropout_add_fused_train")
@mock.patch("megatron.training.initialize.bias_geglu")
@mock.patch("megatron.training.initialize.bias_gelu")
@mock.patch("megatron.training.initialize.torch.cuda.empty_cache")
@mock.patch("megatron.training.initialize.torch.rand", side_effect=_fake_rand)
@mock.patch("megatron.training.initialize.bias_swiglu")
@mock.patch("megatron.training.initialize.get_args")
def test_calls_bias_swiglu_when_fusion_enabled(
    mock_get_args, mock_swiglu, mock_torch_rand, _empty_cache, mock_gelu, mock_geglu, mock_dropout
):
    config = _config(activation_func=torch.nn.functional.silu, gated_linear_unit=True)
    mock_swiglu.return_value = torch.tensor(0.0)
    _warmup_jit_function(config, sequence_length=8, micro_batch_size=1)
    assert mock_swiglu.call_count == 10  # zip([True,True],[False,True]) × range(5)
    mock_gelu.assert_not_called()
    mock_geglu.assert_not_called()
    assert mock_dropout.call_count == 10  # unconditional for RNG parity
    assert mock_torch_rand.call_count == 5  # all draws unconditional


@mock.patch("megatron.training.initialize.bias_geglu")
@mock.patch("megatron.training.initialize.torch.cuda.empty_cache")
@mock.patch("megatron.training.initialize.torch.rand", side_effect=_fake_rand)
@mock.patch("megatron.training.initialize.bias_dropout_add_fused_train")
@mock.patch("megatron.training.initialize.get_args")
def test_dropout_warmup_and_geglu_cat(mock_get_args, mock_dropout, mock_rand, _cache, mock_geglu):
    """Gated GEGLU warmup uses torch.cat (RNG-free) to get 2x-ffn tensors, and
    the fused dropout warmup runs even when bias_dropout_fusion is False so
    RNG consumption stays identical to the historical behavior."""
    config = _config(gated_linear_unit=True, bias_activation_fusion=True, bias_dropout_fusion=False)
    _warmup_jit_function(config, sequence_length=8, micro_batch_size=2)
    assert mock_geglu.call_count == 10
    # bias_geglu receives 2x-ffn tensors built via torch.cat
    bias_arg, input_arg = mock_geglu.call_args[0]
    assert bias_arg.shape[-1] == 128  # 2 * (ffn_hidden_size // tp) = 2 * 64
    assert input_arg.shape[-1] == 128
    assert input_arg.shape[1] == 2  # The caller supplies cfg.train's resolved batch size.
    assert mock_dropout.call_count == 10  # unconditional for RNG parity
    assert mock_rand.call_count == 5
