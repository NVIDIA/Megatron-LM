# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests that JIT warmup skips fused kernels when disabled via arguments.

Uses mocks only (no CUDA required). Patches use
``megatron.training.initialize.<name>`` because ``_warmup_jit_function`` closes
over that module's globals (imported ``get_args``, ``torch``, fusion ops).
"""

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.training.initialize import _warmup_jit_function


def _args(**overrides):
    base = dict(
        bf16=False,
        fp16=False,
        use_te_activation_func=False,
        gated_linear_unit=False,
        quick_geglu=False,
        swiglu=False,
        bias_swiglu_fusion=True,
        bias_gelu_fusion=True,
        bias_dropout_fusion=False,
        sequence_parallel=False,
        ffn_hidden_size=64,
        tensor_model_parallel_size=1,
        seq_length=8,
        context_parallel_size=1,
        micro_batch_size=1,
        hidden_size=32,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


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
@mock.patch("megatron.training.initialize.get_args")
class TestJitWarmupSkippedWhenFusionDisabled:
    @mock.patch("megatron.training.initialize.bias_swiglu")
    def test_skips_bias_swiglu(self, mock_swiglu, mock_get_args, _empty_cache):
        mock_get_args.return_value = _args(swiglu=True, bias_swiglu_fusion=False)
        _warmup_jit_function()
        mock_swiglu.assert_not_called()

    @mock.patch("megatron.training.initialize.bias_gelu")
    def test_skips_bias_gelu(self, mock_gelu, mock_get_args, _empty_cache):
        mock_get_args.return_value = _args(
            swiglu=False, bias_gelu_fusion=False, gated_linear_unit=False
        )
        _warmup_jit_function()
        mock_gelu.assert_not_called()

    @mock.patch("megatron.training.initialize.bias_geglu")
    def test_skips_bias_geglu(self, mock_geglu, mock_get_args, _empty_cache):
        mock_get_args.return_value = _args(
            swiglu=False, bias_gelu_fusion=False, gated_linear_unit=True
        )
        _warmup_jit_function()
        mock_geglu.assert_not_called()

    @mock.patch("megatron.training.initialize.bias_geglu")
    @mock.patch("megatron.training.initialize.bias_gelu")
    def test_skips_gelu_and_geglu_when_quick_geglu(
        self, mock_gelu, mock_geglu, mock_get_args, _empty_cache
    ):
        # Even with bias_gelu_fusion True, quick_geglu skips gelu and geglu warmups.
        mock_get_args.return_value = _args(
            swiglu=False, quick_geglu=True, bias_gelu_fusion=True, gated_linear_unit=True
        )
        _warmup_jit_function()
        mock_gelu.assert_not_called()
        mock_geglu.assert_not_called()

    @mock.patch("megatron.training.initialize.bias_dropout_add_fused_train")
    def test_skips_bias_dropout_fusion(self, mock_dropout, mock_get_args, _empty_cache):
        # With swiglu=True, gelu/geglu warmups are off; disable swiglu fusion so no CUDA path runs.
        mock_get_args.return_value = _args(
            swiglu=True, bias_swiglu_fusion=False, bias_dropout_fusion=False
        )
        _warmup_jit_function()
        mock_dropout.assert_not_called()

    @mock.patch("megatron.training.initialize.bias_dropout_add_fused_train")
    @mock.patch("megatron.training.initialize.bias_geglu")
    @mock.patch("megatron.training.initialize.bias_gelu")
    @mock.patch("megatron.training.initialize.bias_swiglu")
    def test_te_activation_skips_activation_warmups(
        self, mock_swiglu, mock_gelu, mock_geglu, mock_dropout, mock_get_args, _empty_cache
    ):
        # bias_dropout_fusion is independent of use_te_activation_func; keep False
        # so this test stays CPU-only without mocking the dropout path + mpu.
        mock_get_args.return_value = _args(
            use_te_activation_func=True,
            swiglu=True,
            bias_swiglu_fusion=True,
            gated_linear_unit=True,
            bias_gelu_fusion=True,
            bias_dropout_fusion=False,
        )
        _warmup_jit_function()
        mock_swiglu.assert_not_called()
        mock_gelu.assert_not_called()
        mock_geglu.assert_not_called()
        mock_dropout.assert_not_called()


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
    mock_get_args.return_value = _args(swiglu=True, bias_swiglu_fusion=True)
    mock_swiglu.return_value = torch.tensor(0.0)
    _warmup_jit_function()
    assert mock_swiglu.call_count == 10  # zip([True,True],[False,True]) × range(5)
    mock_gelu.assert_not_called()
    mock_geglu.assert_not_called()
    mock_dropout.assert_not_called()
    assert mock_torch_rand.call_count == 2
