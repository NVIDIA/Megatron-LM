# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-level contract tests for ``_restore_gtp_replicated_bias``.

The distributed TP/GTP suite (``test_tp_gtp.py``) exercises the enabled and disabled bias
paths on real TE modules. These tests pin down the helper's contract itself — grouped
per-GEMM restore, fail-closed handling of contract violations, and placeholder identity —
without requiring GPUs, so they run in any environment with TransformerEngine installed.
"""

import types

import pytest
import torch
from torch.nn import Parameter

pytest.importorskip("transformer_engine")

from megatron.core.extensions.transformer_engine import _restore_gtp_replicated_bias  # noqa: E402


def _linear_module(bias, use_bias=True, bias_names=("bias",)):
    # Mirrors the real TE contract: TE Linear/LayerNormLinear expose ``bias_names``
    # (["bias"] unless parameters_split is used) alongside the ``bias`` attribute.
    return types.SimpleNamespace(use_bias=use_bias, bias=bias, bias_names=list(bias_names))


def _grouped_module(biases, single_grouped_bias=False):
    module = types.SimpleNamespace(
        use_bias=True, num_gemms=len(biases), single_grouped_bias=single_grouped_bias
    )
    for idx, bias in enumerate(biases):
        setattr(module, f"bias{idx}", bias)
    return module


def test_restore_resizes_shard_sized_bias_in_place():
    bias = Parameter(torch.full((576,), 7.0, dtype=torch.bfloat16))
    module = _linear_module(bias)

    _restore_gtp_replicated_bias(module, logical_out_features=1152, out_split_size=1)

    # Same Parameter object (so TE-attached attributes survive), new zeroed storage.
    assert module.bias is bias
    assert tuple(module.bias.shape) == (1152,)
    assert module.bias.dtype == torch.bfloat16
    assert torch.count_nonzero(module.bias) == 0


def test_restore_leaves_correctly_sized_bias_storage_untouched():
    bias = Parameter(torch.full((1152,), 3.0, dtype=torch.bfloat16))
    module = _linear_module(bias)

    _restore_gtp_replicated_bias(module, logical_out_features=2304, out_split_size=2)

    assert module.bias is bias
    # Same storage, values preserved: an already-logical bias must not be re-zeroed.
    assert torch.equal(module.bias.data, torch.full((1152,), 3.0, dtype=torch.bfloat16))


def test_restore_skips_disabled_bias_placeholder():
    placeholder = torch.empty(0)
    module = _linear_module(placeholder, use_bias=False)

    _restore_gtp_replicated_bias(module, logical_out_features=1152, out_split_size=1)

    assert module.bias is placeholder
    assert not isinstance(module.bias, Parameter)
    assert module.bias.numel() == 0


def test_restore_resizes_every_grouped_bias():
    biases = [Parameter(torch.zeros(128, dtype=torch.bfloat16)) for _ in range(3)]
    module = _grouped_module(biases)

    _restore_gtp_replicated_bias(
        module, logical_out_features=512, out_split_size=2, is_grouped=True
    )

    for idx, bias in enumerate(biases):
        restored = getattr(module, f"bias{idx}")
        assert restored is bias
        assert tuple(restored.shape) == (256,)


def test_restore_rejects_missing_grouped_bias_attribute():
    module = _grouped_module([Parameter(torch.zeros(128))])
    module.num_gemms = 2  # bias1 does not exist

    with pytest.raises(AttributeError, match="bias1"):
        _restore_gtp_replicated_bias(
            module, logical_out_features=512, out_split_size=2, is_grouped=True
        )


def test_restore_rejects_single_grouped_bias_storage():
    module = _grouped_module([Parameter(torch.zeros(128))], single_grouped_bias=True)

    with pytest.raises(NotImplementedError, match="single_grouped_bias"):
        _restore_gtp_replicated_bias(
            module, logical_out_features=512, out_split_size=2, is_grouped=True
        )


def test_restore_rejects_te_parameters_split_layout():
    module = _linear_module(
        Parameter(torch.zeros(576)), bias_names=("query_bias", "key_bias", "value_bias")
    )

    with pytest.raises(NotImplementedError, match="parameters_split"):
        _restore_gtp_replicated_bias(module, logical_out_features=1152, out_split_size=1)


def test_restore_rejects_non_parameter_trainable_bias():
    module = _linear_module(torch.zeros(576))

    with pytest.raises(TypeError, match="bias"):
        _restore_gtp_replicated_bias(module, logical_out_features=1152, out_split_size=1)


def test_restore_rejects_scalar_bias():
    module = _linear_module(Parameter(torch.tensor(0.0)))

    with pytest.raises(RuntimeError, match="scalar"):
        _restore_gtp_replicated_bias(module, logical_out_features=1152, out_split_size=1)
