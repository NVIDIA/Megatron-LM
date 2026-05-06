# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

pytestmark = [
    pytest.mark.skipif(not te_ext.HAVE_TE, reason="Transformer Engine is not available"),
    pytest.mark.skipif(
        not is_te_min_version("1.13.0"),
        reason="TE fused ops wrappers require Transformer Engine >= 1.13.0",
    ),
]


def _make_rmsnorm() -> torch.nn.Module:
    return te_ext.TEFusedResidualRMSNorm(
        normalized_shape=16,
        dtype=torch.float32,
        device="cpu",
    )


def test_rmsnorm_fused_impl_aliases_source_weight():
    module = _make_rmsnorm()

    fused_impl = module._get_fused_impl()

    assert fused_impl[1].weight is module.weight


def test_fused_impl_is_cached_and_resettable():
    module = _make_rmsnorm()

    first_impl = module._get_fused_impl()

    assert module._get_fused_impl() is first_impl

    module._reset_fused_impl()

    assert module._fused_impl is None

    second_impl = module._get_fused_impl()

    assert second_impl is not first_impl
    assert second_impl[1].weight is module.weight


def test_fused_impl_is_not_registered_as_module_or_state_dict_source():
    module = _make_rmsnorm()
    expected_state_keys = set(module.state_dict().keys())
    expected_module_keys = tuple(module._modules.keys())

    fused_impl = module._get_fused_impl()

    assert set(module.state_dict().keys()) == expected_state_keys == {"weight"}
    assert tuple(module._modules.keys()) == expected_module_keys
    assert "_fused_impl" not in module._modules
    assert all(child is not fused_impl for child in module.modules())


def test_mcore_te_linear_adapter_rejects_plain_te_linear():
    plain_linear = te_ext.te.pytorch.Linear(16, 16, device="meta")

    with pytest.raises(ValueError) as exc_info:
        te_ext._make_te_ops_basic_linear_from_mcore_te_linear(
            plain_linear,
            module_name="plain_linear",
        )

    message = str(exc_info.value)
    assert "plain_linear" in message
    assert plain_linear.__class__.__name__ in message
    assert "config.tp_comm_overlap" in message


def test_mcore_te_linear_adapter_aliases_source_weight():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )
    linear = te_ext.TEColumnParallelLinear(
        16,
        32,
        config=config,
        init_method=torch.nn.init.zeros_,
        gather_output=False,
        bias=True,
        skip_bias_add=True,
        is_expert=False,
        tp_comm_buffer_name="fc1",
    )

    op = te_ext._make_te_ops_basic_linear_from_mcore_te_linear(
        linear,
        module_name="linear",
        output_features=linear.weight.size(0),
    )

    assert isinstance(op, te_ext.te.pytorch.ops.BasicLinear)
    assert op.weight is linear.weight
