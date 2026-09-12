# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize("softmax_type", ["off-by-one", "learnable"])
def test_cpu_initialization_keeps_softmax_offset_on_cpu(monkeypatch, softmax_type):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        perform_initialization=False,
        softmax_type=softmax_type,
        use_cpu_initialization=True,
    )
    tp_group = SimpleNamespace(size=lambda: 1)
    process_groups = SimpleNamespace(tp=tp_group)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("CPU-initialized attention must not access CUDA"),
    )

    attention = DotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=process_groups,
    )

    assert attention.softmax_offset.shape == (4,)
    assert attention.softmax_offset.device.type == "cpu"
    assert attention.softmax_offset.dtype == config.params_dtype
    assert isinstance(attention.softmax_offset, torch.nn.Parameter) is (softmax_type == "learnable")
    if softmax_type == "off-by-one":
        assert dict(attention.named_buffers())["softmax_offset"] is attention.softmax_offset
        assert "softmax_offset" not in attention.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_initialized_off_by_one_softmax_offset_follows_module_to_cuda():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        perform_initialization=False,
        softmax_type="off-by-one",
        use_cpu_initialization=True,
    )
    tp_group = SimpleNamespace(size=lambda: 1)
    process_groups = SimpleNamespace(tp=tp_group)

    attention = DotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=process_groups,
    ).cuda()

    assert attention.softmax_offset.device.type == "cuda"
    assert "softmax_offset" not in attention.state_dict()
