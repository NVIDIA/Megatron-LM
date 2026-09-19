# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The QSA indexer must restore CP-local plain RoPE when it restores hidden states."""

import os

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.ssm.mamba_context_parallel import split_tensor_cp
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
    _rotary,
)


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two ranks")
def test_qsa_cp2_plain_rope_local_frequency_forward_backward():
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    try:
        config = _make_config(context_parallel_size=2)
        attention = build_module(
            get_qsa_module_spec_for_backend(config), config=config, layer_number=1
        ).cuda()
        attention.core_attention.sparse_backend = "dense_masked"
        hidden = (
            split_tensor_cp(
                torch.randn(52, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16),
                None,
                dim=0,
            )
            .detach()
            .requires_grad_()
        )
        local_freqs = _rotary(config, 52)
        assert local_freqs.shape[0] == hidden.shape[0] == 26
        output, _ = attention(hidden, attention_mask=None, rotary_pos_emb=local_freqs)
        hidden_grad = torch.autograd.grad(output.float().square().mean(), hidden)[0]
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all() and torch.isfinite(hidden_grad).all()
    finally:
        Utils.destroy_model_parallel()
