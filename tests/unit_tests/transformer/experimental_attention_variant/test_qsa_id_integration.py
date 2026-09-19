# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in selected-ID QSA model-path comparisons at short sequence lengths."""

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
    _rotary,
)


@pytest.fixture
def qsa_attention(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "1")
    config = _make_config()
    spec = get_qsa_module_spec_for_backend(config)
    attention = build_module(spec, config=config, layer_number=1).cuda().eval()
    try:
        yield attention, config
    finally:
        Utils.destroy_model_parallel()


def _run_attention(attention, hidden, freqs, packed, backend, grad_output):
    attention.core_attention.sparse_backend = backend
    input_states = hidden.detach().clone().requires_grad_()
    output, _ = attention(
        input_states, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed
    )
    grads = torch.autograd.grad(output, (input_states, attention.linear_qkv.weight), grad_output)
    return output.detach(), tuple(grad.detach() for grad in grads)


@pytest.mark.parametrize("packed_lengths", [None, [21, 30]])
def test_qsa_id_backend_matches_dense_masked_model_forward_backward(qsa_attention, packed_lengths):
    attention, config = qsa_attention
    seq_len = sum(packed_lengths) if packed_lengths else 45
    if packed_lengths:
        cu = torch.tensor([0, packed_lengths[0], seq_len], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(packed_lengths),
            max_seqlen_kv=max(packed_lengths),
        )
        freqs = _rotary(config, max(packed_lengths)).cuda()
    else:
        packed = None
        freqs = _rotary(config, seq_len).cuda()
    torch.manual_seed(97)
    hidden = torch.randn(seq_len, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn_like(hidden)
    dense = _run_attention(attention, hidden, freqs, packed, "dense_masked", grad_output)
    sparse = _run_attention(attention, hidden, freqs, packed, "id_sparse", grad_output)
    torch.testing.assert_close(sparse[0].float(), dense[0].float(), atol=3e-2, rtol=3e-2)
    for actual, expected in zip(sparse[1], dense[1]):
        torch.testing.assert_close(actual.float(), expected.float(), atol=6e-2, rtol=6e-2)
    assert attention.core_attention._selection.selected_bits is None
    assert attention.core_attention._selection.selected_ids is not None
