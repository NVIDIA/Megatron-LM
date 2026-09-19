# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small CUDA gate for the opt-in QSA Stage-2 indexer training path."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
    _rotary,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_stage2_id_sparse_trains_both_indexer_projections_without_main_kl_gradient():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    old_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    saved_scale = old_scale.clone() if old_scale is not None else None
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    try:
        config = _make_config(
            qsa_indexer_loss_coeff=0.0, qsa_force_sparse=True, attention_dropout=0.0
        )
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .train()
        )
        attention.core_attention.sparse_backend = "id_sparse"
        sequence = 45
        freqs = _rotary(config, sequence).cuda()
        torch.manual_seed(199)
        hidden = torch.randn(sequence, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        upstream = torch.randn_like(hidden)
        qk_weight = attention.indexer.index_qk_proj.weight

        def run(coeff, recompute=False):
            config.qsa_indexer_loss_coeff = coeff
            state = hidden.detach().clone().requires_grad_()

            def forward(x):
                return attention(x, attention_mask=None, rotary_pos_emb=freqs)[0]

            output = (
                checkpoint(forward, state, use_reentrant=False) if recompute else forward(state)
            )
            gradients = torch.autograd.grad(
                output,
                (state, attention.linear_qkv.weight, qk_weight),
                grad_outputs=upstream,
                allow_unused=True,
            )
            return output.detach(), gradients

        baseline, baseline_grads = run(0.0)
        actual, actual_grads = run(0.7)
        recomputed, recomputed_grads = run(0.7, recompute=True)
        assert torch.equal(actual, baseline)
        assert torch.equal(recomputed, actual)
        for got, expected in zip(actual_grads[:2], baseline_grads[:2]):
            assert torch.equal(got, expected)
        assert baseline_grads[2] is None
        indexer_grad = actual_grads[2]
        assert indexer_grad is not None and torch.isfinite(indexer_grad).all()
        query_width = config.qsa_indexer_n_heads * config.qsa_indexer_head_dim
        assert indexer_grad[:query_width].float().abs().sum() > 0
        assert indexer_grad[query_width:].float().abs().sum() > 0
        for got, expected in zip(recomputed_grads, actual_grads):
            torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)

        attention.zero_grad(set_to_none=True)
        state = hidden.detach().clone().requires_grad_()

        def reentrant_forward(x):
            return attention(x, attention_mask=None, rotary_pos_emb=freqs)[0]

        reentrant = checkpoint(reentrant_forward, state, use_reentrant=True)
        reentrant.backward(upstream)
        assert torch.equal(reentrant.detach(), actual)
        for got, expected in zip(
            (state.grad, attention.linear_qkv.weight.grad, qk_weight.grad), actual_grads
        ):
            torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
    finally:
        if saved_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        else:
            DSAIndexerLossAutoScaler.set_loss_scale(saved_scale)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_stage2_rejects_packed_keys_before_indexer_forward():
    Utils.initialize_model_parallel(1, 1)
    try:
        config = _make_config(qsa_indexer_loss_coeff=0.7)
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .train()
        )
        attention.core_attention.sparse_backend = "id_sparse"
        packed = PackedSeqParams(qkv_format="thd")
        hidden = torch.randn(8, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="packed/padded keys"):
            attention(
                hidden,
                attention_mask=None,
                rotary_pos_emb=_rotary(config, 8).cuda(),
                packed_seq_params=packed,
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_stage2_long_sequence_rejects_before_projection_and_core(monkeypatch):
    Utils.initialize_model_parallel(1, 1)
    try:
        config = _make_config(qsa_indexer_loss_coeff=0.7, attention_dropout=0.0)
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .train()
        )
        attention.core_attention.sparse_backend = "id_sparse"

        def unexpected(*_args, **_kwargs):
            raise AssertionError("the gated long sequence reached an expensive kernel")

        monkeypatch.setattr(attention.indexer.index_qk_proj, "forward", unexpected)
        monkeypatch.setattr(attention.core_attention, "forward", unexpected)
        hidden = torch.zeros(4097, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="above 4096 tokens"):
            attention(hidden, attention_mask=None, rotary_pos_emb=_rotary(config, 4097).cuda())
    finally:
        Utils.destroy_model_parallel()
