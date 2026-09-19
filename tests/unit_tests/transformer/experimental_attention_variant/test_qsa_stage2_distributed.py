# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-GPU TP and CP probes for local-query QSA Stage-2 KL ownership."""

import os

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_context_parallel import split_tensor_cp
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
    _rotary,
)


def _attention(config):
    model_parallel_cuda_manual_seed(123)
    attention = (
        build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
        .cuda()
        .train()
    )
    attention.core_attention.sparse_backend = "id_sparse"
    return attention


def _indexer_grad(attention, hidden, freqs, packed_seq_params=None):
    input_state = hidden.detach().clone().requires_grad_()
    output, _ = attention(
        input_state, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed_seq_params
    )
    upstream = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, attention.indexer.index_qk_proj.weight, grad_outputs=upstream
    )[0]
    return grad.detach().float()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two GPUs")
def test_qsa_stage2_tp2_explicit_teacher_group_and_replicated_indexer_gradient():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    try:
        config = _make_config(
            tensor_model_parallel_size=2,
            qsa_indexer_loss_coeff=0.7,
            qsa_force_sparse=True,
            attention_dropout=0.0,
        )
        attention = _attention(config)
        assert attention.core_attention.pg_collection.tp is not None
        torch.manual_seed(77)
        hidden = torch.randn(48, 1, config.hidden_size, dtype=torch.bfloat16).cuda()
        grad = _indexer_grad(attention, hidden, _rotary(config, 48).cuda())
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        gathered = [torch.empty_like(grad) for _ in range(2)]
        torch.distributed.all_gather(
            gathered, grad, group=attention.core_attention.pg_collection.tp
        )
        torch.testing.assert_close(gathered[0], gathered[1], rtol=3e-2, atol=3e-2)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two GPUs")
def test_qsa_stage2_tp2_packed_teacher_uses_explicit_group_and_aligned_rows():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    try:
        config = _make_config(
            tensor_model_parallel_size=2,
            qsa_indexer_loss_coeff=0.7,
            qsa_force_sparse=True,
            calculate_per_token_loss=True,
            attention_dropout=0.0,
        )
        attention = _attention(config)
        cu = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=16,
            max_seqlen_kv=16,
        )
        packed.qsa_stage2_valid_lengths = torch.tensor([9, 14], device="cuda", dtype=torch.int32)
        packed.qsa_stage2_layout_cpu = ((0, 16, 32), (9, 14))
        torch.manual_seed(97)
        hidden = torch.randn(32, 1, config.hidden_size, dtype=torch.bfloat16).cuda()
        grad = _indexer_grad(attention, hidden, _rotary(config, 16).cuda(), packed)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        gathered = [torch.empty_like(grad) for _ in range(2)]
        torch.distributed.all_gather(
            gathered, grad, group=attention.core_attention.pg_collection.tp
        )
        torch.testing.assert_close(gathered[0], gathered[1], rtol=3e-2, atol=3e-2)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two GPUs")
@pytest.mark.parametrize("absolute_mrope", [False, True])
def test_qsa_stage2_cp2_packed_per_token_sum_matches_cp1_with_padding(absolute_mrope):
    Utils.initialize_model_parallel(1, 1)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    physical_cu = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=physical_cu,
        cu_seqlens_kv=physical_cu,
        cu_seqlens_q_padded=physical_cu,
        cu_seqlens_kv_padded=physical_cu,
        max_seqlen_q=16,
        max_seqlen_kv=16,
    )
    packed.qsa_stage2_valid_lengths = torch.tensor([9, 14], device="cuda", dtype=torch.int32)
    packed.qsa_stage2_layout_cpu = ((0, 16, 32), (9, 14))
    torch.manual_seed(93)
    global_hidden = torch.randn(32, 1, 64, dtype=torch.bfloat16).cuda()
    try:
        cp1_config = _make_config(
            qsa_indexer_loss_coeff=0.7,
            qsa_force_sparse=True,
            calculate_per_token_loss=True,
            attention_dropout=0.0,
            mrope_section=[2, 1, 1] if absolute_mrope else None,
        )
        cp1_attention = _attention(cp1_config)
        state = {name: value.detach().clone() for name, value in cp1_attention.state_dict().items()}
        freqs = (_rotary(cp1_config, 32) if absolute_mrope else _rotary(cp1_config, 16)).cuda()
        cp1_grad = _indexer_grad(cp1_attention, global_hidden, freqs, packed)
    finally:
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        cp2_config = _make_config(
            context_parallel_size=2,
            qsa_indexer_loss_coeff=0.7,
            qsa_force_sparse=True,
            calculate_per_token_loss=True,
            attention_dropout=0.0,
            mrope_section=[2, 1, 1] if absolute_mrope else None,
        )
        cp2_attention = _attention(cp2_config)
        cp2_attention.load_state_dict(state)
        local_hidden = split_tensor_cp(global_hidden, packed, dim=0)
        local_freqs = split_tensor_cp(freqs, packed, dim=0) if absolute_mrope else freqs
        cp2_grad = _indexer_grad(cp2_attention, local_hidden, local_freqs, packed)
        torch.distributed.all_reduce(cp2_grad, group=cp2_attention.core_attention.pg_collection.cp)
        # finalize_model_grads divides the summed CP/DP gradient by the global
        # supervised-token count; use the same divisor on the CP1 reference.
        supervised_tokens = torch.tensor(17.0, device="cuda")
        torch.testing.assert_close(
            cp2_grad / supervised_tokens, cp1_grad / supervised_tokens, rtol=4e-2, atol=4e-2
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two GPUs")
def test_qsa_stage2_cp2_local_query_mean_matches_cp1_global_mean():
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    torch.manual_seed(77)
    global_hidden = torch.randn(48, 1, 64, dtype=torch.bfloat16).cuda()
    Utils.initialize_model_parallel(1, 1)
    try:
        cp1_config = _make_config(
            qsa_indexer_loss_coeff=0.7, qsa_force_sparse=True, attention_dropout=0.0
        )
        cp1_attention = _attention(cp1_config)
        state = {name: value.detach().clone() for name, value in cp1_attention.state_dict().items()}
        freqs = _rotary(cp1_config, 48).cuda()
        cp1_grad = _indexer_grad(cp1_attention, global_hidden, freqs)
    finally:
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        cp2_config = _make_config(
            context_parallel_size=2,
            qsa_indexer_loss_coeff=0.7,
            qsa_force_sparse=True,
            attention_dropout=0.0,
        )
        cp2_attention = _attention(cp2_config)
        cp2_attention.load_state_dict(state)
        local_hidden = split_tensor_cp(global_hidden, None, dim=0)
        local_freqs = split_tensor_cp(freqs, None, dim=0)
        cp2_grad = _indexer_grad(cp2_attention, local_hidden, local_freqs)
        torch.distributed.all_reduce(cp2_grad, group=cp2_attention.core_attention.pg_collection.cp)
        cp2_grad /= 2
        torch.testing.assert_close(cp2_grad, cp1_grad, rtol=4e-2, atol=4e-2)
    finally:
        Utils.destroy_model_parallel()
