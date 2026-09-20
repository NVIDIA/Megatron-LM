# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small CUDA gate for the opt-in QSA Stage-2 indexer training path."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import (
    _get_experimental_attention_variant_loss_scale_func,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.experimental_attention_variant.qsa import (
    _sequence_layout,
    _stage2_packed_lengths,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
    _rotary,
)


def test_qsa_stage2_packed_cpu_layout_validates_without_tensor_scalar_read(monkeypatch):
    cu = torch.tensor([0, 16, 32], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    packed.qsa_stage2_layout_cpu = ((0, 16, 32), (9, 14))

    def unexpected_item(*_args, **_kwargs):
        raise AssertionError("Stage-2 metadata read a tensor scalar")

    monkeypatch.setattr(torch.Tensor, "item", unexpected_item)
    first = _stage2_packed_lengths(packed, total_tokens=32, device=cu.device)
    second = _stage2_packed_lengths(packed, total_tokens=32, device=cu.device)
    assert first[0] is second[0] and first[1] is second[1]
    assert torch.equal(first[0], cu.long())
    assert torch.equal(first[1], torch.tensor([9, 14]))
    doc_ids, positions, max_doc_len = _sequence_layout(1, 32, packed, cu.device, 16)
    assert max_doc_len == 16
    assert torch.equal(doc_ids[0, 16:], torch.ones(16, dtype=torch.int32))
    assert torch.equal(positions[0, 16:], torch.arange(16, dtype=torch.int32))

    packed.qsa_stage2_layout_cpu = ((0, 16, 32), (13, 2))
    changed = _stage2_packed_lengths(packed, total_tokens=32, device=cu.device)
    assert not torch.equal(changed[1], first[1])


@pytest.mark.parametrize("in_place", [False, True])
def test_qsa_stage2_packed_cpu_layout_rejects_device_cu_mutation(in_place):
    cu = torch.tensor([0, 16, 32], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    packed.qsa_stage2_layout_cpu = ((0, 16, 32), (9, 14))
    _stage2_packed_lengths(packed, total_tokens=32, device=cu.device)
    if in_place:
        cu[1] = 15
    else:
        packed.cu_seqlens_q = cu.clone()
    with pytest.raises(ValueError, match="device cu_seqlens changed"):
        _stage2_packed_lengths(packed, total_tokens=32, device=cu.device)


@pytest.mark.parametrize(
    "layout",
    [
        None,
        ((0, 16, 32), (9,)),
        ((0, 16, 31), (9, 14)),
        ((0, 16, 32), (17, 14)),
        ((0, 16, 16), (9, 0)),
        ((0, 16, 32), (9, -1)),
        ((0, 16, 32), (True, 14)),
    ],
)
def test_qsa_stage2_packed_cpu_layout_rejects_invalid_metadata(layout):
    cu = torch.tensor([0, 16, 32], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    if layout is not None:
        packed.qsa_stage2_layout_cpu = layout
    with pytest.raises(ValueError, match="qsa_stage2_layout_cpu"):
        _stage2_packed_lengths(packed, total_tokens=32, device=cu.device)


def test_qsa_stage2_gdn_registers_aux_scale_only_when_enabled():
    disabled = SimpleNamespace(
        experimental_attention_variant="gdn",
        qsa_indexer_loss_coeff=0.0,
        experimental_attention_variant_loss_scale_func=None,
    )
    enabled = SimpleNamespace(
        experimental_attention_variant="gdn",
        qsa_indexer_loss_coeff=0.7,
        experimental_attention_variant_loss_scale_func=None,
    )
    assert _get_experimental_attention_variant_loss_scale_func(disabled) is None
    hook = _get_experimental_attention_variant_loss_scale_func(enabled)
    assert hook is DSAIndexerLossAutoScaler.set_loss_scale
    old_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    saved_scale = old_scale.clone() if old_scale is not None else None
    try:
        hook(torch.tensor(0.25))
        main = torch.ones(2, requires_grad=True)
        auxiliary = torch.ones((), requires_grad=True)
        DSAIndexerLossAutoScaler.apply(main, auxiliary).sum().backward()
        torch.testing.assert_close(main.grad, torch.ones_like(main))
        torch.testing.assert_close(auxiliary.grad, torch.tensor(0.25))
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = saved_scale


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
def test_qsa_stage2_rejects_packed_without_explicit_real_lengths_before_indexer_forward(
    monkeypatch,
):
    Utils.initialize_model_parallel(1, 1)
    try:
        config = _make_config(qsa_indexer_loss_coeff=0.7, attention_dropout=0.0)
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .train()
        )
        attention.core_attention.sparse_backend = "id_sparse"
        cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=8,
            max_seqlen_kv=8,
        )
        hidden = torch.randn(8, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        def unexpected(*_args, **_kwargs):
            raise AssertionError("missing metadata reached the indexer projection or core")

        monkeypatch.setattr(attention.indexer.index_qk_proj, "forward", unexpected)
        monkeypatch.setattr(attention.core_attention, "forward", unexpected)
        with pytest.raises(ValueError, match="qsa_stage2_layout_cpu"):
            attention(
                hidden,
                attention_mask=None,
                rotary_pos_emb=_rotary(config, 8).cuda(),
                packed_seq_params=packed,
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_stage2_packed_padding_excluded_and_main_gradients_unchanged():
    Utils.initialize_model_parallel(1, 1)
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
        torch.manual_seed(91)
        hidden = torch.randn(32, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        valid = torch.zeros(32, device="cuda", dtype=torch.bool)
        valid[:9] = True
        valid[16:30] = True
        upstream = torch.randn(32, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        upstream[~valid] = 0
        freqs = _rotary(config, 16).cuda()
        qk_weight = attention.indexer.index_qk_proj.weight

        def run(coeff, input_hidden, use_reentrant=False):
            config.qsa_indexer_loss_coeff = coeff
            state = input_hidden.detach().clone().requires_grad_()

            def forward(x):
                return attention(
                    x, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed
                )[0]

            output = (
                checkpoint(forward, state, use_reentrant=True) if use_reentrant else forward(state)
            )
            if use_reentrant:
                attention.zero_grad(set_to_none=True)
                output.backward(upstream)
                grads = (state.grad, attention.linear_qkv.weight.grad, qk_weight.grad)
            else:
                grads = torch.autograd.grad(
                    output,
                    (state, attention.linear_qkv.weight, qk_weight),
                    grad_outputs=upstream,
                    allow_unused=True,
                )
            return output.detach(), grads

        del packed.qsa_stage2_valid_lengths
        del packed.qsa_stage2_layout_cpu
        baseline, base_grads = run(0.0, hidden)
        packed.qsa_stage2_valid_lengths = torch.tensor([9, 14], device="cuda", dtype=torch.int32)
        packed.qsa_stage2_layout_cpu = ((0, 16, 32), (9, 14))
        actual, grads = run(0.7, hidden)
        assert torch.equal(actual, baseline)
        for got, expected in zip(grads[:2], base_grads[:2]):
            assert torch.equal(got, expected)
        assert base_grads[2] is None
        assert grads[2] is not None and torch.isfinite(grads[2]).all()
        query_width = config.qsa_indexer_n_heads * config.qsa_indexer_head_dim
        assert grads[2][:query_width].float().abs().sum() > 0
        assert grads[2][query_width:].float().abs().sum() > 0

        recomputed, recomputed_grads = run(0.7, hidden, use_reentrant=True)
        assert torch.equal(recomputed, actual)
        for got, expected in zip(recomputed_grads, grads):
            torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)

        perturbed = hidden.clone()
        perturbed[~valid] = torch.randn_like(perturbed[~valid]) * 100
        _, perturbed_grads = run(0.7, perturbed)
        torch.testing.assert_close(perturbed_grads[2].float(), grads[2].float(), rtol=0, atol=0)
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_stage2_distinguishes_real_lengths_with_same_physical_cu_and_zero_tail():
    Utils.initialize_model_parallel(1, 1)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    try:
        config = _make_config(
            qsa_indexer_loss_coeff=0.7,
            qsa_force_sparse=True,
            calculate_per_token_loss=True,
            attention_dropout=0.0,
        )
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .train()
        )
        attention.core_attention.sparse_backend = "id_sparse"
        physical_cu = torch.tensor([0, 16, 32, 36], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=physical_cu,
            cu_seqlens_kv=physical_cu,
            cu_seqlens_q_padded=physical_cu,
            cu_seqlens_kv_padded=physical_cu,
            max_seqlen_q=16,
            max_seqlen_kv=16,
        )
        torch.manual_seed(103)
        hidden = torch.randn(36, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        freqs = _rotary(config, 16).cuda()

        def grad(real_lengths):
            packed.qsa_stage2_valid_lengths = torch.tensor(
                real_lengths, device="cuda", dtype=torch.int32
            )
            packed.qsa_stage2_layout_cpu = ((0, 16, 32, 36), tuple(real_lengths))
            output = attention(
                hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed
            )[0]
            result = torch.autograd.grad(
                output, attention.indexer.index_qk_proj.weight, grad_outputs=torch.ones_like(output)
            )[0]
            assert torch.isfinite(result).all()
            return result.detach().float()

        first = grad([9, 3, 0])
        second = grad([13, 2, 0])
        assert not torch.equal(first, second)
        assert first.abs().sum() > 0 and second.abs().sum() > 0
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
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
