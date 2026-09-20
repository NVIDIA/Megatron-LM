# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two outstanding QSA microbatches under selective core-attention recompute."""

import pytest
import torch

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
)


def _case(starts, valid_lengths):
    cu = torch.tensor(starts, dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(end - start for start, end in zip(starts[:-1], starts[1:])),
        max_seqlen_kv=max(end - start for start, end in zip(starts[:-1], starts[1:])),
    )
    packed.qsa_stage2_layout_cpu = (tuple(starts), tuple(valid_lengths))
    hidden = torch.randn(starts[-1], 1, 64, dtype=torch.bfloat16, device="cuda")
    freqs = torch.randn(starts[-1], 1, 1, 8, device="cuda")
    upstream = torch.randn_like(hidden)
    for start, end, valid in zip(starts[:-1], starts[1:], valid_lengths):
        upstream[start + valid : end] = 0
    return hidden, freqs, upstream, packed


def _attention(recompute):
    config = _make_config(
        qsa_indexer_loss_coeff=0.7,
        qsa_force_sparse=True,
        calculate_per_token_loss=True,
        attention_dropout=0.0,
        mrope_section=[2, 1, 1],
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["core_attn"] if recompute else [],
    )
    model_parallel_cuda_manual_seed(123)
    attention = (
        build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
        .cuda()
        .train()
    )
    attention.core_attention.sparse_backend = "id_sparse"
    assert attention.checkpoint_core_attention == recompute
    return attention


def _run_two_microbatches(attention, cases, order):
    attention.zero_grad(set_to_none=True)
    inputs, outputs, selections = [], [], []
    original_forward = attention.indexer.forward

    def capture_selection(*args, **kwargs):
        selection = original_forward(*args, **kwargs)
        assert selection.index_query is not None and selection.compressed_key is not None
        selections.append(selection.selected_ids.detach().clone())
        return selection

    attention.indexer.forward = capture_selection
    try:
        for hidden, freqs, _, packed in cases:
            input_state = hidden.detach().clone().requires_grad_()
            output, _ = attention(
                input_state, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed
            )
            inputs.append(input_state)
            outputs.append(output)
            assert attention.core_attention._selection is None
    finally:
        attention.indexer.forward = original_forward
    assert len(selections) == 2
    assert selections[0].shape[0] != selections[1].shape[0]
    for index in order:
        outputs[index].backward(cases[index][2])
    parameters = (
        attention.linear_qkv.weight,
        attention.indexer.index_qk_proj.weight,
        attention.indexer.q_layernorm.weight,
        attention.indexer.k_layernorm.weight,
    )
    grads = tuple(parameter.grad.detach().float().clone() for parameter in parameters)
    return (
        tuple(output.detach().clone() for output in outputs),
        tuple(input_state.grad.detach().float().clone() for input_state in inputs),
        grads,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backward_order", [(1, 0), (0, 1)])
def test_qsa_stage2_mixed_selective_checkpoint_binds_each_microbatch(backward_order):
    Utils.initialize_model_parallel(1, 1)
    torch.cuda.set_per_process_memory_fraction(0.1, device=torch.cuda.current_device())
    old_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    saved_scale = old_scale.clone() if old_scale is not None else None
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    try:
        torch.manual_seed(283)
        cases = (_case([0, 48, 64], [47, 15]), _case([0, 96, 100, 112], [94, 0, 11]))
        baseline = _attention(recompute=False)
        checkpointed = _attention(recompute=True)
        checkpointed.load_state_dict(baseline.state_dict())
        expected = _run_two_microbatches(baseline, cases, backward_order)
        actual = _run_two_microbatches(checkpointed, cases, backward_order)
        for got, want in zip(actual[0], expected[0]):
            assert torch.equal(got, want)
        for got, want in zip(actual[1], expected[1]):
            torch.testing.assert_close(got, want, atol=8e-2, rtol=8e-2)
        for got, want in zip(actual[2], expected[2]):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, want, atol=8e-2, rtol=8e-2)
        qk_grad = actual[2][1]
        assert qk_grad[:32].abs().sum() > 0
        assert qk_grad[32:].abs().sum() > 0
        assert actual[2][2].abs().sum() > 0
        assert actual[2][3].abs().sum() > 0
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = saved_scale
        Utils.destroy_model_parallel()
