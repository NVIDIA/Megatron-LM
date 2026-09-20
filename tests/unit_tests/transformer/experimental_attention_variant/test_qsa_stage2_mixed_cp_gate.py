# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-GPU mixed THD Stage-2 gate; run only in a coordinated GPU window."""

import os

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_context_parallel import reconstruct_tensor_cp, split_tensor_cp
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
)


def _mixed_packed(cu):
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=96,
        max_seqlen_kv=96,
    )
    return packed


def _attention(config):
    model_parallel_cuda_manual_seed(123)
    attention = (
        build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
        .cuda()
        .train()
    )
    attention.core_attention.sparse_backend = "id_sparse"
    return attention


def _run(attention, config, hidden, freqs, packed, upstream, coeff):
    config.qsa_indexer_loss_coeff = coeff
    input_state = hidden.detach().clone().requires_grad_()
    observed = []
    original_forward = attention.indexer.forward

    def capture_selection(*args, **kwargs):
        selection = original_forward(*args, **kwargs)
        observed.append(
            (
                tuple(selection.selected_ids.shape),
                selection.all_selected,
                (
                    None
                    if selection.compact_block_prefix is None
                    else selection.compact_block_prefix.detach().cpu().tolist()
                ),
                selection.compact_block_starts is not None,
                selection.compressed_key is not None and selection.compressed_key.requires_grad,
            )
        )
        return selection

    attention.indexer.forward = capture_selection
    try:
        output, _ = attention(
            input_state, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed
        )
    finally:
        attention.indexer.forward = original_forward
    assert attention.core_attention._selection is None
    assert len(observed) == 1
    shape, all_selected, prefix, has_starts, has_key_grad = observed[0]
    assert shape[1] == 2 and not all_selected
    if coeff > 0:
        assert prefix == [0, 24, 25, 28]
        assert has_starts and has_key_grad
    parameters = (
        input_state,
        attention.linear_qkv.weight,
        attention.indexer.index_qk_proj.weight,
        attention.indexer.q_layernorm.weight,
        attention.indexer.k_layernorm.weight,
    )
    grads = torch.autograd.grad(output, parameters, grad_outputs=upstream, allow_unused=True)
    return output.detach(), tuple(None if grad is None else grad.detach() for grad in grads)


def _check_indexer_gradients(grads, query_width):
    for grad in grads[2:]:
        assert grad is not None and torch.isfinite(grad).all() and grad.float().abs().sum() > 0
    qk = grads[2].float()
    assert qk[:query_width].abs().sum() > 0
    assert qk[query_width:].abs().sum() > 0


def _record_selection_lifetime(attention, config, hidden, freqs, packed, stage):
    """Measure allocated memory after clearing the selection and after another forward."""
    assert attention.core_attention._selection is None
    torch.cuda.synchronize()
    after_backward = torch.cuda.memory_allocated()
    config.qsa_indexer_loss_coeff = 0.0
    with torch.no_grad():
        next_output, _ = attention(
            hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed
        )
    del next_output
    torch.cuda.synchronize()
    after_next_forward = torch.cuda.memory_allocated()
    assert attention.core_attention._selection is None
    print(
        f"QSA_STAGE2_MIXED_MEMORY rank={os.environ.get('RANK', '0')} stage={stage} "
        f"after_backward={after_backward} after_next_forward={after_next_forward}",
        flush=True,
    )


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2 or not torch.cuda.is_available(),
    reason="requires two CUDA ranks",
)
def test_qsa_stage2_mixed_cp2_bf16_matches_cp1_and_keeps_main_gradients():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(0.1, device=torch.cuda.current_device())
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    old_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    saved_scale = old_scale.clone() if old_scale is not None else None
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    torch.manual_seed(211)
    cu = torch.tensor([0, 96, 100, 112], device="cuda", dtype=torch.int32)
    packed = _mixed_packed(cu)
    global_hidden = torch.randn(112, 1, 64, dtype=torch.bfloat16, device="cuda")
    global_freqs = torch.randn(112, 1, 1, 8, device="cuda")
    upstream = torch.randn_like(global_hidden)
    upstream[94:100] = 0  # first document padding and the entire logical empty document
    upstream[111] = 0  # final document's padding
    try:
        Utils.initialize_model_parallel(1, 1)
        try:
            cp1_config = _make_config(
                qsa_indexer_loss_coeff=0.0,
                qsa_force_sparse=True,
                calculate_per_token_loss=True,
                attention_dropout=0.0,
                mrope_section=[2, 1, 1],
            )
            cp1_attention = _attention(cp1_config)
            state = {
                name: value.detach().clone() for name, value in cp1_attention.state_dict().items()
            }
            baseline = _run(
                cp1_attention, cp1_config, global_hidden, global_freqs, packed, upstream, 0.0
            )
            assert all(grad is None for grad in baseline[1][2:])
            packed.qsa_stage2_layout_cpu = ((0, 96, 100, 112), (94, 0, 11))
            cp1 = _run(
                cp1_attention, cp1_config, global_hidden, global_freqs, packed, upstream, 0.7
            )
            assert torch.equal(cp1[0], baseline[0])
            for index, (actual, expected) in enumerate(zip(cp1[1][:2], baseline[1][:2])):
                # BF16 backward can change a handful of rounded cells when the
                # auxiliary loss is attached. This same difference reproduces at
                # the preceding a7bdeca checkpoint, before selection cleanup.
                mismatch = actual != expected
                delta = (actual.float() - expected.float()).abs()
                assert mismatch.sum().item() <= 8
                assert delta.max().item() <= (2e-5 if index == 0 else 5e-4)
                assert (delta.norm() / expected.float().norm()).item() <= 5e-5
            _check_indexer_gradients(cp1[1], query_width=32)
            _record_selection_lifetime(
                cp1_attention, cp1_config, global_hidden, global_freqs, packed, "cp1"
            )
        finally:
            Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
        try:
            cp2_config = _make_config(
                context_parallel_size=2,
                qsa_indexer_loss_coeff=0.0,
                qsa_force_sparse=True,
                calculate_per_token_loss=True,
                attention_dropout=0.0,
                mrope_section=[2, 1, 1],
            )
            cp2_attention = _attention(cp2_config)
            cp2_attention.load_state_dict(state)
            local_hidden = split_tensor_cp(global_hidden, packed, dim=0)
            local_freqs = split_tensor_cp(global_freqs, packed, dim=0)
            local_upstream = split_tensor_cp(upstream, packed, dim=0)
            cp2_baseline = _run(
                cp2_attention, cp2_config, local_hidden, local_freqs, packed, local_upstream, 0.0
            )
            cp2 = _run(
                cp2_attention, cp2_config, local_hidden, local_freqs, packed, local_upstream, 0.7
            )
            assert torch.equal(cp2[0], cp2_baseline[0])
            for actual, expected in zip(cp2[1][:2], cp2_baseline[1][:2]):
                assert torch.equal(actual, expected)
            _check_indexer_gradients(cp2[1], query_width=32)
            full_output = reconstruct_tensor_cp(cp2[0], packed, dim=0)
            full_hidden_grad = reconstruct_tensor_cp(cp2[1][0], packed, dim=0)
            assert torch.equal(full_output, cp1[0])
            torch.testing.assert_close(
                full_hidden_grad.float(), cp1[1][0].float(), atol=8e-2, rtol=8e-2
            )
            for index in range(1, 5):
                cp2_grad = cp2[1][index].float().clone()
                torch.distributed.all_reduce(
                    cp2_grad, group=cp2_attention.core_attention.pg_collection.cp
                )
                expected = cp1[1][index].float()
                torch.testing.assert_close(cp2_grad, expected, atol=8e-2, rtol=8e-2)
            _record_selection_lifetime(
                cp2_attention, cp2_config, local_hidden, local_freqs, packed, "cp2"
            )
        finally:
            Utils.destroy_model_parallel()
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = saved_scale
