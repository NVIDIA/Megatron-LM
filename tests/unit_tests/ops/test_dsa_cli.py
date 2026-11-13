# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CLI state isolation, target accumulation and summed-KL reference tests."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ops.attention.dsa.cli_state import (
    DSACrossLayerIndexerState,
    cli_group_size,
    get_dsa_cross_layer_state_holder,
)


@pytest.mark.parametrize(
    "layer,total,offset,frequency,expected",
    [
        (1, 9, 1, 4, 4),
        (4, 9, 1, 4, 4),
        (9, 9, 1, 4, 1),
        (1, 8, 3, 4, 1),
        (3, 8, 3, 4, 4),
        (7, 8, 3, 4, 2),
    ],
)
def test_cli_group_sizes(layer, total, offset, frequency, expected):
    assert cli_group_size(layer, total, offset, frequency) == expected


def test_cli_state_is_per_forward_and_releases_tensors():
    first, second = SimpleNamespace(), SimpleNamespace()
    state = DSACrossLayerIndexerState(topk_indices=torch.ones(1), topk_length=torch.ones(1))
    get_dsa_cross_layer_state_holder(first)[1] = [state]
    assert get_dsa_cross_layer_state_holder(first)[1][0] is state
    assert not get_dsa_cross_layer_state_holder(second)
    state.release_forward_values()
    assert state.topk_indices is None
    assert state.topk_length is None
    assert state.deferred_cli_target is None


def test_cli_owner_recompute_uses_original_indices(monkeypatch):
    """Tied scores must never cause another selection during owner recomputation."""
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_cudnn_kernels as kernels,
    )

    original = torch.arange(256, dtype=torch.int32).expand(1, 2, -1).contiguous()
    seen = {}

    def fail_selection(*args, **kwargs):
        raise AssertionError("CLI must not select another Top-K during owner recomputation")

    def predict(q, k, weights, indices, **kwargs):
        del q, k, weights, kwargs
        assert indices is original
        return {"predict": torch.full(original.shape, 1.0 / 256)}

    def loss(**kwargs):
        seen["indices"] = kwargs["topk_indices_cmp"]
        return torch.zeros(()), torch.zeros(1), torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(kernels, "_indexer_topk_bshd", fail_selection)
    monkeypatch.setattr(
        kernels, "_cudnn_dsa", SimpleNamespace(sparse_indexer_score_recompute_wrapper=predict)
    )
    monkeypatch.setattr(kernels, "_compute_sparse_indexer_loss_and_grads", loss)
    monkeypatch.setattr(
        kernels,
        "_run_sparse_attention_forward",
        lambda *args, **kw: (torch.zeros(2, 32, 512), torch.zeros(2, 32)),
    )
    context = SimpleNamespace(
        save_for_backward=lambda *a: None, mark_non_differentiable=lambda *a: None
    )
    kernels.FusedQKTopKWithSparseLossFunc.forward(
        context,
        torch.zeros(2, 1, 32, 128),
        torch.zeros(256, 1, 128),
        torch.zeros(2, 1, 32),
        torch.zeros(2, 1, 32, 576),
        torch.zeros(256, 1, 576),
        256,
        0.1,
        0.001,
        False,
        512,
        torch.zeros(2, dtype=torch.int32),
        torch.full((2,), 256, dtype=torch.int32),
        None,
        False,
        False,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        1,
        None,
        SimpleNamespace(topk_indices=original),
    )
    assert seen["indices"] is original


@pytest.mark.parametrize("chunk_rows", [1, 2, 4])
def test_cli_target_matches_reference(chunk_rows):
    pytest.importorskip("tilelang")
    from megatron.core.ops.attention.dsa.cli_target import accumulate_cli_attention_target

    torch.manual_seed(2468)
    indices = torch.arange(256, device="cuda", dtype=torch.int32).expand(1, 3, -1).clone()
    indices[:, 1, 240:] = -1
    lengths = torch.tensor([[256, 240, 0]], device="cuda", dtype=torch.int32)
    query_valid = torch.tensor([[True, True, False]], device="cuda")
    targets = [torch.rand(1, 3, 256, device="cuda") for _ in range(3)]
    expected_sum = torch.zeros_like(targets[0])
    expected_entropy = torch.zeros((), device="cuda")
    accumulator = None
    for target in targets:
        valid = (indices >= 0) & (torch.arange(256, device="cuda") < lengths[..., None])
        valid &= query_valid[..., None]
        probability = target.masked_fill(~valid, 0)
        probability /= probability.sum(-1, keepdim=True).clamp_min(1e-10)
        expected_sum += probability
        expected_entropy += (probability * probability.clamp_min(1e-10).log()).sum()
        accumulator = accumulate_cli_attention_target(
            target, indices, lengths, query_valid, chunk_rows, accumulator
        ).detach()
    torch.testing.assert_close(accumulator.target_sum, expected_sum)
    torch.testing.assert_close(accumulator.teacher_entropy_sum, expected_entropy)
    assert accumulator.reuse_count == len(targets)
    assert accumulator.topk_indices.data_ptr() == indices.data_ptr()


@pytest.mark.parametrize("group_size", [2, 4])
def test_cli_merged_loss_and_score_gradient_match_sum_of_kl(monkeypatch, group_size):
    """Validate the production merge and scaling independently of the cuDNN derivative."""
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_cudnn_kernels as kernels,
    )

    torch.manual_seed(14)
    rows, topk = 2, 256
    logits = torch.randn(1, rows, topk, requires_grad=True)
    teachers = torch.rand(group_size, 1, rows, topk)
    teachers /= teachers.sum(-1, keepdim=True)
    coefficient = 0.003
    reference = coefficient * sum(
        (p * (p.log() - logits.log_softmax(-1))).sum(-1).mean() for p in teachers
    )
    (reference_gradient,) = torch.autograd.grad(reference, logits)
    captured = {}

    monkeypatch.setattr(kernels, "_compute_attn_target", lambda *a, **kw: teachers[0].clone())
    monkeypatch.setattr(kernels, "_pad_indexer_heads_for_backward", lambda q, w: (q, w, q.size(2)))
    monkeypatch.setattr(kernels, "_slice_indexer_backward_head_grads", lambda q, w, _: (q, w))

    def backward(q, w, k, target, predict, indices, *, loss_coeff, grad_loss, block_i):
        del indices, block_i
        captured["gradient"] = (predict - target) * loss_coeff * grad_loss / rows
        return torch.zeros_like(q), torch.zeros_like(w), torch.zeros_like(k)

    monkeypatch.setattr(kernels, "_run_sparse_indexer_backward", backward)
    accumulator = SimpleNamespace(
        target_sum=teachers[1:].sum(0),
        teacher_entropy_sum=(teachers[1:] * teachers[1:].log()).sum(),
        reuse_count=group_size - 1,
    )
    loss, *_ = kernels._compute_sparse_indexer_loss_and_grads(
        q_idx_bshd=torch.zeros(1, rows, 32, 128),
        k_idx_bsd=torch.zeros(1, topk, 128),
        w_bsh=torch.zeros(1, rows, 32),
        topk_indices_cmp=torch.arange(topk, dtype=torch.int32).expand(1, rows, -1),
        topk_length_cmp=torch.full((1, rows), topk, dtype=torch.int32),
        indexer_score_payload=logits.detach(),
        query=torch.zeros(rows, 1, 32, 576),
        kv_full=torch.zeros(topk, 1, 576),
        lse=torch.zeros(rows, 32),
        softmax_scale=0.1,
        loss_coeff=coefficient,
        query_valid_rows=None,
        calculate_per_token_loss=False,
        tp_group=None,
        cli_target_accumulator=accumulator,
    )
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(captured["gradient"], reference_gradient)
