# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only contract probes for the isolated QSA Stage-2 KL prototype."""

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.experimental_attention_variant.qsa_stage2_loss import (
    qsa_stage2_sparse_kl,
)


def _toy():
    torch.manual_seed(84)
    sequence = 12
    hidden = torch.randn(sequence, 8, dtype=torch.float64, requires_grad=True)
    query_weight = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float64))
    key_weight = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.float64))
    index_query = F.linear(hidden.detach(), query_weight).view(sequence, 2, 4)
    compressed_key = F.linear(hidden.detach(), key_weight).view(3, 4, 4).mean(dim=1)
    teacher_query = torch.randn(sequence, 4, 4, dtype=torch.float64, requires_grad=True)
    teacher_key = torch.randn(sequence, 2, 4, dtype=torch.float64, requires_grad=True)
    block_ids = torch.full((sequence, 2), -1, dtype=torch.long)
    block_ids[3:, 0] = 0
    block_ids[7:, 1] = 1
    return dict(
        hidden=hidden,
        query_weight=query_weight,
        key_weight=key_weight,
        index_query=index_query,
        compressed_key=compressed_key,
        teacher_query=teacher_query,
        teacher_key=teacher_key,
        block_ids=block_ids,
        block_starts=torch.tensor([0, 4, 8]),
        document_starts=torch.zeros(sequence, dtype=torch.long),
        query_positions=torch.arange(sequence),
    )


def _call(toy, **kwargs):
    return qsa_stage2_sparse_kl(
        toy["index_query"],
        toy["compressed_key"],
        toy["block_ids"],
        toy["block_starts"],
        toy["teacher_query"],
        toy["teacher_key"],
        toy["document_starts"],
        toy["query_positions"],
        compress_ratio=4,
        loss_coeff=0.7,
        **kwargs,
    )


def _dense_reference(toy):
    """Small-sequence independent implementation of the selected-support formula."""
    total = toy["index_query"].new_zeros(())
    for row in range(toy["index_query"].size(0)):
        blocks = toy["block_ids"][row]
        blocks = blocks[blocks >= 0]
        if blocks.numel() == 0:
            continue
        selected = torch.cat([toy["block_starts"][block] + torch.arange(4) for block in blocks])
        tail_first = ((row + 1) // 4) * 4
        tail = torch.arange(tail_first, row + 1)
        tokens = torch.cat((selected, tail))
        tq = toy["teacher_query"][row].detach()
        tk = toy["teacher_key"][tokens].detach().repeat_interleave(2, dim=1)
        teacher_score = torch.einsum("hd,khd->hk", tq, tk) / math.sqrt(4)
        probability = teacher_score.softmax(dim=-1).sum(dim=0)
        probability = probability / probability.sum()
        mass = probability[: blocks.numel() * 4].view(-1, 4).amax(dim=-1)
        target = mass / mass.sum()
        student = torch.stack(
            [
                torch.einsum(
                    "hd,hd->h", toy["index_query"][row], toy["compressed_key"][block].expand(2, -1)
                )
                .relu()
                .sum()
                / math.sqrt(4)
                for block in blocks
            ]
        )
        total = total + (target * (target.log() - student.log_softmax(dim=-1))).sum()
    return total / toy["index_query"].size(0) * 0.7


def test_qsa_stage2_teacher_detached_both_indexer_projections_train():
    toy = _toy()
    loss = _call(toy, query_chunk_size=3)
    assert loss.isfinite()
    loss.backward()
    assert toy["query_weight"].grad is not None
    assert toy["query_weight"].grad.abs().sum() > 0
    assert toy["key_weight"].grad is not None
    assert toy["key_weight"].grad.abs().sum() > 0
    assert toy["hidden"].grad is None
    assert toy["teacher_query"].grad is None
    assert toy["teacher_key"].grad is None


def test_qsa_stage2_zero_coefficient_is_exact_baseline():
    toy = _toy()
    baseline = toy["hidden"].square()
    zero_loss = qsa_stage2_sparse_kl(
        toy["index_query"],
        toy["compressed_key"],
        toy["block_ids"],
        toy["block_starts"],
        toy["teacher_query"],
        toy["teacher_key"],
        toy["document_starts"],
        toy["query_positions"],
        compress_ratio=4,
        loss_coeff=0,
    )
    assert zero_loss.item() == 0.0
    output = DSAIndexerLossAutoScaler.apply(baseline, zero_loss)
    assert torch.equal(output, baseline)
    output.sum().backward()
    torch.testing.assert_close(toy["hidden"].grad, toy["hidden"].detach() * 2, rtol=0, atol=0)
    assert toy["query_weight"].grad is None
    assert toy["key_weight"].grad is None


def test_qsa_stage2_chunked_matches_dense_small_sequence_and_gradients():
    toy = _toy()
    actual = _call(toy, query_chunk_size=3)
    dense = _dense_reference(toy)
    torch.testing.assert_close(actual, dense, rtol=1e-13, atol=1e-13)
    grads_actual = torch.autograd.grad(
        actual, (toy["index_query"], toy["compressed_key"]), retain_graph=True
    )
    grads_dense = torch.autograd.grad(dense, (toy["index_query"], toy["compressed_key"]))
    for got, expected in zip(grads_actual, grads_dense):
        torch.testing.assert_close(got, expected, rtol=1e-13, atol=1e-13)


def test_qsa_stage2_rejects_cross_document_selected_block():
    toy = _toy()
    toy["document_starts"][8:] = 8
    toy["query_positions"][8:] -= 8
    try:
        _call(toy)
    except ValueError as error:
        assert "document or causal boundary" in str(error)
    else:
        raise AssertionError("cross-document routes were accepted")


def test_qsa_stage2_autoscaler_uses_main_loss_scale():
    toy = _toy()
    loss = _call(toy, query_chunk_size=3)
    expected = torch.autograd.grad(
        loss, (toy["query_weight"], toy["key_weight"]), retain_graph=True
    )
    old_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    saved_scale = old_scale.clone() if old_scale is not None else None
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(2.0))
    try:
        output = DSAIndexerLossAutoScaler.apply(toy["hidden"], loss)
        assert torch.equal(output, toy["hidden"])
        output.sum().backward()
    finally:
        if saved_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        else:
            DSAIndexerLossAutoScaler.set_loss_scale(saved_scale)
    torch.testing.assert_close(toy["query_weight"].grad, 2 * expected[0], rtol=0, atol=1e-12)
    torch.testing.assert_close(toy["key_weight"].grad, 2 * expected[1], rtol=0, atol=1e-12)
    torch.testing.assert_close(toy["hidden"].grad, torch.ones_like(toy["hidden"]), rtol=0, atol=0)


def test_qsa_stage2_local_query_partition_mean_is_full_query_mean():
    """Algebraic local-row check; this is not a distributed CP runtime test."""
    toy = _toy()
    full = _call(toy, query_chunk_size=3)

    def local(start, end):
        return qsa_stage2_sparse_kl(
            toy["index_query"][start:end],
            toy["compressed_key"],
            toy["block_ids"][start:end],
            toy["block_starts"],
            toy["teacher_query"][start:end],
            toy["teacher_key"],
            toy["document_starts"][start:end],
            toy["query_positions"][start:end],
            compress_ratio=4,
            loss_coeff=0.7,
            query_chunk_size=3,
        )

    # CP gradient averaging would combine these two equal-sized local means.
    local_mean = (local(0, 6) + local(6, 12)) / 2
    torch.testing.assert_close(local_mean, full, rtol=1e-13, atol=1e-13)


def test_qsa_stage2_packed_document_offsets_keep_teacher_inside_documents():
    torch.manual_seed(13)
    index_query = torch.randn(16, 2, 4, dtype=torch.float64, requires_grad=True)
    compressed_key = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    teacher_query = torch.randn(16, 4, 4, dtype=torch.float64, requires_grad=True)
    teacher_key = torch.randn(16, 2, 4, dtype=torch.float64, requires_grad=True)
    starts = torch.tensor([0, 4, 8, 12])
    documents = torch.tensor([0] * 8 + [8] * 8)
    positions = torch.arange(8).repeat(2)
    blocks = torch.full((16, 2), -1, dtype=torch.long)
    blocks[3:8, 0] = 0
    blocks[7, 1] = 1
    blocks[11:16, 0] = 2
    blocks[15, 1] = 3

    def loss(row_slice, key):
        return qsa_stage2_sparse_kl(
            index_query[row_slice],
            compressed_key,
            blocks[row_slice],
            starts,
            teacher_query[row_slice],
            key,
            documents[row_slice],
            positions[row_slice],
            compress_ratio=4,
            loss_coeff=1,
            query_chunk_size=3,
        )

    full = loss(slice(None), teacher_key)
    split = (loss(slice(0, 8), teacher_key) + loss(slice(8, 16), teacher_key)) / 2
    torch.testing.assert_close(full, split, rtol=1e-13, atol=1e-13)
    # Altering the first document's main-attention keys must not change the
    # second document's teacher or KL target.
    altered_key = teacher_key.detach().clone()
    altered_key[:8] += 1000
    second = loss(slice(8, 16), teacher_key)
    torch.testing.assert_close(second, loss(slice(8, 16), altered_key), rtol=0, atol=0)


@pytest.mark.parametrize("bad_scale", [0.0, float("nan")])
def test_qsa_stage2_rejects_invalid_teacher_softmax_scale(bad_scale):
    with pytest.raises(ValueError, match="softmax_scale must be finite and positive"):
        _call(_toy(), softmax_scale=bad_scale)
