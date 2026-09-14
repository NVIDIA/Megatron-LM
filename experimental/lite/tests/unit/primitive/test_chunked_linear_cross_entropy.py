# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import gc
import math
import socket
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.lite.primitive.ops import chunked_linear_cross_entropy as chunked_lce
from megatron.lite.primitive.ops.chunked_linear_cross_entropy import (
    _cuda_tp1_fast_path_eligible,
    _forward_chunk_loss,
    _reuse_logits_storage_for_grad_logits,
    _row_reduce_workspace_shape,
    chunked_vocab_parallel_linear_cross_entropy,
)
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy

pytestmark = pytest.mark.mlite


@pytest.mark.parametrize(
    ("is_cuda", "dtype", "tp_world_size", "triton_available", "expected"),
    [
        (True, torch.bfloat16, 1, True, True),
        (False, torch.bfloat16, 1, True, False),
        (True, torch.float32, 1, True, False),
        (True, torch.float64, 1, True, False),
        (True, torch.bfloat16, 2, True, False),
        (True, torch.bfloat16, 1, False, False),
    ],
)
def test_chunked_linear_cross_entropy_cuda_fast_path_is_tp1_only(
    is_cuda, dtype, tp_world_size, triton_available, expected
):
    assert (
        _cuda_tp1_fast_path_eligible(
            is_cuda=is_cuda,
            dtype=dtype,
            tp_world_size=tp_world_size,
            triton_available=triton_available,
        )
        is expected
    )


def test_chunked_linear_cross_entropy_cuda_row_reduce_workspace_is_not_vocab_shaped():
    rows, vocab, block = 1024, 151936, 4096

    partial_shape, row_shape = _row_reduce_workspace_shape(rows, vocab, block)

    assert partial_shape == (rows, 38)
    assert row_shape == (rows,)
    assert partial_shape != (rows, vocab)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
def test_chunked_linear_cross_entropy_cuda_fast_path_materializes_stride_zero_dloss():
    torch.manual_seed(19)
    hidden = torch.randn(4, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(11, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 11, (4,), device="cuda")
    kernel_strides = []
    original = chunked_lce._cuda_lce.inplace_gradient

    def record_gradient(logits, target, grad_output, temperature):
        kernel_strides.append(grad_output.stride())
        return original(logits, target, grad_output, temperature)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(chunked_lce._cuda_lce, "inplace_gradient", record_gradient)
    try:
        loss = chunked_vocab_parallel_linear_cross_entropy(hidden, weight, labels, chunk_size=2)
        upstream = torch.ones(1, device="cuda").expand_as(loss)
        assert upstream.stride() == (0,)
        loss.backward(upstream)
    finally:
        monkeypatch.undo()

    assert kernel_strides == [(1,), (1,)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
@pytest.mark.parametrize("rows", [14135, 14136, 16384])
def test_chunked_linear_cross_entropy_cuda_row_base_offsets_support_large_vocab_windows(rows):
    vocab = 151936
    logits = torch.zeros((rows, vocab), device="cuda", dtype=torch.bfloat16)
    target = torch.zeros(rows, device="cuda", dtype=torch.long)

    loss = chunked_lce._cuda_lce.token_loss(logits, target)
    torch.cuda.synchronize()
    torch.testing.assert_close(loss, torch.full_like(loss, math.log(vocab)))

    if rows == 16384:
        grad_output = torch.ones(rows, device="cuda", dtype=torch.float32)
        grad_logits = chunked_lce._cuda_lce.inplace_gradient(logits, target, grad_output, 1.0)
        torch.cuda.synchronize()
        assert grad_logits.data_ptr() == logits.data_ptr()
        torch.testing.assert_close(
            grad_logits[:, 0].float(),
            torch.full((rows,), -(1.0 - 1.0 / vocab), device="cuda"),
            rtol=0.0,
            atol=0.004,
        )
        torch.testing.assert_close(
            grad_logits[:, 1].float(),
            torch.full((rows,), 1.0 / vocab, device="cuda"),
            rtol=0.0,
            atol=0.004,
        )


def test_chunked_linear_cross_entropy_reuses_logits_storage_for_bf16_dlogits():
    logits = torch.empty(3, 7, dtype=torch.bfloat16)
    softmax = torch.rand(3, 7, dtype=torch.float32)

    grad_logits = _reuse_logits_storage_for_grad_logits(logits, softmax)

    assert grad_logits.dtype is torch.bfloat16
    assert grad_logits.data_ptr() == logits.data_ptr()
    torch.testing.assert_close(grad_logits.float(), softmax, rtol=0.0, atol=0.004)


def test_chunked_linear_cross_entropy_releases_previous_softmax_before_next_window(monkeypatch):
    torch.manual_seed(11)
    hidden = torch.randn(6, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(9, 4, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 9, (6, 1))
    original = chunked_lce._chunk_loss_and_softmax
    prior_softmax = []

    def checked_chunk_loss(*args, **kwargs):
        gc.collect()
        assert all(ref() is None for ref in prior_softmax)
        result = original(*args, **kwargs)
        if result[1] is not None:
            prior_softmax.append(weakref.ref(result[1]))
        return result

    monkeypatch.setattr(chunked_lce, "_chunk_loss_and_softmax", checked_chunk_loss)
    chunked_vocab_parallel_linear_cross_entropy(
        hidden, weight, labels, chunk_size=2
    ).sum().backward()


def test_chunked_linear_cross_entropy_releases_forward_logits_at_window_scope(monkeypatch):
    torch.manual_seed(13)
    hidden = torch.randn(6, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(9, 4, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 9, (6, 1))
    original = chunked_lce._chunk_loss_and_softmax
    captured_logits = []

    def checked_chunk_loss(logits, *args, **kwargs):
        captured_logits.append(weakref.ref(logits))
        return original(logits, *args, **kwargs)

    monkeypatch.setattr(chunked_lce, "_chunk_loss_and_softmax", checked_chunk_loss)
    with torch.no_grad():
        _forward_chunk_loss(
            hidden.reshape(-1, 4), weight, labels.reshape(-1), 0, 2, None, 0, 9, 1.25
        )
    gc.collect()
    assert captured_logits[0]() is None


def _reference_loss(hidden, weight, labels, temperature, group=None):
    logits = hidden.matmul(weight.t())
    if temperature != 1.0:
        logits = logits / temperature
    logits_fp32 = logits.float()
    row_max = logits_fp32.max(dim=-1).values
    if group is not None:
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=group)
    shifted = logits_fp32 - row_max.unsqueeze(-1)
    exp_shifted = shifted.exp()
    summed = exp_shifted.sum(dim=-1)
    target = labels.clone()
    local_vocab = weight.shape[0]
    rank = dist.get_rank(group) if group is not None else 0
    mask = (target < rank * local_vocab) | (target >= (rank + 1) * local_vocab)
    target = (target - rank * local_vocab).masked_fill(mask, 0)
    predicted = shifted.reshape(-1, local_vocab)[
        torch.arange(target.numel()), target.reshape(-1)
    ].reshape_as(labels)
    predicted.masked_fill_(mask, 0.0)
    if group is not None:
        dist.all_reduce(summed, group=group)
        dist.all_reduce(predicted, group=group)
    return summed.log() - predicted


def test_chunked_linear_cross_entropy_tp1_matches_matmul_ce_and_saves_no_logits():
    torch.manual_seed(7)
    hidden = torch.randn(5, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(9, 4, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 9, (5, 2))
    hidden_ref = hidden.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    saved = []

    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: (saved.append(tensor), tensor)[1], lambda tensor: tensor
    ):
        token_loss = chunked_vocab_parallel_linear_cross_entropy(
            hidden, weight, labels, temperature=1.25, chunk_size=3
        )
    token_loss.sum().backward()

    expected = _reference_loss(hidden_ref, weight_ref, labels, 1.25)
    expected.sum().backward()

    torch.testing.assert_close(token_loss, expected)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=0.03, atol=0.003)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=0.03, atol=0.003)
    assert not any(tensor.shape == (5, 2, 9) for tensor in saved)


def _tp_sp_worker(rank, world_size, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(17)
        full_hidden = torch.randn(6, 1, 4, dtype=torch.bfloat16)
        full_weight = torch.randn(10, 4, dtype=torch.bfloat16)
        labels = torch.randint(0, 10, (6, 1))
        hidden = full_hidden.chunk(world_size, dim=0)[rank].clone().requires_grad_()
        weight = full_weight.chunk(world_size, dim=0)[rank].clone().requires_grad_()

        actual = chunked_vocab_parallel_linear_cross_entropy(
            hidden, weight, labels, tp_group=dist.group.WORLD, sequence_parallel=True, chunk_size=2
        )
        actual.sum().backward()

        # Match _VanillaColParallelMatmulSP.backward exactly: each vocab shard
        # forms a full-sequence dH contribution, then one reduce-scatter both
        # sums TP contributions and returns the local SP sequence slice.
        logits_ref = full_hidden.matmul(weight.detach().t()).requires_grad_()
        expected = vocab_parallel_cross_entropy(logits_ref, labels, dist.group.WORLD)
        expected.sum().backward()
        grad_hidden = torch.empty_like(hidden)
        dist.reduce_scatter_tensor(
            grad_hidden,
            logits_ref.grad.matmul(weight.detach()).contiguous(),
            group=dist.group.WORLD,
        )
        grad_weight = (
            logits_ref.grad.reshape(-1, weight.shape[0])
            .t()
            .matmul(full_hidden.reshape(-1, full_hidden.shape[-1]))
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(hidden.grad, grad_hidden, rtol=0.03, atol=0.003)
        torch.testing.assert_close(weight.grad, grad_weight, rtol=0.03, atol=0.003)
    finally:
        dist.destroy_process_group()


def test_chunked_linear_cross_entropy_matches_two_rank_gloo_tp_sp():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_tp_sp_worker, args=(2, port), nprocs=2, join=True)


@pytest.mark.parametrize(
    ("labels", "use_fused_kernels", "calculate_entropy", "has_chunked_ep", "expected"),
    [
        (True, False, False, True, True),
        (True, True, False, True, False),
        (True, False, True, True, False),
        (True, False, False, False, False),
        (False, False, False, True, False),
    ],
)
def test_qwen3_chunked_head_loss_selection_contract(
    labels, use_fused_kernels, calculate_entropy, has_chunked_ep, expected
):
    from megatron.lite.model.qwen3_moe.lite.head_loss import use_chunked_head_loss

    assert (
        use_chunked_head_loss(
            has_labels=labels,
            use_fused_kernels=use_fused_kernels,
            calculate_entropy=calculate_entropy,
            has_chunked_ep=has_chunked_ep,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("token_count", "chunk_count", "expected"), [(1, 1, 1), (1025, 3, 342), (32768, 2, 16384)]
)
def test_qwen3_head_loss_balances_tokens_across_configured_ep_chunks(
    token_count, chunk_count, expected
):
    from megatron.lite.model.qwen3_moe.lite.head_loss import balanced_head_loss_chunk_size

    assert balanced_head_loss_chunk_size(token_count, chunk_count) == expected
