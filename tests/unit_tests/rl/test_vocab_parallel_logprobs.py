# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for vocab_parallel_selective_log_softmax.

Single-process tests verify forward/backward equivalence against the gathered
`selective_log_softmax` reference (fp32 and bf16, with and without chunking).
The 2-process test spawns a gloo TP group, shards the vocab dimension in
halves, and verifies both the forward logprobs and the per-shard gradients
against a single-rank full-vocab reference.

Run with:
    pytest tests/unit_tests/rl/test_vocab_parallel_logprobs.py -v
"""

import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.rl.rl_utils import (
    selective_log_softmax,
    vocab_parallel_selective_log_softmax,
)

# ---------------------------------------------------------------------------
# Reference implementation (naive, full vocab)
# ---------------------------------------------------------------------------


def _naive_logprobs(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return (
        torch.gather(
            logits.float().log_softmax(-1), dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
    )


# ---------------------------------------------------------------------------
# Single-process tests (tp_group=None -> unsharded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_tokens", [None, 3])
def test_forward_matches_reference_fp32(chunk_tokens):
    torch.manual_seed(0)
    batch, seq, vocab = 2, 17, 64
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    index = torch.randint(0, vocab, (batch, seq))

    got = vocab_parallel_selective_log_softmax(logits, index, chunk_tokens=chunk_tokens)
    want = _naive_logprobs(logits, index)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("chunk_tokens", [None, 3])
def test_forward_matches_reference_bf16(chunk_tokens):
    torch.manual_seed(1)
    batch, seq, vocab = 2, 17, 64
    logits = torch.randn(batch, seq, vocab, dtype=torch.bfloat16)
    index = torch.randint(0, vocab, (batch, seq))

    got = vocab_parallel_selective_log_softmax(logits, index, chunk_tokens=chunk_tokens)
    assert got.dtype == torch.bfloat16
    # Reference computed in fp32 from the same bf16 logits, then cast: the only
    # differences are fp32 reduction order, so tolerances are tight.
    want = _naive_logprobs(logits, index).to(torch.bfloat16)

    torch.testing.assert_close(got.float(), want.float(), atol=1e-2, rtol=1e-2)


def test_matches_selective_log_softmax_fp32():
    """Drop-in parity with the function it replaces (fp32 path)."""
    torch.manual_seed(2)
    batch, seq, vocab = 3, 9, 32
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    index = torch.randint(0, vocab, (batch, seq))

    got = vocab_parallel_selective_log_softmax(logits, index)
    want = selective_log_softmax(logits, index)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_matches_reference(dtype):
    torch.manual_seed(3)
    batch, seq, vocab = 2, 11, 48
    base = torch.randn(batch, seq, vocab, dtype=dtype)
    index = torch.randint(0, vocab, (batch, seq))
    grad_out = torch.randn(batch, seq, dtype=dtype)

    logits_a = base.clone().requires_grad_(True)
    vocab_parallel_selective_log_softmax(logits_a, index, chunk_tokens=4).backward(grad_out)

    logits_b = base.clone().requires_grad_(True)
    _naive_logprobs(logits_b, index).backward(grad_out.float())

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        logits_a.grad.float(), logits_b.grad.float(), atol=atol, rtol=atol
    )


def test_forward_does_not_mutate_fp32_input():
    """The chunked in-place ops must operate on copies, not the saved logits."""
    torch.manual_seed(4)
    logits = torch.randn(2, 5, 16, dtype=torch.float32)
    snapshot = logits.clone()
    index = torch.randint(0, 16, (2, 5))

    vocab_parallel_selective_log_softmax(logits, index, chunk_tokens=2)

    torch.testing.assert_close(logits, snapshot)


def test_sliced_view_input():
    """get_logprobs calls the function on logits[:, :-1, :] — a non-contiguous view."""
    torch.manual_seed(5)
    batch, seq, vocab = 2, 8, 24
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32, requires_grad=True)
    tokens = torch.randint(0, vocab, (batch, seq))

    got = vocab_parallel_selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])
    want = _naive_logprobs(logits[:, :-1, :], tokens[:, 1:])
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    got.sum().backward()
    assert logits.grad is not None
    # No gradient flows to the sliced-away last position.
    assert torch.all(logits.grad[:, -1, :] == 0)


# ---------------------------------------------------------------------------
# 2-process TP test (gloo)
# ---------------------------------------------------------------------------


def _init_pg(rank: int, world_size: int, store_path: str) -> dist.ProcessGroup:
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    return dist.new_group(ranks=list(range(world_size)), backend="gloo")


def _worker_tp2(
    rank: int,
    world_size: int,
    store_path: str,
    result_queue: mp.Queue,
) -> None:
    try:
        tp_group = _init_pg(rank, world_size, store_path)

        torch.manual_seed(0)
        batch, seq, vocab = 2, 13, 64
        logits_full = torch.randn(batch, seq, vocab, dtype=torch.float32)
        index = torch.randint(0, vocab, (batch, seq))
        grad_out = torch.randn(batch, seq, dtype=torch.float32)

        partition = vocab // world_size
        start = rank * partition
        local_logits = (
            logits_full[:, :, start : start + partition].clone().requires_grad_(True)
        )

        local_lp = vocab_parallel_selective_log_softmax(
            local_logits, index, tp_group=tp_group, chunk_tokens=4
        )
        local_lp.backward(grad_out)

        result_queue.put(("ok", rank, local_lp.detach().cpu(), local_logits.grad.cpu()))
    except Exception:  # pragma: no cover
        result_queue.put(("err", rank, traceback.format_exc(), None))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_tp2_matches_full_vocab_reference():
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=True) as f:
        store_path = f.name + ".store"

    procs = [
        ctx.Process(target=_worker_tp2, args=(rank, world_size, store_path, queue))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=120) for _ in range(world_size)]
    for p in procs:
        p.join(timeout=120)

    by_rank = {}
    for status, rank, payload, grad in results:
        assert status == "ok", f"rank {rank} failed:\n{payload}"
        by_rank[rank] = (payload, grad)

    # Single-rank full-vocab reference.
    torch.manual_seed(0)
    batch, seq, vocab = 2, 13, 64
    logits_full = torch.randn(batch, seq, vocab, dtype=torch.float32).requires_grad_(True)
    index = torch.randint(0, vocab, (batch, seq))
    grad_out = torch.randn(batch, seq, dtype=torch.float32)
    ref_lp = _naive_logprobs(logits_full, index)
    ref_lp.backward(grad_out)

    partition = vocab // world_size
    for rank in range(world_size):
        lp, grad = by_rank[rank]
        # Every rank returns the same full-sequence logprobs.
        torch.testing.assert_close(lp, ref_lp.detach(), atol=1e-5, rtol=1e-5)
        # Each rank's grad equals its vocab shard of the reference grad.
        start = rank * partition
        torch.testing.assert_close(
            grad, logits_full.grad[:, :, start : start + partition], atol=1e-5, rtol=1e-5
        )
