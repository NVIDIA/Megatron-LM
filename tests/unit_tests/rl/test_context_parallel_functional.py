# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Functional tests for context-parallel logprob computation.

Each test spawns 2 CPU processes via torch.multiprocessing.spawn, sets up a
minimal gloo process group with CP size = 2, and verifies that the
scatter -> local logprobs -> gather pipeline returns the same tensor as a
single-rank reference computation, for both a single full-length sequence
and a multi-sequence packed bin (per-sequence zigzag layout with real
cu_seqlens_*_padded).

Run with:
    pytest tests/unit_tests/rl/test_context_parallel_functional.py -v
"""

import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PAD = 0
MAX_SEQUENCES_PER_BIN = 8

# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def _stub_triton_if_missing():
    try:  # pragma: no cover
        import triton  # noqa: F401
    except ImportError:
        sys.modules.setdefault('megatron.core.transformer.moe.paged_stash', MagicMock())


def _init_pg(rank: int, world_size: int, store_path: str) -> dist.ProcessGroup:
    """Create an in-process distributed group backed by a file store."""
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )
    return dist.new_group(ranks=list(range(world_size)), backend="gloo")


def _make_bin_params(seq_lengths, bin_size, cp_size):
    from megatron.rl.sequence_packing_utils import (
        PackingInfo,
        create_packed_seq_params_for_bin,
    )

    packing_info = PackingInfo(
        bin_seq_indices=[list(range(len(seq_lengths)))],
        seq_starts={0: []},
        seq_lengths=list(seq_lengths),
        seq_to_bin_idx=[0] * len(seq_lengths),
        packing_algo='fifo',
    )
    return create_packed_seq_params_for_bin(
        packing_info=packing_info,
        bin_idx=0,
        bin_size=bin_size,
        max_sequences_per_bin=MAX_SEQUENCES_PER_BIN,
        device=torch.device('cpu'),
        cp_size=cp_size,
    )


def _make_inputs(seq_lengths, bin_size, vocab):
    """Deterministic bin tokens (unique ids), position ids and a per-token-id
    logits table shared by all ranks and the reference."""
    total = sum(seq_lengths)
    tokens = torch.full((1, bin_size), PAD, dtype=torch.long)
    tokens[0, :total] = torch.arange(1, total + 1)
    position_ids = torch.zeros(1, bin_size, dtype=torch.long)
    start = 0
    for length in seq_lengths:
        position_ids[0, start : start + length] = torch.arange(length)
        start += length
    torch.manual_seed(0)
    logits_table = torch.randn(1, vocab, vocab)
    return tokens, position_ids, logits_table


def _worker_get_logprobs_cp(
    rank: int,
    world_size: int,
    store_path: str,
    seq_lengths: list,
    bin_size: int,
    result_queue: mp.Queue,
) -> None:
    """Worker: scatter, compute local logprobs from a shared per-token-id
    logits table, gather, and return the full-sequence logprob tensor."""
    try:
        _stub_triton_if_missing()
        cp_group = _init_pg(rank, world_size, store_path)

        from megatron.rl.rl_utils import (
            _gather_logprobs_context_parallel,
            _scatter_for_context_parallel,
            selective_log_softmax,
        )

        vocab = sum(seq_lengths) + 2
        tokens, position_ids, logits_table = _make_inputs(seq_lengths, bin_size, vocab)
        psp = _make_bin_params(seq_lengths, bin_size, world_size)

        # Patch mpu to return our synthetic CP group.
        with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
            mock_mpu.get_context_parallel_world_size.return_value = world_size
            mock_mpu.get_context_parallel_rank.return_value       = rank
            mock_mpu.get_context_parallel_group.return_value      = cp_group

            local_tokens, _, _, local_labels, cp_gather_index = (
                _scatter_for_context_parallel(
                    tokens, position_ids, psp, world_size, pad_token=PAD
                )
            )
            # "Model output" for a token is its row of the shared logits
            # table, so the local slice is exactly the padded-frame slice.
            local_logits = logits_table[:, local_tokens[0], :]
            local_lp = selective_log_softmax(local_logits, local_labels)
            full_lp = _gather_logprobs_context_parallel(
                local_lp, cp_gather_index, no_grad=True
            )

        result_queue.put(("ok", full_lp.cpu()))

    except Exception:  # pragma: no cover
        import traceback
        result_queue.put(("err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helper to run a 2-process test
# ---------------------------------------------------------------------------

def _run_2rank_test(worker_fn, **kwargs) -> list:
    """Spawn world_size=2 workers and collect their results."""
    world_size = 2
    ctx        = mp.get_context("spawn")
    queue      = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=True) as f:
        store_path = f.name + ".store"
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker_fn,
            args=(rank, world_size, store_path),
            kwargs={**kwargs, "result_queue": queue},
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    results = [queue.get_nowait() for _ in range(world_size)]
    return results


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

class TestCPLogprobsFunctional:

    @staticmethod
    def _reference_logprobs(seq_lengths, bin_size):
        _stub_triton_if_missing()
        from megatron.rl.rl_utils import selective_log_softmax

        vocab = sum(seq_lengths) + 2
        tokens, _, logits_table = _make_inputs(seq_lengths, bin_size, vocab)
        return selective_log_softmax(logits_table[:, tokens[0, :-1], :], tokens[:, 1:])

    @staticmethod
    def _valid_mask(seq_lengths, bin_size):
        """Positions whose shifted label stays within the same sequence — the
        only positions the loss mask can select."""
        valid = torch.zeros(bin_size - 1, dtype=torch.bool)
        start = 0
        for length in seq_lengths:
            valid[start : start + length - 1] = True
            start += length
        return valid

    def _check(self, seq_lengths, bin_size):
        expected = self._reference_logprobs(seq_lengths, bin_size)
        valid = self._valid_mask(seq_lengths, bin_size)
        results = _run_2rank_test(
            _worker_get_logprobs_cp, seq_lengths=seq_lengths, bin_size=bin_size
        )
        payloads = []
        for status, payload in results:
            assert status == "ok", f"Worker failed:\n{payload}"
            assert payload.shape == expected.shape
            torch.testing.assert_close(
                payload[0, valid], expected[0, valid], atol=1e-5, rtol=1e-4
            )
            payloads.append(payload)
        # Both CP ranks must return the identical full tensor.
        torch.testing.assert_close(payloads[0], payloads[1])

    def test_cp2_single_sequence(self):
        """CP=2, one full-length sequence (no per-sequence padding)."""
        self._check(seq_lengths=[16], bin_size=16)

    def test_cp2_packed_bin_odd_lengths(self):
        """CP=2, multi-sequence bin with odd lengths (per-sequence padding)."""
        self._check(seq_lengths=[7, 5, 9], bin_size=32)

    def test_cp2_packed_bin_aligned_lengths(self):
        """CP=2, multi-sequence bin whose lengths are already 2*CP multiples."""
        self._check(seq_lengths=[8, 4, 12], bin_size=32)
