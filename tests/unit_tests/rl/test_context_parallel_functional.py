# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Functional tests for context-parallel logprob computation.

Each test spawns 2 CPU processes via torch.multiprocessing, sets up a gloo
process group with CP size = 2, and verifies that the real
_scatter_for_context_parallel / _gather_logprobs_context_parallel pair (with
rl_utils.tex patched by a pure-torch reference partitioner — see
test_context_parallel.py) returns the same tensor as a single-rank reference
computation, including for packed multi-sequence bins.

Run with:
    pytest tests/unit_tests/rl/test_context_parallel_functional.py -v
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def _reference_thd_partition_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
    """Pure-torch reference for tex.thd_get_partitioned_indices (per-seq zigzag)."""
    idxs = []
    cu = [int(x) for x in cu_seqlens_padded]
    for s, e in zip(cu[:-1], cu[1:]):
        slot_len = e - s
        if slot_len == 0:
            continue
        chunk = slot_len // (2 * cp_size)
        idxs += list(range(s + cp_rank * chunk, s + (cp_rank + 1) * chunk))
        idxs += list(
            range(s + (2 * cp_size - cp_rank - 1) * chunk, s + (2 * cp_size - cp_rank) * chunk)
        )
    return torch.tensor(idxs, dtype=torch.int64)


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


def _worker_get_logprobs_cp(
    rank: int,
    world_size: int,
    store_path: str,
    seq_len: int,
    batch: int,
    vocab: int,
    cu_padded: list,
    result_queue: mp.Queue,
) -> None:
    """Worker function: set up CP group, run scatter + gather, put result in queue."""
    try:
        cp_group = _init_pg(rank, world_size, store_path)

        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.rl.rl_utils import (
            _gather_logprobs_context_parallel,
            _scatter_for_context_parallel,
            selective_log_softmax,
        )

        torch.manual_seed(0)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        # Simulate full logits known to all ranks (deterministic, same on both)
        torch.manual_seed(0)
        logits_full = torch.randn(batch, seq_len, vocab)

        cu = torch.tensor(cu_padded, dtype=torch.int32)
        psp = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu, cu_seqlens_kv=cu,
            max_seqlen_q=seq_len, max_seqlen_kv=seq_len,
            total_tokens=seq_len,
        )

        fake_tex = SimpleNamespace(thd_get_partitioned_indices=_reference_thd_partition_indices)

        # Patch mpu to return our synthetic CP group and tex with the reference.
        with (
            patch('megatron.rl.rl_utils.mpu') as mock_mpu,
            patch('megatron.rl.rl_utils.tex', fake_tex),
        ):
            mock_mpu.get_context_parallel_world_size.return_value = world_size
            mock_mpu.get_context_parallel_rank.return_value = rank
            mock_mpu.get_context_parallel_group.return_value = cp_group

            local_tokens, local_pos, cp_psp, local_labels, idx = _scatter_for_context_parallel(
                tokens, position_ids, psp, world_size
            )
            # The model's output on the partitioned input is the rank's slice of
            # the full logits (ring attention makes each local position's output
            # equal that of the full-sequence forward).
            local_logits = logits_full.index_select(1, idx)

            local_lp = selective_log_softmax(local_logits, local_labels)
            full_lp = _gather_logprobs_context_parallel(local_lp, idx, seq_len, no_grad=True)

        result_queue.put(("ok", full_lp.cpu()))

    except Exception as exc:  # pragma: no cover
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
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
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

    def _check_results(self, results, expected_logprobs):
        """Assert all workers returned successfully and matching logprobs."""
        for status, payload in results:
            assert status == "ok", f"Worker failed:\n{payload}"
            torch.testing.assert_close(payload, expected_logprobs, atol=1e-5, rtol=1e-4)

    def _reference_logprobs(self, seq_len, batch, vocab):
        torch.manual_seed(0)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        torch.manual_seed(0)
        logits = torch.randn(batch, seq_len, vocab)
        from megatron.rl.rl_utils import selective_log_softmax
        return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])

    def test_cp2_single_sequence(self):
        """CP=2 logprobs must match the reference single-rank computation."""
        seq_len, batch, vocab = 8, 1, 16
        expected = self._reference_logprobs(seq_len, batch, vocab)
        results = _run_2rank_test(
            _worker_get_logprobs_cp,
            seq_len=seq_len, batch=batch, vocab=vocab, cu_padded=[0, seq_len],
        )
        self._check_results(results, expected)

    def test_cp2_larger_sequence(self):
        """CP=2 with a larger sequence (16 tokens, batch 2)."""
        seq_len, batch, vocab = 16, 2, 32
        expected = self._reference_logprobs(seq_len, batch, vocab)
        results = _run_2rank_test(
            _worker_get_logprobs_cp,
            seq_len=seq_len, batch=batch, vocab=vocab, cu_padded=[0, seq_len],
        )
        self._check_results(results, expected)

    def test_cp2_packed_bin(self):
        """CP=2 with a packed bin: two aligned slots plus a trailing ghost slot."""
        seq_len, batch, vocab = 24, 1, 16
        expected = self._reference_logprobs(seq_len, batch, vocab)
        results = _run_2rank_test(
            _worker_get_logprobs_cp,
            seq_len=seq_len, batch=batch, vocab=vocab, cu_padded=[0, 8, 16, 24],
        )
        self._check_results(results, expected)

    def test_cp2_all_ranks_agree(self):
        """Both CP ranks must return the identical full-sequence logprob tensor."""
        seq_len, batch, vocab = 8, 1, 16
        results = _run_2rank_test(
            _worker_get_logprobs_cp,
            seq_len=seq_len, batch=batch, vocab=vocab, cu_padded=[0, seq_len],
        )
        statuses = [r[0] for r in results]
        assert all(s == "ok" for s in statuses), str(results)
        lp0, lp1 = results[0][1], results[1][1]
        torch.testing.assert_close(lp0, lp1)
