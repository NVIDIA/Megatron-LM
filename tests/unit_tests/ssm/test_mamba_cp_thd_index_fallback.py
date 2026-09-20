# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parity for THD CP indices when TE's whole-cu shared-memory table is too large."""

import json
import os
import time

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm import mamba_context_parallel as mcp
from tests.unit_tests.test_utilities import Utils


def _layout(cp_size, device):
    lengths = [0, 4 * cp_size, 2 * cp_size, 0, 6 * cp_size]
    valid = [0, lengths[1] - 2, lengths[2] - 1, 0, lengths[4] - 1]
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=device
    )
    cu_valid = torch.tensor(
        [0] + list(torch.tensor(valid).cumsum(0).tolist()), dtype=torch.int32, device=device
    )
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_valid,
        cu_seqlens_kv=cu_valid,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )
    return lengths, cu, packed


def _reference_indices(lengths, cp_size, cp_rank):
    indices = []
    start = 0
    for length in lengths:
        chunk = length // (2 * cp_size)
        for part in (cp_rank, 2 * cp_size - 1 - cp_rank):
            indices.extend(range(start + part * chunk, start + (part + 1) * chunk))
        start += length
    return torch.tensor(indices, dtype=torch.int32)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_thd_fallback_indices_match_manual_zigzag_with_empty_and_padded_docs(monkeypatch, cp_size):
    lengths, cu, packed = _layout(cp_size, "cpu")
    total = sum(lengths)
    monkeypatch.setattr(mcp, "_THD_TE_DEFAULT_SHARED_BYTES", 0)
    parts = [mcp._thd_partitioned_indices(cu, total, cp_size, rank) for rank in range(cp_size)]
    for rank, actual in enumerate(parts):
        assert actual.dtype == torch.int32
        torch.testing.assert_close(actual, _reference_indices(lengths, cp_size, rank))
    assert torch.equal(torch.sort(torch.cat(parts)).values, torch.arange(total, dtype=torch.int32))

    base = torch.randn(total, 3, dtype=torch.float32)
    weights = torch.randn_like(base)
    full = base.detach().clone().requires_grad_()
    reordered = mcp._redo_attention_load_balancing(full, cp_size, packed)
    expected = base.index_select(0, torch.cat(parts))
    torch.testing.assert_close(reordered, expected, atol=0, rtol=0)
    restored = mcp._undo_attention_load_balancing(reordered, cp_size, packed)
    torch.testing.assert_close(restored, base, atol=0, rtol=0)
    (restored * weights).sum().backward()
    torch.testing.assert_close(full.grad, weights, atol=0, rtol=0)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_thd_fallback_random_document_boundaries_match_manual_chunks(monkeypatch, cp_size):
    monkeypatch.setattr(mcp, "_THD_TE_DEFAULT_SHARED_BYTES", 0)
    generator = torch.Generator().manual_seed(257)
    for _ in range(20):
        lengths = (torch.randint(0, 7, (11,), generator=generator) * (2 * cp_size)).tolist()
        lengths[0] = 2 * cp_size
        cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
        for rank in range(cp_size):
            actual = mcp._thd_partitioned_indices(cu, sum(lengths), cp_size, rank)
            torch.testing.assert_close(actual, _reference_indices(lengths, cp_size, rank))


def test_thd_fallback_automatically_handles_cu_above_default_shared_limit():
    documents = 12288
    cu = 4 * torch.arange(documents + 1, dtype=torch.int32)
    assert cu.numel() * cu.element_size() > mcp._THD_TE_DEFAULT_SHARED_BYTES
    actual = mcp._thd_partitioned_indices(cu, documents * 4, 2, 0)
    torch.testing.assert_close(actual, _reference_indices([4] * documents, 2, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and TE")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_thd_fallback_matches_te_indices_and_forward_backward(monkeypatch, cp_size):
    lengths, cu, packed = _layout(cp_size, "cuda")
    total = sum(lengths)
    for rank in range(cp_size):
        expected = mcp.tex.thd_get_partitioned_indices(cu, total, cp_size, rank)
        monkeypatch.setattr(mcp, "_THD_TE_DEFAULT_SHARED_BYTES", 0)
        actual = mcp._thd_partitioned_indices(cu, total, cp_size, rank)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    x = torch.randn(total, 8, dtype=torch.float32, device="cuda")
    weights = torch.randn_like(x)
    with monkeypatch.context() as patcher:
        patcher.setattr(mcp, "_THD_TE_DEFAULT_SHARED_BYTES", 0)
        fallback_input = x.detach().clone().requires_grad_()
        fallback = mcp._undo_attention_load_balancing(
            mcp._redo_attention_load_balancing(fallback_input, cp_size, packed), cp_size, packed
        )
        (fallback * weights).sum().backward()
    with monkeypatch.context() as patcher:
        patcher.setattr(mcp, "_THD_TE_DEFAULT_SHARED_BYTES", 48 * 1024)
        te_input = x.detach().clone().requires_grad_()
        te_result = mcp._undo_attention_load_balancing(
            mcp._redo_attention_load_balancing(te_input, cp_size, packed), cp_size, packed
        )
        (te_result * weights).sum().backward()
    torch.testing.assert_close(fallback, te_result, atol=0, rtol=0)
    torch.testing.assert_close(fallback_input.grad, te_input.grad, atol=0, rtol=0)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2 or not torch.cuda.is_available(),
    reason="requires two CUDA ranks",
)
@pytest.mark.parametrize("many_documents", [False, True])
def test_thd_fallback_cp2_reconstruct_remote_gradient(monkeypatch, many_documents):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(0.1, device=torch.cuda.current_device())
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        cp_rank = int(os.environ["RANK"])
        if many_documents:
            lengths = [131072] + [4] * 32768
            valid_lengths = [131072] + [1] * 32768
        else:
            lengths = [0, 16, 8, 0, 24]
            valid_lengths = [0, 14, 7, 0, 23]
        total = sum(lengths)
        cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device="cuda")
        cu_valid = torch.tensor(
            [0] + list(torch.tensor(valid_lengths).cumsum(0).tolist()), device="cuda"
        )
        cu, cu_valid = cu.int(), cu_valid.int()
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_valid,
            cu_seqlens_kv=cu_valid,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        full = torch.arange(total, device="cuda", dtype=torch.float32).unsqueeze(-1)
        if many_documents:
            assert cu.numel() * cu.element_size() > mcp._THD_TE_DEFAULT_SHARED_BYTES

            def reject_te_partition(*args, **kwargs):
                raise AssertionError("large cu table must use the searchsorted fallback")

            monkeypatch.setattr(mcp.tex, "thd_get_partitioned_indices", reject_te_partition)
        else:
            monkeypatch.setattr(mcp, "_THD_TE_DEFAULT_SHARED_BYTES", 0)
        expected_indices = _reference_indices(lengths, 2, cp_rank).to("cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        forward_start = time.perf_counter()
        local = mcp.split_tensor_cp(full, packed).detach().requires_grad_()
        rebuilt = mcp.reconstruct_tensor_cp(local, packed, differentiable=True)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_start
        torch.testing.assert_close(local, full.index_select(0, expected_indices), atol=0, rtol=0)
        torch.testing.assert_close(rebuilt, full, atol=0, rtol=0)
        backward_start = time.perf_counter()
        (rebuilt.sum() if cp_rank == 0 else rebuilt.sum() * 0).backward()
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - backward_start
        torch.testing.assert_close(local.grad, torch.ones_like(local), atol=0, rtol=0)
        if many_documents:
            print(
                "THD_CP2_FALLBACK_RESULT "
                + json.dumps(
                    {
                        "rank": cp_rank,
                        "documents": len(lengths),
                        "total_tokens": total,
                        "cu_bytes": cu.numel() * cu.element_size(),
                        "forward_seconds": forward_seconds,
                        "backward_seconds": backward_seconds,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "auto_fallback": True,
                    }
                ),
                flush=True,
            )
        if not many_documents:
            monkeypatch.setattr(mcp, "_THD_TE_DEFAULT_SHARED_BYTES", 48 * 1024)
            te_local = mcp.split_tensor_cp(full, packed).detach().requires_grad_()
            te_rebuilt = mcp.reconstruct_tensor_cp(te_local, packed, differentiable=True)
            (te_rebuilt.sum() if cp_rank == 0 else te_rebuilt.sum() * 0).backward()
            torch.testing.assert_close(te_local, local, atol=0, rtol=0)
            torch.testing.assert_close(te_rebuilt, rebuilt, atol=0, rtol=0)
            torch.testing.assert_close(te_local.grad, local.grad, atol=0, rtol=0)
    finally:
        Utils.destroy_model_parallel()
