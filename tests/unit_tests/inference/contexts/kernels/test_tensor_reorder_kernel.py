# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.kernels.tensor_reorder_kernel import (
    triton_move_bookkeeping,
    triton_swap_bookkeeping,
)
from megatron.core.inference.utils import tensor_swap


def _make_tensors(max_requests, max_kv_blocks, num_spec, is_hybrid, has_next_tokens=True):
    """Create all bookkeeping tensors with random data."""
    device = "cuda"

    def rand_i32(shape):
        return torch.randint(0, 1000, shape, dtype=torch.int32, device=device)

    def rand_i64(shape):
        return torch.randint(0, 50000, shape, dtype=torch.long, device=device)

    core = {
        "request_kv_length_offsets": rand_i32((max_requests,)),
        "request_in_prefill_status_tensor": rand_i32((max_requests,)),
        "request_query_lengths": rand_i32((max_requests,)),
        "request_output_lengths": rand_i32((max_requests,)),
        "request_ids": rand_i32((max_requests,)),
        "request_kv_block_counts": rand_i32((max_requests,)),
        "request_last_kv_block_id": rand_i32((max_requests,)),
        "request_last_kv_block_offset": rand_i32((max_requests,)),
        "request_to_kv_block_ids": rand_i32((max_requests, max_kv_blocks)),
    }

    metadata = {
        "temperature": torch.rand(max_requests, dtype=torch.float32, device=device),
        "top_k": rand_i32((max_requests,)),
        "top_p": torch.rand(max_requests, dtype=torch.float32, device=device),
        "termination_id": rand_i64((max_requests,)),
        "return_log_probs": torch.randint(0, 2, (max_requests,), dtype=torch.bool, device=device),
        "skip_prompt_log_probs": torch.randint(
            0, 2, (max_requests,), dtype=torch.bool, device=device
        ),
        "top_n_logprobs": rand_i32((max_requests,)),
    }

    next_tokens = rand_i64((max_requests,)) if has_next_tokens else None
    new_spec = rand_i64((num_spec, max_requests)) if num_spec > 0 else None
    mamba_idx = rand_i32((max_requests,)) if is_hybrid else None

    return core, metadata, next_tokens, new_spec, mamba_idx


def _clone_all(core, metadata, next_tokens, new_spec, mamba_idx):
    """Deep-clone all tensors."""
    c = {k: v.clone() for k, v in core.items()}
    m = {k: v.clone() for k, v in metadata.items()}
    nt = next_tokens.clone() if next_tokens is not None else None
    ns = new_spec.clone() if new_spec is not None else None
    mi = mamba_idx.clone() if mamba_idx is not None else None
    return c, m, nt, ns, mi


def _reference_move(core, metadata, next_tokens, new_spec, mamba_idx, src_idxs, dst_idxs):
    """Python reference for move."""
    for t in core.values():
        t[dst_idxs] = t[src_idxs]
    if next_tokens is not None:
        next_tokens[dst_idxs] = next_tokens[src_idxs]
    if new_spec is not None:
        new_spec[:, dst_idxs] = new_spec[:, src_idxs]
    for t in metadata.values():
        t[dst_idxs] = t[src_idxs]
    if mamba_idx is not None:
        mamba_idx[dst_idxs] = mamba_idx[src_idxs]


def _reference_swap(core, metadata, next_tokens, new_spec, mamba_idx, src_idxs, dst_idxs):
    """Python reference for swap."""
    for t in core.values():
        tensor_swap(t, src_idxs, dst_idxs)
    if next_tokens is not None:
        tensor_swap(next_tokens, src_idxs, dst_idxs)
    if new_spec is not None:
        tensor_swap(new_spec.t(), src_idxs, dst_idxs)
    for t in metadata.values():
        tensor_swap(t, src_idxs, dst_idxs)
    if mamba_idx is not None:
        tensor_swap(mamba_idx, src_idxs, dst_idxs)


def _call_triton(fn, core, metadata, next_tokens, new_spec, mamba_idx, src_idxs, dst_idxs,
                 num_spec, is_hybrid, max_kv_blocks):
    """Call the triton move or swap wrapper."""
    fn(
        src_idxs=src_idxs,
        dst_idxs=dst_idxs,
        next_tokens=next_tokens,
        new_speculative_tokens=new_spec,
        request_kv_length_offsets=core["request_kv_length_offsets"],
        request_in_prefill_status_tensor=core["request_in_prefill_status_tensor"],
        request_query_lengths=core["request_query_lengths"],
        request_output_lengths=core["request_output_lengths"],
        request_ids=core["request_ids"],
        request_kv_block_counts=core["request_kv_block_counts"],
        request_last_kv_block_id=core["request_last_kv_block_id"],
        request_last_kv_block_offset=core["request_last_kv_block_offset"],
        request_to_kv_block_ids=core["request_to_kv_block_ids"],
        request_metadata=metadata,
        mamba_state_idx=mamba_idx,
        is_hybrid_model=is_hybrid,
        num_speculative_tokens=num_spec,
        max_kv_block_count=max_kv_blocks,
    )


def _compare_all(ref_core, ref_meta, ref_nt, ref_ns, ref_mi,
                 kern_core, kern_meta, kern_nt, kern_ns, kern_mi):
    """Assert bitwise equality on all tensors."""
    for name in ref_core:
        assert torch.equal(ref_core[name], kern_core[name]), f"Mismatch in {name}"
    for name in ref_meta:
        assert torch.equal(ref_meta[name], kern_meta[name]), f"Mismatch in metadata {name}"
    if ref_nt is not None:
        assert torch.equal(ref_nt, kern_nt), "Mismatch in next_tokens"
    if ref_ns is not None:
        assert torch.equal(ref_ns, kern_ns), "Mismatch in new_speculative_tokens"
    if ref_mi is not None:
        assert torch.equal(ref_mi, kern_mi), "Mismatch in mamba_state_idx"


class TestTensorReorderKernel:

    @pytest.mark.parametrize("num_indices", [1, 3, 10])
    @pytest.mark.parametrize("num_spec", [0, 2])
    @pytest.mark.parametrize("is_hybrid", [True, False])
    @pytest.mark.parametrize("max_kv_blocks", [8, 256])
    def test_move_matches_reference(self, num_indices, num_spec, is_hybrid, max_kv_blocks):
        max_requests = 64
        core, meta, nt, ns, mi = _make_tensors(max_requests, max_kv_blocks, num_spec, is_hybrid)
        ref_core, ref_meta, ref_nt, ref_ns, ref_mi = _clone_all(core, meta, nt, ns, mi)

        # Ensure disjoint src/dst — kernel programs run in parallel so overlapping
        # indices would race. All actual call sites use disjoint indices.
        perm = torch.randperm(max_requests, device="cuda")[:num_indices * 2].to(torch.long)
        src_idxs = perm[:num_indices]
        dst_idxs = perm[num_indices:]

        _reference_move(ref_core, ref_meta, ref_nt, ref_ns, ref_mi, src_idxs, dst_idxs)
        _call_triton(triton_move_bookkeeping, core, meta, nt, ns, mi, src_idxs, dst_idxs,
                     num_spec, is_hybrid, max_kv_blocks)

        _compare_all(ref_core, ref_meta, ref_nt, ref_ns, ref_mi, core, meta, nt, ns, mi)

    @pytest.mark.parametrize("num_indices", [1, 3, 10])
    @pytest.mark.parametrize("num_spec", [0, 2])
    @pytest.mark.parametrize("is_hybrid", [True, False])
    @pytest.mark.parametrize("max_kv_blocks", [8, 256])
    def test_swap_matches_reference(self, num_indices, num_spec, is_hybrid, max_kv_blocks):
        max_requests = 64
        core, meta, nt, ns, mi = _make_tensors(max_requests, max_kv_blocks, num_spec, is_hybrid)
        ref_core, ref_meta, ref_nt, ref_ns, ref_mi = _clone_all(core, meta, nt, ns, mi)

        # Ensure disjoint src/dst for swap
        perm = torch.randperm(max_requests, device="cuda")[:num_indices * 2].to(torch.long)
        src_idxs = perm[:num_indices]
        dst_idxs = perm[num_indices:]

        _reference_swap(ref_core, ref_meta, ref_nt, ref_ns, ref_mi, src_idxs, dst_idxs)
        _call_triton(triton_swap_bookkeeping, core, meta, nt, ns, mi, src_idxs, dst_idxs,
                     num_spec, is_hybrid, max_kv_blocks)

        _compare_all(ref_core, ref_meta, ref_nt, ref_ns, ref_mi, core, meta, nt, ns, mi)

    def test_swap_no_tokens(self):
        """Chunked prefill relocation: swap with next_tokens=None."""
        max_requests = 16
        core, meta, _, _, mi = _make_tensors(max_requests, 8, 0, False, has_next_tokens=False)
        ref_core, ref_meta, _, _, ref_mi = _clone_all(core, meta, None, None, mi)

        src_idxs = torch.tensor([3], device="cuda", dtype=torch.long)
        dst_idxs = torch.tensor([7], device="cuda", dtype=torch.long)

        _reference_swap(ref_core, ref_meta, None, None, ref_mi, src_idxs, dst_idxs)
        _call_triton(triton_swap_bookkeeping, core, meta, None, None, mi, src_idxs, dst_idxs,
                     0, False, 8)

        _compare_all(ref_core, ref_meta, None, None, ref_mi, core, meta, None, None, mi)

    def test_metadata_dtypes_preserved(self):
        """Verify float32 and bool metadata survive bitwise copy."""
        max_requests = 16
        core, meta, nt, _, _ = _make_tensors(max_requests, 8, 0, False)

        # Set specific float32 values
        meta["temperature"][0] = 0.7
        meta["top_p"][0] = 0.95
        meta["return_log_probs"][0] = True

        ref_core, ref_meta, ref_nt, _, _ = _clone_all(core, meta, nt, None, None)

        src_idxs = torch.tensor([0], device="cuda", dtype=torch.long)
        dst_idxs = torch.tensor([5], device="cuda", dtype=torch.long)

        _reference_move(ref_core, ref_meta, ref_nt, None, None, src_idxs, dst_idxs)
        _call_triton(triton_move_bookkeeping, core, meta, nt, None, None, src_idxs, dst_idxs,
                     0, False, 8)

        assert meta["temperature"][5] == ref_meta["temperature"][5] == 0.7
        assert meta["top_p"][5] == ref_meta["top_p"][5] == 0.95
        assert meta["return_log_probs"][5] == ref_meta["return_log_probs"][5] == True

    def test_zero_indices(self):
        """Empty src/dst should be a no-op."""
        max_requests = 8
        core, meta, nt, _, _ = _make_tensors(max_requests, 8, 0, False)
        orig_core, orig_meta, orig_nt, _, _ = _clone_all(core, meta, nt, None, None)

        empty = torch.empty(0, device="cuda", dtype=torch.long)

        _call_triton(triton_move_bookkeeping, core, meta, nt, None, None, empty, empty,
                     0, False, 8)

        _compare_all(orig_core, orig_meta, orig_nt, None, None, core, meta, nt, None, None)

    def test_large_kv_blocks(self):
        """Exercise tiled 2D loop with many KV blocks."""
        max_requests = 8
        max_kv_blocks = 1024
        core, meta, nt, _, _ = _make_tensors(max_requests, max_kv_blocks, 0, False)
        ref_core, ref_meta, ref_nt, _, _ = _clone_all(core, meta, nt, None, None)

        src_idxs = torch.tensor([0, 2, 4], device="cuda", dtype=torch.long)
        dst_idxs = torch.tensor([1, 3, 5], device="cuda", dtype=torch.long)

        _reference_swap(ref_core, ref_meta, ref_nt, None, None, src_idxs, dst_idxs)
        _call_triton(triton_swap_bookkeeping, core, meta, nt, None, None, src_idxs, dst_idxs,
                     0, False, max_kv_blocks)

        _compare_all(ref_core, ref_meta, ref_nt, None, None, core, meta, nt, None, None)
