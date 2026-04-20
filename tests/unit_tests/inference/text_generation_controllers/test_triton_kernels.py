# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MTP Triton kernels.

Each test provides a pure-PyTorch reference implementation of the operation,
runs both the reference and the Triton kernel on the same inputs, and asserts
that the outputs match exactly.
"""

import math

import pytest
import torch

from megatron.core.inference.text_generation_controllers.triton_kernels import (
    mamba_state_selective_copy,
    prepare_next_forward_pass,
    rewind_kv_cache,
    verify_speculative_tokens,
)


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------


def rewind_kv_cache_pytorch(
    accepted_counts,
    prefill_status,
    last_kv_block_offset,
    kv_length_offsets,
    kv_block_counts,
    last_kv_block_id,
    kv_block_ids,
    num_speculative_tokens,
    block_size_tokens,
    num_active_requests=None,
):
    """Pure-PyTorch reference for the KV-cache rewind operation.

    Mirrors the original ``TextGenerationController._rewind_kv_cache`` logic
    (KV-cache portion only, no Mamba state updates).  Mutates the input tensors
    in-place, just like the Triton kernel.

    Returns (blocks_to_release, remove_mask).
    """
    N = accepted_counts.shape[0]
    if num_active_requests is None:
        num_active_requests = N

    blocks_to_release = torch.empty_like(last_kv_block_id)
    remove_mask = torch.empty(N, device=accepted_counts.device, dtype=torch.bool)

    for i in range(N):
        if i >= num_active_requests:
            blocks_to_release[i] = 0
            remove_mask[i] = False
            continue

        accepted = accepted_counts[i].item()
        prefill = prefill_status[i].item()
        last_offset = last_kv_block_offset[i].item()
        kv_length = kv_length_offsets[i].item()
        block_count = kv_block_counts[i].item()
        last_block = last_kv_block_id[i].item()

        num_to_rewind = 0 if prefill == 1 else num_speculative_tokens - accepted
        diff = last_offset - num_to_rewind
        remove = diff < 0

        new_offset = diff % block_size_tokens
        last_kv_block_offset[i] = new_offset
        kv_length_offsets[i] = kv_length - num_to_rewind

        blocks_to_release[i] = last_block

        new_block_count = block_count - 1 if remove else block_count
        kv_block_counts[i] = new_block_count

        prev_idx = max(new_block_count - 1, 0)
        prev_block_id = kv_block_ids[i, prev_idx].item()

        last_kv_block_id[i] = prev_block_id if remove else last_block

        scatter_idx = min(new_block_count, kv_block_ids.shape[1] - 1)
        if remove:
            kv_block_ids[i, scatter_idx] = -1

        remove_mask[i] = remove

    return blocks_to_release, remove_mask


def verify_speculative_tokens_pytorch(
    input_tokens, output_tokens, num_decode_requests, num_prefill_requests, num_speculative_tokens
):
    """Pure-PyTorch reference for speculative token verification.

    Mirrors the original ``TextGenerationController._verify_speculative_tokens``
    logic.
    """
    if input_tokens.ndim == 2:
        input_tokens = input_tokens.squeeze(0)

    stride = num_speculative_tokens + 1
    active_request_count = num_decode_requests + num_prefill_requests
    decode_len = num_decode_requests * stride

    accepted_tokens_mask = torch.zeros_like(input_tokens, dtype=torch.bool)

    decode_mask_2d = None
    if num_decode_requests > 0:
        decode_inputs = input_tokens[:decode_len].reshape(num_decode_requests, stride)
        decode_outputs = output_tokens[:decode_len].reshape(num_decode_requests, stride)

        decode_outputs_shifted = decode_outputs.roll(1, dims=1)
        decode_mask_2d = decode_inputs == decode_outputs_shifted
        decode_mask_2d[:, 0] = True
        decode_mask_2d = decode_mask_2d.cummin(dim=1).values
        accepted_tokens_mask[:decode_len] = decode_mask_2d.flatten()

    if num_prefill_requests > 0:
        accepted_tokens_mask[decode_len:] = True

    last_one_indices = torch.full(
        (active_request_count,), -1, device=input_tokens.device, dtype=torch.long
    )

    if num_decode_requests > 0:
        local_last_indices = decode_mask_2d.sum(dim=1) - 1
        row_offsets = torch.arange(num_decode_requests, device=input_tokens.device) * stride
        last_one_indices[:num_decode_requests] = row_offsets + local_last_indices

    if num_prefill_requests > 0:
        prefill_valid = torch.nonzero(accepted_tokens_mask[decode_len:]).squeeze(-1) + decode_len
        last_one_indices[num_decode_requests:] = prefill_valid

    return last_one_indices, accepted_tokens_mask, input_tokens


def prepare_next_forward_pass_pytorch(
    num_decode_requests,
    output_tokens,
    required_logit_indices,
    last_one_indices,
    accepted_tokens_mask,
    input_tokens,
    sampled_tokens_buf,
    last_accepted_seq_buf,
    accepted_tokens_per_request,
    accepted_token_counts,
    num_speculative_tokens,
):
    """Pure-PyTorch reference for preparing the next forward pass.

    Mirrors the original ``_dynamic_step_sample_logits_and_verify_tokens``
    post-verification logic.
    """
    active_request_count = last_one_indices.shape[0]
    stride = num_speculative_tokens + 1

    for pid in range(active_request_count):
        idx = last_one_indices[pid].item()
        sampled_tokens_buf[pid] = output_tokens[idx]
        last_accepted_seq_buf[pid] = required_logit_indices[idx]

        if pid < num_decode_requests:
            base = pid * stride
            for s in range(num_speculative_tokens):
                pos = base + 1 + s
                if accepted_tokens_mask[pos]:
                    accepted_tokens_per_request[pid, s] = input_tokens[pos]
                else:
                    accepted_tokens_per_request[pid, s] = -1

            count = 0
            for s in range(num_speculative_tokens):
                if accepted_tokens_per_request[pid, s].item() != -1:
                    count += 1
            accepted_token_counts[pid] = count


def mamba_state_selective_copy_pytorch(
    intermediate_states, current_states, prefill_status, state_idx, accepted_counts, num_layers
):
    """Pure-PyTorch reference for Mamba state selective copy.

    For each decode request, copies
    ``intermediate[layer, slot, accepted_count, ...]`` →
    ``current[layer, slot, ...]`` for every Mamba layer.
    """
    N = prefill_status.shape[0]
    for i in range(N):
        if prefill_status[i].item() == 1:
            continue
        slot = state_idx[i].item()
        accepted = accepted_counts[i].item()
        for layer in range(num_layers):
            current_states[layer, slot] = intermediate_states[layer, slot, accepted]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda"


def _clone_tensors(*tensors):
    """Return a tuple of cloned tensors (for running reference vs kernel on the same data)."""
    return tuple(t.clone() for t in tensors)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRewindKvCache:
    """Tests for the rewind_kv_cache Triton kernel."""

    @pytest.mark.parametrize("num_requests", [1, 4, 16])
    @pytest.mark.parametrize("num_speculative_tokens", [1, 2, 4])
    @pytest.mark.parametrize("block_size_tokens", [8, 16, 64])
    def test_basic(self, num_requests, num_speculative_tokens, block_size_tokens):
        N = num_requests
        max_blocks = 8

        accepted_counts = torch.randint(0, num_speculative_tokens + 1, (N,), device=DEVICE)
        prefill_status = torch.zeros(N, dtype=torch.int32, device=DEVICE)

        last_kv_block_offset = torch.randint(0, block_size_tokens, (N,), device=DEVICE)
        kv_length_offsets = torch.randint(block_size_tokens, block_size_tokens * 4, (N,), device=DEVICE)
        kv_block_counts = torch.randint(2, max_blocks, (N,), device=DEVICE)
        last_kv_block_id = torch.randint(0, 100, (N,), device=DEVICE)
        kv_block_ids = torch.randint(0, 100, (N, max_blocks), device=DEVICE)

        ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )
        tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )

        ref_release, ref_mask = rewind_kv_cache_pytorch(
            accepted_counts.clone(), prefill_status.clone(),
            ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids,
            num_speculative_tokens, block_size_tokens,
        )

        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(), prefill_status.clone(),
            tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids,
            num_speculative_tokens, block_size_tokens,
        )

        torch.testing.assert_close(tri_offset, ref_offset)
        torch.testing.assert_close(tri_kv_len, ref_kv_len)
        torch.testing.assert_close(tri_block_counts, ref_block_counts)
        torch.testing.assert_close(tri_last_block, ref_last_block)
        torch.testing.assert_close(tri_block_ids, ref_block_ids)
        torch.testing.assert_close(tri_release, ref_release)
        torch.testing.assert_close(tri_mask, ref_mask)

    def test_prefill_requests_skip_rewind(self):
        N = 4
        num_spec = 3
        block_size = 16

        accepted_counts = torch.tensor([1, 0, 2, 0], device=DEVICE)
        prefill_status = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
        last_kv_block_offset = torch.tensor([5, 10, 2, 7], device=DEVICE)
        kv_length_offsets = torch.tensor([100, 200, 300, 400], device=DEVICE)
        kv_block_counts = torch.tensor([3, 4, 2, 5], device=DEVICE)
        last_kv_block_id = torch.tensor([10, 20, 30, 40], device=DEVICE)
        kv_block_ids = torch.randint(0, 50, (N, 8), device=DEVICE)

        ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )
        tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )

        ref_release, ref_mask = rewind_kv_cache_pytorch(
            accepted_counts.clone(), prefill_status.clone(),
            ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids,
            num_spec, block_size,
        )
        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(), prefill_status.clone(),
            tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids,
            num_spec, block_size,
        )

        # Prefill requests (indices 1, 3) should be unchanged.
        for idx in [1, 3]:
            assert ref_kv_len[idx] == kv_length_offsets[idx]
            assert ref_offset[idx] == last_kv_block_offset[idx]

        torch.testing.assert_close(tri_offset, ref_offset)
        torch.testing.assert_close(tri_kv_len, ref_kv_len)
        torch.testing.assert_close(tri_block_counts, ref_block_counts)
        torch.testing.assert_close(tri_last_block, ref_last_block)
        torch.testing.assert_close(tri_block_ids, ref_block_ids)
        torch.testing.assert_close(tri_mask, ref_mask)

    def test_block_boundary_crossing(self):
        """When offset - rewind < 0, a block boundary is crossed."""
        N = 2
        num_spec = 3
        block_size = 16

        accepted_counts = torch.tensor([0, 0], device=DEVICE)
        prefill_status = torch.zeros(N, dtype=torch.int32, device=DEVICE)
        last_kv_block_offset = torch.tensor([1, 10], device=DEVICE)
        kv_length_offsets = torch.tensor([100, 200], device=DEVICE)
        kv_block_counts = torch.tensor([3, 4], device=DEVICE)
        last_kv_block_id = torch.tensor([50, 60], device=DEVICE)
        kv_block_ids = torch.tensor(
            [[10, 20, 50, -1, -1, -1, -1, -1], [15, 25, 35, 60, -1, -1, -1, -1]], device=DEVICE
        )

        ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )
        tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )

        rewind_kv_cache_pytorch(
            accepted_counts.clone(), prefill_status.clone(),
            ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids,
            num_spec, block_size,
        )
        rewind_kv_cache(
            accepted_counts.clone(), prefill_status.clone(),
            tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids,
            num_spec, block_size,
        )

        # Request 0: offset 1 - 3 = -2 → crosses boundary.
        assert ref_block_counts[0] == 2
        assert tri_block_counts[0] == 2
        assert ref_last_block[0] == 20  # previous block
        assert tri_last_block[0] == 20

        # Request 1: offset 10 - 3 = 7 → no crossing.
        assert ref_block_counts[1] == 4
        assert tri_block_counts[1] == 4

        torch.testing.assert_close(tri_offset, ref_offset)
        torch.testing.assert_close(tri_kv_len, ref_kv_len)
        torch.testing.assert_close(tri_block_ids, ref_block_ids)

    def test_padding_programs(self):
        """Padding slots (pid >= num_active_requests) must produce safe no-ops."""
        N = 8  # grid size
        active = 3
        num_spec = 2
        block_size = 16

        accepted_counts = torch.randint(0, num_spec + 1, (N,), device=DEVICE)
        prefill_status = torch.zeros(N, dtype=torch.int32, device=DEVICE)
        last_kv_block_offset = torch.randint(0, block_size, (N,), device=DEVICE)
        kv_length_offsets = torch.randint(block_size, block_size * 4, (N,), device=DEVICE)
        kv_block_counts = torch.randint(2, 6, (N,), device=DEVICE)
        last_kv_block_id = torch.randint(0, 100, (N,), device=DEVICE)
        kv_block_ids = torch.randint(0, 100, (N, 8), device=DEVICE)

        ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )
        tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )

        rewind_kv_cache_pytorch(
            accepted_counts.clone(), prefill_status.clone(),
            ref_offset, ref_kv_len, ref_block_counts, ref_last_block, ref_block_ids,
            num_spec, block_size, num_active_requests=active,
        )
        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(), prefill_status.clone(),
            tri_offset, tri_kv_len, tri_block_counts, tri_last_block, tri_block_ids,
            num_spec, block_size, num_active_requests=active,
        )

        # Active slots should match.
        torch.testing.assert_close(tri_offset[:active], ref_offset[:active])
        torch.testing.assert_close(tri_kv_len[:active], ref_kv_len[:active])
        torch.testing.assert_close(tri_block_counts[:active], ref_block_counts[:active])
        torch.testing.assert_close(tri_last_block[:active], ref_last_block[:active])
        torch.testing.assert_close(tri_block_ids[:active], ref_block_ids[:active])

        # Padding slots: release=0, mask=False.
        assert (tri_release[active:] == 0).all()
        assert (~tri_mask[active:]).all()

    def test_empty(self):
        N = 0
        blocks_to_release, remove_mask = rewind_kv_cache(
            torch.empty(0, device=DEVICE, dtype=torch.int64),
            torch.empty(0, device=DEVICE, dtype=torch.int32),
            torch.empty(0, device=DEVICE, dtype=torch.int64),
            torch.empty(0, device=DEVICE, dtype=torch.int64),
            torch.empty(0, device=DEVICE, dtype=torch.int64),
            torch.empty(0, device=DEVICE, dtype=torch.int64),
            torch.empty(0, 8, device=DEVICE, dtype=torch.int64),
            num_speculative_tokens=2,
            block_size_tokens=16,
        )
        assert blocks_to_release.shape[0] == 0
        assert remove_mask.shape[0] == 0


class TestVerifySpeculativeTokens:
    """Tests for the verify_speculative_tokens Triton kernel."""

    def _make_scenario(self, num_decode, num_prefill, num_spec, *, match_pattern=None):
        """Build input/output token tensors for testing.

        Args:
            match_pattern: list of ints per decode request indicating how many
                speculative tokens should match (0 means only base accepted).
                If None, generates random matches.
        """
        stride = num_spec + 1
        decode_len = num_decode * stride
        total_len = decode_len + num_prefill

        input_tokens = torch.randint(1, 1000, (total_len,), device=DEVICE)
        output_tokens = torch.randint(1, 1000, (total_len,), device=DEVICE)

        if match_pattern is not None:
            assert len(match_pattern) == num_decode
            for req_idx, num_match in enumerate(match_pattern):
                base = req_idx * stride
                for s in range(num_match):
                    output_tokens[base + s] = input_tokens[base + s + 1]

        return input_tokens, output_tokens

    @pytest.mark.parametrize("num_decode,num_prefill,num_spec", [
        (1, 0, 2),
        (3, 0, 2),
        (3, 2, 2),
        (0, 3, 2),
        (5, 3, 4),
    ])
    def test_basic(self, num_decode, num_prefill, num_spec):
        input_tokens, output_tokens = self._make_scenario(num_decode, num_prefill, num_spec)

        ref_last, ref_mask, ref_input = verify_speculative_tokens_pytorch(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )
        tri_last, tri_mask, tri_input = verify_speculative_tokens(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )

        torch.testing.assert_close(tri_mask, ref_mask)
        torch.testing.assert_close(tri_last, ref_last)

    def test_all_accepted(self):
        """All speculative tokens match → all accepted."""
        num_decode, num_prefill, num_spec = 3, 0, 3
        input_tokens, output_tokens = self._make_scenario(
            num_decode, num_prefill, num_spec, match_pattern=[3, 3, 3]
        )

        ref_last, ref_mask, _ = verify_speculative_tokens_pytorch(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )
        tri_last, tri_mask, _ = verify_speculative_tokens(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )

        assert ref_mask.all()
        torch.testing.assert_close(tri_mask, ref_mask)
        torch.testing.assert_close(tri_last, ref_last)

    def test_none_accepted(self):
        """No speculative tokens match → only base tokens accepted."""
        num_decode, num_prefill, num_spec = 3, 0, 3
        input_tokens, output_tokens = self._make_scenario(
            num_decode, num_prefill, num_spec, match_pattern=[0, 0, 0]
        )

        ref_last, ref_mask, _ = verify_speculative_tokens_pytorch(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )
        tri_last, tri_mask, _ = verify_speculative_tokens(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )

        stride = num_spec + 1
        for req in range(num_decode):
            base = req * stride
            assert ref_mask[base].item() is True
            assert not ref_mask[base + 1 : base + stride].any()

        torch.testing.assert_close(tri_mask, ref_mask)
        torch.testing.assert_close(tri_last, ref_last)

    def test_mixed_match_pattern(self):
        """Different acceptance counts per request."""
        num_decode, num_prefill, num_spec = 3, 1, 3
        input_tokens, output_tokens = self._make_scenario(
            num_decode, num_prefill, num_spec, match_pattern=[1, 3, 0]
        )

        ref_last, ref_mask, _ = verify_speculative_tokens_pytorch(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )
        tri_last, tri_mask, _ = verify_speculative_tokens(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )

        torch.testing.assert_close(tri_mask, ref_mask)
        torch.testing.assert_close(tri_last, ref_last)

    def test_2d_input(self):
        """Input tokens with shape [1, total_len] should be squeezed."""
        num_decode, num_prefill, num_spec = 2, 1, 2
        input_tokens, output_tokens = self._make_scenario(num_decode, num_prefill, num_spec)
        input_2d = input_tokens.unsqueeze(0)

        ref_last, ref_mask, _ = verify_speculative_tokens_pytorch(
            input_2d.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )
        tri_last, tri_mask, _ = verify_speculative_tokens(
            input_2d.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )

        torch.testing.assert_close(tri_mask, ref_mask)
        torch.testing.assert_close(tri_last, ref_last)


class TestPrepareNextForwardPass:
    """Tests for the prepare_next_forward_pass Triton kernel."""

    def _setup(self, num_decode, num_prefill, num_spec):
        stride = num_spec + 1
        active = num_decode + num_prefill
        decode_len = num_decode * stride
        total_len = decode_len + num_prefill

        output_tokens = torch.randint(1, 1000, (total_len,), device=DEVICE, dtype=torch.int64)
        required_logit_indices = torch.arange(total_len, device=DEVICE, dtype=torch.int64)
        input_tokens = torch.randint(1, 1000, (total_len,), device=DEVICE, dtype=torch.int64)

        accepted_mask = torch.zeros(total_len, device=DEVICE, dtype=torch.bool)
        last_one_indices = torch.empty(active, device=DEVICE, dtype=torch.int64)

        for req in range(num_decode):
            base = req * stride
            num_match = torch.randint(0, num_spec + 1, (1,)).item()
            for j in range(stride):
                if j <= num_match:
                    accepted_mask[base + j] = True
            last_one_indices[req] = base + num_match

        for p in range(num_prefill):
            idx = decode_len + p
            accepted_mask[idx] = True
            last_one_indices[num_decode + p] = idx

        return output_tokens, required_logit_indices, input_tokens, accepted_mask, last_one_indices

    @pytest.mark.parametrize("num_decode,num_prefill,num_spec", [
        (1, 0, 2),
        (3, 0, 2),
        (3, 2, 2),
        (0, 3, 2),
        (5, 3, 4),
    ])
    def test_basic(self, num_decode, num_prefill, num_spec):
        (
            output_tokens, required_logit_indices, input_tokens,
            accepted_mask, last_one_indices,
        ) = self._setup(num_decode, num_prefill, num_spec)

        active = num_decode + num_prefill

        ref_sampled = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        ref_last_seq = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        ref_accepted = torch.full((num_decode, num_spec), -1, device=DEVICE, dtype=torch.int64)
        ref_counts = torch.zeros(num_decode, device=DEVICE, dtype=torch.int64)

        tri_sampled = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        tri_last_seq = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        tri_accepted = torch.full((max(num_decode, 1), num_spec), -1, device=DEVICE, dtype=torch.int64)
        tri_counts = torch.zeros(max(num_decode, 1), device=DEVICE, dtype=torch.int64)

        prepare_next_forward_pass_pytorch(
            num_decode, output_tokens, required_logit_indices,
            last_one_indices, accepted_mask, input_tokens,
            ref_sampled, ref_last_seq, ref_accepted, ref_counts, num_spec,
        )

        prepare_next_forward_pass(
            num_decode, output_tokens, required_logit_indices,
            last_one_indices, accepted_mask, input_tokens,
            tri_sampled, tri_last_seq, tri_accepted, tri_counts, num_spec,
        )

        torch.testing.assert_close(tri_sampled, ref_sampled)
        torch.testing.assert_close(tri_last_seq, ref_last_seq)
        if num_decode > 0:
            torch.testing.assert_close(tri_accepted[:num_decode], ref_accepted[:num_decode])
            torch.testing.assert_close(tri_counts[:num_decode], ref_counts[:num_decode])

    def test_empty(self):
        """Zero active requests should be a no-op."""
        last_one_indices = torch.empty(0, device=DEVICE, dtype=torch.int64)
        prepare_next_forward_pass(
            num_decode_requests=0,
            output_tokens=torch.empty(0, device=DEVICE, dtype=torch.int64),
            required_logit_indices=torch.empty(0, device=DEVICE, dtype=torch.int64),
            last_one_indices=last_one_indices,
            accepted_tokens_mask=torch.empty(0, device=DEVICE, dtype=torch.bool),
            input_tokens=torch.empty(0, device=DEVICE, dtype=torch.int64),
            sampled_tokens_buf=torch.empty(0, device=DEVICE, dtype=torch.int64),
            last_accepted_seq_buf=torch.empty(0, device=DEVICE, dtype=torch.int64),
            accepted_tokens_per_request=torch.empty(0, 2, device=DEVICE, dtype=torch.int64),
            accepted_token_counts=torch.empty(0, device=DEVICE, dtype=torch.int64),
            num_speculative_tokens=2,
        )


class TestMambaStateSelectiveCopy:
    """Tests for the mamba_state_selective_copy Triton kernel."""

    @pytest.mark.parametrize("num_requests", [1, 4, 8])
    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_basic(self, num_requests, num_layers):
        N = num_requests
        M = N  # 1:1 request-to-slot mapping for simplicity
        S = 4  # speculative tokens + 1
        state_shape = (16, 32)  # arbitrary state dimensions

        intermediate = torch.randn(num_layers, M, S, *state_shape, device=DEVICE)
        current_ref = torch.randn(num_layers, M, *state_shape, device=DEVICE)
        current_tri = current_ref.clone()

        prefill_status = torch.zeros(N, dtype=torch.int32, device=DEVICE)
        state_idx = torch.arange(N, device=DEVICE, dtype=torch.int64)
        accepted_counts = torch.randint(0, S, (N,), device=DEVICE, dtype=torch.int64)

        mamba_state_selective_copy_pytorch(
            intermediate, current_ref, prefill_status, state_idx, accepted_counts, num_layers
        )
        mamba_state_selective_copy(
            intermediate, current_tri, prefill_status, state_idx, accepted_counts, num_layers
        )

        torch.testing.assert_close(current_tri, current_ref)

    def test_prefill_skipped(self):
        N = 4
        num_layers = 2
        M = N
        S = 3
        state_shape = (8,)

        intermediate = torch.randn(num_layers, M, S, *state_shape, device=DEVICE)
        current_ref = torch.randn(num_layers, M, *state_shape, device=DEVICE)
        current_tri = current_ref.clone()
        current_orig = current_ref.clone()

        prefill_status = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
        state_idx = torch.arange(N, device=DEVICE, dtype=torch.int64)
        accepted_counts = torch.tensor([1, 0, 2, 0], device=DEVICE, dtype=torch.int64)

        mamba_state_selective_copy_pytorch(
            intermediate, current_ref, prefill_status, state_idx, accepted_counts, num_layers
        )
        mamba_state_selective_copy(
            intermediate, current_tri, prefill_status, state_idx, accepted_counts, num_layers
        )

        # Prefill slots should be unchanged from original.
        for layer in range(num_layers):
            for slot in [1, 3]:
                torch.testing.assert_close(current_ref[layer, slot], current_orig[layer, slot])
                torch.testing.assert_close(current_tri[layer, slot], current_orig[layer, slot])

        torch.testing.assert_close(current_tri, current_ref)

    def test_noncontiguous_state_idx(self):
        """state_idx does not have to be a simple arange."""
        N = 3
        num_layers = 2
        M = 6  # more slots than requests
        S = 3
        state_shape = (8, 4)

        intermediate = torch.randn(num_layers, M, S, *state_shape, device=DEVICE)
        current_ref = torch.randn(num_layers, M, *state_shape, device=DEVICE)
        current_tri = current_ref.clone()

        prefill_status = torch.zeros(N, dtype=torch.int32, device=DEVICE)
        state_idx = torch.tensor([1, 4, 0], device=DEVICE, dtype=torch.int64)
        accepted_counts = torch.tensor([2, 0, 1], device=DEVICE, dtype=torch.int64)

        mamba_state_selective_copy_pytorch(
            intermediate, current_ref, prefill_status, state_idx, accepted_counts, num_layers
        )
        mamba_state_selective_copy(
            intermediate, current_tri, prefill_status, state_idx, accepted_counts, num_layers
        )

        torch.testing.assert_close(current_tri, current_ref)

    def test_empty(self):
        """Zero requests should be a no-op."""
        num_layers = 2
        state_shape = (8,)
        intermediate = torch.randn(num_layers, 4, 3, *state_shape, device=DEVICE)
        current = torch.randn(num_layers, 4, *state_shape, device=DEVICE)
        current_before = current.clone()

        mamba_state_selective_copy(
            intermediate, current,
            torch.empty(0, dtype=torch.int32, device=DEVICE),
            torch.empty(0, dtype=torch.int64, device=DEVICE),
            torch.empty(0, dtype=torch.int64, device=DEVICE),
            num_layers,
        )

        torch.testing.assert_close(current, current_before)


class TestStressRandom:
    """Randomized stress tests running all four kernels with varied inputs."""

    @pytest.mark.parametrize("trial", range(5))
    def test_rewind_random(self, trial):
        torch.manual_seed(42 + trial)
        N = torch.randint(1, 32, (1,)).item()
        num_spec = torch.randint(1, 6, (1,)).item()
        block_size = 2 ** torch.randint(3, 7, (1,)).item()
        max_blocks = torch.randint(4, 16, (1,)).item()

        accepted_counts = torch.randint(0, num_spec + 1, (N,), device=DEVICE)
        prefill_status = (torch.rand(N, device=DEVICE) > 0.7).to(torch.int32)
        last_kv_block_offset = torch.randint(0, block_size, (N,), device=DEVICE)
        kv_length_offsets = torch.randint(block_size, block_size * 4, (N,), device=DEVICE)
        kv_block_counts = torch.randint(2, max_blocks, (N,), device=DEVICE)
        last_kv_block_id = torch.randint(0, 200, (N,), device=DEVICE)
        kv_block_ids = torch.randint(0, 200, (N, max_blocks), device=DEVICE)

        ref_args = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )
        tri_args = _clone_tensors(
            last_kv_block_offset, kv_length_offsets, kv_block_counts, last_kv_block_id, kv_block_ids
        )

        ref_release, ref_mask = rewind_kv_cache_pytorch(
            accepted_counts.clone(), prefill_status.clone(), *ref_args,
            num_spec, block_size,
        )
        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(), prefill_status.clone(), *tri_args,
            num_spec, block_size,
        )

        for r, t in zip(ref_args, tri_args):
            torch.testing.assert_close(t, r)
        torch.testing.assert_close(tri_release, ref_release)
        torch.testing.assert_close(tri_mask, ref_mask)

    @pytest.mark.parametrize("trial", range(5))
    def test_verify_random(self, trial):
        torch.manual_seed(42 + trial)
        num_decode = torch.randint(0, 16, (1,)).item()
        num_prefill = torch.randint(0, 8, (1,)).item()
        if num_decode == 0 and num_prefill == 0:
            num_prefill = 1
        num_spec = torch.randint(1, 6, (1,)).item()

        stride = num_spec + 1
        total_len = num_decode * stride + num_prefill

        input_tokens = torch.randint(1, 500, (total_len,), device=DEVICE)
        output_tokens = torch.randint(1, 500, (total_len,), device=DEVICE)

        # Randomly make some speculative tokens match.
        for req in range(num_decode):
            base = req * stride
            num_match = torch.randint(0, num_spec + 1, (1,)).item()
            for s in range(num_match):
                output_tokens[base + s] = input_tokens[base + s + 1]

        ref_last, ref_mask, _ = verify_speculative_tokens_pytorch(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )
        tri_last, tri_mask, _ = verify_speculative_tokens(
            input_tokens.clone(), output_tokens.clone(), num_decode, num_prefill, num_spec
        )

        torch.testing.assert_close(tri_mask, ref_mask)
        torch.testing.assert_close(tri_last, ref_last)
