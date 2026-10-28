# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MTP Triton kernels.

Each test runs both the pure-PyTorch reference (from mtp_utils_pytorch) and
the Triton kernel (from mtp_utils_triton) on the same inputs, and asserts
that the outputs match exactly.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.mtp_metadata import MTPMetadata
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    mamba_state_selective_copy as mamba_state_selective_copy_pytorch,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    prepare_next_forward_pass as prepare_next_forward_pass_pytorch,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    rewind_kv_cache as rewind_kv_cache_pytorch,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    verify_speculative_tokens as verify_speculative_tokens_pytorch,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    mamba_state_selective_copy,
    mtp_commit_step_metadata,
    mtp_draft_step_metadata,
    prepare_next_forward_pass,
    rewind_kv_cache,
    verify_speculative_tokens,
)

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
        kv_length_offsets = torch.randint(
            block_size_tokens, block_size_tokens * 4, (N,), device=DEVICE
        )
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
            accepted_counts.clone(),
            prefill_status.clone(),
            ref_offset,
            ref_kv_len,
            ref_block_counts,
            ref_last_block,
            ref_block_ids,
            num_speculative_tokens,
            block_size_tokens,
        )

        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(),
            prefill_status.clone(),
            tri_offset,
            tri_kv_len,
            tri_block_counts,
            tri_last_block,
            tri_block_ids,
            num_speculative_tokens,
            block_size_tokens,
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
            accepted_counts.clone(),
            prefill_status.clone(),
            ref_offset,
            ref_kv_len,
            ref_block_counts,
            ref_last_block,
            ref_block_ids,
            num_spec,
            block_size,
        )
        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(),
            prefill_status.clone(),
            tri_offset,
            tri_kv_len,
            tri_block_counts,
            tri_last_block,
            tri_block_ids,
            num_spec,
            block_size,
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
            accepted_counts.clone(),
            prefill_status.clone(),
            ref_offset,
            ref_kv_len,
            ref_block_counts,
            ref_last_block,
            ref_block_ids,
            num_spec,
            block_size,
        )
        rewind_kv_cache(
            accepted_counts.clone(),
            prefill_status.clone(),
            tri_offset,
            tri_kv_len,
            tri_block_counts,
            tri_last_block,
            tri_block_ids,
            num_spec,
            block_size,
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
            accepted_counts.clone(),
            prefill_status.clone(),
            ref_offset,
            ref_kv_len,
            ref_block_counts,
            ref_last_block,
            ref_block_ids,
            num_spec,
            block_size,
            num_active_requests=active,
        )
        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(),
            prefill_status.clone(),
            tri_offset,
            tri_kv_len,
            tri_block_counts,
            tri_last_block,
            tri_block_ids,
            num_spec,
            block_size,
            num_active_requests=active,
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

    @pytest.mark.parametrize(
        "num_decode,num_prefill,num_spec", [(1, 0, 2), (3, 0, 2), (3, 2, 2), (0, 3, 2), (5, 3, 4)]
    )
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

    @pytest.mark.parametrize(
        "num_decode,num_prefill,num_spec", [(1, 0, 2), (3, 0, 2), (3, 2, 2), (0, 3, 2), (5, 3, 4)]
    )
    def test_basic(self, num_decode, num_prefill, num_spec):
        (output_tokens, required_logit_indices, input_tokens, accepted_mask, last_one_indices) = (
            self._setup(num_decode, num_prefill, num_spec)
        )

        active = num_decode + num_prefill

        ref_sampled = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        ref_last_seq = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        ref_accepted = torch.full((num_decode, num_spec), -1, device=DEVICE, dtype=torch.int64)
        ref_counts = torch.zeros(num_decode, device=DEVICE, dtype=torch.int64)

        tri_sampled = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        tri_last_seq = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        tri_accepted = torch.full(
            (max(num_decode, 1), num_spec), -1, device=DEVICE, dtype=torch.int64
        )
        tri_counts = torch.zeros(max(num_decode, 1), device=DEVICE, dtype=torch.int64)

        prepare_next_forward_pass_pytorch(
            num_decode,
            output_tokens,
            required_logit_indices,
            last_one_indices,
            accepted_mask,
            input_tokens,
            ref_sampled,
            ref_last_seq,
            ref_accepted,
            ref_counts,
            num_spec,
        )

        prepare_next_forward_pass(
            num_decode,
            output_tokens,
            required_logit_indices,
            last_one_indices,
            accepted_mask,
            input_tokens,
            tri_sampled,
            tri_last_seq,
            tri_accepted,
            tri_counts,
            num_spec,
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
            intermediate,
            current,
            torch.empty(0, dtype=torch.int32, device=DEVICE),
            torch.empty(0, dtype=torch.int64, device=DEVICE),
            torch.empty(0, dtype=torch.int64, device=DEVICE),
            num_layers,
        )

        torch.testing.assert_close(current, current_before)


class TestMtpDraftStepMetadata:
    """Tests for the per-depth MTP draft metadata Triton kernel.

    The reference is the eager path the kernel replaces: `stage_decode_lengths` +
    `write_token_maps` + `write_mha_metadata`. Both write into sentinel-filled buffers, so a
    write the reference does not make (or a missed one) shows up as a mismatch.
    """

    MAX_REQUESTS = 64
    DUMMY_BLOCK = 999
    SENTINEL = -7

    def _make_metadata(self, block_size_tokens, num_cols):
        meta = MTPMetadata(
            enabled=True,
            max_requests=self.MAX_REQUESTS,
            max_kv_block_count=num_cols,
            block_size_tokens=block_size_tokens,
            dummy_block_idx=self.DUMMY_BLOCK,
            block_table_dtype=torch.int32,
            hidden_size=8,
            hidden_dtype=torch.bfloat16,
        )
        meta.allocate(device=DEVICE)
        return meta

    def _make_gpu_view(self, num_cols):
        r, s = self.MAX_REQUESTS, self.SENTINEL

        def full(n, dtype):
            return torch.full((n,), s, dtype=dtype, device=DEVICE)

        return SimpleNamespace(
            token_to_block_idx=full(r, torch.int64),
            token_to_local_position_within_kv_block=full(r, torch.int32),
            token_to_request_idx=full(r, torch.int32),
            token_to_position_in_request=full(r, torch.int32),
            token_to_pos_ids=full(r, torch.int64),
            mha_query_lengths=full(r, torch.int32),
            mha_cu_query_seq_lengths=full(r + 1, torch.int32),
            mha_kv_seq_lengths=full(r, torch.int32),
            mha_cu_kv_seq_lengths=full(r + 1, torch.int32),
            mha_block_table=torch.full((r, num_cols), s, dtype=torch.int32, device=DEVICE),
        )

    @staticmethod
    def _reference_step(meta, gpu_view):
        n = meta.active_request_count
        query_lengths, kv_lengths = meta.stage_decode_lengths()
        meta.write_token_maps(
            gpu_view=gpu_view,
            rows=meta.row_ids[:n],
            positions=meta.active_offsets,
            block_table=meta.active_block_table,
            padded_token_count=meta.padded_count,
        )
        meta.write_mha_metadata(
            gpu_view=gpu_view,
            query_lengths=query_lengths,
            kv_lengths=kv_lengths,
            block_table=meta.active_block_table,
            padded_request_count=meta.padded_count,
        )

    @staticmethod
    def _kernel_step(meta, gpu_view):
        mtp_draft_step_metadata(
            gpu_view=gpu_view,
            offsets=meta.offsets,
            block_table=meta.block_table,
            staged_query_lengths=meta.query_lengths,
            staged_kv_lengths=meta.kv_lengths,
            active_request_count=meta.active_request_count,
            padded_count=meta.padded_count,
            block_size_tokens=meta.block_size_tokens,
            dummy_block_idx=meta.dummy_block_idx,
        )

    def _assert_equal(self, ref_meta, ref_view, tri_meta, tri_view):
        for name, ref in vars(ref_view).items():
            assert torch.equal(getattr(tri_view, name), ref), name
        n = ref_meta.active_request_count
        assert torch.equal(tri_meta.query_lengths[:n], ref_meta.query_lengths[:n])
        assert torch.equal(tri_meta.kv_lengths[:n], ref_meta.kv_lengths[:n])

    def _run_draft_loop(
        self,
        active_request_count,
        padded_count,
        block_size_tokens,
        num_cols,
        start_positions,
        num_depths,
    ):
        block_table_src = torch.randint(
            0, 500, (self.MAX_REQUESTS, num_cols), dtype=torch.int32, device=DEVICE
        )
        ref_meta = self._make_metadata(block_size_tokens, num_cols)
        tri_meta = self._make_metadata(block_size_tokens, num_cols)
        ref_view = self._make_gpu_view(num_cols)
        tri_view = self._make_gpu_view(num_cols)
        for meta in (ref_meta, tri_meta):
            meta.begin_decode(
                active_request_count=active_request_count,
                padded_count=padded_count,
                start_positions=start_positions,
                block_table_src=block_table_src,
                graphed=True,
            )
        for _ in range(num_depths):
            self._reference_step(ref_meta, ref_view)
            self._kernel_step(tri_meta, tri_view)
            self._assert_equal(ref_meta, ref_view, tri_meta, tri_view)
            ref_meta.advance_decode_step()
            tri_meta.advance_decode_step()

    @pytest.mark.parametrize(
        "active_request_count, padded_count", [(1, 1), (5, 8), (8, 8), (13, 16), (64, 64)]
    )
    @pytest.mark.parametrize("block_size_tokens", [16, 64])
    @pytest.mark.parametrize("num_cols", [4, 33, 130])
    def test_matches_reference(
        self, active_request_count, padded_count, block_size_tokens, num_cols
    ):
        torch.manual_seed(active_request_count * 1000 + num_cols)
        max_position = block_size_tokens * num_cols - 13
        start_positions = torch.randint(
            0, max_position, (self.MAX_REQUESTS,), dtype=torch.int32, device=DEVICE
        )
        self._run_draft_loop(
            active_request_count,
            padded_count,
            block_size_tokens,
            num_cols,
            start_positions,
            num_depths=12,
        )

    def test_block_boundary_crossing(self):
        # Each request's 12 depths straddle a block boundary at a different depth.
        block_size_tokens = 16
        start_positions = torch.tensor(
            [15 - i for i in range(12)] + [0] * (self.MAX_REQUESTS - 12),
            dtype=torch.int32,
            device=DEVICE,
        )
        self._run_draft_loop(12, 16, block_size_tokens, 8, start_positions, num_depths=12)

    def test_all_padding(self):
        # No request drafts, but the forward still launches over padded slots.
        ref_meta = self._make_metadata(16, 8)
        tri_meta = self._make_metadata(16, 8)
        ref_view = self._make_gpu_view(8)
        tri_view = self._make_gpu_view(8)
        for meta in (ref_meta, tri_meta):
            meta._enter_decode(active_request_count=0, padded_count=16, graphed=True)
        self._reference_step(ref_meta, ref_view)
        self._kernel_step(tri_meta, tri_view)
        self._assert_equal(ref_meta, ref_view, tri_meta, tri_view)

    def test_capture_staging(self):
        # `begin_decode_for_capture` treats every padded slot as active at the dummy block.
        ref_meta = self._make_metadata(16, 8)
        tri_meta = self._make_metadata(16, 8)
        ref_view = self._make_gpu_view(8)
        tri_view = self._make_gpu_view(8)
        for meta in (ref_meta, tri_meta):
            meta.begin_decode_for_capture(padded_count=16)
        self._reference_step(ref_meta, ref_view)
        self._kernel_step(tri_meta, tri_view)
        self._assert_equal(ref_meta, ref_view, tri_meta, tri_view)

    def test_dispatcher_uses_kernel(self):
        # `MTPMetadata.write_draft_step_metadata` must produce the reference result.
        ref_meta = self._make_metadata(16, 8)
        tri_meta = self._make_metadata(16, 8)
        ref_view = self._make_gpu_view(8)
        tri_view = self._make_gpu_view(8)
        start_positions = torch.arange(3, 3 + self.MAX_REQUESTS, dtype=torch.int32, device=DEVICE)
        block_table_src = torch.randint(
            0, 500, (self.MAX_REQUESTS, 8), dtype=torch.int32, device=DEVICE
        )
        for meta in (ref_meta, tri_meta):
            meta.begin_decode(
                active_request_count=6,
                padded_count=8,
                start_positions=start_positions,
                block_table_src=block_table_src,
                graphed=False,
            )
        self._reference_step(ref_meta, ref_view)
        kv_lengths = tri_meta.write_draft_step_metadata(tri_view)
        self._assert_equal(ref_meta, ref_view, tri_meta, tri_view)
        assert torch.equal(kv_lengths, ref_meta.kv_lengths[:6])

    def test_padding_count_zero_is_noop(self):
        tri_view = self._make_gpu_view(8)
        before = {k: v.clone() for k, v in vars(tri_view).items()}
        tri_meta = self._make_metadata(16, 8)
        tri_meta.begin_decode_for_capture(padded_count=0)
        self._kernel_step(tri_meta, tri_view)
        for name, ref in before.items():
            assert torch.equal(getattr(tri_view, name), ref), name


class TestMtpCommitStepMetadata:
    """Tests for the CUDA-graphed commit-pass metadata Triton kernel.

    The reference spells out the fixed-slot layout directly: request r owns rows
    r*S .. r*S+S-1, slot j writes position base_r - 1 - a_r + j when j < a_r (else the dummy
    block), and each real request attends with S query rows over start_r + S keys.
    """

    MAX_REQUESTS = 64
    DUMMY_BLOCK = 999
    SENTINEL = -7

    def _make_gpu_view(self, num_tokens, num_cols):
        r, s = self.MAX_REQUESTS, self.SENTINEL

        def full(n, dtype):
            return torch.full((n,), s, dtype=dtype, device=DEVICE)

        return SimpleNamespace(
            token_to_block_idx=full(num_tokens, torch.int64),
            token_to_local_position_within_kv_block=full(num_tokens, torch.int32),
            token_to_request_idx=full(num_tokens, torch.int32),
            token_to_position_in_request=full(num_tokens, torch.int32),
            token_to_pos_ids=full(num_tokens, torch.int64),
            mha_query_lengths=full(r, torch.int32),
            mha_cu_query_seq_lengths=full(r + 1, torch.int32),
            mha_kv_seq_lengths=full(r, torch.int32),
            mha_cu_kv_seq_lengths=full(r + 1, torch.int32),
            mha_block_table=torch.full((r, num_cols), s, dtype=torch.int32, device=DEVICE),
        )

    def _reference(self, view, accepted, base, block_table, n, padded, slots, block_size):
        dummy = self.DUMMY_BLOCK
        accepted = accepted.long().cpu()
        base = base.long().cpu()
        bt = block_table.long().cpu()
        cu_q = cu_kv = 0
        view.mha_cu_query_seq_lengths[0] = 0
        view.mha_cu_kv_seq_lengths[0] = 0
        for r in range(padded):
            real = r < n
            start = int(base[r] - 1 - accepted[r]) if real else 0
            for j in range(slots):
                t = r * slots + j
                pos = start + j
                writes = real and j < int(accepted[r])
                view.token_to_block_idx[t] = int(bt[r, pos // block_size]) if writes else dummy
                view.token_to_local_position_within_kv_block[t] = pos % block_size if writes else 0
                view.token_to_request_idx[t] = r
                view.token_to_position_in_request[t] = pos if real else 0
                view.token_to_pos_ids[t] = pos if real else 0
            q_len = slots if real else 0
            kv_len = start + slots if real else 0
            cu_q += q_len
            cu_kv += kv_len
            view.mha_query_lengths[r] = q_len
            view.mha_kv_seq_lengths[r] = kv_len
            view.mha_cu_query_seq_lengths[r + 1] = cu_q
            view.mha_cu_kv_seq_lengths[r + 1] = cu_kv
            view.mha_block_table[r] = block_table[r].to(torch.int32) if real else -1

    def _check(self, n, padded, slots, block_size, num_cols, accepted, base, seed=0):
        torch.manual_seed(seed)
        block_table = torch.randint(
            0, 500, (self.MAX_REQUESTS, num_cols), dtype=torch.int32, device=DEVICE
        )
        num_tokens = self.MAX_REQUESTS * slots
        ref_view = self._make_gpu_view(num_tokens, num_cols)
        tri_view = self._make_gpu_view(num_tokens, num_cols)
        self._reference(ref_view, accepted, base, block_table, n, padded, slots, block_size)
        mtp_commit_step_metadata(
            gpu_view=tri_view,
            accepted_counts=accepted,
            base_positions=base,
            block_table=block_table,
            active_request_count=n,
            padded_count=padded,
            num_slots=slots,
            block_size_tokens=block_size,
            dummy_block_idx=self.DUMMY_BLOCK,
        )
        for name, ref in vars(ref_view).items():
            assert torch.equal(getattr(tri_view, name), ref), name

    @pytest.mark.parametrize(
        "active_request_count, padded_count", [(1, 1), (5, 8), (8, 8), (13, 16), (64, 64)]
    )
    @pytest.mark.parametrize("num_slots", [1, 2, 3, 12])
    @pytest.mark.parametrize("block_size_tokens", [16, 64])
    def test_matches_reference(
        self, active_request_count, padded_count, num_slots, block_size_tokens
    ):
        num_cols = 33
        torch.manual_seed(active_request_count * 100 + num_slots)
        accepted = torch.randint(
            0, num_slots + 1, (self.MAX_REQUESTS,), dtype=torch.int64, device=DEVICE
        )
        # base - 1 - accepted >= 0, and base + num_slots stays inside the block table.
        base = torch.randint(
            num_slots + 1,
            block_size_tokens * num_cols - num_slots,
            (self.MAX_REQUESTS,),
            dtype=torch.int64,
            device=DEVICE,
        )
        self._check(
            active_request_count,
            padded_count,
            num_slots,
            block_size_tokens,
            num_cols,
            accepted,
            base,
        )

    def test_block_boundary_crossing(self):
        # Refreshed runs that straddle a block boundary at every offset.
        slots, block_size = 12, 16
        accepted = torch.full((self.MAX_REQUESTS,), slots, dtype=torch.int64, device=DEVICE)
        base = torch.arange(20, 20 + self.MAX_REQUESTS, dtype=torch.int64, device=DEVICE)
        self._check(16, 16, slots, block_size, 8, accepted, base)

    def test_capture_inputs(self):
        # Capture / EP-dummy staging: nothing accepted, base position 1, all rows active.
        accepted = torch.zeros(self.MAX_REQUESTS, dtype=torch.int64, device=DEVICE)
        base = torch.ones(self.MAX_REQUESTS, dtype=torch.int64, device=DEVICE)
        self._check(16, 16, 12, 16, 8, accepted, base)

    def test_padding_count_zero_is_noop(self):
        view = self._make_gpu_view(self.MAX_REQUESTS, 8)
        before = {k: v.clone() for k, v in vars(view).items()}
        zeros = torch.zeros(self.MAX_REQUESTS, dtype=torch.int64, device=DEVICE)
        block_table = torch.zeros((self.MAX_REQUESTS, 8), dtype=torch.int32, device=DEVICE)
        mtp_commit_step_metadata(view, zeros, zeros + 1, block_table, 0, 0, 12, 16, 0)
        for name, ref in before.items():
            assert torch.equal(getattr(view, name), ref), name


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
            accepted_counts.clone(), prefill_status.clone(), *ref_args, num_spec, block_size
        )
        tri_release, tri_mask = rewind_kv_cache(
            accepted_counts.clone(), prefill_status.clone(), *tri_args, num_spec, block_size
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
