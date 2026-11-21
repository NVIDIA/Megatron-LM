# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MTP Triton kernels.

Each test runs both the pure-PyTorch reference (from mtp_utils_pytorch) and
the Triton kernel (from mtp_utils_triton) on the same inputs, and asserts
that the outputs match exactly.
"""

import pytest
import torch

from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    mamba_state_selective_copy as mamba_state_selective_copy_pytorch,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    prepare_next_forward_pass as prepare_next_forward_pass_pytorch,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    verify_speculative_tokens as verify_speculative_tokens_pytorch,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    mamba_state_selective_copy,
    prepare_next_forward_pass,
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
        "num_decode,num_prefill,num_spec",
        [(1, 0, 2), (3, 0, 2), (3, 2, 2), (2, 2, 3), (0, 3, 2), (5, 3, 4)],
    )
    def test_basic(self, num_decode, num_prefill, num_spec):
        output_tokens, required_logit_indices, input_tokens, accepted_mask, last_one_indices = (
            self._setup(num_decode, num_prefill, num_spec)
        )

        active = num_decode + num_prefill

        ref_sampled = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        ref_last_seq = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        ref_accepted = torch.full((active, num_spec), 999, device=DEVICE, dtype=torch.int64)
        ref_counts = torch.full((active,), 999, device=DEVICE, dtype=torch.int64)

        tri_sampled = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        tri_last_seq = torch.zeros(active, device=DEVICE, dtype=torch.int64)
        tri_accepted = torch.full((active, num_spec), 999, device=DEVICE, dtype=torch.int64)
        tri_counts = torch.full((active,), 999, device=DEVICE, dtype=torch.int64)

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
        torch.testing.assert_close(tri_accepted, ref_accepted)
        torch.testing.assert_close(tri_counts, ref_counts)
        if num_prefill > 0:
            assert (tri_accepted[num_decode:] == -1).all()
            assert (tri_counts[num_decode:] == 0).all()

    def test_mixed_async_prefill_row_clears_false_termination_match(self):
        """A prefill row cannot inherit a stale accepted termination token."""
        termination_id = 91
        accepted_tokens = torch.tensor([[7, -1], [termination_id, 92]], device=DEVICE)
        accepted_counts = torch.tensor([1, 2], device=DEVICE)

        def prefill_has_valid_termination_match() -> bool:
            valid_count = int(accepted_counts[1].item())
            valid_tokens = accepted_tokens[1, :valid_count]
            return bool((valid_tokens == termination_id).any().item())

        assert accepted_tokens[1].tolist() == [91, 92]
        assert accepted_counts[1].item() == 2
        assert prefill_has_valid_termination_match()

        prepare_next_forward_pass_pytorch(
            num_decode_requests=1,
            output_tokens=torch.tensor([0, 40, 0, 41], device=DEVICE),
            required_logit_indices=torch.arange(4, device=DEVICE),
            last_one_indices=torch.tensor([1, 3], device=DEVICE),
            accepted_tokens_mask=torch.tensor([True, True, False, True], device=DEVICE),
            input_tokens=torch.tensor([0, 7, 0, 0], device=DEVICE),
            sampled_tokens_buf=torch.zeros(2, dtype=torch.long, device=DEVICE),
            last_accepted_seq_buf=torch.zeros(2, dtype=torch.long, device=DEVICE),
            accepted_tokens_per_request=accepted_tokens,
            accepted_token_counts=accepted_counts,
            num_speculative_tokens=2,
        )

        assert accepted_tokens.tolist() == [[7, -1], [-1, -1]]
        assert accepted_counts.tolist() == [1, 0]
        assert not prefill_has_valid_termination_match()

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


class TestStressRandom:
    """Randomized stress tests of verify_speculative_tokens with varied inputs."""

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
