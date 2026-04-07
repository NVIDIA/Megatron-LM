# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.kernels.token_setup_kernel import triton_token_setup


def _reference_token_setup(
    request_kv_length_offsets,
    request_query_lengths,
    request_last_kv_block_offset,
    request_last_kv_block_id,
    next_tokens,
    new_speculative_tokens,
    prev_last_block_ids,
    token_to_input_ids,
    token_to_pos_ids,
    token_to_request_idx,
    token_to_position_in_request,
    token_to_local_position_within_kv_block,
    token_to_block_idx,
    paused_request_count,
    total_request_count,
    block_size_tokens,
    num_speculative_tokens,
):
    """Python reference implementation matching the original dynamic_context.py logic."""
    active_request_count = total_request_count - paused_request_count
    num_generated_tokens = 1 + num_speculative_tokens

    request_kv_length_offsets[paused_request_count:total_request_count].add_(
        request_query_lengths[paused_request_count:total_request_count]
    )
    request_query_lengths[paused_request_count:total_request_count].fill_(num_generated_tokens)

    old_offsets = request_last_kv_block_offset[paused_request_count:total_request_count].clone()
    request_last_kv_block_offset[paused_request_count:total_request_count] = (
        old_offsets + num_generated_tokens
    ) % block_size_tokens

    active_token_count = active_request_count * num_generated_tokens
    sampled_tokens = next_tokens[paused_request_count:total_request_count]

    if num_speculative_tokens > 0:
        sampled_speculative_tokens = new_speculative_tokens[
            :, paused_request_count:total_request_count
        ]
        interleaved = torch.vstack(
            [sampled_tokens.unsqueeze(0), sampled_speculative_tokens]
        ).T.reshape(-1)
    else:
        interleaved = sampled_tokens

    token_to_input_ids[:active_token_count] = interleaved

    token_to_pos_ids[:active_token_count] = request_kv_length_offsets[
        paused_request_count:total_request_count
    ].repeat_interleave(num_generated_tokens) + torch.arange(
        num_generated_tokens, device=next_tokens.device
    ).repeat(active_request_count)

    token_to_request_idx[:active_token_count] = torch.arange(
        paused_request_count, total_request_count, device=next_tokens.device
    ).repeat_interleave(num_generated_tokens)

    token_to_position_in_request[:active_token_count] = token_to_pos_ids[:active_token_count]

    token_to_local_position_within_kv_block[:active_token_count] = (
        token_to_pos_ids[:active_token_count] % block_size_tokens
    )

    current_block_ids = request_last_kv_block_id[paused_request_count:total_request_count]

    raw_positions = (
        old_offsets[:, None]
        + 1
        + torch.arange(num_generated_tokens, device=next_tokens.device)[None, :]
    )
    crosses_boundary = raw_positions >= block_size_tokens

    if not crosses_boundary.any() or num_speculative_tokens == 0:
        token_to_block_idx[:active_token_count] = request_last_kv_block_id[
            paused_request_count:total_request_count
        ].repeat_interleave(num_generated_tokens)
    else:
        prev_block_ids = prev_last_block_ids[paused_request_count:total_request_count]
        request_has_crossing = crosses_boundary.any(dim=1)
        block_idx = current_block_ids[:, None].expand(-1, num_generated_tokens)
        use_prev_block = request_has_crossing[:, None] & ~crosses_boundary
        prev_block_ids_expanded = prev_block_ids[:, None].expand(-1, num_generated_tokens)
        block_idx = torch.where(use_prev_block, prev_block_ids_expanded, block_idx)
        token_to_block_idx[:active_token_count] = block_idx.flatten()

    return active_token_count


def _make_tensors(
    max_requests,
    max_tokens,
    active_request_count,
    paused_request_count,
    block_size_tokens,
    num_speculative_tokens,
    force_boundary_crossing=False,
):
    """Create a matched pair of tensor sets for reference and kernel comparison."""
    device = "cuda"
    total_request_count = active_request_count + paused_request_count

    # Per-request tensors (int32)
    request_kv_length_offsets = torch.randint(
        1, 512, (max_requests,), dtype=torch.int32, device=device
    )
    request_query_lengths = torch.ones(max_requests, dtype=torch.int32, device=device)
    request_last_kv_block_offset = torch.randint(
        0, block_size_tokens, (max_requests,), dtype=torch.int32, device=device
    )
    request_last_kv_block_id = torch.randint(
        0, 1000, (max_requests,), dtype=torch.int32, device=device
    )

    if force_boundary_crossing and num_speculative_tokens > 0:
        num_generated_tokens = 1 + num_speculative_tokens
        for i in range(paused_request_count, total_request_count):
            request_last_kv_block_offset[i] = block_size_tokens - 2

    # Token inputs
    next_tokens = torch.randint(0, 50000, (total_request_count,), dtype=torch.long, device=device)

    if num_speculative_tokens > 0:
        new_speculative_tokens = torch.randint(
            0, 50000, (num_speculative_tokens, total_request_count), dtype=torch.long, device=device
        )
        prev_last_block_ids = torch.randint(
            0, 1000, (max_requests,), dtype=torch.int32, device=device
        )
    else:
        new_speculative_tokens = None
        prev_last_block_ids = None

    # Per-token output tensors (int64)
    def make_output():
        return torch.zeros(max_tokens, dtype=torch.long, device=device)

    token_to_input_ids = make_output()
    token_to_pos_ids = make_output()
    token_to_request_idx = make_output()
    token_to_position_in_request = make_output()
    token_to_local_position_within_kv_block = make_output()
    token_to_block_idx = make_output()

    return {
        "request_kv_length_offsets": request_kv_length_offsets,
        "request_query_lengths": request_query_lengths,
        "request_last_kv_block_offset": request_last_kv_block_offset,
        "request_last_kv_block_id": request_last_kv_block_id,
        "next_tokens": next_tokens,
        "new_speculative_tokens": new_speculative_tokens,
        "prev_last_block_ids": prev_last_block_ids,
        "token_to_input_ids": token_to_input_ids,
        "token_to_pos_ids": token_to_pos_ids,
        "token_to_request_idx": token_to_request_idx,
        "token_to_position_in_request": token_to_position_in_request,
        "token_to_local_position_within_kv_block": token_to_local_position_within_kv_block,
        "token_to_block_idx": token_to_block_idx,
        "paused_request_count": paused_request_count,
        "total_request_count": total_request_count,
        "block_size_tokens": block_size_tokens,
        "num_speculative_tokens": num_speculative_tokens,
    }


def _clone_tensors(tensors):
    """Deep-clone all tensor values for independent reference run."""
    cloned = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            cloned[k] = v.clone()
        else:
            cloned[k] = v
    return cloned


def _compare_outputs(ref, kern, active_token_count, paused_request_count, total_request_count):
    """Assert bitwise equality on all outputs."""
    # Token outputs (compare only the active slice)
    for name in [
        "token_to_input_ids",
        "token_to_pos_ids",
        "token_to_request_idx",
        "token_to_position_in_request",
        "token_to_local_position_within_kv_block",
        "token_to_block_idx",
    ]:
        assert torch.equal(ref[name][:active_token_count], kern[name][:active_token_count]), (
            f"Mismatch in {name}:\n"
            f"  ref:  {ref[name][:active_token_count]}\n"
            f"  kern: {kern[name][:active_token_count]}"
        )

    # Request tensor updates (compare the active slice)
    for name in [
        "request_kv_length_offsets",
        "request_query_lengths",
        "request_last_kv_block_offset",
    ]:
        r = ref[name][paused_request_count:total_request_count]
        k = kern[name][paused_request_count:total_request_count]
        assert torch.equal(r, k), (
            f"Mismatch in {name}[{paused_request_count}:{total_request_count}]:\n"
            f"  ref:  {r}\n"
            f"  kern: {k}"
        )


class TestTokenSetupKernel:

    @pytest.mark.parametrize("num_speculative_tokens", [0, 1, 2, 3])
    @pytest.mark.parametrize("active_request_count", [1, 4, 16, 64])
    @pytest.mark.parametrize("paused_request_count", [0, 2])
    @pytest.mark.parametrize("block_size_tokens", [64, 256])
    def test_token_setup_matches_reference(
        self, num_speculative_tokens, active_request_count, paused_request_count, block_size_tokens
    ):
        max_requests = active_request_count + paused_request_count + 16
        num_generated_tokens = 1 + num_speculative_tokens
        max_tokens = max_requests * num_generated_tokens + 32

        tensors = _make_tensors(
            max_requests=max_requests,
            max_tokens=max_tokens,
            active_request_count=active_request_count,
            paused_request_count=paused_request_count,
            block_size_tokens=block_size_tokens,
            num_speculative_tokens=num_speculative_tokens,
        )
        ref_tensors = _clone_tensors(tensors)

        ref_count = _reference_token_setup(**ref_tensors)
        kern_count = triton_token_setup(**tensors)

        assert ref_count == kern_count
        _compare_outputs(
            ref_tensors, tensors, ref_count, paused_request_count,
            active_request_count + paused_request_count,
        )

    def test_boundary_crossing_specific(self):
        """Verify block boundary crossing with speculative tokens."""
        block_size = 64
        num_spec = 2
        max_requests = 8
        max_tokens = 32

        tensors = _make_tensors(
            max_requests=max_requests,
            max_tokens=max_tokens,
            active_request_count=3,
            paused_request_count=0,
            block_size_tokens=block_size,
            num_speculative_tokens=num_spec,
            force_boundary_crossing=True,
        )
        ref_tensors = _clone_tensors(tensors)

        ref_count = _reference_token_setup(**ref_tensors)
        kern_count = triton_token_setup(**tensors)

        assert ref_count == kern_count
        _compare_outputs(ref_tensors, tensors, ref_count, 0, 3)

    def test_single_request_no_speculative(self):
        max_requests = 4
        max_tokens = 4

        tensors = _make_tensors(
            max_requests=max_requests,
            max_tokens=max_tokens,
            active_request_count=1,
            paused_request_count=0,
            block_size_tokens=128,
            num_speculative_tokens=0,
        )
        ref_tensors = _clone_tensors(tensors)

        ref_count = _reference_token_setup(**ref_tensors)
        kern_count = triton_token_setup(**tensors)

        assert ref_count == kern_count == 1
        _compare_outputs(ref_tensors, tensors, ref_count, 0, 1)

    def test_large_batch(self):
        active = 512
        max_requests = active + 16
        max_tokens = active + 32

        tensors = _make_tensors(
            max_requests=max_requests,
            max_tokens=max_tokens,
            active_request_count=active,
            paused_request_count=0,
            block_size_tokens=64,
            num_speculative_tokens=0,
        )
        ref_tensors = _clone_tensors(tensors)

        ref_count = _reference_token_setup(**ref_tensors)
        kern_count = triton_token_setup(**tensors)

        assert ref_count == kern_count == active
        _compare_outputs(ref_tensors, tensors, ref_count, 0, active)

    def test_zero_active_requests(self):
        max_requests = 8
        max_tokens = 8

        tensors = _make_tensors(
            max_requests=max_requests,
            max_tokens=max_tokens,
            active_request_count=0,
            paused_request_count=2,
            block_size_tokens=64,
            num_speculative_tokens=0,
        )

        result = triton_token_setup(**tensors)
        assert result == 0
