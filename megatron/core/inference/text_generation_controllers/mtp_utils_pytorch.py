# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch


def rewind_kv_cache(
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
    """Update the KV cache bookkeeping for speculative decoding.

    After forward pass with speculative tokens, some tokens may be rejected.
    This function "rewinds" the KV cache bookkeeping to reflect only the accepted tokens.

    When speculative tokens are rejected, we need to:
    1. Update kv_length_offsets (total sequence length)
    2. Update last_kv_block_offset (position within last block)
    3. If rewinding crosses a block boundary:
       - Reduce kv_block_counts
       - Update last_kv_block_id to point to the previous block
       - Clear the entry in kv_block_ids for the released block

    Mutates the input tensors in-place.

    Returns (blocks_to_release, remove_mask).
    """
    N = accepted_counts.shape[0]
    if num_active_requests is None:
        num_active_requests = N

    # Bulk-extract scalars once via .tolist() instead of per-element .item().
    # Avoids N round-trips through the Python/C++ boundary inside the loop.
    accepted_list = accepted_counts.tolist()
    prefill_list = prefill_status.tolist()
    offset_list = last_kv_block_offset.tolist()
    length_list = kv_length_offsets.tolist()
    block_count_list = kv_block_counts.tolist()
    last_block_list = last_kv_block_id.tolist()
    kv_block_ids_list = kv_block_ids.tolist()
    max_blocks = kv_block_ids.shape[1]

    blocks_to_release = torch.empty_like(last_kv_block_id)
    remove_mask = torch.empty(N, device=accepted_counts.device, dtype=torch.bool)

    for i in range(N):
        if i >= num_active_requests:
            blocks_to_release[i] = 0
            remove_mask[i] = False
            continue

        accepted = accepted_list[i]
        prefill = prefill_list[i]
        last_offset = offset_list[i]
        kv_length = length_list[i]
        block_count = block_count_list[i]
        last_block = last_block_list[i]

        # Number of tokens to rewind (rejected speculative tokens).
        # For prefill requests, no speculative tokens were forwarded through the model,
        # so there is nothing to rewind.
        num_to_rewind = 0 if prefill == 1 else num_speculative_tokens - accepted

        # Save the original offset BEFORE modifying to correctly detect block boundary crossing.
        # A request crosses back to a previous block if: original_offset - num_to_rewind < 0
        diff = last_offset - num_to_rewind
        remove = diff < 0

        # Update the offsets
        new_offset = diff % block_size_tokens
        last_kv_block_offset[i] = new_offset
        kv_length_offsets[i] = kv_length - num_to_rewind

        # For requests that crossed back to a previous block, we need to:
        # 1. Reduce the block count by 1
        # 2. Get the block ID to release (current last_kv_block_id)
        # 3. Update last_kv_block_id to point to the previous block
        # 4. Clear the entry in kv_block_ids for the released block
        # 5. Release the block back to the allocator
        blocks_to_release[i] = last_block

        # Reduce block counts for requests that crossed back
        new_block_count = block_count - 1 if remove else block_count
        kv_block_counts[i] = new_block_count

        # Update last_kv_block_id to point to the previous block (at index new_count - 1)
        prev_idx = max(new_block_count - 1, 0)
        prev_block_id = kv_block_ids_list[i][prev_idx]
        last_kv_block_id[i] = prev_block_id if remove else last_block

        # Clear the released block entry (at index new_count, which was the old last block)
        scatter_idx = min(new_block_count, max_blocks - 1)
        if remove:
            kv_block_ids[i, scatter_idx] = -1

        remove_mask[i] = remove

    return blocks_to_release, remove_mask


# pylint: disable=line-too-long
def verify_speculative_tokens(
    input_tokens, output_tokens, num_decode_requests, num_prefill_requests, num_speculative_tokens
):
    """Verify speculative tokens against input tokens and compute acceptance.

    Creates an accepted tokens mask where:
    - For prefill requests, the token is always accepted.
    - For decode requests, the first token (base token) is always accepted, then we compare
      sampled tokens with input tokens and accept consecutive matches.
    Then finds the index of the last accepted token per request.

    Example (assume 1, 2, and 0 spec tokens are accepted in the first 3 decode requests):
        input_tokens_required:              [ a5  a6s  a7s |  b3    b4s  b5s   |  c6   c7s   c8s   |     d2      |         e4         ]  # Size 11
        Output tokens                       [ a6o a7o  a8o |  b40   b5o  b6o   |  c7o  c8o   c9o   |     d3o     |         e5o        ]
        Output tokens right shift           [ d3o a6o  a7o |  a8o   b40  b5o   |  b6o  c7o   c8o   |     c9o     |         d3o        ]
        Accepted tokens  mask               [  1   1    0  |  1      1    1    |   1    0     0    |      1      |         1          ]
        Last one indices                    [      1       |         5         |        6          |      9      |         10         ]

    Returns:
        tuple: (last_one_indices, accepted_tokens_mask, input_tokens) where
            last_one_indices contains the index of the last accepted token per request.
    """
    if input_tokens.ndim == 2:
        input_tokens = input_tokens.squeeze(0)

    stride = num_speculative_tokens + 1
    active_request_count = num_decode_requests + num_prefill_requests
    decode_len = num_decode_requests * stride

    # Initialize mask with False to prevent boundary bleed
    accepted_tokens_mask = torch.zeros_like(input_tokens, dtype=torch.bool)

    # Safe decode token verification without cross-batch boundary contamination
    decode_mask_2d = None
    if num_decode_requests > 0:
        decode_inputs = input_tokens[:decode_len].reshape(num_decode_requests, stride)
        decode_outputs = output_tokens[:decode_len].reshape(num_decode_requests, stride)

        # Shift outputs right by 1 *within* each request to align sampled tokens with input targets
        decode_outputs_shifted = decode_outputs.roll(1, dims=1)
        decode_mask_2d = decode_inputs == decode_outputs_shifted
        # The first token (base token) is always accepted
        decode_mask_2d[:, 0] = True
        # Enforce consecutive acceptance: cummin propagates False to the right
        decode_mask_2d = decode_mask_2d.cummin(dim=1).values
        accepted_tokens_mask[:decode_len] = decode_mask_2d.flatten()

    # Make all prefill tokens accepted
    if num_prefill_requests > 0:
        accepted_tokens_mask[decode_len:] = True

    last_one_indices = torch.full(
        (active_request_count,), -1, device=input_tokens.device, dtype=torch.long
    )

    if num_decode_requests > 0:
        # Summing the consecutive mask gives the count; subtract 1 for the local index
        local_last_indices = decode_mask_2d.sum(dim=1) - 1
        row_offsets = torch.arange(num_decode_requests, device=input_tokens.device) * stride
        last_one_indices[:num_decode_requests] = row_offsets + local_last_indices

    if num_prefill_requests > 0:
        prefill_valid = torch.nonzero(accepted_tokens_mask[decode_len:]).squeeze(-1) + decode_len
        last_one_indices[num_decode_requests:] = prefill_valid

    return last_one_indices, accepted_tokens_mask, input_tokens


# pylint: disable=line-too-long
def prepare_next_forward_pass(
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
    """Prepare data for the next forward pass after speculative token verification.

    For each active request:
    - Store the final sampled tokens for the next forward pass.
    - Store the last accepted positions in the packed sequence for serial
      MTP computation after verification.

    For decode requests, extract accepted tokens and counts:
        input_tokens_required:              [ a5  a6s  a7s |  b3    b4s  b5s   |  c6   c7s   c8s   |     d2      |         e4         ]
        Accepted tokens  mask               [  1   1    0  |  1      1    1    |   1    0     0    |      1      |         1          ]
        Accepted tokens                     [   [a6s  -1]  |     [b4s  b5s]    |     [-1  -1]      ]  # Only decode requests (prefill defaults to -1)
        Accepted token counts               [      1       |         2         |         0         ]  # Prefill defaults to 0

    Writes results into the pre-allocated buffers provided by the caller.
    """
    active_request_count = last_one_indices.shape[0]
    stride = num_speculative_tokens + 1

    for pid in range(active_request_count):
        idx = last_one_indices[pid].item()

        # Store the final sampled tokens for the next forward pass.
        sampled_tokens_buf[pid] = output_tokens[idx]

        # Store the last accepted positions in the packed sequence for serial
        # MTP computation after verification.
        last_accepted_seq_buf[pid] = required_logit_indices[idx]

        # Extract accepted tokens and counts for decode requests.
        # For prefill it is always set to 1. For decode, the first token is always accepted,
        # then we compare with input tokens and accept the next tokens if its a match.
        if pid < num_decode_requests:
            base = pid * stride
            # Skip the first token of every decode request (i.e a5, b3, c6)
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


def mamba_state_selective_copy(
    intermediate_states, current_states, prefill_status, state_idx, accepted_counts, num_layers
):
    """Mamba speculative rewind state update.

    For each decode request, copies
    `intermediate[layer, slot, accepted_count, ...]` →
    `current[layer, slot, ...]` for every Mamba layer.
    """
    N = prefill_status.shape[0]
    for i in range(N):
        if prefill_status[i].item() == 1:
            continue
        slot = state_idx[i].item()
        accepted = accepted_counts[i].item()
        for layer in range(num_layers):
            current_states[layer, slot] = intermediate_states[layer, slot, accepted]
