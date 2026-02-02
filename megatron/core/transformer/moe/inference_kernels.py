import torch
import triton
import triton.language as tl
import pytest
import torch

@triton.jit
def moe_permute_kernel(
    hidden_ptr, mask_ptr_T, dest_idx_ptr, output_ptr,
    stride_h_t, num_tokens, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if not tl.load(mask_ptr_T + pid): return
    
    out_row_idx = tl.load(dest_idx_ptr + pid) - 1
    token_idx = pid % num_tokens

    offsets = tl.arange(0, BLOCK_SIZE)
    mask_h = offsets < hidden_size
    
    row_data = tl.load(hidden_ptr + (token_idx * stride_h_t) + offsets, mask=mask_h)
    tl.store(output_ptr + (out_row_idx * hidden_size) + offsets, row_data, mask=mask_h)

@triton.jit
def moe_unpermute_kernel(
    permuted_ptr, mask_ptr_T, dest_idx_ptr, output_ptr,
    stride_out_t, num_tokens, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if not tl.load(mask_ptr_T + pid): return
    
    src_row_idx = tl.load(dest_idx_ptr + pid) - 1
    token_idx = pid % num_tokens

    offsets = tl.arange(0, BLOCK_SIZE)
    mask_h = offsets < hidden_size

    # Load as current dtype
    row_data = tl.load(permuted_ptr + (src_row_idx * hidden_size) + offsets, mask=mask_h)
    
    # Cast to float32 for the accumulation to avoid BF16 rounding errors
    row_data_fp32 = row_data.to(tl.float32)
    
    # Atomic add in FP32 (Triton handles the casting/locking)
    tl.atomic_add(output_ptr + (token_idx * stride_out_t) + offsets, row_data_fp32, mask=mask_h)

def launch_moe_kernels(hidden_states, mask, static_buffer, unpermute=False):
    T, H = hidden_states.shape
    E = mask.size(1)
    mask_T = mask.t().contiguous()
    dest_indices = torch.cumsum(mask_T.view(-1).long(), dim=0).to(torch.int32)
    
    grid = (E * T,)
    BLOCK_SIZE = triton.next_power_of_2(H)

    if not unpermute:
        moe_permute_kernel[grid](
            hidden_states, mask_T, dest_indices, static_buffer,
            hidden_states.stride(0), T, H, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # For unpermute, hidden_states is the 'output' we write back into
        # ensure that hidden states is zeroed out before accumulation
        moe_unpermute_kernel[grid](
            static_buffer, mask_T, dest_indices, hidden_states,
            hidden_states.stride(0), T, H, BLOCK_SIZE=BLOCK_SIZE
        )


import triton
import triton.language as tl

@triton.jit
def moe_extract_probs_kernel(
    probs_ptr_T,      # [Experts, Tokens] (Transposed & Contiguous)
    mask_ptr_T,       # [Experts, Tokens] (Transposed & Contiguous)
    dest_idx_ptr,     # [Experts * Tokens] (Cumsum of mask_ptr_T)
    out_probs_ptr,    # [MAX_OUT] (Static 1D Buffer)
    num_tokens,
):
    # pid follows Expert-major order: expert_idx * num_tokens + token_idx
    pid = tl.program_id(0)
    
    # 1. Check if this expert-token pair is active
    mask_val = tl.load(mask_ptr_T + pid)
    if not mask_val:
        return

    # 2. Get the destination index in the 1D static buffer
    # out_row_idx corresponds to the row index in the permuted hidden states
    out_idx = tl.load(dest_idx_ptr + pid) - 1

    # 3. Load the probability and store it in the static output buffer
    prob = tl.load(probs_ptr_T + pid)
    tl.store(out_probs_ptr + out_idx, prob)

def launch_extract_probs(probs, mask, static_prob_buffer):
    T, E = probs.shape
    
    # Match the permutation layout: Experts first
    probs_T = probs.t().contiguous()
    mask_T = mask.t().contiguous()
    
    # Reuse the same cumsum logic from your permutation step
    dest_indices = torch.cumsum(mask_T.view(-1).long(), dim=0).to(torch.int32)
    
    grid = (E * T,)
    
    moe_extract_probs_kernel[grid](
        probs_T,
        mask_T,
        dest_indices,
        static_prob_buffer,
        T
    )
    return static_prob_buffer

@pytest.mark.parametrize("T, E, H", [
    (1, 1, 128),       # Minimal case
    (64, 8, 512),      # Standard small
    (128, 16, 1024),   # Medium
    (256, 32, 2048),   # Large (LLM Scale)
    (1024, 1, 128),    # Single Expert
    (32, 64, 64),      # High expert count
])
@pytest.mark.parametrize("sparsity", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_moe_cycle(T, E, H, sparsity, dtype):
    device = "cuda"
    MAX_OUT = T * E
    
    # Setup inputs
    hidden_states = torch.randn(T, H, device=device, dtype=dtype) * 1e-3
    mask = torch.rand(T, E, device=device) > sparsity
    
    # We need a small prob scaling for unpermute to be realistic, 
    # but for pure permutation test, we'll stick to raw values.
    static_buffer = torch.zeros((MAX_OUT, H), device=device, dtype=dtype)
    
    # 1. Test Permute
    launch_moe_kernels(hidden_states, mask, static_buffer, unpermute=False)
    
    # Verification of Grouped-by-Expert layout
    buffer_idx = 0
    for e_idx in range(E):
        for t_idx in range(T):
            if mask[t_idx, e_idx]:
                assert torch.allclose(static_buffer[buffer_idx], hidden_states[t_idx], atol=1e-5)
                buffer_idx += 1
    
    assert static_buffer[buffer_idx:].sum() == 0, "Stale data found in buffer tail"

    # 2. Test Un-permute (Gather)
    # We'll create a new tensor to receive the gathered data
    # (Using zeros because unpermute uses atomic_add to handle Top-K)
    output_states = torch.zeros_like(hidden_states)
    launch_moe_kernels(output_states, mask, static_buffer, unpermute=True)
    
    # Validation: If a token went to N experts, it should be N * original_value
    expert_counts_per_token = mask.sum(dim=1)
    expected_output = hidden_states * expert_counts_per_token.unsqueeze(-1)
    
    # Instead of allclose, use this for BF16
    if dtype == torch.bfloat16:
        # rtol=1.6e-2 is the standard epsilon for bfloat16
        torch.testing.assert_close(output_states, expected_output, rtol=0.016, atol=0.005)
    else:
        torch.testing.assert_close(output_states, expected_output, rtol=1e-5, atol=1e-5)