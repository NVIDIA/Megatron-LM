import torch
import triton
import triton.language as tl
import pytest
import torch


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

def launch_unpermute_kernel(hidden_states, mask, static_buffer,
                       mask_T, dest_indices):
    """
    Launch the unpermute kernel.

    Args:
        hidden_states: Input tensor for permute, output tensor for unpermute
        mask: Boolean mask [T, E]
        static_buffer: Output buffer for permute, input buffer for unpermute
        unpermute: If True, run unpermute kernel
        mask_T: Optional pre-computed transposed mask (avoids recomputation)
        dest_indices: Optional pre-computed cumsum indices (avoids recomputation)
    """
    T, H = hidden_states.shape
    E = mask.size(1)

    grid = (E * T,)
    BLOCK_SIZE = triton.next_power_of_2(H)

    
    # For unpermute, hidden_states is the 'output' we write back into
    # ensure that hidden states is zeroed out before accumulation
    moe_unpermute_kernel[grid](
        static_buffer, mask_T, dest_indices, hidden_states,
        hidden_states.stride(0), T, H, BLOCK_SIZE=BLOCK_SIZE
    )



@triton.jit
def moe_fused_permute_extract_kernel(
    hidden_ptr, probs_ptr_T, mask_ptr_T, dest_idx_ptr,
    out_hidden_ptr, out_probs_ptr,
    stride_h_t, num_tokens, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: permute hidden states AND extract probs in one pass.

    This eliminates the need for separate kernel launches and avoids
    recomputing mask_T and dest_indices twice.
    """
    pid = tl.program_id(0)

    # Early exit if this expert-token pair is inactive
    if not tl.load(mask_ptr_T + pid):
        return

    out_row_idx = tl.load(dest_idx_ptr + pid) - 1
    token_idx = pid % num_tokens

    # 1. Permute hidden states (vectorized load/store)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask_h = offsets < hidden_size
    row_data = tl.load(hidden_ptr + (token_idx * stride_h_t) + offsets, mask=mask_h)
    tl.store(out_hidden_ptr + (out_row_idx * hidden_size) + offsets, row_data, mask=mask_h)

    # 2. Extract probability (single scalar per program)
    prob = tl.load(probs_ptr_T + pid)
    tl.store(out_probs_ptr + out_row_idx, prob)


def launch_fused_permute_and_probs(hidden_states, probs, mask,
                                   hidden_workspace, probs_workspace):
    """
    Fused launcher that:
    1. Computes mask_T, probs_T, and dest_indices ONCE
    2. Launches a single fused kernel for both permute + prob extraction

    Returns:
        mask_T: Transposed mask (cached for potential reuse in unpermute)
        dest_indices: Cumsum indices (cached for potential reuse in unpermute)
    """
    T, H = hidden_states.shape
    E = mask.size(1)

    # Compute shared intermediates once
    mask_T = mask.t().contiguous()
    probs_T = probs.t().contiguous()
    dest_indices = torch.cumsum(mask_T.view(-1).long(), dim=0).to(torch.int32)

    grid = (E * T,)
    BLOCK_SIZE = triton.next_power_of_2(H)

    moe_fused_permute_extract_kernel[grid](
        hidden_states, probs_T, mask_T, dest_indices,
        hidden_workspace, probs_workspace,
        hidden_states.stride(0), T, H, BLOCK_SIZE=BLOCK_SIZE
    )

    # Return cached values for potential reuse in unpermute
    return mask_T, dest_indices



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
def test_fused_permute_and_probs(T, E, H, sparsity, dtype):
    """
    Test that the fused kernel produces identical results to separate kernels.
    """
    device = "cuda"
    MAX_OUT = T * E

    # Setup inputs
    hidden_states = torch.randn(T, H, device=device, dtype=dtype) * 1e-3
    probs = torch.rand(T, E, device=device, dtype=dtype)
    mask = torch.rand(T, E, device=device) > sparsity

    # Ensure at least one active token-expert pair
    if not mask.any():
        mask[0, 0] = True

    # --- Reference: Separate kernel launches ---
    ref_hidden_buffer = torch.zeros((MAX_OUT, H), device=device, dtype=dtype)
    ref_probs_buffer = torch.zeros(MAX_OUT, device=device, dtype=dtype)

    launch_moe_kernels(hidden_states, mask, ref_hidden_buffer, unpermute=False)
    launch_extract_probs(probs, mask, ref_probs_buffer)

    # --- Test: Fused kernel launch ---
    fused_hidden_buffer = torch.zeros((MAX_OUT, H), device=device, dtype=dtype)
    fused_probs_buffer = torch.zeros(MAX_OUT, device=device, dtype=dtype)

    mask_T, dest_indices = launch_fused_permute_and_probs(
        hidden_states, probs, mask,
        fused_hidden_buffer, fused_probs_buffer
    )

    # --- Verify outputs match ---
    num_active = mask.sum().item()

    # Compare hidden states (only active portion)
    torch.testing.assert_close(
        fused_hidden_buffer[:num_active],
        ref_hidden_buffer[:num_active],
        rtol=1e-5, atol=1e-5
    )

    # Compare probs (only active portion)
    torch.testing.assert_close(
        fused_probs_buffer[:num_active],
        ref_probs_buffer[:num_active],
        rtol=1e-5, atol=1e-5
    )

    # Verify cached values are valid for reuse
    assert mask_T.shape == (E, T), f"mask_T shape mismatch: {mask_T.shape}"
    assert dest_indices.shape == (E * T,), f"dest_indices shape mismatch: {dest_indices.shape}"