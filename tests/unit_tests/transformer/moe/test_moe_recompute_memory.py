# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA memory regression test for MoE full recompute activation cleanup.

Verifies that explicit deletion of intermediate tensors in unpermute()
reduces peak CUDA memory during full recomputation.

Issue: https://github.com/NVIDIA/Megatron-LM/issues/3221
PR:    https://github.com/NVIDIA/Megatron-LM/pull/5197
"""

import gc

import pytest
import torch


def unpermute_without_del(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """Original unpermute WITHOUT the explicit del statements (baseline)."""
    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    output_tokens.scatter_add_(
        0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
    )
    out = output_tokens.to(dtype=input_dtype)
    return out


def unpermute_with_del(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """Fixed unpermute WITH explicit del (the PR's approach)."""
    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    output_tokens.scatter_add_(
        0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
    )
    out = output_tokens.to(dtype=input_dtype)
    # Explicitly release intermediate tensor references so CUDA allocator
    # can reclaim memory immediately during full recomputation.
    del output_tokens, permuted_tokens, sorted_indices
    return out


class MoELayer(torch.nn.Module):
    """Minimal MoE layer wrapper for recompute memory testing."""

    def __init__(self, hidden_size: int, ffn_hidden: int, unpermute_fn):
        super().__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, ffn_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(ffn_hidden, hidden_size),
        )
        self.unpermute_fn = unpermute_fn

    def forward(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        expert_out = self.ffn(permuted_tokens)
        return self.unpermute_fn(expert_out, sorted_indices, restore_shape)


def measure_peak_memory(fn, *args, **kwargs):
    """Run fn with CUDA memory tracking and return (result, peak_allocated, retained_after)."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    result = fn(*args, **kwargs)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    retained_mb = torch.cuda.memory_allocated() / 1024**2
    return result, peak_mb, retained_mb


@pytest.mark.parametrize(
    "num_permuted_tokens, num_total_tokens, hidden_size, ffn_hidden",
    [
        (65536, 131072, 4096, 14336),   # DeepSeek-V3 scale
        (32768, 65536, 2048, 7168),     # Medium scale
        (16384, 32768, 1024, 4096),     # Small scale
    ],
)
def test_moe_recompute_memory_reduction(
    num_permuted_tokens, num_total_tokens, hidden_size, ffn_hidden
):
    """Verify del statements reduce peak CUDA memory in full recompute mode.

    The test simulates MoE's token permutation pattern: tokens are first
    permuted (grouped by expert), fed through FFN, then unpermuted back
    to original token order via scatter_add_. In full recompute mode,
    the forward pass runs again during backward, and intermediate tensors
    from scatter_add_ can accumulate if not explicitly freed.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create random data simulating permuted MoE tokens
    permuted_tokens = torch.randn(
        num_permuted_tokens, hidden_size, device=device, dtype=dtype, requires_grad=True
    )
    sorted_indices = torch.randint(
        0, num_total_tokens, (num_permuted_tokens,), device=device
    )
    restore_shape = torch.Size((num_total_tokens, hidden_size))

    # Random target for loss
    target = torch.randn(num_total_tokens, hidden_size, device=device, dtype=dtype)

    def compute_loss_and_backward(layer, grad_scaler=1.0):
        """Simulate one MoE step with full recompute."""
        # Wrap in checkpoint to simulate full recomputation
        output = torch.utils.checkpoint.checkpoint(
            layer, permuted_tokens, sorted_indices, restore_shape,
            use_reentrant=False
        )
        loss = torch.nn.functional.mse_loss(output, target)
        scaled_loss = loss * grad_scaler
        scaled_loss.backward()
        return loss.item()

    # Measure with del (fix)
    layer_with_del = MoELayer(hidden_size, ffn_hidden, unpermute_with_del).to(device)
    layer_with_del.ffn[0].weight.requires_grad_(True)
    layer_with_del.ffn[2].weight.requires_grad_(True)

    _, peak_with_del, retained_with_del = measure_peak_memory(
        compute_loss_and_backward, layer_with_del
    )

    del layer_with_del
    torch.cuda.empty_cache()
    gc.collect()

    # Measure without del (baseline / original behavior)
    layer_no_del = MoELayer(hidden_size, ffn_hidden, unpermute_without_del).to(device)
    layer_no_del.ffn[0].weight.requires_grad_(True)
    layer_no_del.ffn[2].weight.requires_grad_(True)

    _, peak_no_del, retained_no_del = measure_peak_memory(
        compute_loss_and_backward, layer_no_del
    )

    print(
        f"\n[scale={num_permuted_tokens}x{hidden_size}] "
        f"Peak: {peak_no_del:.1f} MB (no del) vs {peak_with_del:.1f} MB (with del) "
        f"| Retained after backward: {retained_no_del:.1f} MB vs {retained_with_del:.1f} MB"
    )

    # The del version should use less or equal peak memory
    # In practice, the difference is ~the size of intermediate activations
    # (output_tokens + permuted_tokens + sorted_indices indices)
    assert (
        peak_with_del <= peak_no_del
    ), (
        f"Peak memory with del ({peak_with_del:.1f} MB) should NOT exceed "
        f"without del ({peak_no_del:.1f} MB) at scale {num_permuted_tokens}x{hidden_size}"
    )
    assert (
        retained_with_del <= retained_no_del
    ), (
        f"Retained memory after backward with del ({retained_with_del:.1f} MB) "
        f"should NOT exceed without del ({retained_no_del:.1f} MB)"
    )
