# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the batch-invariant grouped GEMM Triton kernel.

Test A: Correctness – the kernel produces outputs equivalent to a reference
        (expert-loop torch.mm) implementation and to torch._grouped_mm when
        available.
Test B: Batch invariance – permuting the order of tokens across experts yields
        bitwise-identical per-token outputs.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    grouped_gemm_batch_invariant,
    is_batch_invariant_mode_enabled,
    set_batch_invariant_mode,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import is_te_min_version, is_torch_min_version


def _ref_grouped_gemm(a, b, batch_sizes, bias=None, trans_b=False):
    """Reference grouped GEMM using a simple expert loop with torch.mm.

    Args:
        a: [total_tokens, K]
        b: [E, K, N] (trans_b=False) or [E, N, K] (trans_b=True)
        batch_sizes: [E] token counts per expert
        bias: optional [E, N]
        trans_b: whether b is [E, N, K]

    Returns:
        c: [total_tokens, N]
    """
    K = a.size(1)
    if trans_b:
        N = b.size(1)
    else:
        N = b.size(2)

    c = torch.empty(a.size(0), N, device=a.device, dtype=a.dtype)
    offset = 0
    for e, m in enumerate(batch_sizes.tolist()):
        m = int(m)
        if m == 0:
            continue
        a_e = a[offset : offset + m]  # [m, K]
        if trans_b:
            # b[e] is [N, K], need [K, N] for mm
            b_e = b[e].t()
        else:
            b_e = b[e]  # [K, N]
        out_e = torch.mm(a_e.float(), b_e.float()).to(a.dtype)
        if bias is not None:
            out_e = out_e + bias[e].unsqueeze(0)
        c[offset : offset + m] = out_e
        offset += m
    return c


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_DTYPES = [torch.bfloat16]
_NUM_EXPERTS = [1, 4, 8]
_HIDDEN_SIZES = [128, 256]


def _make_inputs(num_experts, K, N, dtype, min_tokens_per_expert=2, max_tokens_per_expert=32):
    """Generate random inputs for grouped GEMM tests."""
    torch.manual_seed(42)
    batch_sizes = torch.randint(
        min_tokens_per_expert, max_tokens_per_expert + 1, (num_experts,), device="cuda"
    ).to(torch.int64)
    total_tokens = batch_sizes.sum().item()

    a = torch.randn(total_tokens, K, device="cuda", dtype=dtype)
    b = torch.randn(num_experts, N, K, device="cuda", dtype=dtype)  # [E, N, K] (trans_b=True)
    return a, b, batch_sizes


# ============================================================================
# Test A: Correctness vs reference implementation
# ============================================================================


class TestGroupedGemmBatchInvariantCorrectness:
    """Verify the Triton kernel matches a reference expert-loop implementation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("num_experts", _NUM_EXPERTS)
    @pytest.mark.parametrize("K,N", [(128, 256), (256, 128), (256, 256)])
    def test_correctness_vs_reference(self, dtype, num_experts, K, N):
        """Triton grouped GEMM should match expert-loop torch.mm reference."""
        a, b, batch_sizes = _make_inputs(num_experts, K, N, dtype)
        total_tokens = a.size(0)

        # Reference
        c_ref = _ref_grouped_gemm(a, b, batch_sizes, trans_b=True)

        # Triton kernel
        c_triton = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c_triton, batch_sizes, trans_b=True)

        torch.testing.assert_close(c_triton, c_ref, atol=1e-1, rtol=1e-1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_experts", [2, 4])
    def test_correctness_with_bias(self, num_experts):
        """Triton grouped GEMM with bias should match reference."""
        K, N, dtype = 128, 256, torch.bfloat16
        a, b, batch_sizes = _make_inputs(num_experts, K, N, dtype)
        total_tokens = a.size(0)
        bias = torch.randn(num_experts, N, device="cuda", dtype=dtype)

        c_ref = _ref_grouped_gemm(a, b, batch_sizes, bias=bias, trans_b=True)

        c_triton = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c_triton, batch_sizes, bias=bias, trans_b=True)

        torch.testing.assert_close(c_triton, c_ref, atol=1e-1, rtol=1e-1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_correctness_non_transposed(self):
        """Triton grouped GEMM with trans_b=False should match reference."""
        num_experts, K, N, dtype = 4, 128, 256, torch.bfloat16
        torch.manual_seed(42)
        batch_sizes = torch.randint(2, 33, (num_experts,), device="cuda").to(torch.int64)
        total_tokens = batch_sizes.sum().item()

        a = torch.randn(total_tokens, K, device="cuda", dtype=dtype)
        b = torch.randn(num_experts, K, N, device="cuda", dtype=dtype)  # [E, K, N]

        c_ref = _ref_grouped_gemm(a, b, batch_sizes, trans_b=False)
        c_triton = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c_triton, batch_sizes, trans_b=False)

        torch.testing.assert_close(c_triton, c_ref, atol=1e-1, rtol=1e-1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_empty_experts(self):
        """Experts with zero tokens should be handled gracefully."""
        num_experts, K, N, dtype = 4, 128, 256, torch.bfloat16
        torch.manual_seed(42)
        # Some experts have 0 tokens
        batch_sizes = torch.tensor([0, 10, 0, 5], device="cuda", dtype=torch.int64)
        total_tokens = batch_sizes.sum().item()

        a = torch.randn(total_tokens, K, device="cuda", dtype=dtype)
        b = torch.randn(num_experts, N, K, device="cuda", dtype=dtype)

        c_ref = _ref_grouped_gemm(a, b, batch_sizes, trans_b=True)
        c_triton = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c_triton, batch_sizes, trans_b=True)

        torch.testing.assert_close(c_triton, c_ref, atol=1e-1, rtol=1e-1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_all_experts_empty(self):
        """All experts with zero tokens should return the output buffer unchanged."""
        num_experts, K, N, dtype = 4, 128, 256, torch.bfloat16
        batch_sizes = torch.zeros(num_experts, device="cuda", dtype=torch.int64)
        a = torch.empty(0, K, device="cuda", dtype=dtype)
        b = torch.randn(num_experts, N, K, device="cuda", dtype=dtype)
        c = torch.empty(0, N, device="cuda", dtype=dtype)

        result = grouped_gemm_batch_invariant(a, b, c, batch_sizes, trans_b=True)
        assert result.shape == (0, N)

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and is_torch_min_version("2.10")),
        reason="Requires CUDA and PyTorch >= 2.10 for torch._grouped_mm",
    )
    def test_correctness_vs_torch_grouped_mm(self):
        """Triton grouped GEMM should match torch._grouped_mm."""
        num_experts, K, N, dtype = 4, 256, 256, torch.bfloat16
        a, b, batch_sizes = _make_inputs(num_experts, K, N, dtype)
        total_tokens = a.size(0)

        # torch._grouped_mm path
        offs = batch_sizes.cumsum(0).to(torch.int32)
        # _grouped_mm expects weight as [E, K, N] (already transposed)
        c_torch = torch._grouped_mm(a, b.transpose(1, 2), offs=offs)

        # Triton kernel (b is [E, N, K], trans_b=True)
        c_triton = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c_triton, batch_sizes, trans_b=True)

        torch.testing.assert_close(c_triton, c_torch, atol=1e-1, rtol=1e-1)


# ============================================================================
# Test B: Batch invariance property
# ============================================================================


class TestGroupedGemmBatchInvariance:
    """Verify the batch invariance property: permuting token order within the
    batch dimension produces bitwise-identical per-token outputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_permutation_invariance(self, num_experts):
        """Permuting the assignment of tokens to experts should produce the
        same per-token output when the same (token, expert) pair is maintained."""
        K, N, dtype = 256, 256, torch.bfloat16
        torch.manual_seed(123)

        # Build inputs with fixed tokens per expert
        batch_sizes = torch.randint(4, 17, (num_experts,), device="cuda").to(torch.int64)
        total_tokens = batch_sizes.sum().item()
        a = torch.randn(total_tokens, K, device="cuda", dtype=dtype)
        b = torch.randn(num_experts, N, K, device="cuda", dtype=dtype)

        # Run 1: original order
        c1 = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c1, batch_sizes, trans_b=True)

        # Run 2: permute expert order (shuffle which expert's tokens come first)
        perm = torch.randperm(num_experts)
        inv_perm = torch.argsort(perm)

        # Build permuted a and batch_sizes
        offsets = torch.zeros(num_experts + 1, dtype=torch.int64)
        offsets[1:] = batch_sizes.cpu().cumsum(0)

        perm_batch_sizes = batch_sizes[perm]
        perm_a_chunks = [a[offsets[e] : offsets[e + 1]] for e in perm.tolist()]
        perm_a = torch.cat(perm_a_chunks, dim=0)
        perm_b = b[perm]

        c2_perm = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(perm_a, perm_b, c2_perm, perm_batch_sizes, trans_b=True)

        # Un-permute the output to match original expert order
        perm_offsets = torch.zeros(num_experts + 1, dtype=torch.int64)
        perm_offsets[1:] = perm_batch_sizes.cpu().cumsum(0)

        c2_chunks = [c2_perm[perm_offsets[i] : perm_offsets[i + 1]] for i in range(num_experts)]
        # c2_chunks[i] corresponds to expert perm[i]. Reorder to original expert order.
        c2_reordered_chunks = [None] * num_experts
        for i in range(num_experts):
            c2_reordered_chunks[perm[i].item()] = c2_chunks[i]
        c2 = torch.cat(c2_reordered_chunks, dim=0)

        assert torch.equal(
            c1, c2
        ), f"Batch invariance violated: max diff = {(c1 - c2).abs().max().item()}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_token_permutation_within_expert(self):
        """Permuting the order of tokens within each expert should produce the
        same per-token output (just reordered)."""
        num_experts, K, N, dtype = 4, 256, 256, torch.bfloat16
        torch.manual_seed(456)

        batch_sizes = torch.randint(8, 33, (num_experts,), device="cuda").to(torch.int64)
        total_tokens = batch_sizes.sum().item()
        a = torch.randn(total_tokens, K, device="cuda", dtype=dtype)
        b = torch.randn(num_experts, N, K, device="cuda", dtype=dtype)

        # Run 1: original order
        c1 = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c1, batch_sizes, trans_b=True)

        # Run 2: shuffle tokens within each expert
        offsets = torch.zeros(num_experts + 1, dtype=torch.int64)
        offsets[1:] = batch_sizes.cpu().cumsum(0)

        perm_indices = torch.empty(total_tokens, dtype=torch.long, device="cuda")
        for e in range(num_experts):
            start, end = offsets[e].item(), offsets[e + 1].item()
            m = end - start
            if m > 0:
                local_perm = torch.randperm(m, device="cuda") + start
                perm_indices[start:end] = local_perm

        a_shuffled = a[perm_indices]
        c2 = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a_shuffled, b, c2, batch_sizes, trans_b=True)

        # Un-shuffle the output
        inv_perm = torch.argsort(perm_indices)
        c2_unshuffled = c2[inv_perm]

        # The outputs should be bitwise identical because the same (token, expert)
        # pairs are computed; only the row ordering within each expert changed.
        # Note: within-expert ordering does NOT affect other experts' results
        # since the kernel processes each expert's tiles independently.
        # However, within the same expert, floating-point accumulation order in
        # the M dimension is tile-local, so row permutation should not change results.
        assert torch.equal(c1, c2_unshuffled), (
            f"Within-expert permutation changed results: "
            f"max diff = {(c1 - c2_unshuffled).abs().max().item()}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_different_batch_compositions_same_tokens(self):
        """Running the same set of tokens under different batch compositions
        (i.e. different batch_sizes that sum to the same total but redistribute
        tokens across experts) should produce identical results for tokens that
        remain assigned to the same expert."""
        K, N, dtype = 256, 128, torch.bfloat16
        num_experts = 4
        torch.manual_seed(789)

        # Composition 1: [8, 8, 8, 8]
        batch_sizes_1 = torch.tensor([8, 8, 8, 8], device="cuda", dtype=torch.int64)
        total_tokens = 32
        a = torch.randn(total_tokens, K, device="cuda", dtype=dtype)
        b = torch.randn(num_experts, N, K, device="cuda", dtype=dtype)

        c1 = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a, b, c1, batch_sizes_1, trans_b=True)

        # Composition 2: [4, 12, 4, 12] — keep only the tokens that stay with
        # the same expert (first 4 of expert 0, first 8 of expert 1, etc.)
        # For simplicity, just re-run with different batch sizes on fresh tokens
        # but verify each expert's computation is independent.
        batch_sizes_2 = torch.tensor([4, 12, 4, 12], device="cuda", dtype=torch.int64)
        a2_parts = []
        offset = 0
        for e in range(num_experts):
            m1 = batch_sizes_1[e].item()
            m2 = batch_sizes_2[e].item()
            # Take original expert tokens, pad/truncate to new size
            expert_tokens = a[offset : offset + m1]
            if m2 <= m1:
                a2_parts.append(expert_tokens[:m2])
            else:
                extra = torch.randn(m2 - m1, K, device="cuda", dtype=dtype)
                a2_parts.append(torch.cat([expert_tokens, extra], dim=0))
            offset += m1
        a2 = torch.cat(a2_parts, dim=0)

        c2 = torch.empty(a2.size(0), N, device="cuda", dtype=dtype)
        grouped_gemm_batch_invariant(a2, b, c2, batch_sizes_2, trans_b=True)

        # Verify that tokens shared between compositions get the same result.
        # The shared tokens are the first min(m1, m2) tokens of each expert.
        offset1 = 0
        offset2 = 0
        for e in range(num_experts):
            m1 = batch_sizes_1[e].item()
            m2 = batch_sizes_2[e].item()
            shared = min(m1, m2)
            assert torch.equal(c1[offset1 : offset1 + shared], c2[offset2 : offset2 + shared]), (
                f"Expert {e}: shared tokens differ across batch compositions. "
                f"max diff = {(c1[offset1:offset1+shared] - c2[offset2:offset2+shared]).abs().max().item()}"
            )
            offset1 += m1
            offset2 += m2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_repeated_runs_deterministic(self):
        """Multiple runs with the same inputs must produce bitwise-identical results."""
        num_experts, K, N, dtype = 4, 256, 256, torch.bfloat16
        a, b, batch_sizes = _make_inputs(num_experts, K, N, dtype)
        total_tokens = a.size(0)

        results = []
        for _ in range(5):
            c = torch.empty(total_tokens, N, device="cuda", dtype=dtype)
            grouped_gemm_batch_invariant(a, b, c, batch_sizes, trans_b=True)
            results.append(c.clone())

        for i in range(1, len(results)):
            assert torch.equal(
                results[0], results[i]
            ), f"Run 0 vs run {i}: max diff = {(results[0] - results[i]).abs().max().item()}"


# ============================================================================
# Integration test: InferenceGroupedMLP with batch invariant mode
# ============================================================================


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_te_min_version("1.9.0.dev0")),
    reason="Requires CUDA and TE >= 1.9.0.dev0 for TEGroupedLinear",
)
class TestInferenceGroupedMLPBatchInvariant:
    """Test the batch-invariant path integrated into InferenceGroupedMLP."""

    def setup_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(42)

        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.mlp import MLPSubmodules
        from megatron.core.transformer.moe.experts import InferenceGroupedMLP
        from megatron.core.transformer.transformer_config import TransformerConfig

        self.num_experts = 4
        self.hidden_size = 128
        self.dtype = torch.bfloat16

        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=False,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=self.dtype,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            moe_grouped_gemm=True,
            batch_invariant_mode=True,
            attention_backend=AttnBackend.flash,
        )

        submodules = MLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'ep', 'expt_tp', 'expt_dp']
        )

        self.mlp = (
            InferenceGroupedMLP(
                num_local_experts=self.num_experts,
                config=self.config,
                submodules=submodules,
                pg_collection=pg_collection,
            )
            .cuda()
            .eval()
        )

        self.ffn_hidden = self.config.ffn_hidden_size

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    def test_batch_invariant_forward_matches_te_forward(self):
        """Batch-invariant Triton path should produce similar results to TE's path."""
        torch.manual_seed(100)
        tokens_per_expert = torch.tensor([8, 12, 4, 16], device="cuda", dtype=torch.int64)
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(total_tokens, self.hidden_size, device="cuda", dtype=self.dtype)
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        # Run with TE path (parent forward)
        from megatron.core.transformer.moe.experts import TEGroupedMLP

        with torch.no_grad():
            out_te, _ = TEGroupedMLP.forward(self.mlp, hidden_states, tokens_per_expert, probs)

        # Run with batch-invariant path
        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out_bi, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        torch.testing.assert_close(out_bi, out_te, atol=1e-1, rtol=1e-1)

    def test_batch_invariant_forward_permutation(self):
        """Permuting expert order should produce identical per-token outputs
        when using the batch-invariant forward path."""
        torch.manual_seed(200)
        tokens_per_expert = torch.tensor([10, 6, 14, 8], device="cuda", dtype=torch.int64)
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(total_tokens, self.hidden_size, device="cuda", dtype=self.dtype)
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        # Run 1: original order
        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out1, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        # Run 2: permute expert ordering in the batch
        perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        offsets = torch.zeros(self.num_experts + 1, dtype=torch.int64)
        offsets[1:] = tokens_per_expert.cpu().cumsum(0)

        perm_chunks = [hidden_states[offsets[e] : offsets[e + 1]] for e in perm.tolist()]
        perm_hidden = torch.cat(perm_chunks, dim=0)
        perm_tpe = tokens_per_expert[perm]

        # We need to also permute the weights to match
        # Save and permute fc1/fc2 weights
        orig_fc1 = self.mlp._fc1_weight.data.clone()
        orig_fc2 = self.mlp._fc2_weight.data.clone()

        self.mlp._fc1_weight.data.copy_(orig_fc1[perm])
        self.mlp._fc2_weight.data.copy_(orig_fc2[perm])

        perm_probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out2_perm, _ = self.mlp._triton_batch_invariant_forward(
                    perm_hidden, perm_tpe, perm_probs
                )

        # Restore weights
        self.mlp._fc1_weight.data.copy_(orig_fc1)
        self.mlp._fc2_weight.data.copy_(orig_fc2)

        # Un-permute outputs
        perm_offsets = torch.zeros(self.num_experts + 1, dtype=torch.int64)
        perm_offsets[1:] = perm_tpe.cpu().cumsum(0)

        out2_chunks = [
            out2_perm[perm_offsets[i] : perm_offsets[i + 1]] for i in range(self.num_experts)
        ]
        # out2_chunks[i] corresponds to expert perm[i]
        reordered = [None] * self.num_experts
        for i in range(self.num_experts):
            reordered[perm[i].item()] = out2_chunks[i]
        out2 = torch.cat(reordered, dim=0)

        assert torch.equal(out1, out2), (
            f"InferenceGroupedMLP batch invariance violated: "
            f"max diff = {(out1 - out2).abs().max().item()}"
        )

    def test_dispatch_uses_batch_invariant_path(self):
        """When batch_invariant_mode is enabled, forward() should dispatch to
        the Triton batch-invariant path (not torch._grouped_mm or TE)."""
        torch.manual_seed(300)
        tokens_per_expert = torch.tensor([4, 4, 4, 4], device="cuda", dtype=torch.int64)
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(total_tokens, self.hidden_size, device="cuda", dtype=self.dtype)
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        # Run via the direct batch-invariant method
        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out_direct, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        # Run via forward() dispatch (should use batch-invariant path)
        self.mlp.training = False
        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out_dispatch, _ = self.mlp.forward(hidden_states, tokens_per_expert, probs)

        assert torch.equal(
            out_direct, out_dispatch
        ), "forward() dispatch did not use batch-invariant path"


# ============================================================================
# Expert parallel integration tests (multi-GPU)
# ============================================================================


def _build_inference_grouped_mlp(num_moe_experts, hidden_size, dtype, pg_collection):
    """Helper to build an InferenceGroupedMLP with the given config and process groups."""
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    ep_size = parallel_state.get_expert_model_parallel_world_size()
    num_local_experts = num_moe_experts // ep_size

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_moe_experts=num_moe_experts,
        expert_model_parallel_size=ep_size,
        use_cpu_initialization=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=dtype,
        moe_router_load_balancing_type="sinkhorn",
        moe_router_topk=1,
        moe_grouped_gemm=True,
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
    )

    submodules = MLPSubmodules(
        linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
    )

    mlp = (
        InferenceGroupedMLP(
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
        )
        .cuda()
        .eval()
    )
    return mlp, config, num_local_experts


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_te_min_version("1.9.0.dev0")),
    reason="Requires CUDA and TE >= 1.9.0.dev0 for TEGroupedLinear",
)
class TestInferenceGroupedMLPExpertParallel:
    """Test the batch-invariant path with expert parallelism (EP > 1).

    These tests require multiple GPUs and validate that each EP rank
    independently computes correct batch-invariant results for its
    local subset of experts.
    """

    def setup_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, expert_model_parallel_size=2
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(42)

        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

        self.pg_collection = get_default_pg_collection()
        self.num_moe_experts = 8
        self.hidden_size = 128
        self.dtype = torch.bfloat16
        self.ep_size = parallel_state.get_expert_model_parallel_world_size()
        self.ep_rank = parallel_state.get_expert_model_parallel_rank()

        self.mlp, self.config, self.num_local_experts = _build_inference_grouped_mlp(
            self.num_moe_experts, self.hidden_size, self.dtype, self.pg_collection
        )

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_ep_local_expert_correctness(self):
        """Each EP rank's batch-invariant forward should match the TE forward
        for its local experts."""
        from megatron.core.transformer.moe.experts import TEGroupedMLP

        torch.manual_seed(100 + self.ep_rank)
        tokens_per_expert = torch.tensor(
            [8, 6, 10, 4], device="cuda", dtype=torch.int64
        )[: self.num_local_experts]
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            out_te, _ = TEGroupedMLP.forward(
                self.mlp, hidden_states, tokens_per_expert, probs
            )

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out_bi, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        torch.testing.assert_close(out_bi, out_te, atol=1e-1, rtol=1e-1)

    @pytest.mark.internal
    def test_ep_batch_invariance(self):
        """Batch invariance should hold independently on each EP rank:
        permuting local expert order produces identical per-token outputs."""
        torch.manual_seed(200 + self.ep_rank)
        num_local = self.num_local_experts
        tokens_per_expert = torch.randint(
            4, 17, (num_local,), device="cuda", dtype=torch.int64
        )
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        # Run 1: original order
        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out1, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        # Run 2: permute local expert order
        perm = torch.randperm(num_local)
        offsets = torch.zeros(num_local + 1, dtype=torch.int64)
        offsets[1:] = tokens_per_expert.cpu().cumsum(0)

        perm_chunks = [
            hidden_states[offsets[e] : offsets[e + 1]] for e in perm.tolist()
        ]
        perm_hidden = torch.cat(perm_chunks, dim=0)
        perm_tpe = tokens_per_expert[perm]

        orig_fc1 = self.mlp._fc1_weight.data.clone()
        orig_fc2 = self.mlp._fc2_weight.data.clone()
        self.mlp._fc1_weight.data.copy_(orig_fc1[perm])
        self.mlp._fc2_weight.data.copy_(orig_fc2[perm])

        perm_probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out2_perm, _ = self.mlp._triton_batch_invariant_forward(
                    perm_hidden, perm_tpe, perm_probs
                )

        # Restore weights
        self.mlp._fc1_weight.data.copy_(orig_fc1)
        self.mlp._fc2_weight.data.copy_(orig_fc2)

        # Un-permute outputs
        perm_offsets = torch.zeros(num_local + 1, dtype=torch.int64)
        perm_offsets[1:] = perm_tpe.cpu().cumsum(0)

        out2_chunks = [
            out2_perm[perm_offsets[i] : perm_offsets[i + 1]]
            for i in range(num_local)
        ]
        reordered = [None] * num_local
        for i in range(num_local):
            reordered[perm[i].item()] = out2_chunks[i]
        out2 = torch.cat(reordered, dim=0)

        assert torch.equal(out1, out2), (
            f"EP rank {self.ep_rank}: batch invariance violated, "
            f"max diff = {(out1 - out2).abs().max().item()}"
        )

    @pytest.mark.internal
    def test_ep_determinism_across_runs(self):
        """Multiple runs on each EP rank must produce bitwise-identical results."""
        torch.manual_seed(300 + self.ep_rank)
        tokens_per_expert = torch.tensor(
            [6, 10, 8, 12], device="cuda", dtype=torch.int64
        )[: self.num_local_experts]
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        results = []
        for _ in range(5):
            with torch.no_grad():
                with set_batch_invariant_mode(True):
                    out, _ = self.mlp._triton_batch_invariant_forward(
                        hidden_states, tokens_per_expert, probs
                    )
            results.append(out.clone())

        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), (
                f"EP rank {self.ep_rank}: run 0 vs run {i}, "
                f"max diff = {(results[0] - results[i]).abs().max().item()}"
            )

    @pytest.mark.internal
    def test_ep_dispatch_uses_batch_invariant_path(self):
        """forward() should dispatch to the batch-invariant Triton path on each
        EP rank when batch_invariant_mode is enabled."""
        torch.manual_seed(400 + self.ep_rank)
        tokens_per_expert = torch.tensor(
            [4, 4, 4, 4], device="cuda", dtype=torch.int64
        )[: self.num_local_experts]
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out_direct, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        self.mlp.training = False
        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out_dispatch, _ = self.mlp.forward(
                    hidden_states, tokens_per_expert, probs
                )

        assert torch.equal(out_direct, out_dispatch), (
            f"EP rank {self.ep_rank}: forward() dispatch did not use "
            f"batch-invariant path"
        )

    @pytest.mark.internal
    def test_ep_ranks_produce_independent_results(self):
        """Verify that EP ranks with different local experts produce different
        outputs for the same input tokens (sanity check that expert weights
        are actually partitioned)."""
        torch.manual_seed(500)
        tokens_per_expert = torch.tensor(
            [8, 8, 8, 8], device="cuda", dtype=torch.int64
        )[: self.num_local_experts]
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                local_out, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        # Gather outputs from all EP ranks to rank 0
        ep_group = parallel_state.get_expert_model_parallel_group()
        gathered = [torch.zeros_like(local_out) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered, local_out, group=ep_group)

        # Each rank should produce different outputs (different local expert weights)
        if self.ep_rank == 0:
            for i in range(1, self.ep_size):
                assert not torch.equal(gathered[0], gathered[i]), (
                    f"EP rank 0 and rank {i} produced identical outputs — "
                    f"experts may not be properly partitioned"
                )

    @pytest.mark.internal
    def test_ep_empty_local_experts(self):
        """Each EP rank should handle zero tokens for some local experts gracefully."""
        torch.manual_seed(600 + self.ep_rank)
        # Create tokens_per_expert with some zeros
        tokens_per_expert = torch.zeros(
            self.num_local_experts, device="cuda", dtype=torch.int64
        )
        # Assign tokens to only the first local expert
        tokens_per_expert[0] = 16
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        assert out.shape == (total_tokens, self.hidden_size)
        assert not torch.isnan(out).any(), "NaN in output with sparse token assignment"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_te_min_version("1.9.0.dev0")),
    reason="Requires CUDA and TE >= 1.9.0.dev0 for TEGroupedLinear",
)
class TestInferenceGroupedMLPExpertParallelWithTP:
    """Test the batch-invariant path with combined expert and tensor parallelism
    (EP=2, TP=2, requiring 4 GPUs)."""

    def setup_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, expert_model_parallel_size=2
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(42)

        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

        self.pg_collection = get_default_pg_collection()
        self.num_moe_experts = 8
        self.hidden_size = 128
        self.dtype = torch.bfloat16
        self.ep_size = parallel_state.get_expert_model_parallel_world_size()
        self.ep_rank = parallel_state.get_expert_model_parallel_rank()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()

        self.mlp, self.config, self.num_local_experts = _build_inference_grouped_mlp(
            self.num_moe_experts, self.hidden_size, self.dtype, self.pg_collection
        )

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_tp_ep_local_expert_correctness(self):
        """With TP=2, EP=2: each rank's batch-invariant forward should match TE."""
        from megatron.core.transformer.moe.experts import TEGroupedMLP

        torch.manual_seed(100 + self.ep_rank)
        tokens_per_expert = torch.tensor(
            [8, 6, 10, 4], device="cuda", dtype=torch.int64
        )[: self.num_local_experts]
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            out_te, _ = TEGroupedMLP.forward(
                self.mlp, hidden_states, tokens_per_expert, probs
            )

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out_bi, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        torch.testing.assert_close(out_bi, out_te, atol=1e-1, rtol=1e-1)

    @pytest.mark.internal
    def test_tp_ep_batch_invariance(self):
        """With TP=2, EP=2: batch invariance should hold on each rank."""
        torch.manual_seed(200 + self.ep_rank)
        num_local = self.num_local_experts
        tokens_per_expert = torch.randint(
            4, 17, (num_local,), device="cuda", dtype=torch.int64
        )
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out1, _ = self.mlp._triton_batch_invariant_forward(
                    hidden_states, tokens_per_expert, probs
                )

        # Permute local expert order
        perm = torch.randperm(num_local)
        offsets = torch.zeros(num_local + 1, dtype=torch.int64)
        offsets[1:] = tokens_per_expert.cpu().cumsum(0)

        perm_chunks = [
            hidden_states[offsets[e] : offsets[e + 1]] for e in perm.tolist()
        ]
        perm_hidden = torch.cat(perm_chunks, dim=0)
        perm_tpe = tokens_per_expert[perm]

        orig_fc1 = self.mlp._fc1_weight.data.clone()
        orig_fc2 = self.mlp._fc2_weight.data.clone()
        self.mlp._fc1_weight.data.copy_(orig_fc1[perm])
        self.mlp._fc2_weight.data.copy_(orig_fc2[perm])

        perm_probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                out2_perm, _ = self.mlp._triton_batch_invariant_forward(
                    perm_hidden, perm_tpe, perm_probs
                )

        self.mlp._fc1_weight.data.copy_(orig_fc1)
        self.mlp._fc2_weight.data.copy_(orig_fc2)

        perm_offsets = torch.zeros(num_local + 1, dtype=torch.int64)
        perm_offsets[1:] = perm_tpe.cpu().cumsum(0)

        out2_chunks = [
            out2_perm[perm_offsets[i] : perm_offsets[i + 1]]
            for i in range(num_local)
        ]
        reordered = [None] * num_local
        for i in range(num_local):
            reordered[perm[i].item()] = out2_chunks[i]
        out2 = torch.cat(reordered, dim=0)

        assert torch.equal(out1, out2), (
            f"TP+EP rank (ep={self.ep_rank}): batch invariance violated, "
            f"max diff = {(out1 - out2).abs().max().item()}"
        )

    @pytest.mark.internal
    def test_tp_ep_determinism(self):
        """With TP=2, EP=2: repeated runs must be bitwise-identical."""
        torch.manual_seed(300 + self.ep_rank)
        tokens_per_expert = torch.tensor(
            [6, 10, 8, 12], device="cuda", dtype=torch.int64
        )[: self.num_local_experts]
        total_tokens = tokens_per_expert.sum().item()
        hidden_states = torch.randn(
            total_tokens, self.hidden_size, device="cuda", dtype=self.dtype
        )
        probs = torch.ones(total_tokens, device="cuda", dtype=torch.float32)

        results = []
        for _ in range(5):
            with torch.no_grad():
                with set_batch_invariant_mode(True):
                    out, _ = self.mlp._triton_batch_invariant_forward(
                        hidden_states, tokens_per_expert, probs
                    )
            results.append(out.clone())

        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), (
                f"TP+EP rank (ep={self.ep_rank}): run 0 vs run {i}, "
                f"max diff = {(results[0] - results[i]).abs().max().item()}"
            )
