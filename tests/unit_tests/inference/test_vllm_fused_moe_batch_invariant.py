# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for batch-invariant mode on the vLLM Triton fused-MoE backend.

Covers:
- CUDA-graph bucket token counts floored to 64-multiples under batch-invariant mode
- swiglu_with_probs / weighted_silu_mul_bounded: training-parity rounding vs reference
- _moe_sum apply_weights / acc_fp64 options
- vllm_fused_moe end-to-end batch invariance for gated (SwiGLU) models
- te_native backend registration
"""

import os
import tempfile

os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton_test_cache"))

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import CUDAGraphBatchDimensionBuilder
from megatron.core.inference.moe.batch_invariant import HAVE_TRITON
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    _BATCH_INVARIANT_BACKENDS,
    set_batch_invariant_mode,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON,
    reason="batch-invariant MoE kernels require CUDA and Triton",
)


def _vt(n):
    return torch.tensor(n, dtype=torch.int32, device="cuda")


# ---------------------------------------------------------------------------
# CUDA-graph bucket 64-multiple floor
# ---------------------------------------------------------------------------


class TestCudaGraphBucket64Floor:

    @pytest.mark.parametrize("num_cuda_graphs", [-1, 8, 16])
    def test_bucket_token_counts_are_64_multiples(self, num_cuda_graphs):
        with set_batch_invariant_mode(True, backend="triton"):
            dims, token_counts = (
                CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
                    tp_size=1,
                    num_cuda_graphs=num_cuda_graphs,
                    cuda_graph_max_tokens=2048,
                    cuda_graph_mixed_prefill_request_count=None,
                    max_requests=512,
                    max_tokens=2048,
                    max_sequence_length=4096,
                    use_cuda_graphs_for_non_decode_steps=False,
                )
            )
        for bd in dims:
            assert bd.token_count % 64 == 0 and bd.token_count >= 64, (
                f"bucket token_count {bd.token_count} violates the 64-multiple floor "
                f"(num_cuda_graphs={num_cuda_graphs})"
            )

    def test_non_multiple_max_requests_keeps_largest_decode_bucket(self):
        # Regression: the floor must PAD buckets, not invalidate them. With
        # max_requests=100 (not a 64-multiple), the largest decode bucket must
        # survive with its request budget intact and an aligned token count.
        with set_batch_invariant_mode(True, backend="triton"):
            dims, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
                tp_size=1,
                num_cuda_graphs=16,
                cuda_graph_max_tokens=2048,
                cuda_graph_mixed_prefill_request_count=None,
                max_requests=100,
                max_tokens=2048,
                max_sequence_length=4096,
                use_cuda_graphs_for_non_decode_steps=False,
            )
        decode_dims = [d for d in dims if d.prefill_req_count == 0 and d.decode_req_count > 0]
        assert decode_dims, "no decode buckets survived"
        largest = max(decode_dims, key=lambda d: d.decode_req_count)
        assert (
            largest.decode_req_count == 100
        ), f"largest decode bucket lost its request budget: {largest}"
        assert (
            largest.token_count % 64 == 0 and largest.token_count >= 128
        ), f"largest decode bucket not aligned/padded: {largest}"

    def test_auto_sizing_injects_small_buckets_without_bi(self):
        # Guard for the non-BI behavior this floor exists to counteract: the
        # auto (-1) ladder includes 1/2-token buckets when BI mode is off.
        dims, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=1,
            num_cuda_graphs=-1,
            cuda_graph_max_tokens=2048,
            cuda_graph_mixed_prefill_request_count=None,
            max_requests=512,
            max_tokens=2048,
            max_sequence_length=4096,
            use_cuda_graphs_for_non_decode_steps=False,
        )
        assert any(bd.token_count < 64 for bd in dims)


# ---------------------------------------------------------------------------
# Training-parity weighted SwiGLU kernels
# ---------------------------------------------------------------------------


def _weighted_swiglu_reference(y, probs_flat):
    """bf16(fp32 silu(gate) * up * prob) with a single final rounding."""
    half = y.shape[1] // 2
    gate = y[:, :half].float()
    up = y[:, half:].float()
    return (torch.nn.functional.silu(gate) * up * probs_flat[:, None]).to(y.dtype)


class TestWeightedSwigluKernels:

    def test_swiglu_with_probs_value_deterministic_and_row_local(self):
        from megatron.core.inference.moe import batch_invariant

        torch.manual_seed(7)
        rows, ffn = 512, 256
        y = (torch.randn(rows, 2 * ffn, device="cuda") * 2.0).bfloat16()
        probs = torch.rand(rows, device="cuda", dtype=torch.float32)
        perm_map = torch.arange(rows, device="cuda", dtype=torch.int32)
        n_used = _vt(rows)
        out = batch_invariant.swiglu_with_probs(y, perm_map, n_used, probs)
        # value correctness (tolerance-based: sigmoid instruction sequences may
        # legitimately differ from the torch reference by 1 ulp on rare values)
        # compare at bf16 (bf16 tolerances): sigmoid instruction sequences may
        # legitimately differ from the torch reference by 1 bf16 ulp
        torch.testing.assert_close(out, _weighted_swiglu_reference(y, probs))
        # bitwise repeat-determinism
        for _ in range(5):
            assert torch.equal(batch_invariant.swiglu_with_probs(y, perm_map, n_used, probs), out)
        # row-locality: a row's bits do not depend on co-batch size
        half_out = batch_invariant.swiglu_with_probs(
            y[:128].contiguous(), perm_map[:128], _vt(128), probs[:128]
        )
        assert torch.equal(half_out, out[:128])

    def test_weighted_silu_mul_bounded_bound_and_invariance(self):
        from megatron.core.inference.moe import batch_invariant

        torch.manual_seed(8)
        rows, ffn, live = 512, 256, 300
        y = (torch.randn(rows, 2 * ffn, device="cuda") * 2.0).bfloat16()
        probs = torch.rand(rows, device="cuda", dtype=torch.float32)
        bound = torch.tensor(live * ffn, dtype=torch.int64, device="cuda")
        out = batch_invariant.weighted_silu_mul_bounded(y, probs, bound)
        torch.testing.assert_close(out[:live], _weighted_swiglu_reference(y[:live], probs[:live]))
        # rows beyond the device bound are neither read nor written: NaN-poison
        # the tail and require the live rows to stay BITWISE identical
        y2 = y.clone()
        y2[live:] = float("nan")
        out2 = batch_invariant.weighted_silu_mul_bounded(y2, probs, bound)
        assert torch.equal(out2[:live], out[:live])

    def test_weighted_silu_mul_bounded_grid_size_is_bit_inert(self):
        # num_programs is an occupancy knob only: the kernel is elementwise
        # with disjoint per-program index ranges, so any grid size must give
        # bitwise-identical output (incl. 1184 = the B200 Inductor capture).
        from megatron.core.inference.moe import batch_invariant

        torch.manual_seed(11)
        rows, ffn, live = 512, 256, 300
        y = (torch.randn(rows, 2 * ffn, device="cuda") * 2.0).bfloat16()
        probs = torch.rand(rows, device="cuda", dtype=torch.float32)
        bound = torch.tensor(live * ffn, dtype=torch.int64, device="cuda")
        ref = batch_invariant.weighted_silu_mul_bounded(y, probs, bound)  # derived default
        for np_ in (1, 148, 1184, 4096):
            out = batch_invariant.weighted_silu_mul_bounded(y, probs, bound, num_programs=np_)
            assert torch.equal(out[:live], ref[:live]), f"bits changed at num_programs={np_}"


# ---------------------------------------------------------------------------
# _moe_sum options
# ---------------------------------------------------------------------------


class TestMoeSumOptions:

    def _setup(self):
        torch.manual_seed(9)
        max_tokens, topk, K, E = 64, 4, 128, 8
        inp = (torch.randn(max_tokens * topk, K, device="cuda")).bfloat16()
        probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing = torch.randint(0, E, (max_tokens, topk), device="cuda", dtype=torch.int64)
        return inp, probs, routing, max_tokens, topk, K, E

    def test_unit_weights_fp64_matches_fp64_reference(self):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        inp, probs, routing, max_tokens, topk, K, E = self._setup()
        out = _moe_sum(
            inp,
            probs,
            max_tokens,
            topk,
            K,
            _vt(max_tokens),
            routing,
            0,
            E,
            apply_weights=False,
            acc_fp64=True,
        )
        ref = inp.view(max_tokens, topk, K).to(torch.float64).sum(dim=1).to(torch.float32)
        assert torch.equal(out, ref)

    def test_unit_weights_fp32_matches_sequential_reference(self):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        inp, probs, routing, max_tokens, topk, K, E = self._setup()
        out = _moe_sum(
            inp,
            probs,
            max_tokens,
            topk,
            K,
            _vt(max_tokens),
            routing,
            0,
            E,
            apply_weights=False,
            acc_fp64=False,
        )
        # unit weights => pure sequential fp32 adds (no FMA), so a same-order
        # torch reference is bitwise-reproducible
        ref = torch.zeros(max_tokens, K, device="cuda", dtype=torch.float32)
        for t in range(topk):
            ref += inp.view(max_tokens, topk, K)[:, t].float()
        assert torch.equal(out, ref)

    def test_default_weighted_fp32_deterministic_and_correct(self):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        inp, probs, routing, max_tokens, topk, K, E = self._setup()
        out = _moe_sum(inp, probs, max_tokens, topk, K, _vt(max_tokens), routing, 0, E)
        # value correctness (tolerance-based: the kernel's acc += v*w compiles
        # to FMA, which a mul-then-add torch reference cannot match bitwise)
        ref = torch.zeros(max_tokens, K, device="cuda", dtype=torch.float32)
        for t in range(topk):
            ref += inp.view(max_tokens, topk, K)[:, t].float() * probs[:, t : t + 1]
        torch.testing.assert_close(out, ref)
        # bitwise repeat-determinism of the default path
        for _ in range(5):
            assert torch.equal(
                _moe_sum(inp, probs, max_tokens, topk, K, _vt(max_tokens), routing, 0, E), out
            )


# ---------------------------------------------------------------------------
# End-to-end batch invariance (gated / SwiGLU)
# ---------------------------------------------------------------------------


class TestVllmFusedMoeBatchInvariance:

    def test_same_tokens_bitwise_across_cobatch_sizes(self):
        from megatron.core.inference.moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        torch.manual_seed(11)
        max_tokens, K, ffn, E, topk = 256, 128, 64, 8, 4
        hidden = (torch.randn(max_tokens, K, device="cuda") * 0.05).bfloat16()
        fc1 = (torch.randn(E, 2 * ffn, K, device="cuda") * 0.02).bfloat16()
        fc2 = (torch.randn(E, K, ffn, device="cuda") * 0.02).bfloat16()
        routing = torch.stack(
            [torch.randperm(E, device="cuda")[:topk] for _ in range(max_tokens)]
        ).long()
        probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        probs = probs / probs.sum(-1, keepdim=True)

        def run(valid, hint):
            with set_batch_invariant_mode(True, backend="triton"):
                return vllm_fused_moe(
                    hidden,
                    probs,
                    fc1,
                    fc2,
                    activation_type=ActivationType.SWIGLU,
                    num_local_experts=E,
                    local_expert_start=0,
                    valid_tokens=_vt(valid),
                    routing_map=routing,
                    num_tokens_hint=hint,
                )

        # the first 64 tokens, computed in co-batches of different sizes and
        # hint classes, must be bitwise identical
        small = run(valid=64, hint=64)[:64]
        large = run(valid=256, hint=256)[:64]
        assert torch.equal(small, large)


# ---------------------------------------------------------------------------
# te_native backend registration
# ---------------------------------------------------------------------------


class TestTeNativeBackend:

    def test_backend_registered(self):
        assert "te_native" in _BATCH_INVARIANT_BACKENDS

    def test_enable_disable_roundtrip(self):
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            disable_batch_invariant_mode,
            enable_batch_invariant_mode,
            get_batch_invariant_backend,
            is_batch_invariant_mode_enabled,
        )

        try:
            import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm_mod

            ws_fn_before = te_gemm_mod.get_cublas_workspace_size_bytes
            have_te = True
        except ImportError:
            have_te = False
        env_before = os.environ.get("CUBLASLT_WORKSPACE_SIZE")
        try:
            enable_batch_invariant_mode("te_native")
            assert is_batch_invariant_mode_enabled()
            assert get_batch_invariant_backend() == "te_native"
            assert os.environ.get("CUBLASLT_WORKSPACE_SIZE") == "0"
            if have_te:
                assert te_gemm_mod.get_cublas_workspace_size_bytes() == 1024
            # te_native must NOT reroute aten::mm — native kernels stay
            a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
            torch.mm(a, b)  # should not raise / not require DeepGEMM
        finally:
            disable_batch_invariant_mode()
        assert not is_batch_invariant_mode_enabled()
        # the workspace patch and env pin must be fully restored (no leak
        # into subsequent non-BI work in the same process)
        if have_te:
            assert te_gemm_mod.get_cublas_workspace_size_bytes is ws_fn_before
        assert os.environ.get("CUBLASLT_WORKSPACE_SIZE") == env_before
