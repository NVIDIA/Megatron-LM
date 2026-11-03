# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for batch-invariant mode on the vLLM Triton fused-MoE backend.

Covers:
- CUDA-graph bucket token counts floored to 64-multiples under batch-invariant mode
- swiglu_with_probs / weighted_silu_mul_bounded: training-parity rounding vs reference
- _moe_sum apply_weights / acc_fp64 options
- vllm_fused_moe end-to-end batch invariance for SwiGLU and squared-ReLU models
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

    def test_swiglu_with_probs_matches_fused_training_order(self):
        """Inference must reproduce TEGroupedMLP's weighted SwiGLU fusion exactly."""
        from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
        from megatron.core.inference.moe import batch_invariant

        torch.manual_seed(6)
        rows, ffn = 256, 128
        y = (torch.randn(rows, 2 * ffn, device="cuda") * 2.0).bfloat16()
        probs = torch.rand(rows, device="cuda", dtype=torch.float32)
        perm_map = torch.arange(rows, device="cuda", dtype=torch.int32)
        inference = batch_invariant.swiglu_with_probs(y, perm_map, _vt(rows), probs)

        training = weighted_bias_swiglu_impl(y, None, probs[:, None])

        assert torch.equal(training, inference), (
            "fused training and inference SwiGLU differ; max abs diff: "
            f"{(training.float() - inference.float()).abs().max().item()}"
        )

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

    def test_weighted_silu_mul_bounded_large_input_offset(self):
        from megatron.core.inference.moe import batch_invariant

        # The final input row begins at element 2**31.
        rows, ffn = 262145, 4096
        if torch.cuda.mem_get_info()[0] < 6.5 * 1024**3:
            pytest.skip("requires 6.5 GiB of free GPU memory")
        y = torch.zeros((rows, 2 * ffn), device="cuda", dtype=torch.bfloat16)
        probs = torch.ones(rows, device="cuda", dtype=torch.float32)
        y[-1] = 1
        bound = torch.tensor(rows * ffn, device="cuda", dtype=torch.int64)
        expected = _weighted_swiglu_reference(y[-1:], probs[-1:])[0]
        out = batch_invariant.weighted_silu_mul_bounded(y, probs, bound)
        assert torch.equal(out[-1], expected)


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


# End-to-end batch invariance across physical shapes and routing layouts


def _assert_target_variation_witnesses(witnesses):
    """Require physical and routing variation in target-invariance rows."""
    physical_shapes = {
        (w["hidden_shape"], w["routing_shape"]) for w in witnesses if w["position"] == "front"
    }
    assert physical_shapes == {
        ((64, 128), (64, 2)),
        ((128, 128), (128, 2)),
        ((256, 128), (256, 2)),
    }, f"physical shape witness did not cover 64/128/256 rows: {physical_shapes}"

    by_size = {w["hidden_shape"][0]: w for w in witnesses if w["position"] == "front"}
    load_signatures = {w["expert_loads"] for w in by_size.values()}
    offset_signatures = {w["target_segment_offsets"] for w in by_size.values()}
    assert len(load_signatures) > 1 or len(offset_signatures) > 1, (
        "routing/load witness did not change across physical batch shapes: "
        f"loads={load_signatures}, offsets={offset_signatures}"
    )
    for witness in witnesses:
        loads = witness["expert_loads"]
        assert max(loads) > min(loads), f"expert load is not unequal: {loads}"

    by_layout = {(w["hidden_shape"][0], w["position"]): w for w in witnesses}
    assert len(by_layout) == len(witnesses) == 6
    for physical_rows in (64, 128, 256):
        front, back = (by_layout[physical_rows, position] for position in ("front", "back"))
        assert front["target_rows"] == tuple(range(8))
        assert back["target_rows"] == tuple(range(physical_rows - 8, physical_rows))


class TestVllmFusedMoeBatchInvariance:

    @pytest.mark.parametrize("activation_name", ["SWIGLU", "SQUARED_RELU"])
    def test_target_bitwise_across_physical_shapes_and_routing(self, monkeypatch, activation_name):
        import importlib

        from megatron.core.inference.moe import ActivationType
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            get_batch_invariant_backend,
            get_batch_invariant_collective,
            is_batch_invariant_mode_enabled,
        )

        vllm_module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
        original_invoke = vllm_module._invoke_fused_moe_kernel
        kernel_calls = []

        def invoke_with_witness(*args, **kwargs):
            result = original_invoke(*args, **kwargs)
            # Synchronize the real production launch before witnessing its device-built routing.
            torch.cuda.synchronize()
            block_m = kwargs["config"]["BLOCK_SIZE_M"]
            num_post_padded = int(args[6].item())
            num_blocks = (num_post_padded + block_m - 1) // block_m
            kernel_calls.append(
                {
                    "a_shape": tuple(args[0].shape),
                    "b_shape": tuple(args[1].shape),
                    "top_k": kwargs["top_k"],
                    "grid_size": kwargs["grid_size"],
                    "block_m": block_m,
                    "sorted_ids": args[4][:num_post_padded].cpu(),
                    "expert_ids": args[5][:num_blocks].cpu(),
                }
            )
            return result

        monkeypatch.setattr(vllm_module, "_invoke_fused_moe_kernel", invoke_with_witness)

        target_count, hidden_size, ffn, num_experts, topk = 8, 128, 64, 8, 2
        activation_type = getattr(ActivationType, activation_name)
        fc1_width = 2 * ffn if activation_type == ActivationType.SWIGLU else ffn
        witnesses = []
        target_outputs = []

        # BI vLLM FC1/FC2 use Triton regardless of the dense-GEMM backend, so this is one MoE row.
        # Dense TE contrasts live in dynamic model tests; registration below is separate.
        backend = "triton"
        assert not is_batch_invariant_mode_enabled(), "backend state leaked from another row"
        with set_batch_invariant_mode(True, backend=backend):
            assert get_batch_invariant_backend() == backend
            assert get_batch_invariant_collective() == "ordered"
            torch.manual_seed(11)
            target_hidden = (
                torch.randn(target_count, hidden_size, device="cuda") * 0.05
            ).bfloat16()
            target_probs = torch.tensor(
                [[0.25, 0.75], [0.4, 0.6], [0.6, 0.4], [0.75, 0.25]] * 2,
                device="cuda",
                dtype=torch.float32,
            )
            target_routing = torch.tensor(
                [[4, 6], [5, 7], [4, 7], [5, 6]] * 2, device="cuda", dtype=torch.long
            )

            max_cobatch = 256 - target_count
            cobatch_hidden = (
                torch.randn(max_cobatch, hidden_size, device="cuda") * 0.05
            ).bfloat16()
            cobatch_probs = torch.rand(max_cobatch, topk, device="cuda", dtype=torch.float32)
            cobatch_probs /= cobatch_probs.sum(-1, keepdim=True)
            cobatch_index = torch.arange(max_cobatch, device="cuda")
            first_expert = cobatch_index % num_experts
            # Skew the second route toward expert 0, keeping two distinct experts per co-batch row.
            second_expert = torch.where(first_expert == 0, 1, 0)
            cobatch_routing = torch.stack((first_expert, second_expert), dim=1).long()
            weight_options = dict(device="cuda", dtype=torch.bfloat16)
            fc1 = torch.randn(num_experts, fc1_width, hidden_size, **weight_options) * 0.02
            fc2 = torch.randn(num_experts, hidden_size, ffn, **weight_options) * 0.02

            for physical_rows in (64, 128, 256):
                cobatch_count = physical_rows - target_count
                permutation = torch.roll(
                    torch.arange(cobatch_count, device="cuda").flip(0), shifts=physical_rows // 8
                )
                co_hidden = cobatch_hidden[:cobatch_count].index_select(0, permutation)
                co_probs = cobatch_probs[:cobatch_count].index_select(0, permutation)
                co_routing = cobatch_routing[:cobatch_count].index_select(0, permutation)

                for position in ("front", "back"):
                    hidden, probs, routing = (
                        torch.cat(
                            (target, neighbors) if position == "front" else (neighbors, target)
                        )
                        for target, neighbors in zip(
                            (target_hidden, target_probs, target_routing),
                            (co_hidden, co_probs, co_routing),
                        )
                    )
                    start = 0 if position == "front" else physical_rows - target_count
                    target_rows = torch.arange(start, start + target_count, device="cuda")

                    assert hidden.shape == (physical_rows, hidden_size)
                    assert routing.shape == (physical_rows, topk)
                    assert torch.equal(hidden.index_select(0, target_rows), target_hidden)
                    assert torch.equal(probs.index_select(0, target_rows), target_probs)
                    assert torch.equal(routing.index_select(0, target_rows), target_routing)

                    call_start = len(kernel_calls)
                    output = vllm_module.vllm_fused_moe(
                        hidden,
                        probs,
                        fc1,
                        fc2,
                        activation_type=activation_type,
                        num_local_experts=num_experts,
                        local_expert_start=0,
                        valid_tokens=_vt(physical_rows),
                        routing_map=routing,
                        num_tokens_hint=physical_rows,
                    )
                    target_outputs.append(output.index_select(0, target_rows).clone())
                    run_calls = kernel_calls[call_start:]
                    assert len(run_calls) == 2, "FC1 and FC2 production kernels must both execute"
                    assert [call["top_k"] for call in run_calls] == [topk, 1]
                    assert run_calls[0]["a_shape"] == (physical_rows, hidden_size)
                    assert run_calls[1]["a_shape"] == (physical_rows * topk, ffn)

                    fc1_call = run_calls[0]
                    sorted_ids = fc1_call["sorted_ids"]
                    live_ids = sorted_ids[sorted_ids < physical_rows * topk].long()
                    assert live_ids.numel() == physical_rows * topk
                    actual_loads = torch.bincount(
                        routing.reshape(-1).cpu().index_select(0, live_ids), minlength=num_experts
                    )
                    expected_loads = torch.bincount(
                        routing.reshape(-1).cpu(), minlength=num_experts
                    )
                    assert torch.equal(actual_loads, expected_loads)

                    target_pair_ids = (
                        target_rows[:, None] * topk + torch.arange(topk, device="cuda")[None, :]
                    ).reshape(-1)
                    assert set(target_pair_ids.cpu().tolist()).issubset(set(live_ids.tolist()))
                    expert_ids = fc1_call["expert_ids"]
                    target_segment_offsets = []
                    for expert in sorted(set(target_routing.reshape(-1).cpu().tolist())):
                        expert_blocks = torch.nonzero(expert_ids == expert).flatten()
                        assert expert_blocks.numel() > 0
                        target_segment_offsets.append(int(expert_blocks[0]) * fc1_call["block_m"])

                    witness = {
                        "backend": backend,
                        "activation": activation_name,
                        "position": position,
                        "hidden_shape": tuple(hidden.shape),
                        "routing_shape": tuple(routing.shape),
                        "target_rows": tuple(target_rows.cpu().tolist()),
                        "expert_loads": tuple(actual_loads.tolist()),
                        "target_segment_offsets": tuple(target_segment_offsets),
                        "kernel_calls": tuple(
                            (call["a_shape"], call["b_shape"], call["grid_size"])
                            for call in run_calls
                        ),
                    }
                    witnesses.append(witness)
                    print("batch_invariant_moe_witness", witness)
        reference = target_outputs[0]
        assert reference.abs().max() > 0, "target oracle received only zero expert outputs"
        for target_output in target_outputs[1:]:
            assert torch.equal(target_output, reference)

        _assert_target_variation_witnesses(witnesses)

        # Hiding physical/routing variation must trip its witness oracle without changing outputs.
        for fields, message in (
            (("hidden_shape", "routing_shape"), "physical shape witness"),
            (("expert_loads", "target_segment_offsets"), "routing/load witness"),
        ):
            frozen = {name: witnesses[0][name] for name in fields}
            with pytest.raises(AssertionError, match=message):
                _assert_target_variation_witnesses([{**w, **frozen} for w in witnesses])


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
            te_supports_batch_invariant_grouped_gemm,
        )

        have_grouped_workspace = False
        try:
            import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm_mod

            ws_fn_before = te_gemm_mod.get_cublas_workspace_size_bytes
            have_te = True
            have_grouped_workspace = hasattr(te_gemm_mod, "_get_grouped_cublas_workspace")
        except ImportError:
            have_te = False
        if have_grouped_workspace:
            grouped_ws_fn_before = te_gemm_mod._get_grouped_cublas_workspace
        env_before = os.environ.get("CUBLASLT_WORKSPACE_SIZE")
        try:
            enable_batch_invariant_mode("te_native")
            assert is_batch_invariant_mode_enabled()
            assert get_batch_invariant_backend() == "te_native"
            assert os.environ.get("CUBLASLT_WORKSPACE_SIZE") == "0"
            if have_te:
                assert te_gemm_mod.get_cublas_workspace_size_bytes() == 1024
            if have_grouped_workspace:
                device = torch.cuda.current_device()
                if te_supports_batch_invariant_grouped_gemm(device):
                    workspace = te_gemm_mod._get_grouped_cublas_workspace(device, "TN")
                    assert workspace.numel() == ws_fn_before()
                else:
                    with pytest.raises(RuntimeError, match="moe_use_grouped_tensor=False"):
                        te_gemm_mod._get_grouped_cublas_workspace(device, "TN")
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
        if have_grouped_workspace:
            assert te_gemm_mod._get_grouped_cublas_workspace is grouped_ws_fn_before
        assert os.environ.get("CUBLASLT_WORKSPACE_SIZE") == env_before
