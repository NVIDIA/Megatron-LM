# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Executed graph, prefix-release and learnable-sink batch-invariance policies."""

import pytest
import torch

from megatron.core.inference.config import CudaGraphSizingDistribution, PrefixCachingEvictionPolicy
from tests.unit_tests.inference.engines import batch_invariant_test_utils as f

REF_ZERO = PrefixCachingEvictionPolicy.REF_ZERO


@pytest.mark.parametrize("policy", ["exp", "bounded", "all-prefill", "ref-zero", "learnable"])
def test_batch_invariant_policy(policy, monkeypatch):
    options, all_prefill = {}, policy == "all-prefill"
    if policy in ("exp", "bounded", "all-prefill"):
        options.update(num_cuda_graphs=4, max_tokens=256)
    if policy == "exp":
        options["cuda_graph_sizing_distribution"] = CudaGraphSizingDistribution.EXPONENTIAL
    if policy in ("bounded", "all-prefill"):
        options.update(
            cuda_graph_max_tokens=128,
            cuda_graph_all_prefills=all_prefill,
            use_cuda_graphs_for_non_decode_steps=True,
            enable_chunked_prefill=True,
        )
    if policy == "ref-zero":
        options.update(enable_prefix_caching=True, prefix_caching_eviction_policy=REF_ZERO)
    length = 369 if policy in ("bounded", "all-prefill") else 273 if policy == "ref-zero" else 17
    case = f.Case(
        policy,
        context=options,
        model={"softmax_type": "learnable"} if policy == "learnable" else {},
        sampling={"return_log_probs": False} if policy == "ref-zero" else {},
        prompt_length=length,
    )
    build, get_offset = f.build_engine, f.attention.Attention._get_inference_softmax_offset
    offsets = []

    def observe_offset(layer):
        value = get_offset(layer)
        if value is not None:
            offsets.append(value.detach().clone())
        return value

    def build_observed(*args):
        engine = build(*args)
        ctx, model = engine.context, engine.controller.inference_wrapped_model.model
        engine._bi_dims = []
        engine._bi_after_forward = lambda: engine._bi_dims.append(ctx.padded_batch_dimensions)
        if policy == "learnable":
            for name, parameter in model.named_parameters():
                if name.endswith("softmax_offset"):
                    parameter.data.copy_(parameter.new_tensor([1, 2, 3, 4]))
        if policy == "ref-zero":
            allocator, released = ctx.kv_block_allocator, []
            release = allocator.release_memory_blocks

            def observe_release(blocks):
                registered = bool((allocator.block_hashes[blocks] >= 0).any())
                release(blocks)
                if registered:
                    assert not allocator.kv_hash_to_block_id
                    assert (allocator.block_ref_counts[blocks] == 0).all()
                    released.append(True)

            monkeypatch.setattr(allocator, "release_memory_blocks", observe_release)
            params = f.SamplingParams(num_tokens_to_generate=1, top_k=1, termination_id=-1)
            engine.add_request(99, f.target_prompt(273), params)
            engine.step_modern()
            assert released and not engine.has_unfinished_requests()
        return engine

    monkeypatch.setattr(f, "build_engine", build_observed)
    monkeypatch.setattr(f.attention.Attention, "_get_inference_softmax_offset", observe_offset)
    with f.invariant_runtime(case) as (backend, version):
        reference = f.run_order(case, backend, version, "solo")
        actual = f.run_order(case, backend, version, "back")
        f.assert_same_target(reference, actual)
        assert {s["physical"] for s in reference[1].steps if s["decode"]} == {64}
        contrasts = actual[1].steps
        assert 128 in {s["physical"] for s in contrasts if s["decode"] and s["target_row"] > 0}
        for result in (reference, actual):
            steps = result[1].steps
            if policy in ("bounded", "all-prefill"):
                wide = [s for s in steps if not s["decode"] and s["physical"] > 128]
                assert wide and all(bool(s["captured_graphs"]) == all_prefill for s in wide)
            if policy == "ref-zero":
                assert min(int(s["positions"].min()) for s in steps) == 0
            if policy == "learnable":
                assert {s["decode"] for s in steps if s["sinks"]} == {False, True}
        if policy == "exp":
            dims = reference[1].engine._bi_dims
            assert {d.decode_req_count for d in dims if d.prefill_req_count == 0} == {64}
        if policy == "learnable":
            assert offsets and all(torch.equal(v, v.new_tensor([1, 2, 3, 4])) for v in offsets)
        print("BI_POLICY_WITNESS", policy, backend, version)
