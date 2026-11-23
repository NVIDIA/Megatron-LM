# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Target-owned batch invariance across executed dynamic-inference features.

Run backend/version groups in fresh torch.distributed.run processes using
MCORE_BI_TEST_BACKEND and MCORE_BI_TEST_FA_VERSION. Start te_native with
CUBLASLT_WORKSPACE_SIZE=0; missing required capabilities must fail, not skip.
"""

from functools import partial

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, CudaGraphSizingDistribution
from tests.unit_tests.inference.engines.batch_invariant_test_utils import (
    Case,
    assert_same_target,
    invariant_runtime,
    labeled,
    new_engine,
    run_order,
)

DENSE_CASES = [
    Case("dense"),
    Case("async", context={"async_sched_mode": AsyncScheduleMode.ASYNC}),
    Case("local", model={"transformer_impl": "local"}),
    Case("optimized", model={"transformer_impl": "inference_optimized", "add_bias_linear": False}),
    Case(
        "last-logits",
        context={"materialize_only_last_token_logits": True},
        sampling={"return_log_probs": False},
    ),
    Case("chunked", context={"enable_chunked_prefill": True, "max_tokens": 128}, prompt_length=273),
    Case(
        "chunked-async",
        context={
            "enable_chunked_prefill": True,
            "max_tokens": 128,
            "async_sched_mode": AsyncScheduleMode.ASYNC,
        },
        prompt_length=273,
    ),
    Case(
        "prefix",
        context={"enable_prefix_caching": True},
        sampling={"return_log_probs": False},
        prompt_length=273,
        warm_prefix=True,
    ),
    Case(
        "sink",
        model={"window_size": (8, 0), "window_attn_skip_freq": 1, "softmax_type": "off-by-one"},
    ),
    Case("decode-graph", context={"num_cuda_graphs": 4, "max_tokens": 128}),
    Case(
        "decode-graph-linear",
        context={
            "num_cuda_graphs": 4,
            "max_tokens": 128,
            "cuda_graph_sizing_distribution": CudaGraphSizingDistribution.LINEAR,
        },
    ),
    Case(
        "mixed-graph",
        context={
            "num_cuda_graphs": 4,
            "max_tokens": 128,
            "use_cuda_graphs_for_non_decode_steps": True,
            "cuda_graph_all_prefills": True,
            "cuda_graph_mixed_prefill_count": 2,
            "enable_chunked_prefill": True,
        },
        prompt_length=273,
    ),
]


@pytest.mark.single_rank
@pytest.mark.parametrize("case", DENSE_CASES, ids=lambda case: case.name)
def test_dense_dynamic_feature_batch_invariance(case):
    with invariant_runtime(case) as (backend, version):
        # The orders differ only in their requests, so all four run on one engine.
        engine = new_engine(case, backend, version)
        reference = run_order(case, backend, version, "solo", engine=engine)
        contrasts = []
        contrast_runs = {}
        for order in ("front", "back", "staggered"):
            result = run_order(case, backend, version, order, engine=engine)
            assert_same_target(reference, result)
            contrasts.extend(result[1].steps)
            contrast_runs[order] = result[1]
            assert any(
                s["decode"] and s["physical"] == 128 and s["requests"] == 65
                for s in result[1].steps
            ), (case.name, order)
        # Witness target movement and changed decode shape during execution with live neighbors.
        assert any(step["target_row"] > 0 for step in contrasts)
        ref_decode = {s["physical"] for s in reference[1].steps if s["decode"]}
        wide_decode = {s["physical"] for s in contrasts if s["decode"] and s["requests"] > 64}
        assert ref_decode == {64}, ref_decode
        assert 128 in wide_decode, wide_decode
        assert any(
            step["neighbor_queries"].get(202) == 33 for step in contrasts
        ), "long neighbor prompt never shared a target forward"
        assert any(
            event["target_active"] and event["generated"] == 2
            for event in contrast_runs["staggered"].retirement_events
        )
        if case.prompt_length == 17:
            assert any(not s["decode"] and s["physical"] == 128 for s in contrasts)
        print("BI_WITNESS", case.name, backend, version, "decode_shapes", ref_decode, wide_decode)


SAMPLING_FILTERS = {"top-k": dict(top_k=7, top_p=0.0), "top-p": dict(top_k=0, top_p=0.8)}


def _check_fixed_history_sampling(run, backend, filters, mode, offset_by_dp):
    # Stop the target after one real sample. No stochastic sampled token enters
    # a subsequent target forward, so solo and co-batched histories are identical.
    params = dict(num_tokens_to_generate=1, temperature=0.7, skip_prompt_log_probs=True, **filters)
    reference = run("solo", sampling=params)
    actual = run("back", sampling=params)
    assert_same_target(reference, actual, require_trajectory=False)
    ref_samples, samples = reference[1].sample_steps, actual[1].sample_steps
    assert len(ref_samples) == len(samples) == 1
    expected, observed = ref_samples[0], samples[0]
    assert observed["sampler"] == ("TorchSampling" if backend == "torch" else "FlashInferSampling")
    assert observed["requests"] == 65 and not any(observed["filters"])
    assert torch.equal(expected["logits"], observed["logits"])
    assert torch.equal(expected["log_probs"], observed["log_probs"])
    assert {s["physical"] for s in reference[1].steps} == {64}
    assert any(s["physical"] == 128 and s["requests"] == 65 for s in actual[1].steps)
    # TP=PP=1 in this case, so WORLD rank is the actual DP rank.
    seed = 333 + (torch.distributed.get_rank() if offset_by_dp else 0)
    for result, sample in ((reference, expected), (actual, observed)):
        assert result[1].engine.controller._sampling._rng.initial_seed() == seed
        assert len(result[0].generated_tokens) == 1
        token = result[0].generated_tokens[0]
        assert token == sample["token"]
        assert torch.isfinite(sample["log_probs"][token]), "sampled outside the actual distribution"
        assert len(result[0].generated_log_probs) == 1
    print("BI_SAMPLING_WITNESS", backend, filters, mode, observed["requests"], observed["filters"])


@pytest.mark.parametrize("backend", ["torch", "flashinfer"])
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
@pytest.mark.parametrize("offset_by_dp", [False, True])
def test_sampling_fixed_history_batch_invariance(backend, mode, offset_by_dp):
    case = Case("sampling", context={"sampling_backend": backend, "logprobs_mode": mode})
    case.context["offset_sampling_seed_by_dp_rank"] = offset_by_dp
    with invariant_runtime(case) as (gemm, version):
        # The filters are request options, so both run on one engine; no check compares
        # sampled tokens across runs, so the sampler's advanced generator state is harmless.
        run = partial(run_order, case, gemm, version, engine=new_engine(case, gemm, version))
        for name, filters in SAMPLING_FILTERS.items():
            with labeled(name):
                _check_fixed_history_sampling(run, backend, filters, mode, offset_by_dp)


def test_parallel_batch_invariance():
    # One layout covers tensor, sequence and pipeline parallelism together: the assertions are the
    # same for each, and a two-rank reduction gives the same result in either order.
    case = Case("tp2-pp2-sp", tp=2, pp=2, sp=True)
    with invariant_runtime(case) as (backend, version):
        engine = new_engine(case, backend, version)
        reference = run_order(case, backend, version, "solo", engine=engine)
        actual = run_order(case, backend, version, "back", engine=engine)
        assert_same_target(reference, actual)
        assert any(
            s["decode"] and s["physical"] == 128 and s["requests"] == 65 for s in actual[1].steps
        )
