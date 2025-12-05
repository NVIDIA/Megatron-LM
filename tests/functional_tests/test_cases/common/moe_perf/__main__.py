# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU performance regression tests for the MoE layer."""

from __future__ import annotations

import gc
import json
import os
import statistics
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, cast

import pytest  # type: ignore[import]
import torch

from megatron.core.config import set_experimental_flag
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP, HAVE_HYBRIDEP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import RandomSTE
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

from .test_cases import PERFORMANCE_CASES, MoEPerformanceCase

# NOTE: Performance regression threshold
DEFAULT_MAX_REGRESSION_RATIO = 1.02
DEFAULT_MAX_VARIANCE_RATIO = 0.02  # The std/mean should be less than 2%
WARMUP_ITERS = 5
MEASURE_ITERS = 20


BASELINES_PATH = Path(__file__).resolve().parent / "baseline.json"
UPDATE_BASELINES_ENV = "MEGATRON_UPDATE_PERF_BASELINES"


def _build_transformer_config(case: MoEPerformanceCase) -> TransformerConfig:
    model = case.model
    config_kwargs = dict(
        num_layers=1,
        hidden_size=model.hidden_size,
        moe_ffn_hidden_size=model.moe_ffn_hidden_size,
        num_attention_heads=model.num_attention_heads,
        # MoE Arguments
        num_moe_experts=model.num_experts,
        moe_router_topk=model.router_topk,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1.0,
        moe_token_dispatcher_type=case.token_dispatcher,
        moe_flex_dispatcher_backend=case.moe_flex_dispatcher_backend,
        use_cpu_initialization=True,
        add_bias_linear=False,
        # Router Arguments
        moe_router_num_groups=model.moe_router_num_groups,
        moe_router_group_topk=model.moe_router_group_topk,
        moe_router_score_function=model.moe_router_score_function,
        moe_router_dtype=model.moe_router_dtype,
        moe_router_enable_expert_bias=model.moe_router_enable_expert_bias,
        # Parallelism Arguments
        sequence_parallel=case.tensor_model_parallel_size > 1,
        tensor_model_parallel_size=case.tensor_model_parallel_size,
        pipeline_model_parallel_size=case.pipeline_model_parallel_size,
        expert_model_parallel_size=case.expert_model_parallel_size,
        expert_tensor_parallel_size=case.expert_tensor_parallel_size,
        context_parallel_size=case.context_parallel_size,
        params_dtype=case.input_dtype,
        bf16=True,
        fp8=case.fp8,
        moe_permute_fusion=case.moe_permute_fusion,
        moe_router_fusion=case.moe_router_fusion,
        moe_router_force_load_balancing=case.moe_router_force_load_balancing,
    )

    if case.fp8:
        config_kwargs.update(
            dict(fp8="hybrid", fp8_margin=0, fp8_interval=1, fp8_recipe="blockwise")
        )

    return TransformerConfig(**config_kwargs)


# NOTE: Only TE backend is covered in this test.
def _resolve_moe_submodules(case: MoEPerformanceCase):
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=case.model.num_experts, moe_grouped_gemm=True
    )
    return layer_spec.submodules.mlp.submodules


def _load_baselines() -> Dict[str, Dict[str, float]]:
    if not BASELINES_PATH.exists():
        return {}
    with BASELINES_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _persist_baselines(data: Dict[str, Dict[str, float]]) -> None:
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BASELINES_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _serialize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    forward_ms = metrics["forward_ms"]
    backward_ms = metrics["backward_ms"]
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "max_allocated_bytes": metrics["max_allocated_bytes"],
        "max_regression_ratio": DEFAULT_MAX_REGRESSION_RATIO,
    }


def _assert_within_baseline(
    case_name: str, metrics: Mapping[str, Any], baselines: Dict[str, Dict[str, float]]
):
    baseline = baselines.get(case_name)
    if baseline is None:
        pytest.fail(
            f"Missing baseline data for {case_name}. Set {UPDATE_BASELINES_ENV}=1 to record."
        )

    max_ratio = baseline.get("max_regression_ratio", DEFAULT_MAX_REGRESSION_RATIO)

    def _limit(metric_name: str) -> float:
        baseline_value = baseline.get(metric_name)
        if baseline_value is None:
            return float("inf")
        ratio_limit = baseline_value * max_ratio
        return ratio_limit

    fwd_limit = _limit("forward_ms")
    bwd_limit = _limit("backward_ms")
    mem_limit = _limit("max_allocated_bytes")

    forward_ms = cast(float, metrics["forward_ms"])
    backward_ms = cast(float, metrics["backward_ms"])
    max_allocated_bytes = cast(float, metrics["max_allocated_bytes"])

    forward_std_ms = cast(float, metrics.get("forward_std_ms", 0.0))
    backward_std_ms = cast(float, metrics.get("backward_std_ms", 0.0))
    forward_timings = cast(Sequence[float], metrics.get("forward_timings", ()))
    backward_timings = cast(Sequence[float], metrics.get("backward_timings", ()))

    assert (
        forward_ms <= fwd_limit
    ), f"Forward pass for {case_name} regressed: {forward_ms:.3f} ms (limit {fwd_limit:.3f} ms)."
    assert (
        backward_ms <= bwd_limit
    ), f"Backward pass for {case_name} regressed: {backward_ms:.3f} ms (limit {bwd_limit:.3f} ms)."

    if forward_ms > 0.0:
        assert forward_std_ms / forward_ms <= DEFAULT_MAX_VARIANCE_RATIO, (
            "Forward pass for "
            f"{case_name} has high variance: {forward_std_ms:.3f} ms "
            f"(limit {DEFAULT_MAX_VARIANCE_RATIO:.3f} of {forward_ms:.3f} ms). "
            f"The full timings are {list(forward_timings)}."
        )
    if backward_ms > 0.0:
        assert backward_std_ms / backward_ms <= DEFAULT_MAX_VARIANCE_RATIO, (
            "Backward pass for "
            f"{case_name} has high variance: {backward_std_ms:.3f} ms "
            f"(limit {DEFAULT_MAX_VARIANCE_RATIO:.3f} of {backward_ms:.3f} ms). "
            f"The full timings are {list(backward_timings)}."
        )
    assert max_allocated_bytes <= mem_limit, (
        "Max allocated memory for "
        f"{case_name} regressed: {max_allocated_bytes / (1024 ** 2):.3f} MiB "
        f"(limit {mem_limit / (1024 ** 2):.3f} MiB)."
    )


def _benchmark_moe_layer(layer: MoELayer, case: MoEPerformanceCase):
    torch.cuda.synchronize()
    set_experimental_flag(True)

    forward_timings = []
    backward_timings = []
    max_allocated_bytes = []

    generator = torch.Generator(device="cuda").manual_seed(1234)
    model = case.model

    if case.manual_gc:
        torch.cuda.empty_cache()
        gc.disable()
        gc.collect()

    # NOTE: Using the same input tensor for all iterations to prevent different routing results,
    # which may lead to different kernels and library load/compile overhead.
    input_tensor = torch.randn(
        model.seq_length,
        model.micro_batch_size,
        model.hidden_size,
        device="cuda",
        dtype=case.input_dtype,
        generator=generator,
    )
    input_tensor.requires_grad_(True)
    for iteration in range(WARMUP_ITERS + MEASURE_ITERS):
        if RandomSTE.generator is not None:
            RandomSTE.generator.manual_seed(RandomSTE.generator.initial_seed())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.nvtx.range_push(f"({case.name}) iteration {iteration}")
        # Use a long CUDA kernel to hide the router launch overhead
        with torch.cuda.nvtx.range("(dummy GEMM)"):
            dummy_tensor = torch.randn(8192, 8192, device="cuda")
            torch.matmul(dummy_tensor, dummy_tensor)
            del dummy_tensor
        input_tensor.grad = None
        layer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        context = get_fp8_context(layer.config) if case.fp8 else nullcontext()
        with context:
            fwd_start.record()
            output, _ = layer(input_tensor)
            fwd_end.record()

            backward_grad = torch.randn_like(output)
            bwd_start.record()
            output.backward(backward_grad)
            bwd_end.record()

        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

        if iteration >= WARMUP_ITERS:
            forward_timings.append(fwd_start.elapsed_time(fwd_end))
            backward_timings.append(bwd_start.elapsed_time(bwd_end))
            max_allocated_bytes.append(torch.cuda.max_memory_allocated())

    # Exclude the top 3 values from timings lists to avoid outliers
    forward_timings_sorted = sorted(forward_timings)[:-3]
    backward_timings_sorted = sorted(backward_timings)[:-3]
    forward_ms = statistics.mean(forward_timings)
    backward_ms = statistics.mean(backward_timings)
    max_allocated_bytes = statistics.mean(max_allocated_bytes)

    if case.manual_gc:
        gc.collect()
        gc.enable()

    if Utils.rank == 0:
        print(f"({case.name}) forward times {forward_timings}")
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "forward_std_ms": statistics.pstdev(forward_timings) if len(forward_timings) > 1 else 0.0,
        "backward_std_ms": (
            statistics.pstdev(backward_timings) if len(backward_timings) > 1 else 0.0
        ),
        "max_allocated_bytes": max_allocated_bytes,
        "forward_timings": forward_timings,
        "backward_timings": backward_timings,
    }


def _maybe_update_baseline(
    case: MoEPerformanceCase, metrics: Dict[str, float], baselines: Dict[str, Dict[str, float]]
):
    forward_ms = metrics["forward_ms"]
    backward_ms = metrics["backward_ms"]
    forward_std_ms = metrics["forward_std_ms"]
    backward_std_ms = metrics["backward_std_ms"]
    assert forward_std_ms / forward_ms <= DEFAULT_MAX_VARIANCE_RATIO, (
        "Forward pass for "
        f"{case.name} has high variance: {forward_std_ms:.3f} ms "
        f"(limit {DEFAULT_MAX_VARIANCE_RATIO:.3f} of {forward_ms:.3f} ms)."
    )
    assert backward_std_ms / backward_ms <= DEFAULT_MAX_VARIANCE_RATIO, (
        "Backward pass for "
        f"{case.name} has high variance: {backward_std_ms:.3f} ms "
        f"(limit {DEFAULT_MAX_VARIANCE_RATIO:.3f} of {backward_ms:.3f} ms)."
    )
    baselines[case.name] = _serialize_metrics(metrics)
    _persist_baselines(baselines)


def _prepare_moe_layer(case: MoEPerformanceCase) -> MoELayer:
    config = _build_transformer_config(case)
    submodules = _resolve_moe_submodules(case)
    layer = MoELayer(config=config, submodules=submodules).cuda().to(dtype=torch.bfloat16)

    layer.train()
    return layer


def _check_env():
    NCCL_MAX_NCHANNELS = os.environ.get("NCCL_MAX_NCHANNELS")
    if NCCL_MAX_NCHANNELS is not None:
        pytest.fail(
            f"NCCL_MAX_NCHANNELS is set to {NCCL_MAX_NCHANNELS}, this may lead to performance regression"
        )


def _check_dependencies(case: MoEPerformanceCase):
    if case.token_dispatcher == "flex":
        if case.moe_flex_dispatcher_backend == "deepep":
            if not HAVE_DEEP_EP:
                pytest.skip("DeepEP is not available")
        elif case.moe_flex_dispatcher_backend == "hybridep":
            if not HAVE_HYBRIDEP:
                pytest.skip("HybridEP is not available")


@pytest.mark.flaky(reruns=10)
@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for MoE performance benchmarking"
)
@pytest.mark.parametrize("perf_case", PERFORMANCE_CASES, ids=lambda c: c.name)
def test_moe_layer_performance(perf_case: MoEPerformanceCase, debug_mode: bool = False):
    _check_env()
    _check_dependencies(perf_case)
    if not perf_case.is_current_platform():
        pytest.skip(
            "GPU platform mismatch: "
            f"expected '{perf_case.gpu_platform}', "
            f"found '{torch.cuda.get_device_name(torch.cuda.current_device())}'."
        )

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=perf_case.tensor_model_parallel_size,
        pipeline_model_parallel_size=perf_case.pipeline_model_parallel_size,
        expert_model_parallel_size=perf_case.expert_model_parallel_size,
        context_parallel_size=perf_case.context_parallel_size,
        expert_tensor_parallel_size=perf_case.expert_tensor_parallel_size,
    )

    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        torch.cuda.reset_peak_memory_stats()
        layer = _prepare_moe_layer(perf_case)
        with torch.cuda.nvtx.range(f"({perf_case.name})"):
            metrics = _benchmark_moe_layer(layer, perf_case)

        summary = (
            f"MoE layer performance ({perf_case.name}): forward {metrics['forward_ms']:.3f} ms "
            f"(σ={metrics['forward_std_ms']:.3f}), backward {metrics['backward_ms']:.3f} ms "
            f"(σ={metrics['backward_std_ms']:.3f}), max mem {metrics['max_allocated_bytes'] / (1024 ** 2):.3f} MiB"
        )
        if Utils.rank == 0:
            print(summary)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Don't check performance if profiling is enabled
        baseline_failed = False
        baseline_failure_message = ""

        # Only rank 0 checks the baseline
        if Utils.rank == 0 and not debug_mode:
            baselines = _load_baselines()
            try:
                if os.getenv(UPDATE_BASELINES_ENV) == "1":
                    _maybe_update_baseline(perf_case, metrics, baselines)
                else:
                    _assert_within_baseline(perf_case.name, metrics, baselines)
            except AssertionError as exc:
                baseline_failed = True
                baseline_failure_message = str(exc)

        failure_tensor = torch.tensor(
            [1 if baseline_failed else 0],
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(failure_tensor, op=torch.distributed.ReduceOp.MAX)
        baseline_failed = bool(failure_tensor.item())

        if baseline_failed:
            if Utils.rank != 0:
                baseline_failure_message = "Baseline regression detected on rank 0."
                pytest.fail(baseline_failure_message, pytrace=False)
            else:
                pytest.fail(baseline_failure_message, pytrace=True)

    finally:
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()


# Main entry for local performance testing
# Commands to run with nsys profiling:
# nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
#         -f true -x true \
#         --cuda-graph-trace=node \
#         --capture-range=cudaProfilerApi \
#         --capture-range-end=stop \
#         -o output \
#         uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 -m tests.functional_tests.test_cases.common.moe_perf
# Commands to run with pytest:
# export MEGATRON_UPDATE_PERF_BASELINES=0 # set to 1 to update baseline perf numbers
# uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 -m tests.functional_tests.test_cases.common.moe_perf
if __name__ == "__main__":
    pytest.main(["-x", "-v", "-s", __file__])  # -xvs
    # torch.cuda.cudart().cudaProfilerStart()
    # torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
    # for case in PERFORMANCE_CASES:
    #     if case.name == "mixtral_a2a_tp1ep8_fp8":
    #         test_moe_layer_performance(case, debug_mode=True)
    # torch.cuda.cudart().cudaProfilerStop()
    # torch.distributed.destroy_process_group()
