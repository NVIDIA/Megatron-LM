# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""GPU performance regression tests for the MoE layer."""

from __future__ import annotations

import gc
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pytest  # type: ignore[import]
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


try:  # pragma: no cover - runtime availability check
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except ImportError:  # pragma: no cover - runtime availability check
    HAVE_TE = False

# NOTE: Performance regression threshold
DEFAULT_MAX_REGRESSION_RATIO = 1.02
WARMUP_ITERS = 3
MEASURE_ITERS = 10


BASELINES_PATH = (
    Path(__file__).resolve().parent / "baselines" / "moe_layer.json"
)
UPDATE_BASELINES_ENV = "MEGATRON_UPDATE_PERF_BASELINES"


@dataclass(frozen=True)
class MoEModelConfig:
    seq_length: int
    micro_batch_size: int
    hidden_size: int
    moe_ffn_hidden_size: int
    num_experts: int
    router_topk: int
    num_attention_heads: int = 8
    moe_router_load_balancing_type: str = "aux_loss"


@dataclass(frozen=True)
class MoEPerformanceCase:
    """Describes a single MoE performance configuration to exercise."""

    name: str
    model: MoEModelConfig

    # Token dispatcher related
    token_dispatcher: str

    # FP8 related
    fp8: bool = False

    # Tested GPU platform
    gpu_platform: str = "H100"

    # Parallelism related
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    # kernel fusion related
    moe_permute_fusion: bool = True
    moe_router_fusion: bool = True

    # Performance stability related
    moe_router_force_load_balancing: bool = True 
    manual_gc: bool = True

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def is_current_platform(self) -> bool:
        if self.gpu_platform is None:
            return True
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return self.gpu_platform.lower() in device_name.lower()


MIXTRAL_PROXY = MoEModelConfig(
    seq_length=4096,
    micro_batch_size=1,
    hidden_size=4096,
    moe_ffn_hidden_size=14336,
    num_experts=8,
    router_topk=2,
    num_attention_heads=32,
    moe_router_load_balancing_type="aux_loss",
)

# DEEPSEEK_PROXY = MoEModelConfig(
#     seq_length=4096,
#     micro_batch_size=1,
#     hidden_size=5120,
#     num_experts=4,
#     router_topk=2,
#     num_attention_heads=40,
# )


PERFORMANCE_CASES: Iterable[MoEPerformanceCase] = (
    MoEPerformanceCase(
        name="mixtral_a2a_tp1ep8_bf16",
        token_dispatcher="alltoall",
        model=MIXTRAL_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
    ),
    # MoEPerformanceCase(
    #     name="mixtral_deepep_tp1ep8_bf16",
    #     token_dispatcher="flex",
    #     model=MIXTRAL_PROXY,
    #     tensor_parallel_size=1,
    #     expert_parallel_size=8,
    # ),
)


def _build_transformer_config(case: MoEPerformanceCase) -> TransformerConfig:
    model = case.model
    config_kwargs = dict(
        num_layers=1,
        hidden_size=model.hidden_size,
        moe_ffn_hidden_size=model.moe_ffn_hidden_size,
        num_attention_heads=model.num_attention_heads,
        num_moe_experts=model.num_experts,
        moe_router_topk=model.router_topk,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type=case.token_dispatcher,
        use_cpu_initialization=True,
        add_bias_linear=False,
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
            dict(
                fp8="hybrid",
                fp8_margin=0,
                fp8_interval=1,
                fp8_recipe="blockwise",
            )
        )

    return TransformerConfig(**config_kwargs)


def _resolve_moe_submodules(case: MoEPerformanceCase):
    layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts=case.model.num_experts)
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
        "max_regression_ratio": DEFAULT_MAX_REGRESSION_RATIO,
    }


def _assert_within_baseline(case_name: str, metrics: Dict[str, float], baselines: Dict[str, Dict[str, float]]):
    baseline = baselines.get(case_name)
    if baseline is None:
        pytest.error(f"Missing baseline data for {case_name}. Set {UPDATE_BASELINES_ENV}=1 to record.")

    max_ratio = baseline.get("max_regression_ratio", DEFAULT_MAX_REGRESSION_RATIO)

    def _limit(metric_name: str) -> float:
        baseline_value = baseline.get(metric_name)
        ratio_limit = baseline_value * max_ratio
        return ratio_limit

    fwd_limit = _limit("forward_ms")
    bwd_limit = _limit("backward_ms")

    forward_ms = metrics["forward_ms"]
    backward_ms = metrics["backward_ms"]

    assert forward_ms <= fwd_limit, (
        f"Forward pass for {case_name} regressed: {forward_ms:.3f} ms (limit {fwd_limit:.3f} ms)."
    )
    assert backward_ms <= bwd_limit, (
        f"Backward pass for {case_name} regressed: {backward_ms:.3f} ms (limit {bwd_limit:.3f} ms)."
    )


def _benchmark_moe_layer(layer: MoELayer, case: MoEPerformanceCase) -> Dict[str, float]:
    torch.cuda.synchronize()

    forward_timings = []
    backward_timings = []

    generator = torch.Generator(device="cuda").manual_seed(1234)
    model = case.model
    
    if case.manual_gc:
        torch.cuda.empty_cache()
        gc.disable()
        gc.collect()

    for iteration in range(WARMUP_ITERS + MEASURE_ITERS):
        input_tensor = torch.randn(
            model.seq_length,
            model.micro_batch_size,
            model.hidden_size,
            device="cuda",
            dtype=case.input_dtype,
            generator=generator,
        )
        input_tensor.requires_grad_(True)

        layer.zero_grad(set_to_none=True)

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        fwd_start.record()
        output, _ = layer(input_tensor)
        fwd_end.record()

        loss = output.sum()
        bwd_start.record()
        loss.backward()
        bwd_end.record()

        torch.cuda.synchronize()

        if iteration >= WARMUP_ITERS:
            forward_timings.append(fwd_start.elapsed_time(fwd_end))
            backward_timings.append(bwd_start.elapsed_time(bwd_end))

    forward_ms = statistics.mean(forward_timings)
    backward_ms = statistics.mean(backward_timings)

    if case.manual_gc:
        gc.collect()
        gc.enable()

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "forward_std_ms": statistics.pstdev(forward_timings) if len(forward_timings) > 1 else 0.0,
        "backward_std_ms": statistics.pstdev(backward_timings) if len(backward_timings) > 1 else 0.0,
    }


def _maybe_update_baseline(case: MoEPerformanceCase, metrics: Dict[str, float], baselines: Dict[str, Dict[str, float]]):
    baselines[case.name] = _serialize_metrics(metrics)
    _persist_baselines(baselines)


def _prepare_moe_layer(case: MoEPerformanceCase) -> MoELayer:
    config = _build_transformer_config(case)
    submodules = _resolve_moe_submodules(case)
    layer = MoELayer(config=config, submodules=submodules).cuda().to(dtype=torch.bfloat16)

    layer.train()
    return layer


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MoE performance benchmarking")
@pytest.mark.parametrize("perf_case", PERFORMANCE_CASES, ids=lambda c: c.name)
def test_moe_layer_performance(perf_case: MoEPerformanceCase):
    if perf_case.fp8 and not (HAVE_TE and is_te_min_version("1.7.0")):
        pytest.skip("TransformerEngine with FP8 support is required for this configuration")

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
        metrics = _benchmark_moe_layer(layer, perf_case)

        summary = (
            f"MoE layer performance ({perf_case.name}): forward {metrics['forward_ms']:.3f} ms "
            f"(σ={metrics['forward_std_ms']:.3f}), backward {metrics['backward_ms']:.3f} ms "
            f"(σ={metrics['backward_std_ms']:.3f})"
        )
        print(summary)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if Utils.rank == 0:
            baselines = _load_baselines()
            if os.getenv(UPDATE_BASELINES_ENV) == "1":
                _maybe_update_baseline(perf_case, metrics, baselines)
            else:
                _assert_within_baseline(perf_case.name, metrics, baselines)

    finally:
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()


# Main entry for local performance testing
if __name__ == "__main__":
    test_moe_layer_performance(PERFORMANCE_CASES[0])