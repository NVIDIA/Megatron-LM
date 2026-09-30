# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import runpy
import sys
from unittest.mock import Mock

import pytest
import torch

from megatron.core import config
from megatron.core.tensor_parallel.random import (
    CudaRNGStatesTracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.transformer.moe import moe_utils
from tests.functional_tests.test_cases.common.moe_perf.test_cases import (
    MoEModelConfig,
    MoEPerformanceCase,
)

BENCHMARK_MODULE = "tests.functional_tests.test_cases.common.moe_perf.__main__"


@pytest.mark.parametrize("exit_code", [pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED])
def test_benchmark_entrypoint_preserves_pytest_exit_code(monkeypatch, exit_code):
    pytest_main = Mock(return_value=exit_code)
    monkeypatch.setattr(pytest, "main", pytest_main)
    monkeypatch.delitem(sys.modules, BENCHMARK_MODULE, raising=False)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(BENCHMARK_MODULE, run_name="__main__")

    assert exc_info.value.code == exit_code
    pytest_main.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG streams require a GPU")
def test_benchmark_repeats_routing_without_resetting_other_rng_streams(monkeypatch):
    benchmark = importlib.import_module(BENCHMARK_MODULE)
    monkeypatch.setattr(benchmark, "WARMUP_ITERS", 1)
    monkeypatch.setattr(benchmark, "MEASURE_ITERS", 3)
    monkeypatch.setattr(config, "ENABLE_EXPERIMENTAL", False)
    tracker = CudaRNGStatesTracker()
    monkeypatch.setattr(benchmark, "get_cuda_rng_tracker", lambda: tracker)
    monkeypatch.setattr(moe_utils, "get_cuda_rng_tracker", lambda: tracker)

    randn = torch.randn

    def small_dummy_gemm_input(*size, **kwargs):
        # Keep the benchmark's real CUDA timing and backward path, but bound its GEMM.
        if size == (8192, 8192):
            size = (8, 8)
        return randn(*size, **kwargs)

    monkeypatch.setattr(torch, "randn", small_dummy_gemm_input)

    class SyntheticLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.routing_logits = []
            self.other_draws = []
            self.backward_grads = []

        def forward(self, input_tensor):
            output = moe_utils.RandomSTE.apply(input_tensor)
            self.routing_logits.append(output.detach().clone())
            with tracker.fork("other-stream"):
                self.other_draws.append(torch.rand(4, device=input_tensor.device))
            output.register_hook(self.backward_grads.append)
            return output, None

    case = MoEPerformanceCase(
        name="rng-regression",
        model=MoEModelConfig(
            seq_length=2,
            micro_batch_size=1,
            hidden_size=4,
            moe_ffn_hidden_size=8,
            num_experts=4,
            router_topk=1,
        ),
        token_dispatcher="alltoall",
        manual_gc=False,
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        tracker.add(get_expert_parallel_rng_tracker_name(), 123)
        tracker.add("other-stream", 456)
        reference_generator = torch.Generator(device="cuda").manual_seed(456)
        layer = SyntheticLayer()

        metrics = benchmark._benchmark_moe_layer(layer, case)

        assert len(layer.routing_logits) == 4
        assert len(layer.backward_grads) == 4
        for logits in layer.routing_logits:
            torch.testing.assert_close(logits, layer.routing_logits[0], rtol=0, atol=0)
        for draw in layer.other_draws:
            expected = torch.rand(4, device="cuda", generator=reference_generator)
            torch.testing.assert_close(draw, expected, rtol=0, atol=0)
        assert len(metrics["forward_timings"]) == 3
        assert len(metrics["backward_timings"]) == 3
