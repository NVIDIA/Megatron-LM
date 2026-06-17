# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused GatedDeltaNet CUDA optimization coverage.

This keeps the optimized-kernel correctness and optional perf check separate
from the generic GatedDeltaNet unit tests.
"""

import os

import pytest
import torch

from tests.unit_tests.ssm import bench_gdn_cuda_opt as runner

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


def _scenario_items():
    keys = [
        key.strip()
        for key in os.environ.get(
            "MCORE_GDN_UNIT_TEST_SCENARIOS", "baseline,all_four_dv_dhu"
        ).split(",")
        if key.strip()
    ]
    if "baseline" not in keys:
        keys.insert(0, "baseline")
    unknown = [key for key in keys if key not in runner.SCENARIOS]
    if unknown:
        raise ValueError(f"unknown GDN CUDA opt scenarios: {unknown}")
    return [(key, runner.SCENARIOS[key]) for key in keys]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.internal
def test_gated_delta_net_cuda_opt_correctness_and_optional_perf(dtype):
    scenario_items = _scenario_items()
    runner.validate_fla_dispatch_sources(scenario_items)

    torch.manual_seed(123)
    runner.set_env({})
    try:
        model = runner.make_model(dtype).eval()
        seq_len = int(os.environ.get("MCORE_GDN_UNIT_TEST_T", "8192"))
        batch = int(os.environ.get("MCORE_GDN_UNIT_TEST_B", "2"))
        x = torch.randn(seq_len, batch, 128, device="cuda", dtype=dtype)

        atol = float(os.environ.get("MCORE_GDN_UNIT_TEST_ATOL", "5e-3"))
        rtol = float(os.environ.get("MCORE_GDN_UNIT_TEST_RTOL", "5e-3"))
        loss = os.environ.get("MCORE_GDN_UNIT_TEST_LOSS", "sum")
        accuracy_rows = runner.check_accuracy(
            model, x, scenario_items, loss=loss, atol=atol, rtol=rtol, use_nvtx=False
        )
        failed = [row for row in accuracy_rows if row.status != "PASS"]
        assert not failed, "\n".join(
            f"{row.name}: output={row.output_max_abs:.9f} "
            f"input_grad={row.input_grad_max_abs:.9f} "
            f"{row.worst_param}={row.worst_param_max_abs:.9f}"
            for row in failed
        )

        if os.environ.get("MCORE_GDN_UNIT_TEST_PERF", "0") == "1":
            perf_rows = runner.benchmark(
                model,
                x,
                scenario_items,
                loss=loss,
                warmup=int(os.environ.get("MCORE_GDN_UNIT_TEST_WARMUP", "5")),
                repeats=int(os.environ.get("MCORE_GDN_UNIT_TEST_REPEATS", "20")),
                rounds=int(os.environ.get("MCORE_GDN_UNIT_TEST_ROUNDS", "3")),
                use_nvtx=True,
            )
            for row in perf_rows:
                print(
                    f"PERF {row.name}: mean_us={row.mean_us:.3f} "
                    f"speedup_vs_baseline={row.speedup:.3f}"
                )
    finally:
        runner.set_env({})
        runner.Utils.destroy_model_parallel()
