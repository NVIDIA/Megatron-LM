# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Direct GatedDeltaNet CUDA optimization correctness and performance runner.

This runner intentionally uses installed packages and normal project imports.
Install `mcore_gdn_opt` and FLA in editable mode before running it.
"""

import argparse
import importlib.util
import os
import statistics
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


FLAGS = (
    "MCORE_GDN_USE_OPT_WRAPPER",
    "MCORE_GDN_OPT_BACKEND",
    "MCORE_GDN_OPT_WARN_FALLBACK",
    "MCORE_GDN_OPT_ENABLE_FWD_H",
    "MCORE_GDN_OPT_ENABLE_WY_BWD",
    "MCORE_GDN_OPT_ENABLE_DV_DHU",
    "MCORE_GDN_OPT_ENABLE_DHU",
    "MCORE_GDN_OPT_ENABLE_DQKWG",
    "MCORE_GDN_OPT_ENABLE_DHU_DQKWG",
    "FLA_CUTE_FWD_H",
    "CHUNK_DELTA_FWD_USE_BWD_PORT",
    "FLA_CUTE_WY_BWD",
    "FLA_CUTE_BWD_DV_DHU",
    "FLA_CUTE_BWD_DHU",
    "FLA_CUTE_BWD_DQKWG",
    "FLA_CUTE_BWD_DHU_DQKWG",
    "FLA_CUTE_BWD_DHU_DQKWG_KERNEL",
    "FLA_CUTE_BWD_DHU_DQKWG_DIRECT",
)


SCENARIOS = {
    "baseline": ("Triton baseline", {}),
    "wrapper_fla": (
        "MCore wrapper forced FLA",
        {"MCORE_GDN_USE_OPT_WRAPPER": "1", "MCORE_GDN_OPT_BACKEND": "fla"},
    ),
    "wrapper_auto": (
        "MCore wrapper auto",
        {"MCORE_GDN_USE_OPT_WRAPPER": "1", "MCORE_GDN_OPT_BACKEND": "auto"},
    ),
    "wrapper_cuda": (
        "MCore wrapper forced CUDA",
        {"MCORE_GDN_USE_OPT_WRAPPER": "1", "MCORE_GDN_OPT_BACKEND": "cuda"},
    ),
    "wy": (
        "CUDA wy_bwd",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_FWD_H": "0",
            "MCORE_GDN_OPT_ENABLE_DV_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DQKWG": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
    "dv_dhu": (
        "CUDA dv_local+delta_h fused",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_FWD_H": "0",
            "MCORE_GDN_OPT_ENABLE_WY_BWD": "0",
            "MCORE_GDN_OPT_ENABLE_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DQKWG": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
    "dhu": (
        "CUDA delta_h",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_FWD_H": "0",
            "MCORE_GDN_OPT_ENABLE_WY_BWD": "0",
            "MCORE_GDN_OPT_ENABLE_DV_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DQKWG": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
    "dqkwg": (
        "CUDA dqkwg",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_FWD_H": "0",
            "MCORE_GDN_OPT_ENABLE_WY_BWD": "0",
            "MCORE_GDN_OPT_ENABLE_DV_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
    "fused": (
        "CUDA wy+dhu+dqkwg fused",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_FWD_H": "0",
            "MCORE_GDN_OPT_ENABLE_DV_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DQKWG": "0",
        },
    ),
    "separate": (
        "CUDA all three separate",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_FWD_H": "0",
            "MCORE_GDN_OPT_ENABLE_DV_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
    "dv_dhu_dqkwg": (
        "CUDA dv_local+delta_h fused + dqkwg",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_FWD_H": "0",
            "MCORE_GDN_OPT_ENABLE_WY_BWD": "0",
            "MCORE_GDN_OPT_ENABLE_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
    "all_four": (
        "CUDA all four",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_DV_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
    "all_four_dv_dhu": (
        "CUDA fwd_h+wy+dv_dhu+dqkwg",
        {
            "MCORE_GDN_USE_OPT_WRAPPER": "1",
            "MCORE_GDN_OPT_BACKEND": "cuda",
            "MCORE_GDN_OPT_ENABLE_DHU": "0",
            "MCORE_GDN_OPT_ENABLE_DHU_DQKWG": "0",
        },
    ),
}


@dataclass
class AccuracyRow:
    name: str
    status: str
    output_max_abs: float
    input_grad_max_abs: float
    worst_param: str
    worst_param_max_abs: float


@dataclass
class PerfRow:
    name: str
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    speedup: float


def set_env(overrides):
    for flag in FLAGS:
        os.environ.pop(flag, None)
    if "MCORE_GDN_USE_OPT_WRAPPER" not in overrides:
        os.environ["MCORE_GDN_USE_OPT_WRAPPER"] = "0"
    if "MCORE_GDN_OPT_BACKEND" not in overrides:
        os.environ["MCORE_GDN_OPT_BACKEND"] = "fla"
    os.environ.update(overrides)


def set_model_dispatch(model):
    if os.environ.get("MCORE_GDN_USE_OPT_WRAPPER", "0") == "1":
        from mcore_gdn_opt.gated_delta_rule import chunk_gated_delta_rule
    else:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    model.gated_delta_rule = chunk_gated_delta_rule


def validate_dispatch_sources(scenario_items):
    if any("MCORE_GDN_OPT_BACKEND" in env for _, (_, env) in scenario_items):
        for module_name in (
            "mcore_gdn_opt.gated_delta_rule.chunk",
            "mcore_gdn_opt.gated_delta_rule.backward",
        ):
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"cannot locate required mcore_gdn_opt module {module_name!r}")
            print(f"MCORE_GDN_OPT_DISPATCH_SOURCE module={module_name} path={spec.origin}", flush=True)


def nvtx_range(label, enabled=True):
    if enabled and torch.cuda.is_available():
        return torch.cuda.nvtx.range(label)
    return nullcontext()


def scenario_label(index, name):
    safe_name = name.replace(" ", "_").replace("+", "plus").replace("/", "_")
    return f"gdn_only/{index:02d}_{safe_name}"


def make_model(dtype):
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    cfg = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=64,
        linear_num_value_heads=64,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=64,
        activation_func=F.silu,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    submodules = get_experimental_attention_variant_module_spec(config=cfg).submodules
    return GatedDeltaNet(
        cfg,
        submodules=submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    ).cuda().to(dtype)


def zero_grads(model):
    model.zero_grad(set_to_none=True)


def compute_loss(output, loss):
    if loss == "sum":
        return output.float().sum()
    if loss == "square_mean":
        return output.float().square().mean()
    raise ValueError(f"unknown loss: {loss}")


def run_once(model, x, env, loss, nvtx_label=None, use_nvtx=True):
    set_env(env)
    set_model_dispatch(model)
    print(
        "RUN_ONCE "
        f"label={nvtx_label or 'none'} "
        f"use_wrapper={os.environ.get('MCORE_GDN_USE_OPT_WRAPPER', '')} "
        f"backend={os.environ.get('MCORE_GDN_OPT_BACKEND', '')}",
        flush=True,
    )
    zero_grads(model)
    inp = x.detach().clone().requires_grad_(True)
    with nvtx_range(nvtx_label, enabled=use_nvtx and nvtx_label is not None):
        out, _ = model(inp, attention_mask=None)
        compute_loss(out, loss).backward()
    torch.cuda.synchronize()
    grads = {
        name: param.grad.detach().float().clone().cpu()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return out.detach().float().clone().cpu(), inp.grad.detach().float().clone().cpu(), grads


def diff_max_abs(actual, expected):
    return float((actual - expected).abs().max().item())


def allclose(actual, expected, atol, rtol):
    return bool(torch.isfinite(actual).all().item()) and bool(
        torch.allclose(actual, expected, atol=atol, rtol=rtol)
    )


def check_accuracy(model, x, scenario_items, loss, atol, rtol, use_nvtx=True):
    base_name, base_env = SCENARIOS["baseline"]
    base_out, base_grad, base_params = run_once(
        model, x, base_env, loss, "gdn_only/00_accuracy_reference/Triton_baseline", use_nvtx
    )
    rows = []
    for scenario_idx, (_, (name, env)) in enumerate(scenario_items, start=1):
        label = f"{scenario_label(scenario_idx, name)}/accuracy"
        out, grad, params = run_once(model, x, env, loss, label, use_nvtx)
        output_ok = allclose(out, base_out, atol, rtol)
        grad_ok = allclose(grad, base_grad, atol, rtol)
        worst_param = ""
        worst_param_abs = 0.0
        params_ok = True
        for param_name, expected in base_params.items():
            actual = params[param_name]
            params_ok = params_ok and allclose(actual, expected, atol, rtol)
            param_abs = diff_max_abs(actual, expected)
            if param_abs > worst_param_abs:
                worst_param = param_name
                worst_param_abs = param_abs
        rows.append(
            AccuracyRow(
                name=name,
                status="PASS" if output_ok and grad_ok and params_ok else "FAIL",
                output_max_abs=diff_max_abs(out, base_out),
                input_grad_max_abs=diff_max_abs(grad, base_grad),
                worst_param=worst_param,
                worst_param_max_abs=worst_param_abs,
            )
        )
    return rows


def fwd_bwd(model, x, env, loss, nvtx_label=None, use_nvtx=True):
    set_env(env)
    set_model_dispatch(model)
    zero_grads(model)
    inp = x.detach().requires_grad_(True)
    with nvtx_range(nvtx_label, enabled=use_nvtx and nvtx_label is not None):
        out, _ = model(inp, attention_mask=None)
        compute_loss(out, loss).backward()


def benchmark(model, x, scenario_items, loss, warmup, repeats, rounds, use_nvtx=True):
    rows = []
    baseline_us = None
    for scenario_idx, (_, (name, env)) in enumerate(scenario_items, start=1):
        base_label = scenario_label(scenario_idx, name)
        for warmup_idx in range(warmup):
            fwd_bwd(model, x, env, loss, f"{base_label}/warmup_{warmup_idx:02d}", use_nvtx)
        torch.cuda.synchronize()
        samples = []
        for round_idx in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with nvtx_range(f"{base_label}/round_{round_idx:02d}/measured_{repeats}iters", enabled=use_nvtx):
                start.record()
                for iter_idx in range(repeats):
                    fwd_bwd(model, x, env, loss, f"{base_label}/round_{round_idx:02d}/iter_{iter_idx:02d}", use_nvtx)
                end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / repeats)
        mean_us = statistics.mean(samples)
        if baseline_us is None:
            baseline_us = mean_us
        rows.append(
            PerfRow(
                name=name,
                mean_us=mean_us,
                median_us=statistics.median(samples),
                min_us=min(samples),
                max_us=max(samples),
                speedup=baseline_us / mean_us,
            )
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--loss", choices=("sum", "square_mean"), default="square_mean")
    parser.add_argument("--scenarios", default="baseline,fused,separate,all_four")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--fail-on-accuracy", action="store_true")
    parser.add_argument("--no-nvtx", dest="use_nvtx", action="store_false", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    keys = [key.strip() for key in args.scenarios.split(",") if key.strip()]
    if "baseline" not in keys:
        keys.insert(0, "baseline")
    unknown = [key for key in keys if key not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}; choices={sorted(SCENARIOS)}")
    scenario_items = [(key, SCENARIOS[key]) for key in keys]
    validate_dispatch_sources(scenario_items)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    torch.manual_seed(123)
    set_env({})
    print(
        f"DEVICE {torch.cuda.get_device_name(0)} SHAPE B=2 T=8192 H=64 D=128 "
        f"dtype={args.dtype} loss={args.loss}"
    )
    try:
        model = make_model(dtype).eval()
        x = torch.randn(8192, 2, 128, device="cuda", dtype=dtype)
        accuracy_rows = check_accuracy(model, x, scenario_items, args.loss, args.atol, args.rtol, args.use_nvtx)
        for row in accuracy_rows:
            print(
                f"ACCURACY name={row.name!r} status={row.status} "
                f"output_max_abs={row.output_max_abs:.9f} "
                f"input_grad_max_abs={row.input_grad_max_abs:.9f} "
                f"worst_param={row.worst_param} "
                f"worst_param_max_abs={row.worst_param_max_abs:.9f}"
            )
        perf_rows = benchmark(model, x, scenario_items, args.loss, args.warmup, args.repeats, args.rounds, args.use_nvtx)
        for row in perf_rows:
            print(
                f"PERF name={row.name!r} mean_us={row.mean_us:.3f} "
                f"median_us={row.median_us:.3f} min_us={row.min_us:.3f} "
                f"max_us={row.max_us:.3f} speedup_vs_baseline={row.speedup:.3f}"
            )
        if args.fail_on_accuracy and any(row.status != "PASS" for row in accuracy_rows):
            raise SystemExit(1)
    finally:
        set_env({})
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
