"""CUDA-graph latency benchmark for the GDP decode kernels on GB200.

Run inside a GPU allocation with the Megatron-LM CI container::

    uv run python tests/unit_tests/ssm/ops/gdp/benchmark_gdp_inference.py

Each JSON line records independent graph-replay samples in microseconds per
call. The 32-request case matches the GDP 800M inference recipe's kernel shape.
"""

import argparse
import json
import subprocess

import torch

from megatron.core.ssm.ops.gdp.decode_prepare import gdp_decode_prepare
from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update


def make_case(batch, tokens, padding=False):
    torch.manual_seed(2026)
    m, h, groups, state_dim, value_dim = 3, 32, 8, 128, 64
    width = m * h * value_dim + (m + 1) * groups * state_dim
    packed = torch.randn(batch, tokens, width + (m + 1) * h, device="cuda", dtype=torch.bfloat16)
    x, ba = packed.split([width, (m + 1) * h], dim=-1)
    a_log = torch.zeros(h, device="cuda", dtype=torch.float32)
    dt_bias = torch.ones(h, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(batch, h, state_dim, value_dim, device="cuda", dtype=torch.float32)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    if padding:
        indices[-1] = -1
    snapshots = (
        torch.empty(batch, tokens, h, state_dim, value_dim, device="cuda", dtype=torch.float32)
        if tokens > 1
        else None
    )

    def prepare():
        return gdp_decode_prepare(x, ba, a_log, dt_bias, m, h, groups, value_dim, state_dim)

    q, k, v, beta, g = prepare()

    def recurrent():
        return fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=state,
            state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            intermediate_states=snapshots,
            steps_per_token=m,
        )

    return prepare, recurrent


def bench_graph(fn, graph_calls, samples):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000 / graph_calls)
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-kernel", choices=["prepare", "recurrent"])
    parser.add_argument("--batch", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--graph-calls", type=int, default=64)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "run on a GPU compute node"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    cases = (
        [(args.batch, args.tokens, False)]
        if args.batch is not None and args.tokens is not None
        else [(batch, tokens, False) for batch in (1, 8, 32) for tokens in (1, 4)]
        + [(32, 1, True), (32, 4, True)]
    )
    for batch, tokens, padding in cases:
        prepare, recurrent = make_case(batch, tokens, padding)
        for name, fn in (("prepare", prepare), ("recurrent", recurrent)):
            if args.profile_kernel:
                if name == args.profile_kernel:
                    fn()
                    torch.cuda.synchronize()
                continue
            samples = bench_graph(fn, args.graph_calls, args.samples)
            print(
                json.dumps(
                    {
                        "revision": revision,
                        "gpu": torch.cuda.get_device_name(),
                        "kernel": name,
                        "batch": batch,
                        "tokens": tokens,
                        "householder": 3,
                        "heads": 32,
                        "groups": 8,
                        "state_dim": 128,
                        "value_dim": 64,
                        "activation_dtype": "bfloat16",
                        "state_dtype": "float32",
                        "padding_slot": padding,
                        "snapshots": tokens > 1,
                        "graph_calls": args.graph_calls,
                        "samples_us": samples,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
