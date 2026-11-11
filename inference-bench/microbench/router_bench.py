"""Sigmoid top-k router: fused tl.topk kernel vs the compiled inference router.

Correctness: same expert set per token as the compiled path, probabilities within fp32 rounding
(compared as dense [tokens, experts] matrices, so output order does not matter).
Timing: 48 calls (one per MoE layer) replayed from a CUDA graph, per-call us.

    python inference-bench/microbench/router_bench.py
"""

import statistics

import torch

from megatron.core.inference.moe.sigmoid_topk import sigmoid_topk_routing
from megatron.core.transformer.moe.router import InferenceTopKRouter

CALLS = 48
SHAPES = [("ultra", 512, 22, 5.0), ("nano", 128, 6, 2.5)]
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096]
CONFIGS = [(1, 4), (1, 8), (2, 4), (4, 4), (4, 8)]  # (tokens per program, warps)


def compiled(logits, bias, topk, scale):
    return InferenceTopKRouter._compiled_topk_routing(
        logits, topk, False, None, None, scale, "sigmoid", bias, False, None, True, None
    )


def dense(probs, idx, num_experts):
    out = torch.zeros(probs.shape[0], num_experts, device=probs.device)
    return out.scatter_(1, idx, probs)


def time_graph(fn, inputs, reps=50):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x in inputs[:3]:
            fn(*x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x in inputs:
            fn(*x)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / len(inputs))
    return statistics.median(times)


def main():
    torch._dynamo.config.cache_size_limit = 256
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()}; per-call us, {CALLS} calls per CUDA graph")
    for name, experts, topk, scale in SHAPES:
        print(f"\n== {name}: {experts} experts, top-{topk}")
        # Correctness first.
        worst = 0.0
        for t in TOKENS:
            logits = torch.randn(t, experts, device="cuda") * 2
            bias = torch.randn(experts, device="cuda") * 0.05
            with torch.inference_mode():
                rp, ri = compiled(logits, bias, topk, scale)
                kp, ki = sigmoid_topk_routing(logits, bias, topk, scale)
            same_set = torch.equal(torch.sort(ri, dim=1).values, torch.sort(ki, dim=1).values)
            diff = (dense(rp, ri, experts) - dense(kp, ki, experts)).abs().max().item()
            worst = max(worst, diff)
            assert same_set, f"expert sets differ at {t} tokens"
        print(f"   correctness: identical expert sets at all token counts; max |prob diff| {worst:.2e}")

        header = f"{'tokens':>7} | {'compiled':>9} | " + " | ".join(f"T{b}w{w}".rjust(7) for b, w in CONFIGS)
        print(header)
        print("-" * len(header))
        for t in TOKENS:
            inputs = [
                (torch.randn(t, experts, device="cuda") * 2, torch.randn(experts, device="cuda") * 0.05)
                for _ in range(CALLS)
            ]
            with torch.inference_mode():
                ref = time_graph(lambda lg, b: compiled(lg, b, topk, scale), inputs)
                cells = []
                for block, warps in CONFIGS:
                    us = time_graph(
                        lambda lg, b, block=block, warps=warps: sigmoid_topk_routing(
                            lg, b, topk, scale, block_tokens=block, num_warps=warps
                        ),
                        inputs,
                    )
                    cells.append(f"{us:7.2f}")
            print(f"{t:7d} | {ref:9.2f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
