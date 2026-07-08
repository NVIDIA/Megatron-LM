# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static throughput/latency benchmark against an OpenAI-compatible completions server.

Fires --batch-size requests simultaneously via asyncio.gather, waits for all to
finish, and reports throughput, latency (avg/p50/p99), and TPOT. Iterates over
warmup + timed batches and emits a JSON results file consumable by
compare_to_baseline.py.

Hits the server's POST /v1/completions endpoint directly via aiohttp — no
`openai` python client dep required. Vendored and adapted from
/Users/shanmugamr/inference-bench/static_benchmark.py.

Prompt source:
  --dataset synthetic   (default for dense models) — uses `"hello " * N` of
                        fixed length `--num-input-tokens`. Cheap, deterministic,
                        but every request is identical → not representative for
                        MoE/hybrid models because routers/dispatchers see the
                        same token-to-expert assignment each time.
  --dataset gsm8k       Loads prompts from data/gsm8k_prompts.jsonl (vendored
                        from openai/gsm8k test split, 256 prompts). Each
                        request in a batch gets the next prompt, cycling.
                        Required for MoE models per reviewer feedback on
                        PR #4917 — synthetic input gives misleading perf for
                        anything with token-dependent routing.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

_GSM8K_PATH = Path(__file__).parent / "data" / "gsm8k_prompts.jsonl"


def _load_gsm8k_prompts() -> list[str]:
    """Load the vendored gsm8k prompts. Returns the list of question strings."""
    if not _GSM8K_PATH.exists():
        raise FileNotFoundError(f"gsm8k prompt file not found at {_GSM8K_PATH}")
    prompts: list[str] = []
    for line in _GSM8K_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = obj.get("prompt") or obj.get("problem") or obj.get("question") or obj.get("text")
        if text:
            prompts.append(text)
    if not prompts:
        raise ValueError(f"no prompts loaded from {_GSM8K_PATH}")
    return prompts


async def _single_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    num_output_tokens: int,
    temperature: float,
) -> tuple[int, int, float]:
    """POST /v1/completions and return (input_tokens, output_tokens, latency_s)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": num_output_tokens,
        "temperature": temperature,
        "ignore_eos": True,
        "stop_token_ids": [],
    }
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=None)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        body = await resp.json()
    latency = time.perf_counter() - t0
    usage = body.get("usage", {})
    actual_input = usage.get("prompt_tokens") or 0
    actual_output = usage.get("completion_tokens")
    if actual_output is not None:
        assert (
            actual_output == num_output_tokens
        ), f"Expected {num_output_tokens} output tokens, got {actual_output}."
    return actual_input, actual_output or num_output_tokens, latency


async def _run_batch(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    url: str,
    prompts: list[str],
    iter_start_index: int,
) -> tuple[list[int], list[int], list[float], float]:
    """Fire batch_size requests in parallel. Cycles through `prompts` deterministically
    starting at `iter_start_index` so each timed iteration sees the same prompt
    distribution (reduces run-to-run variance for gsm8k mode)."""
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            _single_request(
                session,
                url,
                args.model,
                prompts[(iter_start_index + i) % len(prompts)],
                args.num_output_tokens,
                args.temperature,
            )
            for i in range(args.batch_size)
        ]
    )
    wall = time.perf_counter() - t0
    input_counts = [r[0] for r in results]
    output_counts = [r[1] for r in results]
    latencies = [r[2] for r in results]
    return input_counts, output_counts, latencies, wall


def _percentile(sorted_values: list[float], pct: float) -> float:
    idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
    return sorted_values[idx]


async def main(args: argparse.Namespace) -> dict:
    url = f"{args.server_url.rstrip('/')}/completions"

    if args.dataset == "gsm8k":
        prompts = _load_gsm8k_prompts()
        prompt_source = f"gsm8k ({len(prompts)} prompts)"
    elif args.dataset == "synthetic":
        prompts = [("hello " * args.num_input_tokens).strip()]
        prompt_source = f"synthetic (ISL={args.num_input_tokens})"
    else:
        raise ValueError(f"unknown --dataset {args.dataset!r}; expected 'synthetic' or 'gsm8k'")

    print(f"Server          : {args.server_url}")
    print(f"Model           : {args.model}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Dataset         : {prompt_source}")
    print(f"Output tokens   : {args.num_output_tokens}")
    print(f"Warmup iters    : {args.num_warmup_iters}")
    print(f"Timed iters     : {args.num_iters}", flush=True)

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        cursor = 0
        for i in range(args.num_warmup_iters):
            print(f"\nWarmup {i + 1}/{args.num_warmup_iters}...", flush=True)
            await _run_batch(session, args, url, prompts, cursor)
            cursor += args.batch_size

        all_wall: list[float] = []
        all_output_tokens: list[int] = []
        all_input_tokens: list[int] = []
        all_latencies: list[float] = []

        for i in range(args.num_iters):
            input_counts, output_counts, latencies, wall = await _run_batch(
                session, args, url, prompts, cursor
            )
            cursor += args.batch_size
            total_out = sum(output_counts)
            all_wall.append(wall)
            all_output_tokens.append(total_out)
            all_input_tokens.extend(input_counts)
            all_latencies.extend(latencies)
            print(
                f"Iter {i + 1}/{args.num_iters}: "
                f"wall={wall * 1000:.0f} ms, "
                f"throughput={total_out / wall:.1f} tok/s, "
                f"avg_latency={sum(latencies) / len(latencies) * 1000:.0f} ms",
                flush=True,
            )

    avg_wall = sum(all_wall) / len(all_wall)
    avg_out = sum(all_output_tokens) / len(all_output_tokens)
    throughput = avg_out / avg_wall
    sorted_lat = sorted(all_latencies)
    avg_latency_ms = sum(sorted_lat) / len(sorted_lat) * 1000
    p50_latency_ms = _percentile(sorted_lat, 0.50) * 1000
    p99_latency_ms = _percentile(sorted_lat, 0.99) * 1000
    tpot_ms = avg_wall * 1000 / args.num_output_tokens
    avg_input_tokens = sum(all_input_tokens) / len(all_input_tokens) if all_input_tokens else 0

    summary = {
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "num_input_tokens_avg": avg_input_tokens,
        "num_output_tokens": args.num_output_tokens,
        "num_iters": args.num_iters,
        "throughput_tok_per_sec": throughput,
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p99_latency_ms": p99_latency_ms,
        "tpot_ms_per_tok": tpot_ms,
    }

    print(f"\n{'=' * 50}")
    print(f"RESULTS (avg over {args.num_iters} iters, batch={args.batch_size})")
    print(f"{'=' * 50}")
    for k, v in summary.items():
        print(f"  {k:28s} : {v}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static throughput/latency benchmark")
    parser.add_argument("--server-url", default="http://localhost:5000/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "gsm8k"],
        default="synthetic",
        help="Prompt source. 'synthetic' uses 'hello '*N (deterministic, MoE-misleading); "
        "'gsm8k' uses the vendored 256-prompt gsm8k subset (real tokens, recommended for MoE).",
    )
    parser.add_argument(
        "--num-input-tokens",
        type=int,
        default=512,
        help="Synthetic-prompt length in tokens. Ignored when --dataset=gsm8k.",
    )
    parser.add_argument("--num-output-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-warmup-iters", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write a JSON summary of this batch to the given path. "
        "If the file exists, the new entry is merged in under key 'batch_<size>'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = asyncio.run(main(args))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if out_path.exists():
            existing = json.loads(out_path.read_text())
        existing[f"batch_{args.batch_size}"] = summary
        out_path.write_text(json.dumps(existing, indent=2))
        print(f"\nWrote summary to {out_path}")
