"""Static throughput/latency benchmark against an OpenAI-compatible completions server.

Fires --batch-size requests simultaneously via asyncio.gather, waits for all to
finish, and reports throughput, latency (avg/p50/p99), and TPOT. Iterates over
warmup + timed batches and emits a JSON results file consumable by
compare_to_baseline.py.

Hits the server's POST /v1/completions endpoint directly via aiohttp — no
`openai` python client dep required. Vendored and adapted from
/Users/shanmugamr/inference-bench/static_benchmark.py.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


async def _single_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    expected_input_tokens: int,
    num_output_tokens: int,
    temperature: float,
) -> tuple[int, float]:
    """POST /v1/completions and return (tokens_generated, latency_s)."""
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
    actual_input = usage.get("prompt_tokens")
    actual_output = usage.get("completion_tokens")
    if actual_input is not None:
        assert actual_input == expected_input_tokens, (
            f"Expected {expected_input_tokens} prompt tokens, server saw {actual_input}."
        )
    if actual_output is not None:
        assert actual_output == num_output_tokens, (
            f"Expected {num_output_tokens} output tokens, got {actual_output}."
        )
    return actual_output or num_output_tokens, latency


async def _run_batch(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    url: str,
    prompt: str,
) -> tuple[list[int], list[float], float]:
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            _single_request(
                session,
                url,
                args.model,
                prompt,
                args.num_input_tokens,
                args.num_output_tokens,
                args.temperature,
            )
            for _ in range(args.batch_size)
        ]
    )
    wall = time.perf_counter() - t0
    token_counts = [r[0] for r in results]
    latencies = [r[1] for r in results]
    return token_counts, latencies, wall


def _percentile(sorted_values: list[float], pct: float) -> float:
    idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
    return sorted_values[idx]


async def main(args: argparse.Namespace) -> dict:
    url = f"{args.server_url.rstrip('/')}/completions"
    prompt = ("hello " * args.num_input_tokens).strip()

    print(f"Server          : {args.server_url}")
    print(f"Model           : {args.model}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Input tokens    : {args.num_input_tokens}")
    print(f"Output tokens   : {args.num_output_tokens}")
    print(f"Warmup iters    : {args.num_warmup_iters}")
    print(f"Timed iters     : {args.num_iters}", flush=True)

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(args.num_warmup_iters):
            print(f"\nWarmup {i + 1}/{args.num_warmup_iters}...", flush=True)
            await _run_batch(session, args, url, prompt)

        all_wall: list[float] = []
        all_tokens: list[int] = []
        all_latencies: list[float] = []

        for i in range(args.num_iters):
            token_counts, latencies, wall = await _run_batch(session, args, url, prompt)
            total_tokens = sum(token_counts)
            all_wall.append(wall)
            all_tokens.append(total_tokens)
            all_latencies.extend(latencies)
            print(
                f"Iter {i + 1}/{args.num_iters}: "
                f"wall={wall * 1000:.0f} ms, "
                f"throughput={total_tokens / wall:.1f} tok/s, "
                f"avg_latency={sum(latencies) / len(latencies) * 1000:.0f} ms",
                flush=True,
            )

    avg_wall = sum(all_wall) / len(all_wall)
    avg_tokens = sum(all_tokens) / len(all_tokens)
    throughput = avg_tokens / avg_wall
    sorted_lat = sorted(all_latencies)
    avg_latency_ms = sum(sorted_lat) / len(sorted_lat) * 1000
    p50_latency_ms = _percentile(sorted_lat, 0.50) * 1000
    p99_latency_ms = _percentile(sorted_lat, 0.99) * 1000
    tpot_ms = avg_wall * 1000 / args.num_output_tokens

    summary = {
        "batch_size": args.batch_size,
        "num_input_tokens": args.num_input_tokens,
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
    parser.add_argument("--num-input-tokens", type=int, default=512)
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
