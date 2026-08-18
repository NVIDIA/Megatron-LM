# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Capacity benchmark for the inference frontend and coordinator, with no model.

Answers "how many requests per second can the frontend and coordinator move, and
how many can be in flight before latency falls apart?" without a checkpoint, a
GPU or torch.distributed. A real coordinator process is driven by a real
InferenceClient; the engines are ZMQ stand-ins that hold each request for
``--engine-latency-ms`` and then reply. Because the stand-ins are far faster than
any real engine, the numbers here are an upper bound on the serving path and
regress only when the frontend or coordinator itself gets slower.

Which layers are included depends on the flags:

  ``--mode client``       InferenceClient -> coordinator -> engines. Isolates the
                          ZMQ hop and the coordinator's routing loop.
  ``--mode http``         aiohttp -> Quart app -> InferenceClient -> coordinator
                          -> engines. Adds HTTP parsing, tokenization and OpenAI
                          response assembly.
  ``--mode http --streaming``
                          Also pays one detokenize step and one SSE flush per
                          generated token, normally the frontend's largest
                          per-token cost.

Running all three and comparing localizes a regression to a layer.

The concurrency sweep offers a fixed number of in-flight requests at each level
for ``--seconds-per-level`` and records sustained throughput and latency
percentiles. ``max_stable_concurrency`` is the largest level still served at
near-linear throughput, i.e. where the path saturates.

Note the http path measures a single frontend replica in-process, while
_run_text_gen_server forks four, so production HTTP capacity is higher than what
this reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import sys
import time
from pathlib import Path

# Import the standalone frontend harness that the unit tests use, so both share
# one definition of "a coordinator with fake engines".
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from megatron.core.inference.inference_client import InferenceClient  # noqa: E402
from megatron.core.inference.sampling_params import SamplingParams  # noqa: E402
from tests.unit_tests.inference.frontend_test_utils import (  # noqa: E402
    ByteLevelFastTokenizer,
    standalone_coordinator,
)

_SSE_DONE = b"data: [DONE]"


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def _build_prompt(tokenizer, num_input_tokens: int) -> tuple[list[int], str]:
    """Return a prompt of exactly ``num_input_tokens`` tokens, as ids and as text.

    Both forms describe the same prompt so the client and http modes put an
    identical number of tokens on the wire and their byte counts stay comparable.
    """
    tokens = tokenizer.tokenize("hello " * num_input_tokens)[:num_input_tokens]
    return tokens, tokenizer.detokenize(tokens)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LevelResult:
    """Throughput, latency and wire cost measured at one concurrency level."""

    def __init__(self, concurrency, completed, errors, latencies, wall, wire_delta):
        self.concurrency = concurrency
        self.completed = completed
        self.errors = errors
        self.wall = wall
        sorted_latencies = sorted(latencies)
        self.throughput = completed / wall if wall > 0 else 0.0
        self.avg_ms = 1000 * statistics.fmean(sorted_latencies) if sorted_latencies else 0.0
        self.p50_ms = 1000 * _percentile(sorted_latencies, 0.50)
        self.p99_ms = 1000 * _percentile(sorted_latencies, 0.99)
        self.wire_delta = wire_delta

    def as_metrics(self) -> dict:
        """Return this level's metrics in the flat shape compare_to_baseline reads."""
        per_request = max(1, self.completed)
        return {
            "concurrency": self.concurrency,
            "num_requests": self.completed,
            "errors": self.errors,
            "throughput_req_per_sec": self.throughput,
            "avg_latency_ms": self.avg_ms,
            "p50_latency_ms": self.p50_ms,
            "p99_latency_ms": self.p99_ms,
            "request_bytes_per_request": self.wire_delta["sent_bytes"] / per_request,
            "reply_bytes_per_request": self.wire_delta["received_bytes"] / per_request,
        }


def _wire_snapshot(client: InferenceClient) -> dict:
    snapshot = client.get_wire_metrics()
    return {"sent_bytes": snapshot["sent_bytes"], "received_bytes": snapshot["received_bytes"]}


async def _drive(submit, concurrency: int, seconds: float) -> tuple[int, int, list[float]]:
    """Keep ``concurrency`` requests in flight for ``seconds``.

    Args:
        submit: Zero-argument coroutine function that issues one request and
            returns once that request is complete.
        concurrency: Number of requests to keep outstanding.
        seconds: How long to sustain the offered load.

    Returns:
        tuple[int, int, list[float]]: completed count, error count, latencies.
    """
    deadline = time.perf_counter() + seconds
    latencies: list[float] = []
    errors = 0

    async def worker():
        nonlocal errors
        while time.perf_counter() < deadline:
            started = time.perf_counter()
            try:
                await submit()
            except Exception as exc:  # noqa: BLE001 - errors are a reported metric
                errors += 1
                if errors == 1:
                    print(f"  first error: {type(exc).__name__}: {exc}", flush=True)
            else:
                latencies.append(time.perf_counter() - started)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return len(latencies), errors, latencies


def _sampling_params(args: argparse.Namespace, streaming: bool) -> SamplingParams:
    return SamplingParams(
        num_tokens_to_generate=args.num_output_tokens,
        temperature=0.0,
        top_k=1,
        return_log_probs=False,
        streaming=streaming,
    )


def _make_client_submit(args, client, prompt_tokens):
    """Build a submit callable that goes straight through the InferenceClient."""
    if not args.streaming:

        async def submit():
            await client.add_request(list(prompt_tokens), _sampling_params(args, False))

        return submit

    async def submit_streaming():
        stream = client.add_request_streaming(list(prompt_tokens), _sampling_params(args, True))
        async for _ in stream:
            pass

    return submit_streaming


def _make_http_submit(args, session, url, prompt_text):
    """Build a submit callable that posts to /v1/completions like a real caller."""
    payload = {
        "prompt": prompt_text,
        "max_tokens": args.num_output_tokens,
        "temperature": 0.0,
        "stream": args.streaming,
    }

    if not args.streaming:

        async def submit():
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                await response.read()

        return submit

    async def submit_streaming():
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            async for line in response.content:
                if line.startswith(_SSE_DONE):
                    break

    return submit_streaming


async def _await_health(session, health_url, timeout=60.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            async with session.get(health_url) as response:
                if response.status == 200:
                    return
        except Exception:  # noqa: BLE001 - server is still binding
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError(f"frontend did not become ready at {health_url}")


async def _sweep(args, client, submit) -> list[LevelResult]:
    """Warm up, then measure every requested concurrency level in order."""
    print(
        f"\nWarmup ({args.warmup_seconds:.1f}s at concurrency {args.concurrency[0]})...", flush=True
    )
    await _drive(submit, args.concurrency[0], args.warmup_seconds)

    results = []
    for concurrency in args.concurrency:
        before = _wire_snapshot(client)
        started = time.perf_counter()
        completed, errors, latencies = await _drive(submit, concurrency, args.seconds_per_level)
        wall = time.perf_counter() - started
        after = _wire_snapshot(client)
        wire_delta = {key: after[key] - before[key] for key in before}
        result = LevelResult(concurrency, completed, errors, latencies, wall, wire_delta)
        results.append(result)
        print(
            f"  concurrency={concurrency:5d}  "
            f"throughput={result.throughput:9.1f} req/s  "
            f"p50={result.p50_ms:8.2f} ms  p99={result.p99_ms:8.2f} ms  "
            f"errors={errors}",
            flush=True,
        )
    return results


def _scaling_efficiency(result: LevelResult, base: LevelResult) -> float:
    """How close this level came to the throughput implied by linear scaling.

    Below saturation each added in-flight request adds its own throughput, so the
    expected rate is the lowest level's per-request rate times this level's
    concurrency. The ratio falls away from 1.0 once requests start queueing.

    This leans on the lowest level being both unsaturated and dominated by the
    fake engine's reply delay. Keep --engine-latency-ms well above the engine's
    1 ms poll granularity, or the reference rate comes out optimistically high
    and every later level looks saturated.
    """
    if base.concurrency == 0 or base.throughput == 0:
        return 0.0
    expected = base.throughput / base.concurrency * result.concurrency
    return result.throughput / expected


def _max_stable_concurrency(results: list[LevelResult], scaling_tolerance: float) -> int:
    """Largest offered concurrency still served at near-linear throughput.

    Stops at the first level that either errors or drops below
    ``scaling_tolerance`` of linear scaling, so the answer is where the path
    saturates rather than simply the highest level tried. Preferred over a
    latency-percentile knee because throughput is far less noisy than p99.

    Being a step function over the sampled grid, this is reported but not gated;
    ``peak_throughput_req_per_sec`` is the continuous metric CI compares.
    """
    if not results:
        return 0
    base = results[0]
    stable = base.concurrency
    for result in results:
        if result.errors or _scaling_efficiency(result, base) < scaling_tolerance:
            break
        stable = result.concurrency
    return stable


async def run_client_mode(args) -> list[LevelResult]:
    """Measure InferenceClient -> coordinator -> engines."""
    tokenizer = ByteLevelFastTokenizer()
    prompt_tokens, _ = _build_prompt(tokenizer, args.num_input_tokens)
    with standalone_coordinator(
        max_requests=args.max_requests,
        tokenizer=tokenizer,
        num_engines=args.num_engines,
        num_output_tokens=args.num_output_tokens,
        reply_delay_s=args.engine_latency_ms / 1000.0,
    ) as (addr, _engines):
        client = InferenceClient(addr)
        client.start()
        try:
            submit = _make_client_submit(args, client, prompt_tokens)
            return await _sweep(args, client, submit)
        finally:
            client.stop()


async def run_http_mode(args) -> list[LevelResult]:
    """Measure aiohttp -> Quart frontend -> InferenceClient -> coordinator -> engines."""
    import aiohttp
    from hypercorn.asyncio import serve
    from hypercorn.config import Config as HypercornConfig

    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (  # noqa: E501
        build_app,
    )

    tokenizer = ByteLevelFastTokenizer()
    _, prompt_text = _build_prompt(tokenizer, args.num_input_tokens)
    port = args.port or _free_port()

    with standalone_coordinator(
        max_requests=args.max_requests,
        tokenizer=tokenizer,
        num_engines=args.num_engines,
        num_output_tokens=args.num_output_tokens,
        reply_delay_s=args.engine_latency_ms / 1000.0,
    ) as (addr, _engines):
        client = InferenceClient(addr)
        client.start()
        app = build_app(client, tokenizer)

        config = HypercornConfig()
        config.bind = [f"127.0.0.1:{port}"]
        config.accesslog = None
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(serve(app, config, shutdown_trigger=shutdown.wait))

        connector = aiohttp.TCPConnector(limit=0)
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                base = f"http://127.0.0.1:{port}"
                await _await_health(session, f"{base}/v1/health")
                submit = _make_http_submit(args, session, f"{base}/v1/completions", prompt_text)
                return await _sweep(args, client, submit)
        finally:
            shutdown.set()
            try:
                await asyncio.wait_for(server_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                server_task.cancel()
            client.stop()


async def main(args) -> dict:
    """Run the sweep for one mode and return the results keyed for the baseline."""
    print(f"Mode            : {args.mode}")
    print(f"Streaming       : {args.streaming}")
    print(f"Engines         : {args.num_engines} (reply delay {args.engine_latency_ms} ms)")
    print(f"ISL / OSL       : {args.num_input_tokens} / {args.num_output_tokens}")
    print(f"Concurrency     : {args.concurrency}")
    print(f"Seconds / level : {args.seconds_per_level}", flush=True)

    if args.mode == "client":
        results = await run_client_mode(args)
    else:
        results = await run_http_mode(args)

    prefix = f"{args.mode}_stream" if args.streaming else args.mode
    base = results[0]
    entries = {}
    for result in results:
        metrics = result.as_metrics()
        metrics["scaling_efficiency"] = _scaling_efficiency(result, base)
        entries[f"{prefix}_concurrency_{result.concurrency}"] = metrics
    entries[f"{prefix}_summary"] = {
        "peak_throughput_req_per_sec": max((r.throughput for r in results), default=0.0),
        "max_stable_concurrency": _max_stable_concurrency(results, args.scaling_tolerance),
        "total_errors": sum(r.errors for r in results),
    }

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({prefix})")
    print(f"{'=' * 60}")
    for key, metrics in entries.items():
        print(f"  [{key}]")
        for name, value in metrics.items():
            print(f"    {name:30s} : {value}")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=["client", "http"],
        default="client",
        help="'client' isolates the coordinator; 'http' adds the Quart frontend.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream responses. In http mode this costs one SSE flush per token, "
        "which is the frontend's dominant per-token cost.",
    )
    parser.add_argument(
        "--concurrency",
        # Sorted because the lowest level is the unsaturated reference that
        # scaling_efficiency and max_stable_concurrency are measured against.
        type=lambda value: sorted(int(part) for part in value.split(",")),
        default=[1, 8, 32, 128, 512],
        help="Comma-separated in-flight request counts to sweep.",
    )
    parser.add_argument("--seconds-per-level", type=float, default=3.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--num-input-tokens", type=int, default=512)
    parser.add_argument("--num-output-tokens", type=int, default=64)
    parser.add_argument("--num-engines", type=int, default=1)
    parser.add_argument(
        "--engine-latency-ms",
        type=float,
        default=10.0,
        help="How long a fake engine holds a request before replying. Non-zero "
        "values are what make high concurrency levels meaningful.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1024,
        help="Coordinator max_requests, used by its load-aware routing scores.",
    )
    parser.add_argument(
        "--scaling-tolerance",
        type=float,
        default=0.8,
        help="Fraction of linear throughput scaling a level must still reach to "
        "count as unsaturated when computing max_stable_concurrency.",
    )
    parser.add_argument("--port", type=int, default=None, help="http mode bind port.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results to this path, merging into any existing file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    entries = asyncio.run(main(parsed))
    if parsed.output_json:
        out_path = Path(parsed.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(out_path.read_text()) if out_path.exists() else {}
        existing.update(entries)
        out_path.write_text(json.dumps(existing, indent=2))
        print(f"\nWrote results to {out_path}")
