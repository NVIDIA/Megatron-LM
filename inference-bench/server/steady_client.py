"""Steady-state closed-loop load: throughput and interactivity at fixed concurrency.

At each concurrency level c, c workers each keep one chat request in flight on the math blend
(AIME 2025 + HMMT Feb 2025 + BeyondAIME, see rollouts/collect_rollouts.py), sending the next
problem as soon as the previous reply arrives. No artificial output lengths: the model stops at
its own EOS, with server-default sampling, so long-running requests are part of the load. By
default there is no output cap (the server allows up to its max sequence length); --max-tokens
sets one.

Levels ramp up (c sorted ascending) without draining: going from c1 to c2 adds c2 - c1 workers,
and the in-flight requests carry over into the next level's steady state. Per level:
  - warm up for --warmup seconds, then measure for --window seconds;
  - throughput: the server's /metrics generation-token counter (summed over engines), read at
    the start and end of the window; exact to an engine step;
  - interactivity: per request, end-to-end latency / output tokens (TPOT including TTFT, since
    the user waits for the first token too) over the requests that finished inside the window;
    reported as TPOT percentiles and tokens/s/user = 1000 / TPOT. Unlike InferenceX, TTFT is
    not subtracted. The p90 / p99 TPOT (slowest requests) are the low end of tokens/s/user.

    python3 inference-bench/server/steady_client.py --server http://<host>:5000 \
        --concurrency 4 8 16 --out <exp dir>/steady.jsonl
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rollouts"))
from collect_rollouts import PROMPT_SUFFIX, load_rows  # noqa: E402


# vLLM's names for the mcore:* counters (vLLM's prompt-token counter includes cached tokens).
VLLM_METRICS = {
    "generation_tokens_total": "generation_tokens_total",
    "prompt_tokens_total": "prompt_tokens_total",
    "prompt_tokens_cached_total": "prompt_tokens_cached_total",
    "request_success_total": "requests_finished_total",
    "iteration_tokens_total_count": "engine_steps_total",
    "num_requests_running": "requests_running",
    "num_requests_waiting": "requests_waiting",
    # Disaggregated mcore servers only (prefill-pool requests handed off to decode).
    "prefill_handoffs_total": "prefill_handoffs_total",
}


def load_wildchat(limit=5000, seed=0):
    """First user turns of WildChat-1M (real ChatGPT conversations, many languages and tasks):
    one parquet shard, non-empty, under 8K characters, de-duplicated, shuffled, up to `limit`."""
    from datasets import load_dataset

    # One shard: skip the split-size check, which expects the full dataset.
    ds = load_dataset("allenai/WildChat-1M", data_files="data/train-00000-of-00014.parquet",
                      split="train", verification_mode="no_checks")
    seen, prompts = set(), []
    for conv in ds["conversation"]:
        turn = next((t["content"] for t in conv if t["role"] == "user"), "").strip()
        if turn and len(turn) < 8000 and turn not in seen:
            seen.add(turn)
            prompts.append(turn)
    random.Random(seed).shuffle(prompts)
    return prompts[:limit]


def parse_histograms(text, name="step_time_ms"):
    """mcore histogram `name` from /metrics text: {scope: {class: {"buckets": {le: cumulative
    count}, "sum", "count"}}}, scope an engine identity or "all" (the sum over engines)."""
    out = {}
    for line in text.splitlines():
        if not line.startswith(f"mcore:{name}_"):
            continue
        series, value = line.rsplit(None, 1)
        kind, _, labels = series[len(f"mcore:{name}_"):].partition("{")
        labels = dict(kv.split("=", 1) for kv in labels.rstrip("}").split(","))
        labels = {k: v.strip('"') for k, v in labels.items()}
        hist = out.setdefault(labels.get("engine", "all"), {}).setdefault(
            labels["step"], {"buckets": {}, "sum": 0.0, "count": 0})
        if kind == "bucket":
            hist["buckets"][labels["le"]] = int(float(value))
        elif kind == "sum":
            hist["sum"] = float(value)
        elif kind == "count":
            hist["count"] = int(float(value))
    return out


def diff_histograms(h0, h1):
    """Histograms over a window: h1 - h0 per scope and class (h0 lacks classes not yet seen),
    with bucket counts made per-bucket (not cumulative) and the mean step time added."""
    out = {}
    for scope, classes in h1.items():
        for step_class, hist in classes.items():
            before = h0.get(scope, {}).get(step_class, {"buckets": {}, "sum": 0.0, "count": 0})
            count = hist["count"] - before["count"]
            if count <= 0:
                continue
            cumulative = {le: n - before["buckets"].get(le, 0) for le, n in hist["buckets"].items()}
            buckets, prev = {}, 0
            for le, n in cumulative.items():  # bounds in ascending order, +Inf last
                buckets[le], prev = n - prev, n
            ms = hist["sum"] - before["sum"]
            out.setdefault(scope, {})[step_class] = {
                "count": count, "mean_ms": ms / count, "buckets": buckets}
    return out


STEP_MIX_FIELDS = ("steps", "tokens", "wall_ms")


def parse_step_mix(text):
    """mcore step-mix counters from /metrics text: {scope: {class: {"steps", "tokens",
    "wall_ms"}}}, scope an engine identity (prefill/<id> or decode/<id> behind a disaggregated
    frontend) or "all" (the sum over engines)."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("mcore:step_mix_"):
            continue
        series, value = line.rsplit(None, 1)
        field = series[len("mcore:step_mix_"):].split("{")[0].removesuffix("_total")
        if field not in STEP_MIX_FIELDS:
            continue
        labels = dict(kv.split("=", 1) for kv in series.split("{", 1)[1].rstrip("}").split(","))
        labels = {k: v.strip('"') for k, v in labels.items()}
        entry = out.setdefault(labels.get("engine", "all"), {}).setdefault(
            labels["step"], dict.fromkeys(STEP_MIX_FIELDS, 0.0))
        entry[field] = float(value)
    return out


def _pool(engine):
    """"prefill" / "decode" for an engine of a disaggregated server (main's native PD names
    engines prefill_s<i>_dp<j> / decode_s<i>_dp<j>), else None."""
    for role in ("prefill", "decode"):
        if engine.startswith((f"{role}_", f"{role}/")):
            return role
    return None


def diff_step_mix(m0, m1):
    """Step-mix counters over a window: m1 - m0 per scope and class, with the mean step time
    and each class's share of the scope's steps and wall time. "all" sums engines, so its
    shares are over engine-steps; behind a disaggregated frontend, the per-pool sums are added
    as "prefill" and "decode"."""
    out = {}
    for scope, classes in m1.items():
        for step_class, counters in classes.items():
            before = m0.get(scope, {}).get(step_class, {})
            delta = {f: counters[f] - before.get(f, 0.0) for f in STEP_MIX_FIELDS}
            if delta["steps"] > 0:
                out.setdefault(scope, {})[step_class] = delta
    for scope in [s for s in out if _pool(s)]:
        pool = out.setdefault(_pool(scope), {})
        for step_class, delta in out[scope].items():
            acc = pool.setdefault(step_class, dict.fromkeys(STEP_MIX_FIELDS, 0.0))
            for f in STEP_MIX_FIELDS:
                acc[f] += delta[f]
    for classes in out.values():
        steps = sum(d["steps"] for d in classes.values())
        wall = sum(d["wall_ms"] for d in classes.values())
        for d in classes.values():
            d["mean_ms"] = d["wall_ms"] / d["steps"]
            d["step_share"] = d["steps"] / steps
            d["time_share"] = d["wall_ms"] / wall if wall > 0 else float("nan")
    return out


def read_engine_gauges(server, names=("requests_running", "requests_waiting", "kv_tokens",
                                      "min_context_tokens", "p50_context_tokens",
                                      "p90_context_tokens", "p99_context_tokens",
                                      "max_context_tokens")):
    """{engine: {gauge: value}} from the per-engine (labeled) mcore:* lines of /metrics."""
    with urllib.request.urlopen(f"{server}/metrics", timeout=120) as resp:
        text = resp.read().decode()
    out = {}
    for line in text.splitlines():
        if not line.startswith("mcore:") or '{engine="' not in line:
            continue
        series, value = line.rsplit(None, 1)
        name = series[len("mcore:"):].split("{")[0]
        engine = series.split('engine="', 1)[1].split('"', 1)[0]
        if name == "request_context_tokens":
            # Per active request: {request id: context tokens} under "requests".
            request_id = series.split('request="', 1)[1].split('"', 1)[0]
            out.setdefault(engine, {}).setdefault("requests", {})[request_id] = int(float(value))
        elif name in names:
            out.setdefault(engine, {})[name] = float(value)
    return out


def read_metrics(server):
    """The server's mcore:* metrics (or vLLM's equivalents). mcore prints each metric per
    engine (labeled) and as an unlabeled total, so the total is used when present; vLLM prints
    only labeled lines, which are summed. mcore's step-time histograms are under "step_time_ms"
    (see parse_histograms)."""
    with urllib.request.urlopen(f"{server}/metrics", timeout=120) as resp:
        text = resp.read().decode()
    totals, labeled_sums = {}, {}
    for line in text.splitlines():
        if not line.startswith(("mcore:", "vllm:")):
            continue
        name, value = line.rsplit(None, 1)
        labeled = "{" in name
        prefix, _, name = name.split("{")[0].partition(":")
        if prefix == "vllm":
            name = VLLM_METRICS.get(name)
        if name not in VLLM_METRICS.values():
            continue
        if labeled:
            labeled_sums[name] = labeled_sums.get(name, 0.0) + float(value)
        else:
            totals[name] = float(value)
    metrics = {name: totals.get(name, labeled_sums.get(name, 0.0)) for name in VLLM_METRICS.values()}
    metrics["step_time_ms"] = parse_histograms(text)
    metrics["step_mix"] = parse_step_mix(text)
    return metrics


def percentile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", required=True, help="total in-flight requests per level")
    ap.add_argument("--warmup", type=float, default=60.0, help="seconds before measuring each level")
    ap.add_argument("--window", type=float, default=300.0, help="measured seconds per level")
    ap.add_argument("--max-tokens", type=int, default=0, help="output cap; 0 = none (natural EOS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=1,
                    help="consecutive measurement windows per level (one record each), e.g. 12 x 300 s "
                         "to follow one level for an hour")
    ap.add_argument("--sample-interval", type=float, default=0,
                    help="seconds between samples of each engine's gauges (running and waiting "
                         "requests, KV tokens, context percentiles, and every active request's context) into "
                         "<out>.engines.jsonl; 0 = off")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="ignore EOS, so with --max-tokens every request generates exactly that many "
                         "tokens (a fixed OSL)")
    ap.add_argument("--stream", action="store_true",
                    help="stream responses (server-sent events): also records TTFT and the "
                         "TPOT after the first token")
    ap.add_argument("--stream-interval", type=int, default=1,
                    help="with --stream: tokens per streamed chunk after the first, which mcore "
                         "sends immediately (fewer chunks keep the client off the critical path "
                         "at high concurrency)")
    ap.add_argument("--dataset", choices=["math", "wildchat"], default="math",
                    help="math: the 160-problem math blend; wildchat: WildChat-1M first user turns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dataset == "math":
        rows = load_rows("hard_math")
        random.Random(args.seed).shuffle(rows)
        rows = [{"content": r["problem"] + PROMPT_SUFFIX} for r in rows]
    else:
        rows = [{"content": p} for p in load_wildchat(seed=args.seed)]
    print(f"{len(rows)} {args.dataset} prompts", flush=True)
    lock = threading.Lock()
    next_row = [0]
    completed = []  # (start, end, completion_tokens, prompt_tokens, finish_reason, first token time)
    errors = [0]
    consecutive_errors = [0]
    stop = threading.Event()
    # This many request errors in a row with no success means the server is gone: stop instead
    # of hammering it (a dead server refuses connections instantly).
    max_consecutive_errors = 50

    requests_log = open(os.path.splitext(args.out)[0] + ".requests.jsonl", "a")

    def request_streaming(req):
        """Read the server-sent events; returns (first token time, usage, finish reason)."""
        first, usage, finish = None, {}, None
        with urllib.request.urlopen(req, timeout=4 * 3600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if first is None and (delta.get("content") or delta.get("reasoning_content")
                                          or delta.get("reasoning")):
                        first = time.time()
                    finish = choice.get("finish_reason") or finish
        return first, usage, finish

    def worker():
        while not stop.is_set():
            with lock:
                row = rows[next_row[0] % len(rows)]
                next_row[0] += 1
            payload = {"messages": [{"role": "user", "content": row["content"]}]}
            if args.max_tokens > 0:
                payload["max_tokens"] = args.max_tokens
            if args.ignore_eos:
                payload["ignore_eos"] = True  # with --max-tokens: every request generates exactly that
            if args.stream:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}
                if args.stream_interval > 1:
                    payload["streaming_interval"] = args.stream_interval
            req = urllib.request.Request(f"{args.server}/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            start = time.time()
            try:
                if args.stream:
                    first, usage, finish = request_streaming(req)
                else:
                    with urllib.request.urlopen(req, timeout=4 * 3600) as resp:
                        body = json.loads(resp.read())
                    first, usage = None, body.get("usage", {})
                    finish = body["choices"][0].get("finish_reason")
                rec = (start, time.time(), usage.get("completion_tokens", 0),
                       usage.get("prompt_tokens", 0), finish, first)
                with lock:
                    completed.append(rec)
                    consecutive_errors[0] = 0
                    # Every finished request (any level, warmup included) for its OSL / ISL.
                    requests_log.write(json.dumps({
                        "start": round(rec[0], 3), "end": round(rec[1], 3),
                        "output_tokens": rec[2], "prompt_tokens": rec[3], "finish_reason": rec[4],
                        # Streaming only: when the first content token arrived.
                        "first_token": round(rec[5], 3) if rec[5] is not None else None,
                    }) + "\n")
                    requests_log.flush()
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors[0] += 1
                    consecutive_errors[0] += 1
                    if consecutive_errors[0] >= max_consecutive_errors:
                        stop.set()
                print(f"request error: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(1)

    # Fail fast if the server is not up (e.g. its job ended before this client started).
    try:
        urllib.request.urlopen(f"{args.server}/health", timeout=30).close()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"server {args.server} is not reachable: {type(e).__name__}: {e}")
    print(f"{len(rows)} problems; levels {sorted(args.concurrency)}; warmup {args.warmup:.0f} s, "
          f"window {args.window:.0f} s, max_tokens {args.max_tokens}", flush=True)
    if args.sample_interval > 0:
        samples_path = os.path.splitext(args.out)[0] + ".engines.jsonl"

        def sampler():
            with open(samples_path, "a") as f:
                while not stop.wait(args.sample_interval):
                    try:
                        gauges = read_engine_gauges(args.server)
                    except Exception as e:  # noqa: BLE001
                        print(f"engine sample failed: {type(e).__name__}: {e}", flush=True)
                        continue
                    f.write(json.dumps({"t": round(time.time(), 1), "engines": gauges}) + "\n")
                    f.flush()

        threading.Thread(target=sampler, daemon=True).start()
    threads = []
    out = open(args.out, "a")
    for c in sorted(args.concurrency):
        while len(threads) < c:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
        level_start = time.time()
        print(f"{level_start:.3f} level c={c}: warming up", flush=True)
        if stop.wait(args.warmup):
            break
        m0, t0 = read_metrics(args.server), time.time()
        for window in range(args.windows):
            if stop.wait(args.window):
                break
            m1, t1 = read_metrics(args.server), time.time()
            dt = t1 - t0
            with lock:
                done = [r for r in completed if t0 <= r[1] <= t1]
            tpot_ms = [1000.0 * (r[1] - r[0]) / r[2] for r in done if r[2] > 0]
            rec = {
                "concurrency": c,
                "window_index": window,
                "window_start_s": round(t0 - level_start, 1),
                "window_s": round(dt, 2),
                # Absolute bounds, to pool <out>.requests.jsonl over a level's windows.
                "window_t0": round(t0, 3),
                "window_t1": round(t1, 3),
                "output_tok_s": (m1["generation_tokens_total"] - m0["generation_tokens_total"]) / dt,
                "prompt_tok_s": (m1["prompt_tokens_total"] - m0["prompt_tokens_total"]) / dt,
                "cached_prompt_tok_s": (m1["prompt_tokens_cached_total"] - m0["prompt_tokens_cached_total"]) / dt,
                "requests_finished": int(m1["requests_finished_total"] - m0["requests_finished_total"]),
                "prefill_handoffs": int(m1["prefill_handoffs_total"] - m0["prefill_handoffs_total"]),
                "engine_steps": int(m1["engine_steps_total"] - m0["engine_steps_total"]),
                "running_at_end": int(m1["requests_running"]),
                "waiting_at_end": int(m1["requests_waiting"]),
                "client_finished": len(done),
                "tpot_ms_p50": percentile(tpot_ms, 0.5),
                "tpot_ms_mean": sum(tpot_ms) / len(tpot_ms) if tpot_ms else float("nan"),
                "tpot_ms_p90": percentile(tpot_ms, 0.9),
                "tpot_ms_p99": percentile(tpot_ms, 0.99),
                "output_tokens_p50": percentile([r[2] for r in done], 0.5),
                "truncated": sum(r[4] == "length" for r in done),
                "errors": errors[0],
            }
            # Tokens/s/user = 1000 / TPOT, InferenceX style (median, and the slow tail at p90).
            rec["user_tok_s_p50"] = 1000.0 / rec["tpot_ms_p50"]
            rec["user_tok_s_p90_slow"] = 1000.0 / rec["tpot_ms_p90"]
            if args.stream:
                # Streaming only: TTFT, and the per-token time after the first token (InferenceX TPOT).
                ttft_ms = [1000.0 * (r[5] - r[0]) for r in done if r[5] is not None]
                decode_ms = [1000.0 * (r[1] - r[5]) / (r[2] - 1) for r in done
                             if r[5] is not None and r[2] > 1]
                rec.update(stream=True,
                           ttft_ms_p50=percentile(ttft_ms, 0.5), ttft_ms_p90=percentile(ttft_ms, 0.9),
                           decode_tpot_ms_p50=percentile(decode_ms, 0.5),
                           decode_tpot_ms_p90=percentile(decode_ms, 0.9))
            # Cross-check only: equals the per-user rate only if every worker is always decoding.
            rec["output_tok_s_over_c"] = rec["output_tok_s"] / c
            # Timed (logging) steps in the window by class, per engine and summed ("all").
            rec["step_time_ms"] = diff_histograms(m0["step_time_ms"], m1["step_time_ms"])
            # Every step in the window by class (count, tokens, host wall time), per engine,
            # summed ("all") and, behind a disaggregated frontend, per pool.
            rec["step_mix"] = diff_step_mix(m0["step_mix"], m1["step_mix"])
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"c={c:5d}  output {rec['output_tok_s']:9.1f} tok/s  "
                  f"interactivity p50 {rec['user_tok_s_p50']:6.1f} tok/s/user "
                  f"(TPOT p50 {rec['tpot_ms_p50']:.2f} ms, p90 {rec['tpot_ms_p90']:.2f} ms; "
                  f"{len(done)} finished)  "
                  f"running {rec['running_at_end']} waiting {rec['waiting_at_end']}  "
                  f"truncated {rec['truncated']}  errors {errors[0]}", flush=True)
            if args.stream:
                print(f"         streaming: TTFT p50 {rec['ttft_ms_p50']:.0f} ms, p90 {rec['ttft_ms_p90']:.0f} ms; "
                      f"TPOT after the first token p50 {rec['decode_tpot_ms_p50']:.2f} ms, "
                      f"p90 {rec['decode_tpot_ms_p90']:.2f} ms", flush=True)
            # Disaggregated: one summary per pool; aggregated: the sum over engines.
            scopes = [s for s in ("prefill", "decode") if s in rec["step_mix"]] or ["all"]
            for scope in scopes:
                for step_class, d in sorted(rec["step_mix"].get(scope, {}).items()):
                    print(f"         {scope:7s} {step_class:18s} {int(d['steps']):7d} steps "
                          f"({100 * d['step_share']:5.1f}% of steps, "
                          f"{100 * d['time_share']:5.1f}% of time), mean {d['mean_ms']:7.2f} ms",
                          flush=True)
            for step_class, h in sorted(rec["step_time_ms"].get("all", {}).items()):
                print(f"         {step_class:18s} {h['count']:6d} timed steps, mean {h['mean_ms']:8.2f} ms",
                      flush=True)
            m0, t0 = m1, t1
        if stop.is_set():
            break
    failed = stop.is_set()
    stop.set()
    out.close()
    if failed:
        sys.exit(f"stopped after {max_consecutive_errors} consecutive request errors; is the server up?")
    print("done", flush=True)


if __name__ == "__main__":
    main()
