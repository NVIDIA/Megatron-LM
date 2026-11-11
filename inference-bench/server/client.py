"""Static throughput sweep against a running mcore server.

For each batch size, sends `batch_size` prompts at once with a fixed output length
(ignore_eos), waits for all of them, and reports input and output throughput and ms per output
token. Each iteration uses the next `batch_size` prompts, in dataset order, so runs are
comparable.

Prompts:
  --dataset gsm8k     GSM8K test questions (~100 tokens each, via HF datasets).
  --dataset sharegpt  ShareGPT prompts from --sharegpt-path (<= 1024 tokens each); --min-isl
                      keeps only prompts with at least that many tokens.
  --isl N             joins consecutive prompts into ~N-token prompts (any dataset).
  --exact-isls N ...  sends prompts of exactly N token ids each (needs --tokenizer): the dataset
                      text is tokenized into one stream and cut into N-token slices. Sweeps
                      every N for every batch size; with batch size 1, each request is one
                      prefill step as long as N fits the server's max tokens per step.
Prefill-heavy runs: combine a long prompt setting with --osl 1.

--profile wraps the timed iterations of the FIRST batch size in /start_profile and
/stop_profile (for a server launched under nsys with --capture-range=cudaProfilerApi), then
stops: with --capture-range-end=stop-shutdown the server exits once profiling stops.

    python3 inference-bench/server/client.py --server http://localhost:5000 \
        --batch-sizes 1 16 64 --osl 1024 --iters 3 --warmup 1 --out bench.json
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


SHAREGPT_PATH = "sharegpt_filtered_benchmark.json"


def load_prompts(dataset, isl, min_isl, sharegpt_path):
    """Prompts in dataset order; with `isl`, consecutive prompts joined to ~isl tokens each."""
    if dataset == "gsm8k":
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split="test")
        questions = [row["question"] for row in rows]
    else:
        # Entries are [prompt, prompt_tokens, output_tokens].
        with open(sharegpt_path) as f:
            questions = [p for p, n, _ in json.load(f) if n >= min_isl]
    if not isl:
        return questions
    prompts, current = [], []
    for q in questions * (1 + 4 * isl // sum(map(len, questions))):
        current.append(q)
        if sum(map(len, current)) >= 4 * isl:  # ~4 characters per token
            prompts.append("\n\n".join(current))
            current = []
    return prompts


def token_stream(prompts, tokenizer_path, needed):
    """Concatenated token ids of `prompts` (cycled), at least `needed` long."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ids, i = [], 0
    while len(ids) < needed:
        ids.extend(tokenizer.encode(prompts[i % len(prompts)], add_special_tokens=False))
        i += 1
    return ids


def complete(server, prompt, osl, timeout):
    payload = {
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0.0,
        "ignore_eos": True,
        "stop_token_ids": [],
    }
    req = urllib.request.Request(
        f"{server}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    end = time.time()
    usage = body["usage"]
    return end - start, usage["prompt_tokens"], usage["completion_tokens"], start, end


def post(server, path):
    req = urllib.request.Request(f"{server}{path}", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def run_batch(server, prompts, osl, timeout):
    start = time.time()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(pool.map(lambda p: complete(server, p, osl, timeout), prompts))
    wall = time.time() - start
    latencies = sorted(r[0] for r in results)
    prompt_tokens = sum(r[1] for r in results)
    tokens = sum(r[2] for r in results)
    return {
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "tokens": tokens,
        "input_tok_s": prompt_tokens / wall,
        "throughput_tok_s": tokens / wall,
        "ms_per_output_token": 1000 * wall / osl,
        "p50_latency_s": statistics.median(latencies),
        "p99_latency_s": latencies[int(0.99 * (len(latencies) - 1))],
        # Absolute (epoch) batch and per-request send/receive times, to line up with server logs.
        "start_time": start,
        "requests": [[round(r[3], 3), round(r[4], 3)] for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://localhost:5000")
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--dataset", choices=["gsm8k", "sharegpt"], default="gsm8k")
    parser.add_argument("--sharegpt-path", default=SHAREGPT_PATH)
    parser.add_argument("--min-isl", type=int, default=0, help="sharegpt: min prompt tokens")
    parser.add_argument("--isl", type=int, default=0, help="join prompts to ~isl tokens (0 = off)")
    parser.add_argument("--exact-isls", type=int, nargs="*", default=None)
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer for --exact-isls")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", action="store_true", help="profile the first batch size only")
    parser.add_argument(
        "--profile-all",
        action="store_true",
        help="run every batch size's warmup first, then profile all timed iterations in one "
        "capture, with --pause seconds of idle between batch sizes to tell them apart",
    )
    parser.add_argument("--pause", type=float, default=0.0, help="seconds idle between batch sizes")
    args = parser.parse_args()

    questions = load_prompts(args.dataset, args.isl, args.min_isl, args.sharegpt_path)
    print(f"{len(questions)} prompts from {args.dataset}", flush=True)
    isls = args.exact_isls or [None]
    stream = None
    if args.exact_isls:
        assert args.tokenizer, "--exact-isls needs --tokenizer"
        n_requests = (args.warmup + args.iters) * sum(args.batch_sizes)
        stream = token_stream(questions, args.tokenizer, n_requests * max(isls) * len(isls))
        print(f"token stream: {len(stream)} ids for exact ISLs {isls}", flush=True)
    cursor = 0
    summary = []
    done = False

    def next_prompts(isl, bs):
        nonlocal cursor
        if isl is None:
            prompts = [questions[(cursor + j) % len(questions)] for j in range(bs)]
            cursor += bs
        else:
            prompts = [stream[cursor + j * isl : cursor + (j + 1) * isl] for j in range(bs)]
            cursor += bs * isl
        return prompts

    if args.profile_all:
        # Warm every batch size up front so the capture holds only timed iterations.
        for isl in isls:
            for bs in args.batch_sizes:
                for _ in range(args.warmup):
                    run_batch(args.server, next_prompts(isl, bs), args.osl, args.timeout)
        time.sleep(args.pause)
        print(f"start_profile: {post(args.server, '/start_profile')}", flush=True)
        time.sleep(2)  # the start signal reaches the engines asynchronously
    first = True
    for isl in isls:
        for bs in args.batch_sizes:
            if args.profile_all and not first:
                time.sleep(args.pause)
            first = False
            print(f"{time.time():.3f} start isl={isl} bs={bs}", flush=True)
            timed = []
            # --profile-all already ran the warmups up front.
            for i in range(args.warmup if args.profile_all else 0, args.warmup + args.iters):
                if args.profile and i == args.warmup:
                    print(f"start_profile: {post(args.server, '/start_profile')}", flush=True)
                    time.sleep(2)  # the start signal reaches the engines asynchronously
                stats = run_batch(args.server, next_prompts(isl, bs), args.osl, args.timeout)
                if i >= args.warmup:
                    timed.append(stats)
            if args.profile:
                time.sleep(1)
                print(f"stop_profile: {post(args.server, '/stop_profile')}", flush=True)
            row = {
                "batch_size": bs,
                "osl": args.osl,
                "dataset": args.dataset,
                "isl": args.isl,
                "exact_isl": isl,
                "min_isl": args.min_isl,
                "iters": timed,
                "wall_s": statistics.fmean(s["wall_s"] for s in timed),
                "prompt_tokens": statistics.fmean(s["prompt_tokens"] for s in timed),
                "input_tok_s": statistics.fmean(s["input_tok_s"] for s in timed),
                "throughput_tok_s": statistics.fmean(s["throughput_tok_s"] for s in timed),
                "ms_per_output_token": statistics.fmean(s["ms_per_output_token"] for s in timed),
                "p50_latency_s": statistics.fmean(s["p50_latency_s"] for s in timed),
            }
            summary.append(row)
            walls = " ".join(f"{1000 * s['wall_s']:.1f}" for s in timed)
            print(
                f"isl={isl or '-':>5} bs={bs:4d}  in={row['prompt_tokens']:8.0f} tok  "
                f"wall={1000 * row['wall_s']:8.1f} ms [{walls}]  in={row['input_tok_s']:9.1f} tok/s  "
                f"out={row['throughput_tok_s']:9.1f} tok/s  p50={row['p50_latency_s']:.3f}s",
                flush=True,
            )
            if args.profile:
                done = True
                break
        if done:
            break
    if args.profile_all:
        time.sleep(1)
        print(f"stop_profile: {post(args.server, '/stop_profile')}", flush=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
