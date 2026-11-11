"""InferenceX-style load: random-token prompts at fixed input / output lengths.

Mirrors SemiAnalysis InferenceX (vllm bench serve --dataset-name random): each prompt is ISL'
random token ids and asks for exactly OSL' tokens with ignore_eos, where ISL' and OSL' are drawn
uniformly from [range_ratio * ISL, ISL] and [range_ratio * OSL, OSL]. Per concurrency level c,
--prompts-per-conc x c requests (InferenceX: 10) are sent through /v1/completions with at most c
in flight, streaming so TTFT and TPOT separate. Random tokens defeat prefix caching and make
routing independent of any dataset (no topic skew).

Per level, one JSON line in --out:
  - output_tok_s: generated tokens / wall time of the level (first send to last reply);
  - ttft_ms_{p50,p90,p99}, tpot_ms_{p50,p90} (TPOT = (last - first token) / (tokens - 1));
  - user_tok_s_p50 / user_tok_s_p90_slow = 1000 / TPOT p50 / p90 (InferenceX's interactivity);
  - step_time_ms: the server's step-time histograms over the level (as in steady_client.py).

    python3 inference-bench/server/inferencex_client.py --server http://<host>:5000 \
        --tokenizer <hf id or path> --isl 1024 --osl 1024 --concurrency 64 256 --out ix.jsonl
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steady_client import diff_histograms, percentile, read_metrics  # noqa: E402


def random_prompts(tokenizer, n, isl, osl, ratio, seed):
    """[(token ids, output length)] with lengths uniform in [ratio * len, len]; ids drawn from the
    vocabulary minus special tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    special = set(tok.all_special_ids)
    vocab = [i for i in range(len(tok)) if i not in special]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        n_in = rng.randint(max(1, int(ratio * isl)), isl)
        n_out = rng.randint(max(1, int(ratio * osl)), osl)
        out.append(([rng.choice(vocab) for _ in range(n_in)], n_out))
    return out


def run_request(server, prompt, n_out):
    """Stream one completion; returns (start, first token time, end, chunks with text)."""
    payload = {"prompt": prompt, "max_tokens": n_out, "ignore_eos": True, "stream": True,
               "temperature": 1.0}
    req = urllib.request.Request(f"{server}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    first = None
    with urllib.request.urlopen(req, timeout=4 * 3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            if first is None:
                first = time.time()
    return start, first or time.time(), time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--range-ratio", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, nargs="+", required=True)
    ap.add_argument("--prompts-per-conc", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = open(args.out, "a")
    print(f"InferenceX random: ISL {args.isl} OSL {args.osl} range ratio {args.range_ratio}; "
          f"levels {args.concurrency}; {args.prompts_per_conc} x c prompts each", flush=True)
    for level, c in enumerate(sorted(args.concurrency)):
        n = args.prompts_per_conc * c
        prompts = random_prompts(args.tokenizer, n, args.isl, args.osl, args.range_ratio,
                                 args.seed + level)
        results, errors, lock, cursor = [], [0], threading.Lock(), [0]

        def worker():
            while True:
                with lock:
                    if cursor[0] >= n:
                        return
                    prompt, n_out = prompts[cursor[0]]
                    cursor[0] += 1
                try:
                    start, first, end = run_request(args.server, prompt, n_out)
                    with lock:
                        results.append((start, first, end, len(prompt), n_out))
                except Exception as e:  # noqa: BLE001
                    with lock:
                        errors[0] += 1
                    print(f"request error: {type(e).__name__}: {str(e)[:200]}", flush=True)

        m0 = read_metrics(args.server)
        t0 = time.time()
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(c)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t1 = time.time()
        m1 = read_metrics(args.server)

        ttft = [1000 * (f - s) for s, f, e, _, _ in results]
        tpot = [1000 * (e - f) / (k - 1) for s, f, e, _, k in results if k > 1]
        rec = {
            "concurrency": c, "isl": args.isl, "osl": args.osl, "range_ratio": args.range_ratio,
            "requests": len(results), "errors": errors[0], "duration_s": t1 - t0,
            "output_tok_s": sum(r[4] for r in results) / (t1 - t0),
            "input_tok_s": sum(r[3] for r in results) / (t1 - t0),
            "ttft_ms_p50": percentile(ttft, 0.5), "ttft_ms_p90": percentile(ttft, 0.9),
            "ttft_ms_p99": percentile(ttft, 0.99),
            "tpot_ms_p50": percentile(tpot, 0.5), "tpot_ms_p90": percentile(tpot, 0.9),
            "step_time_ms": diff_histograms(m0["step_time_ms"], m1["step_time_ms"]),
        }
        rec["user_tok_s_p50"] = 1000.0 / rec["tpot_ms_p50"]
        rec["user_tok_s_p90_slow"] = 1000.0 / rec["tpot_ms_p90"]
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"c={c:5d}  output {rec['output_tok_s']:9.1f} tok/s  interactivity p50 "
              f"{rec['user_tok_s_p50']:6.1f} tok/s/user (TPOT p50 {rec['tpot_ms_p50']:.2f} ms, "
              f"p90 {rec['tpot_ms_p90']:.2f} ms)  TTFT p50 {rec['ttft_ms_p50']:.0f} ms  "
              f"{len(results)} done, {errors[0]} errors, {rec['duration_s']:.0f} s", flush=True)
        for step_class, h in sorted(rec["step_time_ms"].get("all", {}).items()):
            print(f"         {step_class:18s} {h['count']:6d} timed steps, mean {h['mean_ms']:8.2f} ms",
                  flush=True)


if __name__ == "__main__":
    main()
