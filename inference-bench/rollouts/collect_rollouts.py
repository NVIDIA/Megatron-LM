"""Collect natural-length reasoning rollouts from a running server, to study generation lengths.

Datasets (--dataset):
  math500    HuggingFaceH4/MATH-500 (500 problems, levels 1-5; mostly short traces)
  hard_math  ByteDance-Seed/BeyondAIME (100) + math-ai/aime25 (30) + MathArena/hmmt_feb_2025 (30):
             160 competition problems with integer answers, for long reasoning traces

Sends the problems to /v1/chat/completions
with the chat template and the model's reasoning on. No ignore_eos and no sampling parameters:
the server applies the model's generation_config.json defaults and every rollout ends at the
model's own EOS. max_tokens is only a high safety cap (the endpoint needs a number to compute
finish_reason); a rollout that reaches it is reported as finish_reason "length".

Closed loop: CONCURRENCY workers each keep one request in flight, so the server sees a steady load
until the queue drains. One JSON line per rollout (written as it completes) with prompt and
completion tokens, reasoning / answer character counts, finish reason, latency, MATH level and
subject, the extracted boxed answer and a rough exact-match check.

    python3 inference-bench/rollouts/collect_rollouts.py --server http://<host>:5000 \
        --out <exp dir>/rollouts.jsonl [--samples 4] [--concurrency 64]
"""

import argparse
import json
import queue
import re
import threading
import time
import urllib.request

PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text):
    """Contents of the last \\boxed{...} in text (balanced braces), or None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed{")
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            if depth == 0:
                return text[j:k]
            depth -= 1
    return None


def normalize(ans):
    return re.sub(r"\s+|\\!|\\,|\\left|\\right|\$", "", ans or "").replace("dfrac", "frac")


def load_rows(dataset):
    """Rows with unique_id, problem, answer (str), level and subject (source for hard_math)."""
    from datasets import load_dataset

    if dataset == "math500":
        return [{**r, "answer": str(r["answer"])} for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    rows = []
    for name, split, source in (
        ("ByteDance-Seed/BeyondAIME", "test", "BeyondAIME"),
        ("math-ai/aime25", "test", "AIME 2025"),
        ("MathArena/hmmt_feb_2025", "train", "HMMT Feb 2025"),
    ):
        for i, r in enumerate(load_dataset(name, split=split)):
            rows.append({"unique_id": f"{source}/{r.get('id', r.get('problem_idx', i))}",
                         "problem": r["problem"], "answer": str(r["answer"]).strip(),
                         "level": source, "subject": source})
    return rows


def chat(server, problem, max_tokens, timeout):
    payload = {"messages": [{"role": "user", "content": problem + PROMPT_SUFFIX}],
               "max_tokens": max_tokens}
    req = urllib.request.Request(f"{server}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body, start, time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--dataset", choices=["math500", "hard_math"], default="math500")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=4, help="rollouts per problem")
    ap.add_argument("--concurrency", type=int, default=64, help="requests kept in flight")
    ap.add_argument("--max-tokens", type=int, default=131072, help="safety cap only")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (0 = all 500)")
    ap.add_argument("--timeout", type=float, default=4 * 3600)
    args = ap.parse_args()

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    work = queue.Queue()
    for s in range(args.samples):  # sample-major, so a partial run still covers every problem
        for row in rows:
            work.put((row, s))
    total = work.qsize()
    print(f"{len(rows)} problems x {args.samples} samples = {total} rollouts, "
          f"{args.concurrency} in flight, max_tokens cap {args.max_tokens}", flush=True)

    lock, done = threading.Lock(), [0]
    out = open(args.out, "a")
    t0 = time.time()

    def worker():
        while True:
            try:
                row, s = work.get_nowait()
            except queue.Empty:
                return
            rec = {"unique_id": row["unique_id"], "sample": s, "level": row["level"],
                   "subject": row["subject"], "gold": row["answer"]}
            try:
                body, start, end = chat(args.server, row["problem"], args.max_tokens, args.timeout)
                choice, usage = body["choices"][0], body.get("usage", {})
                msg = choice.get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                pred = last_boxed(content) or last_boxed(reasoning)
                rec.update({
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": choice.get("finish_reason"),
                    "reasoning_chars": len(reasoning), "answer_chars": len(content),
                    "start": round(start, 3), "latency_s": round(end - start, 3),
                    "pred": pred, "correct": pred is not None and normalize(pred) == normalize(row["answer"]),
                })
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == total:
                    print(f"{done[0]}/{total} rollouts, {time.time() - t0:.0f} s", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.close()
    print(f"done: {total} rollouts in {time.time() - t0:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
