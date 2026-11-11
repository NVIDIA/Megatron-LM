"""Greedy output sanity check against a running server: prompts with checkable answers.

Sends each prompt once alone and once inside a concurrent batch (so multi-token decode steps and
the EP dispatch with uneven per-rank counts are exercised), greedy, and prints the replies.

    python3 inference-bench/server/sanity.py --server http://localhost:5000 [--batch 16]
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    ("count", "Count from 1 to 40, separated by commas. Output only the numbers.",
     ", ".join(str(i) for i in range(1, 41))),
    ("arith", "What is 37 * 43? Answer with just the number.", "1591"),
    ("gsm8k", "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did Natalia sell altogether in April and May? End with "
     "'Answer: <number>'.", "72"),
    ("capital", "What is the capital of Australia? One word.", "Canberra"),
    ("code", "Write a Python function is_prime(n) that returns True if n is prime. Code only.",
     "def is_prime"),
]


def chat(server, prompt, max_tokens):
    # No "model": vLLM rejects names it does not serve and MInf ignores it.
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    # MInf names the reasoning field reasoning_content, vLLM reasoning.
    return msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:5000")
    ap.add_argument("--batch", type=int, default=16, help="concurrent requests in the batched pass")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    results = {}
    for name, prompt, _ in PROMPTS:
        results[(name, "alone")] = chat(args.server, prompt, args.max_tokens)
    jobs = [PROMPTS[i % len(PROMPTS)] for i in range(args.batch)]
    with ThreadPoolExecutor(args.batch) as pool:
        outs = list(pool.map(lambda p: chat(args.server, p[1], args.max_tokens), jobs))
    for (name, _, _), out in zip(jobs, outs):
        results.setdefault((name, "batched"), out)
        if out != results[(name, "batched")]:
            results[(name, "batched-differs")] = out

    passed = 0
    for (name, mode), (content, reasoning) in results.items():
        expected = next(e for n, _, e in PROMPTS if n == name)
        ok = expected.replace(" ", "") in content.replace(" ", "")
        passed += ok and mode == "alone"
        print(f"=== {name} [{mode}] {'PASS' if ok else 'FAIL'} "
              f"(reasoning {len(reasoning)} chars, content {len(content)} chars)")
        print(content.strip()[:600])
        if not content.strip() and reasoning:
            # No answer: show where the reasoning starts and ends (coherent but long, or garbage).
            print(f"[reasoning head] {reasoning[:300]!r}\n[reasoning tail] {reasoning[-300:]!r}")
    print(f"\n{passed}/{len(PROMPTS)} prompts pass alone")


if __name__ == "__main__":
    main()
