# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise completion streaming with legacy and asynchronous scheduling.

The test launches the real OpenAI-compatible inference server twice against the
Qwen3-8B checkpoint, once per scheduling mode. Each launch receives a burst of
concurrent requests, including a streaming request that uses the Hugging Face
fast-tokenizer incremental-detokenization path. Every response must satisfy its
protocol invariants, and the streamed text, tokens, and framing must agree
across scheduling modes.
"""

import argparse
import concurrent.futures
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

READINESS_MARKER = "Running on http"
READINESS_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 180
SHUTDOWN_TIMEOUT_S = 60
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
STREAMING_INTERVAL = 2
STREAMING_MAX_TOKENS = 16


def _build_server_cmd(
    checkpoint_dir: str, tokenizer_model: str, scheduler_mode: str, server_log_dir: str | None
) -> list[str]:
    log_args = ["--log-dir", server_log_dir, "--tee", "3"] if server_log_dir else []
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *log_args,
        "--nproc-per-node=8",
        "-m",
        "examples.inference.launch_inference_server",
        "--use-mcore-models",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        tokenizer_model,
        "--auto-detect-ckpt-format",
        "--max-tokens-to-oom",
        "3600000",
        "--inference-max-seq-length",
        "1024",
        "--attention-backend",
        "flash",
        "--micro-batch-size",
        "1",
        "--no-load-optim",
        "--no-use-tokenizer-model-from-checkpoint-args",
        "--load",
        checkpoint_dir,
        "--distributed-backend",
        "nccl",
        "--transformer-impl",
        "transformer_engine",
        "--tensor-model-parallel-size",
        "2",
        "--pipeline-model-parallel-size",
        "2",
        "--deterministic-mode",
        "--ckpt-format",
        "torch_dist",
        "--bf16",
        # The converted checkpoint stores weights but not the model arguments.
        # Keep this architecture aligned with the existing Qwen3-8B functional tests.
        "--num-layers",
        "36",
        "--hidden-size",
        "4096",
        "--ffn-hidden-size",
        "12288",
        "--num-attention-heads",
        "32",
        "--group-query-attention",
        "--num-query-groups",
        "8",
        "--kv-channels",
        "128",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "0.000001",
        "--qk-layernorm",
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        "1000000",
        "--rotary-percent",
        "1.0",
        "--use-rotary-position-embeddings",
        "--swiglu",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--attention-softmax-in-fp32",
        "--vocab-size",
        "151936",
        "--make-vocab-size-divisible-by",
        "128",
        "--max-position-embeddings",
        "1024",
        "--no-masked-softmax-fusion",
        "--seq-length",
        "1024",
        "--inference-dynamic-batching-buffer-size-gb",
        "20",
        "--inference-dynamic-batching-max-requests",
        "2",
        "--inference-dynamic-batching-async-sched-mode",
        scheduler_mode,
        "--dist-ckpt-strictness",
        "log_unexpected",
        "--inference-ckpt-non-strict",
        "--port",
        str(SERVER_PORT),
        "--host",
        SERVER_HOST,
    ]


def _cleaned_env() -> dict[str, str]:
    env = os.environ.copy()
    for variable in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ):
        env.pop(variable, None)
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NCCL_ALGO"] = "Ring"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _request(body: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        f"http://localhost:{SERVER_PORT}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _open(request: urllib.request.Request):
    try:
        return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise AssertionError(f"server returned HTTP {error.code}: {detail}") from error


def _assert_log_probs(log_probs: Any, expected_count: int) -> None:
    assert isinstance(log_probs, list), "log probabilities are missing"
    assert len(log_probs) == expected_count
    assert all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in log_probs
    ), "log probabilities must be finite numbers"


def _post_completion(prompt: str, max_tokens: int) -> dict[str, Any]:
    request = _request(
        {
            "model": "EMPTY",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "logprobs": 3,
        }
    )
    with _open(request) as response:
        assert response.status == 200
        body = json.loads(response.read())

    choices = body.get("choices") or []
    assert len(choices) == 1, f"expected one completion choice, got: {body}"
    choice = choices[0]
    assert choice["text"], f"completion text is empty: {body}"
    assert choice["finish_reason"] == "length"
    assert len(choice["generation_token_ids"]) == max_tokens
    _assert_log_probs(choice.get("generation_log_probs"), max_tokens)
    return choice


def _post_streaming(prompt: str) -> dict[str, Any]:
    request = _request(
        {
            "model": "EMPTY",
            "prompt": prompt,
            "max_tokens": STREAMING_MAX_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
            "logprobs": 3,
            "stream": True,
            "streaming_interval": STREAMING_INTERVAL,
        }
    )

    deltas: list[str] = []
    delta_token_counts: list[int] = []
    final_choice = None
    saw_done = False
    with _open(request) as response:
        assert response.status == 200
        assert response.headers.get_content_type() == "text/event-stream"
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line:
                continue
            assert line.startswith("data: "), f"unexpected SSE line: {line!r}"
            if line == "data: [DONE]":
                saw_done = True
                continue

            event = json.loads(line[6:])
            assert "error" not in event, f"streaming server error: {event}"
            choices = event.get("choices") or []
            assert len(choices) == 1, f"expected one streaming choice: {event}"
            choice = choices[0]
            if choice["finish_reason"] is None:
                logprobs = choice.get("logprobs")
                assert isinstance(logprobs, dict), "streaming log probabilities are missing"
                tokens = logprobs.get("tokens")
                assert isinstance(tokens, list), "streaming log-probability tokens are missing"
                token_count = len(tokens)
                assert 1 <= token_count <= STREAMING_INTERVAL
                _assert_log_probs(logprobs.get("token_logprobs"), token_count)
                deltas.append(choice["text"])
                delta_token_counts.append(token_count)
            else:
                assert final_choice is None, "stream emitted more than one terminal choice"
                final_choice = choice

    assert saw_done, "stream did not emit the [DONE] sentinel"
    assert final_choice is not None, "stream did not emit a terminal choice"
    assert len(deltas) > 1, "stream did not emit multiple text deltas"
    assert sum(delta_token_counts) == STREAMING_MAX_TOKENS
    assert final_choice["finish_reason"] == "length"
    assert final_choice["generated_length"] == STREAMING_MAX_TOKENS
    assert len(final_choice["generation_token_ids"]) == STREAMING_MAX_TOKENS
    _assert_log_probs(final_choice.get("generation_log_probs"), STREAMING_MAX_TOKENS)
    assert "".join(deltas) == final_choice["generated_text"]
    return {
        "delta_token_counts": delta_token_counts,
        "finish_reason": final_choice["finish_reason"],
        "generated_text": final_choice["generated_text"],
        "generation_token_ids": final_choice["generation_token_ids"],
    }


def _after_barrier(
    barrier: threading.Barrier, function: Callable[..., dict[str, Any]], *args: Any
) -> dict[str, Any]:
    barrier.wait(timeout=REQUEST_TIMEOUT_S)
    return function(*args)


def _wait_until_ready(proc: subprocess.Popen[str], ready: threading.Event) -> None:
    deadline = time.monotonic() + READINESS_TIMEOUT_S
    while not ready.wait(timeout=0.25):
        return_code = proc.poll()
        if return_code is not None:
            raise AssertionError(f"server exited before readiness with status {return_code}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"readiness banner not seen in {READINESS_TIMEOUT_S}s")


def _run_server(args: argparse.Namespace, scheduler_mode: str) -> dict[str, Any]:
    log_dir = os.path.join(args.server_log_dir, scheduler_mode) if args.server_log_dir else None
    cmd = _build_server_cmd(args.checkpoint_dir, args.tokenizer_model, scheduler_mode, log_dir)
    print(f"[sse] spawning {scheduler_mode} server: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_cleaned_env(),
    )
    assert proc.stdout is not None

    ready = threading.Event()

    def watch() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{scheduler_mode}] {line}", end="", flush=True)
            if READINESS_MARKER in line:
                ready.set()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    try:
        _wait_until_ready(proc, ready)
        time.sleep(2)

        prompt_lengths = [
            ("The capital of France is", 24),
            ("Two plus two equals", 6),
            ("A primary color is", 18),
            ("Water freezes at", 8),
            ("The opposite of hot is", 20),
            ("Complete this sentence: the sky is", 10),
        ]
        barrier = threading.Barrier(len(prompt_lengths) + 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=barrier.parties) as executor:
            completions = [
                executor.submit(_after_barrier, barrier, _post_completion, prompt, length)
                for prompt, length in prompt_lengths
            ]
            stream = executor.submit(_after_barrier, barrier, _post_streaming, "Count from one:")
            for future in completions:
                future.result()
            stream_result = stream.result()
        print(f"[sse] {scheduler_mode} mode passed", flush=True)
        return stream_result
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(f"[sse] {scheduler_mode} server ignored SIGTERM; sending SIGKILL")
                proc.kill()
                proc.wait()
        watcher.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--server-log-dir")
    args = parser.parse_args()

    # Ordinary completions provide concurrent load and are validated within each session.
    # Their BF16 outputs, including logprobs, are not invariant to scheduler batch packing.
    # The stream summary excludes numeric logprobs and is the differential SSE oracle.
    legacy_stream = _run_server(args, "legacy")
    asynchronous_stream = _run_server(args, "async")
    assert legacy_stream == asynchronous_stream, (
        "legacy and async streamed text, tokens, or framing differ: "
        f"{legacy_stream!r} != {asynchronous_stream!r}"
    )
    print("[sse] PASS: legacy and async streaming responses match", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
