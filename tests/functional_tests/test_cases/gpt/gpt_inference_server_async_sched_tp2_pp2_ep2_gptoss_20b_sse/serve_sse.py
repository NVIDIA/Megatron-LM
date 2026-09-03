# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise GPTOSS completion streaming with legacy and asynchronous scheduling.

The test launches the real OpenAI-compatible inference server twice against the
GPTOSS-20B checkpoint, once per scheduling mode. Each launch receives a burst
of concurrent requests, including a streaming request that uses the Hugging
Face fast-tokenizer incremental-detokenization path. The responses must satisfy
the SSE protocol invariants and agree across scheduling modes.
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
        "4096",
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
        "--expert-model-parallel-size",
        "2",
        "--expert-tensor-parallel-size",
        "1",
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-grouped-gemm",
        "--deterministic-mode",
        "--ckpt-format",
        "torch_dist",
        "--bf16",
        # The converted checkpoint stores weights but not the model arguments.
        # Keep this architecture aligned with the canonical GPTOSS functional test.
        "--num-layers",
        "24",
        "--hidden-size",
        "2880",
        "--ffn-hidden-size",
        "2880",
        "--num-attention-heads",
        "64",
        "--group-query-attention",
        "--num-query-groups",
        "8",
        "--kv-channels",
        "64",
        "--num-experts",
        "32",
        "--moe-ffn-hidden-size",
        "2880",
        "--moe-router-topk",
        "4",
        "--moe-router-dtype",
        "fp32",
        "--moe-router-score-function",
        "softmax",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-aux-loss-coeff",
        "0.0",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        "--position-embedding-type",
        "yarn",
        "--rotary-base",
        "150000",
        "--rotary-percent",
        "1.0",
        "--rotary-scaling-factor",
        "32.0",
        "--yarn-original-max-position-embeddings",
        "4096",
        "--yarn-beta-fast",
        "32.0",
        "--yarn-beta-slow",
        "1.0",
        "--mscale",
        "1.0",
        "--mscale-all-dim",
        "0.0",
        "--no-yarn-correction-range-round-to-int",
        "--quick-geglu",
        "--glu-linear-offset",
        "1.0",
        "--activation-func-clamp-value",
        "7.0",
        "--softmax-type",
        "learnable",
        "--window-size",
        "127,0",
        "--window-attn-skip-freq",
        "2",
        "--padded-vocab-size",
        "201088",
        "--make-vocab-size-divisible-by",
        "128",
        "--max-position-embeddings",
        "40960",
        "--no-rope-fusion",
        "--no-masked-softmax-fusion",
        "--seq-length",
        "4096",
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
    env["HF_HUB_CACHE"] = "/mnt/artifacts/hf_home/hub"
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
    assert len(choice["generation_log_probs"]) == max_tokens
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
                token_count = len(choice["logprobs"]["tokens"])
                assert 1 <= token_count <= STREAMING_INTERVAL
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
    assert len(final_choice["generation_log_probs"]) == STREAMING_MAX_TOKENS
    assert "".join(deltas) == final_choice["generated_text"]
    return {
        "delta_token_counts": delta_token_counts,
        "finish_reason": final_choice["finish_reason"],
        "generated_text": final_choice["generated_text"],
        "generation_token_ids": final_choice["generation_token_ids"],
        "generation_log_probs": final_choice["generation_log_probs"],
    }


def _after_barrier(
    barrier: threading.Barrier, function: Callable[..., dict[str, Any]], *args: Any
) -> dict[str, Any]:
    barrier.wait(timeout=REQUEST_TIMEOUT_S)
    return function(*args)


def _normalize_completion(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        key: choice[key]
        for key in (
            "text",
            "finish_reason",
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
        )
    }


def _assert_nested_close(
    expected: Any, actual: Any, *, atol: float, path: str = "response"
) -> None:
    assert type(expected) is type(actual), f"{path}: {type(expected)} != {type(actual)}"
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), f"{path}: keys differ"
        for key in expected:
            _assert_nested_close(expected[key], actual[key], atol=atol, path=f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), f"{path}: length {len(expected)} != {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_nested_close(expected_item, actual_item, atol=atol, path=f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(
            expected, actual, rel_tol=0.0, abs_tol=atol
        ), f"{path}: {expected} != {actual} (atol={atol})"
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


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
            result = {
                "completions": [_normalize_completion(future.result()) for future in completions],
                "stream": stream.result(),
            }
        print(f"[sse] {scheduler_mode} mode passed", flush=True)
        return result
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

    legacy = _run_server(args, "legacy")
    asynchronous = _run_server(args, "async")
    _assert_nested_close(legacy, asynchronous, atol=5e-3)
    print("[sse] PASS: legacy and async responses match", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
