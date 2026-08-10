# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from examples.inference.advanced.gpt_dynamic_inference import _assert_nested_close

READINESS_MARKER = "Running on http"
READINESS_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 60
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000


def build_server_cmd(
    checkpoint_dir: str, tokenizer_model: str, scheduler_mode: str, server_log_dir: str = None
) -> list[str]:
    # ``--tee "3"`` writes per-rank stdout+stderr files under ``--log-dir`` (which
    # the JET harness expects at ``logs/*/*/attempt_0/*/std*.log``) while still
    # echoing to this driver's captured stdout so the readiness watcher works.
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
        "--use-checkpoint-args",
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


def cleaned_env() -> dict:
    """Strip torchrun-specific env vars so the spawned server's torchrun
    starts a fresh distributed setup instead of inheriting a stale one.
    """
    env = os.environ.copy()
    for v in (
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
        env.pop(v, None)
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NCCL_ALGO"] = "Ring"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["HF_HOME"] = "/mnt/artifacts/hf_home"
    return env


def post_completion(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": "EMPTY",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 3,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{SERVER_PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        if resp.status != 200:
            raise AssertionError(f"server returned status {resp.status}")
        return json.loads(resp.read())


def post_streaming(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": "EMPTY",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 3,
            "stream": True,
            "streaming_interval": 2,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{SERVER_PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    final = None
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        assert resp.status == 200
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = json.loads(line[6:])["choices"]
            if not choice:
                continue
            choice = choice[0]
            if choice["finish_reason"] is None:
                chunks.append(choice["text"])
            else:
                final = choice
    assert final is not None and len(chunks) > 1, "stream did not emit multiple deltas"
    assert "".join(chunks) == final["generated_text"]
    return final


def normalize_completion(body: dict) -> dict:
    choice = body["choices"][0]
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


def run_server(args, scheduler_mode: str) -> dict:
    log_dir = os.path.join(args.server_log_dir, scheduler_mode) if args.server_log_dir else None
    cmd = build_server_cmd(args.checkpoint_dir, args.tokenizer_model, scheduler_mode, log_dir)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=cleaned_env(),
    )

    ready = threading.Event()

    def watch():
        for line in proc.stdout:
            print(f"[server] {line}", end="", flush=True)
            if READINESS_MARKER in line:
                ready.set()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    try:
        if not ready.wait(READINESS_TIMEOUT_S):
            raise AssertionError(f"readiness banner not seen in {READINESS_TIMEOUT_S}s")

        # Allow a beat after the readiness banner for all 4 frontend replicas
        # to be reachable.
        time.sleep(2)

        prompts = [
            "The capital of France is",
            "Two plus two equals",
            "A primary color is",
            "Water freezes at",
            "The opposite of hot is",
            "Complete this sentence: the sky is",
        ]
        lengths = [24, 6, 18, 8, 20, 10]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts) + 1) as executor:
            requests = [
                executor.submit(post_completion, prompt, length)
                for prompt, length in zip(prompts, lengths)
            ]
            stream_request = executor.submit(post_streaming, "Count from one:", 16)
            completions = [normalize_completion(request.result()) for request in requests]
            stream = stream_request.result()
        assert all(completion["text"] for completion in completions)
        return {"completions": completions, "stream": stream}
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print("[smoke] server didn't exit on SIGTERM; SIGKILL", flush=True)
                proc.kill()
                proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--server-log-dir")
    args = parser.parse_args()

    legacy = run_server(args, "legacy")
    asynchronous = run_server(args, "async")
    _assert_nested_close(legacy, asynchronous, atol=5e-3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
