# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Smoke test for ``examples/inference/launch_inference_server.py``.

Spawns the high-level-API server as a subprocess, optionally with native
disaggregation, sends one OpenAI-compatible ``/v1/completions`` request, and
checks for a non-empty response. The server is then SIGTERM'd and joined.

No golden values: this is a pass/fail HTTP smoke. It validates the daemon-thread
CUDA-device fix, coordinator startup, frontend replicas, and request/response
round-trip end-to-end.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

READINESS_MARKER = "Running on http"
READINESS_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 60
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000


def build_server_cmd(
    checkpoint_dir: str,
    tokenizer_model: str,
    server_log_dir: str = None,
    inference_shards: str = "",
) -> list[str]:
    """Build the ``launch_inference_server.py`` command for Mistral 0.5B.

    The base model arguments match the existing DP=8 inference test. An
    ``inference_shards`` value partitions those ranks into prefill and decode pools.
    """
    # ``--tee "3"`` writes per-rank stdout+stderr files under ``--log-dir`` (which
    # the JET harness expects at ``logs/*/*/attempt_0/*/std*.log``) while still
    # echoing to this driver's captured stdout so the readiness watcher works.
    log_args = ["--log-dir", server_log_dir, "--tee", "3"] if server_log_dir else []
    disagg_args = []
    if inference_shards:
        disagg_args = [
            "--inference-shards",
            inference_shards,
            "--disagg-kv-transport-backend",
            "nccl",
            "--inference-dynamic-batching-prefix-caching",
            "--cuda-graph-impl",
            "local",
            "--inference-cuda-graph-scope",
            "block",
            "--inference-dynamic-batching-num-cuda-graphs",
            "1",
        ]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *log_args,
        "--nproc-per-node=8",
        "-m",
        "examples.inference.launch_inference_server",
        "--tiktoken-pattern",
        "v2",
        "--use-mcore-models",
        "--tokenizer-type",
        "TikTokenizer",
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
        "inference_optimized",
        "--sequence-parallel",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--deterministic-mode",
        "--ckpt-format",
        "torch_dist",
        "--bf16",
        "--num-layers",
        "24",
        "--hidden-size",
        "1152",
        "--num-attention-heads",
        "16",
        "--max-position-embeddings",
        "1024",
        "--seq-length",
        "1024",
        "--inference-dynamic-batching-buffer-size-gb",
        "20",
        "--dist-ckpt-strictness",
        "log_unexpected",
        "--inference-ckpt-non-strict",
        *disagg_args,
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
    return env


def post_completion() -> dict:
    body = json.dumps(
        {"model": "EMPTY", "prompt": "Hello, world!", "max_tokens": 10, "temperature": 0.0}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument(
        "--server-log-dir",
        default=None,
        help="torchrun --log-dir for the spawned server; CI passes the JET assets "
        "dir so per-rank logs land where the harness expects them.",
    )
    parser.add_argument("--inference-shards", default="")
    args = parser.parse_args()

    cmd = build_server_cmd(
        args.checkpoint_dir, args.tokenizer_model, args.server_log_dir, args.inference_shards
    )
    print(f"[smoke] spawning server: {' '.join(cmd)}", flush=True)

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

    rc = 1
    try:
        if not ready.wait(READINESS_TIMEOUT_S):
            print(f"[smoke] FAIL: readiness banner not seen in {READINESS_TIMEOUT_S}s", flush=True)
            return rc

        # Allow a beat after the readiness banner for all 4 frontend replicas
        # to be reachable.
        time.sleep(2)

        print("[smoke] sending /v1/completions request", flush=True)
        body = post_completion()
        choices = body.get("choices") or []
        if not choices:
            print(f"[smoke] FAIL: no choices in response: {body}", flush=True)
            return rc
        text = choices[0].get("text", "")
        if not text:
            print(f"[smoke] FAIL: empty completion text: {body}", flush=True)
            return rc

        print(f"[smoke] PASS: completion={text!r}", flush=True)
        rc = 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print("[smoke] server didn't exit on SIGTERM; SIGKILL", flush=True)
                proc.kill()
                proc.wait()
    return rc


if __name__ == "__main__":
    sys.exit(main())
