# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared local-activation inputs and provenance for replay and phase timings.

Keep this adapter independent of the reporting producer. The head's benchmark
driver can import it while production operators come from a base checkout.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Iterator

ADAPTER = "local_activation_v1"
SEED = 1234
OP_IDS = {
    "bias_swiglu": "fused_bias_swiglu",
    "weighted_swiglu": "fused_bias_swiglu",
    "weighted_squared_relu": "fused_weighted_squared_relu",
}
POLICY_ENV = (
    "NCCL_ALGO",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "MAMBA_DETERMINISTIC",
    "CAUSAL_CONV1D_DETERMINISTIC",
    "TRITON_CACHE_AUTOTUNING",
)


@contextlib.contextmanager
def kernel_policy(torch, deterministic: bool) -> Iterator[None]:
    """Use strict replay/timing policy and restore the caller's Torch settings."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark


def make_case(torch, name: str, tokens: int, hidden: int, dtype, device: str = "cuda"):
    """Return the actual production function and its complete positional inputs."""
    if name not in OP_IDS or min(tokens, hidden) < 1:
        raise ValueError("Unknown kernel case or nonpositive dimensions")
    if name in ("bias_swiglu", "weighted_swiglu"):
        from megatron.core.fusions.fused_bias_swiglu import (
            bias_swiglu_impl,
            weighted_bias_swiglu_impl,
        )

        function = bias_swiglu_impl if name == "bias_swiglu" else weighted_bias_swiglu_impl
    else:
        from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl

        function = weighted_squared_relu_impl
    # Import-time work cannot advance the input generator after this point.
    torch.manual_seed(SEED)
    width = hidden if name == "weighted_squared_relu" else 2 * hidden
    x = torch.randn(tokens, width, device=device, dtype=dtype, requires_grad=True)
    inputs: tuple
    if name == "bias_swiglu":
        bias = torch.randn(width, device=device, dtype=dtype, requires_grad=True)
        inputs = (x, bias)
    else:
        weights = torch.rand(tokens, 1, device=device, dtype=torch.float32, requires_grad=True)
        inputs = (x, None, weights) if name == "weighted_swiglu" else (x, weights)
    return function, inputs, OP_IDS[name]


def case_signature(torch, name: str, inputs: tuple) -> dict:
    """Fingerprint inputs and actual dispatch context outside the timed region.

    Only these noncollective adapters may join replicated multi-rank correctness
    evidence to one-GPU latency. GPU UUID/allocation identity stays in the timing
    report; the portable contract matches GPU model/capability and driver instead.
    """
    if name not in OP_IDS:
        raise ValueError(f"Unsupported adapter case: {name}")
    arguments: list[dict | None] = []
    for value in inputs:
        if value is None:
            arguments.append(None)
            continue
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        arguments.append(
            {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "requires_grad": value.requires_grad,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    versions: dict[str, str | None] = {}
    for package in ("torch", "triton", "transformer-engine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        driver = sorted(
            set(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                )
                .strip()
                .splitlines()
            )
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = None
    environment_keys = set(POLICY_ENV) | {
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_PROTO",
        "TRITON_CACHE_DIR",
    }
    environment_keys.update(key for key in os.environ if key.startswith("TRITON_AUTOTUNE_BLOCK_"))
    cuda = torch.cuda.is_available()
    return {
        "adapter": ADAPTER,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "distribution": "local_without_collectives",
        "case": name,
        "op_id": OP_IDS[name],
        "implementation": "torch.compile:" + name,
        "inputs": arguments,
        "input_seed": SEED,
        "upstream_gradient": "ones_like_output",
        "runtime": {
            "python": platform.python_version(),
            "packages": versions,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if cuda else None,
            "capability": list(torch.cuda.get_device_capability()) if cuda else None,
            "driver": driver,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "autocast": torch.is_autocast_enabled(),
            "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "environment": {key: os.environ.get(key) for key in sorted(environment_keys)},
        },
    }
