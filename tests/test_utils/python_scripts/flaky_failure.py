# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

COMMON_FLAKY_FAILURE_PATTERNS = (
    "The server socket has failed to listen on any local network address.",
    "Some NCCL operations have failed or timed out.",
    "uncorrectable ECC error encountered",
    "illegal memory access",
    "illegal instruction",
    "torch.distributed.DistNetworkError",
    "Segmentation fault",
    "found NaN in",
    "For debugging consider passing CUDA_LAUNCH_BLOCKING=1",
    "double free or corruption",
    "Call to CUDA function failed.",
    "Connection reset by peer",
    "invalid pointer",
    "malloc(): unaligned tcache chunk detected",
    "zmq.error.ZMQError: Address already in use",
    "We couldn't connect to 'https://huggingface.co'",
    "Unpack failed: incomplete input",
    "unspecified launch failure",
    "free(): corrupted unsorted chunks",
    "Segfault encountered",
    # Shared-filesystem failures observed while writing checkpoints.
    "Disk quota exceeded",
    "basic_ios::clear: iostream error",
)

JET_EXTRA_FLAKY_FAILURE_PATTERNS = ("Fatal glibc error",)

NEMO_RUN_EXTRA_FLAKY_FAILURE_PATTERNS = (
    "The read operation timed out",
    "Read timed out",
    "TimeoutError",
    "Connection broken",
    "Temporary failure in name resolution",
    "The following metrics failed",
    "removal of container",
    "is already in progress",
    "Error deleting container",
)


def is_flaky_failure(*logs: str, extra_patterns: tuple[str, ...] = ()) -> bool:
    """Return whether any supplied log contains a known intermittent failure."""

    patterns = COMMON_FLAKY_FAILURE_PATTERNS + extra_patterns
    return any(pattern in log for pattern in patterns for log in logs)
