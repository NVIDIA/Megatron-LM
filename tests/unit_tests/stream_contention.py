# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Side-stream GEMM contention for bit-exact replay tests, importable without side effects.

``tests/unit_tests/determinism/__init__.py`` pins ``CUDA_DEVICE_MAX_CONNECTIONS=1`` for the whole
pytest process at import time (the pre-Blackwell async-TP requirement), and the CUDA driver reads
that variable once, at context creation. With a single hardware queue every stream serialises, so
side-stream work cannot perturb kernel scheduling and contention-based replay tests silently lose
their adversarial component. Importing this module changes nothing. :func:`contention_available`
reports whether contention can do anything in this process; callers keep asserting bit-exactness
across replays either way -- a replay without contention is still a valid replay -- and take more
replays when contention is off (:func:`replay_count`).
"""

import os

import torch


def contention_available() -> bool:
    """True when more than one hardware queue is configured, so side streams can interleave.

    Reads the variable the driver read at context creation. The unit-test bucket pins it to ``1``
    during collection, before CUDA initialises, so the value seen here matches the driver's.
    """
    return os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8") != "1"


def replay_count(with_contention: int = 3, without_contention: int = 6) -> int:
    """Replays to run: fewer when contention perturbs scheduling, more when it cannot."""
    return with_contention if contention_available() else without_contention


class SideStreamContention:
    """Run mixed-size bf16 GEMMs on side streams (half at high priority) while the body runs.

    A no-op when :func:`contention_available` is False, so tests can always wrap replays in it.
    """

    _SIZES = (1024, 2048, 3072)

    def __init__(self, num_streams: int = 4, num_iters: int = 60):
        self.num_streams = num_streams
        self.num_iters = num_iters
        self._streams: list[torch.cuda.Stream] = []
        self._keepalive: list[torch.Tensor] = []

    def __enter__(self):
        if not contention_available():
            return self
        self._streams = [
            torch.cuda.Stream(priority=-1 if i % 2 else 0) for i in range(self.num_streams)
        ]
        mats = {s: torch.ones(s, s, device="cuda", dtype=torch.bfloat16) for s in self._SIZES}
        for i, stream in enumerate(self._streams):
            with torch.cuda.stream(stream):
                result = None
                for j in range(self.num_iters):
                    m = mats[self._SIZES[(i + j) % len(self._SIZES)]]
                    result = torch.matmul(m, m)
                self._keepalive.append(result)
        self._keepalive.extend(mats.values())
        return self

    def __exit__(self, *exc):
        for stream in self._streams:
            stream.synchronize()
        self._streams = []
        self._keepalive = []
        return False


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-level equality where NaN at the same position counts as equal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(((a == b) | (a.isnan() & b.isnan())).all().item())
