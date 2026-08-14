# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Calibration ballast: add N no-op kernel nodes per layer to the decode graph.

This is not an optimization and must stay off in any measured configuration. It
exists to answer one question that four sessions of fusion work have had to
guess at: what does *one more kernel node* cost inside the captured decode
graph? Fusing two kernels into one removes both the second kernel's device time
and its node, and the two cannot be separated after the fact — so measure the
sum directly by adding nodes that do no useful work.

Set ``MCORE_DISPATCH_BALLAST=<n>`` to launch n extra one-element kernels per
transformer layer. Step time against n gives the per-node cost, which converts
any future "this fusion removes K launches" into an expected throughput gain.
"""

import os

import torch

COUNT: int = int(os.environ.get("MCORE_DISPATCH_BALLAST", "0"))

_scratch = None


def tick() -> None:
    """Launch ``COUNT`` minimal kernels, serialized on one scratch element."""
    if COUNT <= 0:
        return
    global _scratch
    if _scratch is None:
        _scratch = torch.zeros(1, device=torch.cuda.current_device())
    for _ in range(COUNT):
        _scratch.add_(1.0)
