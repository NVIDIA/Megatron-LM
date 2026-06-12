# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""CUDA stream helpers for MoT (Mixture of Transformers) und/gen overlap.

A single (und_stream, gen_stream) pair is created lazily per CUDA device and
shared by every MoT layer — layers run sequentially in the forward pass, so
per-layer streams would add no parallelism but real init cost.
"""

from contextlib import contextmanager
from typing import Dict, Tuple

import torch


_DEVICE_STREAMS: Dict[int, Tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}


def get_mot_streams() -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """Return the (und, gen) side-stream pair for the current CUDA device."""
    device = torch.cuda.current_device()
    pair = _DEVICE_STREAMS.get(device)
    if pair is None:
        pair = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
        _DEVICE_STREAMS[device] = pair
    return pair


@contextmanager
def mot_overlap_region():
    """Context manager that yields (und_stream, gen_stream) with fences attached.

    On entry, both side streams wait for the current (default) stream so they
    see all prior work. On exit, the default stream waits for both side streams
    so any downstream op (typically a torch.cat join) sees their results.
    """
    default = torch.cuda.current_stream()
    und_s, gen_s = get_mot_streams()
    und_s.wait_stream(default)
    gen_s.wait_stream(default)
    try:
        yield und_s, gen_s
    finally:
        default.wait_stream(und_s)
        default.wait_stream(gen_s)
