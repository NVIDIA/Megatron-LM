# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact check of kernels launched with programmatic dependent launch (PDL).

A kernel launched with PDL may start while the previous kernel on its stream is still running,
so it is correct only if every block executes ``griddepcontrol.wait`` before it touches memory.
``assert_pdl_launch_bit_exact`` runs the kernel after a producer that lets its dependents launch
at once and writes the kernel's input only after a long spin, with the kernel's outputs filled
with garbage beforehand, and compares the result byte for byte with the kernel launched without
PDL on the same data, both eagerly and inside a CUDA graph. A kernel that read its input or
wrote its output before waiting would see the garbage or be overwritten by the producer.
"""

from typing import Callable, Optional, Sequence

import pytest
import torch

from megatron.core.inference import triton_pdl
from tests.unit_tests.determinism.kernels.harness import bytes_equal

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_launch_dependents

    HAVE_PDL_PRODUCER = True
except ImportError:
    HAVE_PDL_PRODUCER = False

requires_pdl = pytest.mark.skipif(
    not (
        HAVE_PDL_PRODUCER
        and triton_pdl.HAVE_GDC
        and torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 9
    ),
    reason="needs an NVIDIA GPU with programmatic dependent launch and Triton's gdc intrinsics",
)

# Spin iterations of the producer before it stores (about 0.2 ms per MB on current GPUs).
PRODUCER_SPIN = 4000
_NEVER = -123456789

if HAVE_PDL_PRODUCER:

    @triton.jit
    def _late_copy_kernel(src, dst, n, n_spin, never, BLOCK: tl.constexpr):
        gdc_launch_dependents()  # dependents may launch now, long before dst is written
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(src + offsets, mask=mask, other=0)
        acc = x.to(tl.int32)
        for _ in tl.range(0, n_spin):
            acc = acc * 1103515245 + 12345
        # The store depends on the spin result (``never`` is never hit), so it stays after it.
        tl.store(dst + offsets, tl.where(acc == never, x ^ 1, x), mask=mask)


def _late_copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    src_bytes, dst_bytes = src.view(torch.uint8).view(-1), dst.view(torch.uint8).view(-1)
    n = src_bytes.numel()
    _late_copy_kernel[(triton.cdiv(n, 1024),)](
        src_bytes, dst_bytes, n, PRODUCER_SPIN, _NEVER, BLOCK=1024
    )


def _fill_garbage(tensors: Sequence[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.view(torch.uint8).fill_(0xA5)


def _assert_same(outputs, references, what: str) -> None:
    torch.cuda.synchronize()
    assert len(outputs) == len(references), what
    for index, (out, ref) in enumerate(zip(outputs, references)):
        assert bytes_equal(out, ref), f"{what}: output {index} differs from the non-PDL launch"


def assert_pdl_launch_bit_exact(
    call: Callable[[], Sequence[torch.Tensor]],
    root: torch.Tensor,
    make_input: Callable[[int], torch.Tensor],
    outputs: Sequence[torch.Tensor] = (),
    reset: Optional[Callable[[], None]] = None,
    replays: int = 3,
    what: str = "kernel",
) -> None:
    """Assert ``call`` gives the same bytes launched with PDL after a late producer as without PDL.

    Args:
        call: launches the kernel(s) under test and returns the tensors to compare.
        root: the input buffer ``call`` reads; the producer writes ``make_input(i)`` into it.
        make_input: returns the input data of replay ``i``.
        outputs: buffers ``call`` writes in place; filled with garbage before every run.
        reset: restores in-place state (e.g. a recurrent state) before every run.
        replays: runs per mode (reference, eager, CUDA graph).
        what: label for failure messages.
    """
    reset = reset or (lambda: None)
    staging = torch.empty_like(root)
    saved = triton_pdl._enabled
    try:
        triton_pdl._enabled = False
        references = []
        for i in range(replays):
            staging.copy_(make_input(i))
            root.copy_(staging)
            _fill_garbage(outputs)
            reset()
            references.append([out.clone() for out in call()])
        torch.cuda.synchronize()

        triton_pdl._enabled = True
        for i in range(replays):
            staging.copy_(make_input(i))
            _fill_garbage((root, *outputs))
            reset()
            _late_copy(staging, root)
            _assert_same(call(), references[i], f"{what} (eager, replay {i})")

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # compile and warm up outside the capture
            _late_copy(staging, root)
            call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _late_copy(staging, root)
            graph_outputs = call()
        for i in range(replays):
            staging.copy_(make_input(i))
            _fill_garbage((root, *outputs))
            reset()
            graph.replay()
            _assert_same(graph_outputs, references[i], f"{what} (CUDA graph, replay {i})")
    finally:
        triton_pdl._enabled = saved


def seeded_randn(like: torch.Tensor, seed: int, scale: float = 1.0) -> torch.Tensor:
    """Random normal data with the shape and dtype of ``like``, from a seeded generator."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    data = torch.randn(like.shape, generator=generator, device="cuda", dtype=torch.float32)
    return (data * scale).to(like.dtype)
