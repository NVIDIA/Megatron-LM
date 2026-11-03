# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_init_permutation_map_kernel`` (megatron/core/inference/moe/permute.py).

Launcher: ``init_permutation_map(permutation_map, n_used)``; called by ``permute_tokens`` and
``permute_and_quantize_mxfp8`` before the permute kernel claims rows.

Contract (from the launcher's documented API):

* ``permutation_map[i] = -1`` for ``i < n_used``, where ``n_used`` is an int32 device scalar
  (``inclusive_expert_offsets[-1]``, read on device so the launch is CUDA-graph safe); entries
  ``>= n_used`` are left untouched, and nothing outside the ``[output_size]`` buffer is written.
* Inputs: ``permutation_map`` ``[output_size]`` int32 contiguous (pre-allocated at the maximum
  size); ``n_used`` ``[1]`` int32. Precondition: ``n_used <= output_size`` (holds when the
  permute preconditions hold).
* Degenerate: ``n_used = 0`` writes nothing.
* Tolerance class: **exact**.

Oracle: the definition, element by element (``-1`` below ``n_used``, the poison above).

Structure: ``BLOCK_SIZE = 1024``; grid ``cdiv(output_size, 1024)``. Inputs cover
``output_size`` and ``n_used`` at B-1/B/B+1 and 2B, ``n_used`` in {0, 1, output_size - 1,
output_size}, and CUDA-graph replays with ``n_used`` growing and shrinking (the buffer is
re-poisoned between replays, so a write past the new ``n_used`` shows up). The buffer sits
between guard bands and starts poisoned (``INT32_MIN``).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

permute = pytest.importorskip("megatron.core.inference.moe.permute")
pytest.importorskip("triton")


def _check(buf, n_used, what):
    t = buf.tensor
    ou.check_guards(permutation_map=buf)
    ou.assert_exact(t[:n_used], torch.full((n_used,), -1, dtype=torch.int32), what=what)
    ou.assert_poisoned(t[n_used:], what=f"entries past n_used ({what})")


def _cases():
    sizes = ou.boundary_sizes(rr.INIT_BLOCK, degenerate=False, minimum=1) + [1]
    cases = []
    for size in sorted(set(sizes)):
        used = {0, 1, size - 1, size} | {n for n in (1023, 1024, 1025, 2047, 2048) if n <= size}
        cases += [pytest.param(size, n, id=f"size{size}-used{n}") for n in sorted(used)]
    return cases


@pytest.mark.parametrize("output_size,n_used", _cases())
def test_fills_prefix_only(output_size, n_used):
    """``-1`` exactly on ``[0, n_used)``; the rest and the guard bands untouched."""
    dev = rr.device()
    buf = ou.guarded_empty((output_size,), torch.int32, dev, guard_bytes=8192)
    permute.init_permutation_map(buf.tensor, rr.valid_tensor(n_used, dev))
    _check(buf, n_used, f"size={output_size} n_used={n_used}")


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    dev = rr.device()
    buf = ou.guarded_empty((1500,), torch.int32, dev)
    permute.init_permutation_map(buf.tensor, rr.valid_tensor(1100, dev))
    _check(buf, 1100, "sanitizer")


def test_cuda_graph_replay_follows_n_used():
    """One capture at the maximum size, replays with ``n_used`` growing and shrinking."""
    dev = rr.require_cuda()
    size = 2 * rr.INIT_BLOCK + 5
    buf = ou.guarded_empty((size,), torch.int32, dev, guard_bytes=8192)
    n_used = rr.valid_tensor(size, dev)
    permute.init_permutation_map(buf.tensor, n_used)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        permute.init_permutation_map(buf.tensor, n_used)
    for n in [size, 0, 1, 1023, 1024, 1025, size - 1, 7, size]:
        ou.poison_(buf.tensor)
        n_used.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        _check(buf, n, f"replay n_used={n}")
