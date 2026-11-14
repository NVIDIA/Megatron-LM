# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_prefix_sum_kernel`` (megatron/core/inference/moe/permute.py).

Launcher: ``compute_expert_offsets(tokens_per_expert, alignment)``; called by ``permute_tokens``,
``permute_and_quantize_mxfp8`` and the vLLM align path on every non-FlashInfer inference MoE.

Contract (from the launcher's documented API and its callers):

* Each count ``c_e`` is rounded up to a multiple of ``alignment`` (``0`` stays ``0``):
  ``h_e = ceil(c_e / alignment) * alignment``. Outputs ``inclusive[e] = sum_{j <= e} h_j`` and
  ``exclusive[e] = inclusive[e] - h_e``; ``inclusive[-1]`` is the used row count ``n_used``.
* Inputs: ``tokens_per_expert`` ``[E_l]`` int32 (read-only); ``alignment`` a positive int.
  Outputs: two new ``[E_l]`` int32 tensors, every element written.
* Precondition: the sum fits int32 (bounded by the permute buffer sizes).
* Tolerance class: **exact** (integers).

Oracle: a Python loop over the counts (``routing_reference.aligned_offsets``).

Structure: ``alignment`` is a compile-time constant (1 skips the rounding; production 16 torch
BF16, 128 MXFP8, DeepGEMM's m-alignment for batch-invariant BF16, vLLM ``BLOCK_SIZE_M`` in
{16, 32, 64, 128}); one program with ``BLOCK_SIZE = next_pow2(E_l)``. Inputs therefore cover
``E_l`` at powers of two +-1 (1 to 129, and 256), every production alignment plus 1, 2, 8 and
256 (DeepGEMM's runtime m-alignment is not pinned here; powers of two up to 256 bracket it),
counts 0 / 1 / alignment-1 / alignment / alignment+1 / random / aligned sums at 2^31 - 1, and
all-zero vectors.
Outputs come from ``torch.empty`` allocations poisoned by PyTorch's
``fill_uninitialized_memory``, so an unwritten element shows up.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

permute = pytest.importorskip("megatron.core.inference.moe.permute")
pytest.importorskip("triton")

ALIGNMENTS = [1, 2, 8, 16, 32, 64, 128, 256]
NUM_LOCAL = ou.boundary_sizes(2, 4, 8, 32, 128, multiples=(), degenerate=False, minimum=1) + [256]


def _counts(kind, num_local, alignment, g):
    a = alignment
    if kind == "zeros":
        return torch.zeros(num_local, dtype=torch.int32)
    if kind == "edges":
        pool = torch.tensor(sorted({0, 1, max(a - 1, 0), a, a + 1, 2 * a, 2 * a + 1}))
        return pool[torch.randint(0, pool.numel(), (num_local,), generator=g)].to(torch.int32)
    if kind == "random":
        return torch.randint(0, 5000, (num_local,), generator=g, dtype=torch.int32)
    if kind == "large":
        # The largest counts whose aligned sum still fits int32, each a-1 below its padding.
        c = ((2**31 - 1) // num_local // a) * a - (a - 1)
        return torch.full((num_local,), c, dtype=torch.int32)
    raise ValueError(kind)


def _check(counts, alignment, dev, what):
    counts = counts.to(dev)
    before = counts.clone()
    with rr.poisoned_outputs():
        exclusive, inclusive = permute.compute_expert_offsets(counts, alignment=alignment)
    ref_ex, ref_in = rr.aligned_offsets(counts.tolist(), alignment)
    assert exclusive.dtype == torch.int32 and inclusive.dtype == torch.int32
    ou.assert_exact(inclusive, torch.tensor(ref_in), what=f"inclusive offsets ({what})")
    ou.assert_exact(exclusive, torch.tensor(ref_ex), what=f"exclusive offsets ({what})")
    ou.assert_exact(counts, before, what="tokens_per_expert is read-only")


@pytest.mark.parametrize("kind", ["zeros", "edges", "random"])
@pytest.mark.parametrize("alignment", ALIGNMENTS)
@pytest.mark.parametrize("num_local", NUM_LOCAL)
def test_offsets(num_local, alignment, kind):
    """Every ``E_l`` around the ``next_pow2`` block, every alignment variant, count kinds."""
    dev = rr.device()
    g = ou.make_generator()
    _check(_counts(kind, num_local, alignment, g), alignment, dev, f"E={num_local} a={alignment}")


@pytest.mark.parametrize("alignment", [1, 16, 128])
@pytest.mark.parametrize("num_local", [1, 7, 128])
def test_large_counts(num_local, alignment):
    """Aligned sums up to 2^31 - 1 (the int32 limit of the outputs): no overflow in the scan."""
    dev = rr.device()
    _check(_counts("large", num_local, alignment, None), alignment, dev, "large counts")


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    dev = rr.device()
    g = ou.make_generator()
    _check(_counts("edges", 7, 16, g), 16, dev, "sanitizer")
