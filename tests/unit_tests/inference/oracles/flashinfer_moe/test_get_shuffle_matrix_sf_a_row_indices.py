# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the routed-MXFP8 scale repack: FlashInfer's ``get_shuffle_matrix_sf_a_row_indices``
with Megatron's ``_unshuffle_cublas_scale`` and ``_block_scale_interleave``
(``megatron/core/inference/moe/flashinfer_mxfp8.py``, ``_shuffle_routed_scale`` and
``prepare_routed_mxfp8_weights``).

Contract (from the call site, the cuBLAS block-scale layout and TRT-LLM's Major-K layout):

* Canonical MXFP8 weights hold one E8M0 scale per 32 elements along K, stored in the cuBLAS 128x4
  layout (``spec.swizzle_128x4``) of the ``[round_up(M, 128), round_up(K/32, 4)]`` matrix.
* ``_unshuffle_cublas_scale(scale, M, K)`` returns the logical ``[M, K/32]`` scale matrix; it
  raises ValueError when ``K % 32 != 0`` or the byte count is not that of the layout.
* ``_block_scale_interleave(L)`` maps a ``[R, C]`` matrix (``R % 128 == 0``, ``C % 4 == 0``,
  else ValueError) to its 128x4 layout.
* ``get_shuffle_matrix_sf_a_row_indices(L, 128)`` gives, for each destination row, the source row
  of the Major-K shuffle (the same map as the weight data: in each 32-row block, destination
  ``8b + a`` takes source ``4a + b``).
* ``routed.scale[e]`` = 128x4 layout of the logical matrix zero-padded to ``[round_up(M, 128),
  round_up(K, 128) / 32]`` and row-shuffled: the scale of every weight byte moves with that byte.
  Upstream FlashInfer does the same with ``shuffle_matrix_sf_a`` (row gather + its CUDA
  ``block_scale_interleave``); on Blackwell the bytes must match it.
* Round trip: unswizzling the routed scale and undoing the shuffle gives back the canonical
  logical scales exactly, so dequantizing the routed weight gives the canonical values.

Oracle: the per-element layout formula and the literal row map (``spec``). Tolerance: exact.
Variants: one. Hardware: Megatron's helpers on any device (CPU in the CPU checks); the FlashInfer
row indices on any device with FlashInfer; FlashInfer's CUDA interleave on Blackwell.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

# Logical scale rows around the 32-row shuffle block and the 128-row swizzle tile, and scale
# columns (K / 32) around the 4-column swizzle tile, plus production (K = 2688 -> 84).
ROWS = ou.boundary_sizes(32, 128, multiples=(3,), production=(200, 1856), minimum=1)
SCALE_COLS = [1, 3, 4, 5, 8, 84]


def _logical(rows, scale_cols, generator):
    return torch.randint(0, 256, (rows, scale_cols), generator=generator, dtype=torch.uint8)


@pytest.mark.parametrize("scale_cols", SCALE_COLS)
@pytest.mark.parametrize("rows", ROWS)
def test_unshuffle_cublas_scale_inverts_the_cublas_layout(rows, scale_cols):
    from megatron.core.inference.moe.flashinfer_mxfp8 import _unshuffle_cublas_scale

    logical = _logical(rows, scale_cols, ou.make_generator())
    swizzled = spec.swizzle_128x4(logical)
    before = swizzled.clone()
    out = _unshuffle_cublas_scale(swizzled.view(torch.float8_e8m0fnu), rows, 32 * scale_cols)
    ou.assert_exact(out, logical, what="unswizzled scales")
    ou.assert_exact(swizzled, before, what="input is read-only")


@pytest.mark.parametrize("scale_cols", [4, 8, 84])
@pytest.mark.parametrize("rows", [128, 256, 384, 1920])
def test_block_scale_interleave_is_the_128x4_layout(rows, scale_cols):
    from megatron.core.inference.moe.flashinfer_mxfp8 import _block_scale_interleave

    logical = _logical(rows, scale_cols, ou.make_generator())
    ou.assert_exact(_block_scale_interleave(logical), spec.swizzle_128x4(logical), what="layout")


def test_megatron_helpers_reject_bad_shapes():
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        _block_scale_interleave,
        _unshuffle_cublas_scale,
    )

    with pytest.raises(ValueError, match="divisible by 32"):
        _unshuffle_cublas_scale(torch.zeros(128 * 4, dtype=torch.uint8), 128, 48)
    with pytest.raises(ValueError, match="unexpected MXFP8 scale size"):
        _unshuffle_cublas_scale(torch.zeros(128 * 4 - 1, dtype=torch.uint8), 128, 128)
    for shape in ((127, 4), (128, 3)):
        with pytest.raises(ValueError, match="TRT-LLM scale interleave"):
            _block_scale_interleave(torch.zeros(shape, dtype=torch.uint8))


@pytest.mark.parametrize("rows", [128, 256, 384, 1920])
def test_row_indices_are_the_major_k_shuffle(rows):
    spec.require_flashinfer()
    from flashinfer.utils import get_shuffle_matrix_sf_a_row_indices

    logical = _logical(rows, 8, ou.make_generator())
    idx = get_shuffle_matrix_sf_a_row_indices(logical, 128)
    ou.assert_exact(idx.cpu(), spec.trtllm_shuffle_source_rows(rows), what="row indices")


@pytest.mark.parametrize("scale_cols", [4, 8, 84])
@pytest.mark.parametrize("rows", [128, 256, 1920])
def test_shuffle_routed_scale(rows, scale_cols):
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import _shuffle_routed_scale

    logical = _logical(rows, scale_cols, ou.make_generator()).to(spec.device())
    out = _shuffle_routed_scale(logical)
    expected = spec.swizzle_128x4(logical[spec.trtllm_shuffle_source_rows(rows, logical.device)])
    ou.assert_exact(out, expected, what="routed scale bytes")


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"E{c['experts']}-M{c['rows']}-Kb{c['scale_cols']}")
        for c in ou.pairwise_cases({"rows": ROWS, "scale_cols": SCALE_COLS, "experts": [1, 3]})
    ],
)
def test_prepare_routed_mxfp8_weights_scale_round_trip(case):
    """Routed scales equal the spec repack of the canonical scales, and undoing the repack (spec
    unswizzle, inverse shuffle, unpad) gives the canonical logical scales back exactly."""
    _check_scale_round_trip(case["experts"], case["rows"], case["scale_cols"])


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small case for compute-sanitizer, on the GPU: two experts of 97 logical rows (a partial
    32-row shuffle block and a partial 128-row tile) by 5 scale columns (a partial 4-column tile)
    through the routed repack and its round trip, and ``_unshuffle_cublas_scale`` of a device
    tensor of the same shape."""
    from megatron.core.inference.moe.flashinfer_mxfp8 import _unshuffle_cublas_scale

    _check_scale_round_trip(2, 97, 5)
    logical = _logical(97, 5, ou.make_generator())
    swizzled = spec.swizzle_128x4(logical).cuda()
    out = _unshuffle_cublas_scale(swizzled.view(torch.float8_e8m0fnu), 97, 32 * 5)
    ou.assert_exact(out.cpu(), logical, what="unswizzled scales on the GPU")


def _check_scale_round_trip(experts, rows, scale_cols):
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    g = ou.make_generator()
    cols = 32 * scale_cols
    logical = torch.stack([_logical(rows, scale_cols, g) for _ in range(experts)])
    scale = torch.stack([spec.swizzle_128x4(s) for s in logical])
    data = torch.zeros(experts, rows, cols, dtype=torch.uint8)
    dev = spec.device()
    weight = MXFP8Tensor(
        data=data.to(dev).view(torch.float8_e4m3fn),
        scale=scale.to(dev).view(torch.float8_e8m0fnu),
        backend="triton",
    )
    routed = prepare_routed_mxfp8_weights(weight)
    pr, pc = routed.padded_rows, routed.padded_cols // 32
    assert routed.scale.dtype == torch.uint8 and routed.scale.shape == (experts, pr * pc)
    src = spec.trtllm_shuffle_source_rows(pr)
    inverse = torch.argsort(src)
    for e in range(experts):
        padded = torch.zeros(pr, pc, dtype=torch.uint8)
        padded[:rows, :scale_cols] = logical[e]
        got = routed.scale[e].cpu()
        ou.assert_exact(got, spec.swizzle_128x4(padded[src]), what=f"expert {e} routed scales")
        back = spec.unswizzle_128x4(got, pr, pc)[inverse][:rows, :scale_cols]
        ou.assert_exact(back, logical[e], what=f"expert {e} round trip")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("scale_cols", [4, 8, 84])
@pytest.mark.parametrize("rows", [128, 256, 1920])
def test_matches_flashinfer_shuffle_matrix_sf_a(rows, scale_cols):
    """Megatron's private interleave must track FlashInfer's CUDA ``block_scale_interleave``
    (the upstream ``shuffle_matrix_sf_a`` route) byte for byte."""
    fi = spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import _shuffle_routed_scale

    logical = _logical(rows, scale_cols, ou.make_generator()).cuda()
    upstream = fi.shuffle_matrix_sf_a(logical, 128).reshape(-1)
    ou.assert_exact(_shuffle_routed_scale(logical), upstream.view(torch.uint8), what="vs upstream")
