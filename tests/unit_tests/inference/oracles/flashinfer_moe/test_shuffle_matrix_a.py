# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``shuffle_matrix_a`` as Megatron uses it for routed MXFP8 weights
(``megatron/core/inference/moe/flashinfer_mxfp8.py``, ``prepare_routed_mxfp8_weights``).

Contract (written from the call site and TRT-LLM's documented Major-K layout):

* ``prepare_routed_mxfp8_weights(weight)`` takes a stacked canonical MXFP8 weight
  ``data [E, M, K]`` (E4M3 bytes, row-major, Megatron/cuBLAS layout) and returns ``routed.data
  [E, round_up(M, 128), round_up(K, 128)]``: each expert's data zero-padded to 128-aligned rows
  and columns, then row-shuffled for ``epilogue_tile_m = 128``: inside every block of 32 rows,
  destination row ``8b + a`` holds source row ``4a + b`` (a < 8, b < 4). Bytes are moved, never
  requantized. Padding rows and columns are zero bytes, so they add 0 to every GEMM.
* ``shuffle_matrix_a(t, 128)`` itself is that row gather of a ``[M, K]`` byte matrix and
  requires ``M % 32 == 0`` (Megatron always passes 128-aligned rows).
* With ``out=routed`` (refit) the same bytes are written into the existing storage: addresses
  stay fixed for CUDA graphs and every byte is rewritten; changed metadata raises ValueError.
* Inputs are read only. Degenerate: a 0-expert stack gives empty outputs.

Oracle: the row map from the layout definition (``spec.trtllm_shuffle_source_rows``), no
FlashInfer call. Tolerance class: exact (bytes). The semantic check, that the TRT-LLM kernel
consumes these bytes as the canonical weight, is the routed-MoE oracle
(``test_trtllm_fp8_block_scale_routed_moe.py``), which builds its reference from the unshuffled
weights. Variants: one (``epilogue_tile_m = 128``); FC1 and FC2 share the path. Hardware: any
device with FlashInfer (pure tensor indexing; CUDA when available, as in production).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

# Megatron's routed layout pads rows and columns to this (flashinfer_mxfp8.py, prepare_routed_...).
ROUTED_ALIGN = 128


def _canonical(experts, rows, cols, generator, dev):
    """A canonical MXFP8 stack with random data bytes and random (spec-swizzled) scales."""
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    data = torch.randint(0, 256, (experts, rows, cols), generator=generator, dtype=torch.uint8)
    logical = torch.randint(100, 140, (experts, rows, cols // 32), generator=generator)
    scale_len = spec._ceil_to(rows, 128) * spec._ceil_to(cols // 32, 4)
    scale = torch.zeros(experts, scale_len, dtype=torch.uint8)
    for e in range(experts):
        scale[e] = spec.swizzle_128x4(logical[e].to(torch.uint8))
    weight = MXFP8Tensor(
        data=data.to(dev).view(torch.float8_e4m3fn),
        scale=scale.to(dev).view(torch.float8_e8m0fnu),
        backend="triton",
    )
    return weight, data, logical.to(torch.uint8)


def _expected_routed_data(data: torch.Tensor) -> torch.Tensor:
    """Zero-pad each expert to 128-aligned rows/cols, then apply the Major-K row shuffle."""
    experts, rows, cols = data.shape
    pr, pc = spec._ceil_to(rows, ROUTED_ALIGN), spec._ceil_to(cols, ROUTED_ALIGN)
    padded = torch.zeros(experts, pr, pc, dtype=torch.uint8)
    padded[:, :rows, :cols] = data
    return padded[:, spec.trtllm_shuffle_source_rows(pr)]


def test_row_map_is_the_transposed_8x4_grid():
    """CPU check of the oracle's own row map: a permutation of each 32-row block, equal to
    reading a row-major 8x4 grid of source rows column by column."""
    src = spec.trtllm_shuffle_source_rows(96)
    for block in range(3):
        ou.assert_is_permutation(src[32 * block : 32 * block + 32] - 32 * block, 32)
    grid = torch.arange(32).reshape(8, 4)  # grid[a, b] = 4a + b
    ou.assert_exact(src[:32], grid.T.reshape(-1), what="row map")


@pytest.mark.parametrize("cols", [1, 16, 128, 160])
@pytest.mark.parametrize("rows", [32, 64, 96, 128, 256])
def test_shuffle_matrix_a_is_the_major_k_row_gather(rows, cols):
    fi = spec.require_flashinfer()
    g = ou.make_generator()
    dev = spec.device()
    x = torch.randint(0, 256, (rows, cols), generator=g, dtype=torch.uint8).to(dev)
    before = x.clone()
    out = fi.shuffle_matrix_a(x, 128)
    ou.assert_exact(out, x[spec.trtllm_shuffle_source_rows(rows, dev)], what="shuffled rows")
    ou.assert_exact(x, before, what="input is read-only")


@pytest.mark.parametrize("rows", [1, 31, 33, 100])
def test_shuffle_matrix_a_rejects_rows_not_a_multiple_of_32(rows):
    """Precondition (FlashInfer asserts it); Megatron never violates it because it pads to 128."""
    fi = spec.require_flashinfer()
    with pytest.raises(AssertionError):
        fi.shuffle_matrix_a(torch.zeros(rows, 32, dtype=torch.uint8, device=spec.device()), 128)


# Rows around the shuffle block (32) and the routed alignment (128); K around 128 and production.
ROWS = ou.boundary_sizes(32, 128, multiples=(3,), production=(1856,), minimum=1)
COLS = [32, 96, 128, 160, 2688]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"E{c['experts']}-M{c['rows']}-K{c['cols']}")
        for c in ou.pairwise_cases({"rows": ROWS, "cols": COLS, "experts": [1, 3]})
    ],
)
def test_prepare_routed_mxfp8_weights_data_layout(case):
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights

    g = ou.make_generator()
    dev = spec.device()
    weight, data, _ = _canonical(case["experts"], case["rows"], case["cols"], g, dev)
    before = data.clone()
    routed = prepare_routed_mxfp8_weights(weight)
    expected = _expected_routed_data(data)
    assert (routed.logical_rows, routed.logical_cols) == (case["rows"], case["cols"])
    assert (routed.padded_rows, routed.padded_cols) == tuple(expected.shape[1:])
    assert routed.data.dtype == torch.float8_e4m3fn
    ou.assert_exact(spec.as_bytes(routed.data).cpu(), expected, what="routed data bytes")
    ou.assert_exact(spec.as_bytes(weight.data).cpu(), before, what="canonical data is read-only")


@pytest.mark.sanitizer
def test_refit_rewrites_every_byte_in_place():
    """Refit (``out=``) keeps addresses and rewrites all bytes, padding included: storage
    poisoned with 0xFF bytes must come back exactly as a fresh build."""
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights

    g = ou.make_generator()
    dev = spec.device()
    first, _, _ = _canonical(2, 96, 160, g, dev)
    second, data2, _ = _canonical(2, 96, 160, g, dev)
    routed = prepare_routed_mxfp8_weights(first)
    ptrs = (routed.data.data_ptr(), routed.scale.data_ptr())
    spec.as_bytes(routed.data).fill_(0xFF)
    routed.scale.fill_(0xFF)
    refreshed = prepare_routed_mxfp8_weights(second, out=routed)
    assert refreshed is routed
    assert (routed.data.data_ptr(), routed.scale.data_ptr()) == ptrs, "refit moved the storage"
    ou.assert_exact(spec.as_bytes(routed.data).cpu(), _expected_routed_data(data2), what="data")
    fresh = prepare_routed_mxfp8_weights(second)
    ou.assert_exact(routed.scale, fresh.scale, what="scales after refit")


def test_refit_rejects_changed_shapes():
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights

    g = ou.make_generator()
    dev = spec.device()
    routed = prepare_routed_mxfp8_weights(_canonical(2, 96, 160, g, dev)[0])
    for rows, cols in ((128, 160), (96, 128)):
        with pytest.raises(ValueError, match="changed across refit"):
            prepare_routed_mxfp8_weights(_canonical(2, rows, cols, g, dev)[0], out=routed)


def test_zero_experts():
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights

    weight, _, _ = _canonical(0, 96, 160, ou.make_generator(), spec.device())
    routed = prepare_routed_mxfp8_weights(weight)
    assert routed.data.shape == (0, 128, 256) and routed.scale.shape == (0, 128 * 8)
