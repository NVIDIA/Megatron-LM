# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Conformance tests for the CuTeDSL paged-stash copy kernels.

Every case checks the CuTe result against the Triton kernel that owns the contract
(``paged_stash_copy_kernel``), across all allocator branches.
"""

import pytest
import torch

from megatron.core.transformer.moe.ops.paged_stash import GLOBAL_BLOCK_SIZE, paged_stash_copy_kernel

try:
    from megatron.core.transformer.moe.ops import paged_stash_cute_copy as cute_copy

    HAVE_CUTE = True
except ImportError:
    HAVE_CUTE = False

pytestmark = [
    pytest.mark.skipif(not HAVE_CUTE, reason="CuTeDSL (nvidia-cutlass-dsl) not available"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.launch_on_gb200,
]

PAGE = 64
DEV = "cuda"

# (row_bytes, source_rows, cuda_rows, host_rows). Row widths mirror the MXFP8 activations a
# DeepSeek-V3 MoE layer stashes: wide activations plus the narrow columnwise scale-inv tensor,
# whose row count is num_tokens/32 and whose row width is unrelated to the wide ones.
GEOMETRIES = {
    "h7168": (7168, 2048, 16384, 8192),
    "h2048": (2048, 2048, 16384, 8192),
    "colwise_scale_inv_h2048": (2048, 4096, 16384, 8192),
    "h8192": (8192, 4096, 16384, 8192),
}

OUTPUTS = (
    "cuda_stash",
    "host_stash",
    "page_record",
    "overflow",
    "host_spill",
    "spilled_to_host",
    "new_free_list_head",
    "free_list_head_after",
)


def _inputs(row_bytes, source_rows, cuda_rows, host_rows, tokens, head, tail, incoming):
    cuda_pages, host_pages = cuda_rows // PAGE, host_rows // PAGE
    return dict(
        # A permuted free list is essential: with an identity list the destination row equals
        # the source row and an incorrect page mapping stays invisible.
        source=torch.randint(0, 255, (source_rows, row_bytes), device=DEV, dtype=torch.uint8),
        num_tokens=torch.tensor([tokens], device=DEV, dtype=torch.int64),
        free_list_cuda=torch.randperm(cuda_pages, device=DEV).to(torch.int64),
        free_list_host=torch.randperm(host_pages, device=DEV).to(torch.int64),
        free_list_head=torch.tensor(list(head), device=DEV, dtype=torch.int64),
        free_list_tail=torch.tensor(list(tail), device=DEV, dtype=torch.int64),
        free_list_capacity=torch.tensor([cuda_pages, host_pages], device=DEV, dtype=torch.int64),
        overflow_initial=torch.tensor([incoming], device=DEV, dtype=torch.int64),
    )


def _outputs(row_bytes, source_rows, cuda_rows, host_rows, incoming):
    return dict(
        cuda_stash=torch.zeros((cuda_rows, row_bytes), device=DEV, dtype=torch.uint8),
        host_stash=torch.zeros((host_rows, row_bytes), pin_memory=True, dtype=torch.uint8),
        page_record=torch.zeros((source_rows + PAGE - 1) // PAGE, device=DEV, dtype=torch.int64),
        overflow=torch.tensor([incoming], device=DEV, dtype=torch.int64),
        host_spill=torch.zeros(1, device=DEV, dtype=torch.int64),
        spilled_to_host=torch.zeros(1, device=DEV, dtype=torch.int64),
        new_free_list_head=torch.zeros(2, device=DEV, dtype=torch.int64),
        free_list_head_after=torch.zeros(2, device=DEV, dtype=torch.int64),
    )


def _run_triton(inp, out, row_bytes, source_rows):
    paged_stash_copy_kernel[(min(source_rows, 2048),)](
        inp["source"],
        out["cuda_stash"],
        out["host_stash"],
        inp["num_tokens"],
        inp["free_list_cuda"],
        inp["free_list_host"],
        inp["free_list_head"],
        inp["free_list_tail"],
        inp["free_list_capacity"],
        out["page_record"],
        out["overflow"],
        out["host_spill"],
        out["spilled_to_host"],
        out["new_free_list_head"],
        PAGE_SIZE=PAGE,
        HIDDEN_SIZE=row_bytes,
        BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
        HAS_HOST_BUFFER=1,
    )
    out["free_list_head_after"].copy_(out["new_free_list_head"])


def _run_cute(inp, out, impl):
    impl(
        inp["source"],
        inp["num_tokens"],
        inp["free_list_cuda"],
        inp["free_list_host"],
        inp["free_list_head"],
        inp["free_list_tail"],
        inp["free_list_capacity"],
        inp["overflow_initial"],
        out["cuda_stash"],
        out["host_stash"],
        out["page_record"],
        out["overflow"],
        out["host_spill"],
        out["spilled_to_host"],
        out["new_free_list_head"],
        out["free_list_head_after"],
    )


def _scratch_rows(tokens, page_record, nrows):
    """Rows inside pages this activation owns that carry no live token.

    The bulk-copy kernel moves whole CTA tiles, so the tail tile may carry rows past
    ``num_tokens``.  Those rows sit in already-allocated pages and reload stops at
    ``num_tokens``, so they may differ from Triton -- nothing outside these pages may.
    """
    required = (tokens + PAGE - 1) // PAGE
    pages = page_record[:required].cpu()
    rows = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)
    live = rows[:tokens]
    # page_record holds IDs for whichever arena was selected; ignore any that fall outside
    # the arena being checked.
    mask = torch.zeros(nrows, dtype=torch.bool)
    mask[rows[rows < nrows]] = True
    mask[live[live < nrows]] = False
    return mask


def _assert_matches_triton(impl, geometry, tokens, head, tail, incoming):
    row_bytes, source_rows, cuda_rows, host_rows = geometry
    inp = _inputs(row_bytes, source_rows, cuda_rows, host_rows, tokens, head, tail, incoming)
    want = _outputs(row_bytes, source_rows, cuda_rows, host_rows, incoming)
    got = _outputs(row_bytes, source_rows, cuda_rows, host_rows, incoming)

    _run_triton(inp, want, row_bytes, source_rows)
    _run_cute(inp, got, impl)
    torch.cuda.synchronize()

    for field in OUTPUTS:
        actual, expected = got[field].cpu(), want[field].cpu()
        if field in ("cuda_stash", "host_stash"):
            differing = (actual != expected).any(dim=1)
            if not differing.any():
                continue
            scratch = _scratch_rows(tokens, want["page_record"], differing.numel())
            stray = differing & ~scratch
            assert not stray.any(), (
                f"{field}: {int(stray.sum())} rows differ outside the pages this "
                f"activation owns, first at {int(stray.nonzero()[0])}"
            )
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"{field} differs")


@pytest.mark.parametrize("geometry_name", list(GEOMETRIES))
@pytest.mark.parametrize(
    "branch", ["cuda", "cuda_ragged_tail", "free_list_wraparound", "host_spill", "overflow"]
)
def test_matches_triton(geometry_name, branch):
    """Every geometry x allocator branch must agree with the Triton kernel."""
    geometry = GEOMETRIES[geometry_name]
    _, source_rows, cuda_rows, host_rows = geometry
    cuda_pages, host_pages = cuda_rows // PAGE, host_rows // PAGE
    need = (source_rows + PAGE - 1) // PAGE

    cases = {
        "cuda": (source_rows, (0, 0), (cuda_pages, host_pages), 0),
        "cuda_ragged_tail": (source_rows - PAGE - 7, (0, 0), (cuda_pages, host_pages), 0),
        # Head near the end of the ring so page indices wrap.
        "free_list_wraparound": (
            source_rows,
            (cuda_pages - need + 1, 3),
            (2 * cuda_pages, host_pages),
            0,
        ),
        # CUDA arena exhausted -> pinned host.
        "host_spill": (source_rows, (0, 0), (need - 1, host_pages), 0),
        # Neither arena fits -> overflow, nothing copied.
        "overflow": (source_rows, (0, 0), (need - 1, need - 1), 0),
    }
    tokens, head, tail, incoming = cases[branch]
    _assert_matches_triton(cute_copy.run, geometry, tokens, head, tail, incoming)


@pytest.mark.parametrize("geometry_name", list(GEOMETRIES))
def test_direct_kernel_matches_triton(geometry_name):
    """run_direct is the fallback for geometries the bulk path cannot tile; check it directly."""
    geometry = GEOMETRIES[geometry_name]
    _, source_rows, cuda_rows, host_rows = geometry
    _assert_matches_triton(
        cute_copy.run_direct,
        geometry,
        source_rows,
        (0, 0),
        (cuda_rows // PAGE, host_rows // PAGE),
        0,
    )


def test_incoming_overflow_is_a_noop():
    """An overflow latched by an earlier stash must suppress the copy and preserve the heads."""
    _assert_matches_triton(cute_copy.run, GEOMETRIES["h2048"], 2048, (0, 0), (256, 128), 1)


def test_row_offset_beyond_int32():
    """Regression: a stash larger than 2 GiB needs 64-bit destination row offsets.

    ``row * row_bytes`` overruns Int32 once the free-list head walks far enough into a
    production-sized arena, which silently wrapped writes into unrelated memory.
    """
    row_bytes, source_rows = 2048, 1024
    cuda_rows = 1_400_000  # 1.4M * 2048B = 2.87 GiB, so row * row_bytes exceeds 2**31
    assert cuda_rows * row_bytes > 2**31
    cuda_rows -= cuda_rows % PAGE
    free = torch.cuda.mem_get_info()[0]
    if free < 4 * 2**30:
        pytest.skip("needs ~3 GiB free device memory")

    cuda_pages = cuda_rows // PAGE
    need = (source_rows + PAGE - 1) // PAGE
    # Park the head deep in the ring so the allocated pages sit past the Int32 cliff.
    head = cuda_pages - need - 1
    _assert_matches_triton(
        cute_copy.run,
        (row_bytes, source_rows, cuda_rows, PAGE * 32),
        source_rows,
        (head, 0),
        (cuda_pages + head, 32),
        0,
    )
