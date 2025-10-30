# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl
import nvtx


def _assert_1d(t: torch.Tensor, name: str):
    assert t.dim() == 1, f"{name} must be 1D, got shape {tuple(t.shape)}"
    assert t.is_contiguous(), f"{name} must be contiguous"


def _assert_2d(t: torch.Tensor, rows_min: int, cols_min: int, name: str):
    assert t.dim() == 2, f"{name} must be 2D, got shape {tuple(t.shape)}"
    assert t.size(0) >= rows_min and t.size(1) >= cols_min, \
        f"{name} must be at least {(rows_min, cols_min)}, got {tuple(t.shape)}"
    assert t.is_contiguous(), f"{name} must be contiguous"


def _assert_int32_cuda(t: torch.Tensor, name: str, device: torch.device):
    assert t.device == device, f"{name}.device mismatch (got {t.device}, expected {device})"
    assert t.dtype == torch.int32, f"{name}.dtype must be int32, got {t.dtype}"


def _validate_preflight(
    *,
    device: torch.device,
    batch_size: int,
    max_num_blocks: int,
    chunk_size_tokens: int,
    MAX_BATCH_SIZE_CONST: int,
    dc_target_size: int,
    pf_target_size: int,
    # tensors
    qo_indptr: torch.Tensor,
    last_page_len: torch.Tensor,
    indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    pf_qo_indptr: torch.Tensor,
    pf_last_page_len: torch.Tensor,
    pf_indptr: torch.Tensor,
    pf_cum_kv_seq_len: torch.Tensor,
    dc_qo_indptr: torch.Tensor,
    dc_last_page_len: torch.Tensor,
    dc_indptr: torch.Tensor,
    prefill_block_table: torch.Tensor,
    decode_block_table: torch.Tensor,
    full_block_table: torch.Tensor,
    cum_kv_seq_len: torch.Tensor,
    max_metadata: torch.Tensor,
    kv_seq_lengths: torch.Tensor,
    device_decode_prefill: torch.Tensor,
):
    # Basic params
    assert chunk_size_tokens > 0, "CHUNK_SIZE_TOKENS must be > 0"
    assert MAX_BATCH_SIZE_CONST >= batch_size, \
        f"MAX_BATCH_SIZE_CONST ({MAX_BATCH_SIZE_CONST}) must be >= BATCH_SIZE ({batch_size})"

    # Dtype/device/1D checks
    for name, t in [
        ("qo_indptr", qo_indptr),
        ("last_page_len", last_page_len),
        ("indptr", indptr),
        ("kv_indices", kv_indices),
        ("pf_qo_indptr", pf_qo_indptr),
        ("pf_last_page_len", pf_last_page_len),
        ("pf_indptr", pf_indptr),
        ("pf_cum_kv_seq_len", pf_cum_kv_seq_len),
        ("dc_qo_indptr", dc_qo_indptr),
        ("dc_last_page_len", dc_last_page_len),
        ("dc_indptr", dc_indptr),
        ("cum_kv_seq_len", cum_kv_seq_len),
        ("max_metadata", max_metadata),
        ("kv_seq_lengths", kv_seq_lengths),
        ("device_decode_prefill", device_decode_prefill),
    ]:
        _assert_int32_cuda(t, name, device)
        _assert_1d(t, name)

    # Capacity rules (allow oversizing)
    # +1 arrays may write up to index MAX_BATCH_SIZE_CONST → need len >= MAX_BATCH_SIZE_CONST + 1
    assert qo_indptr.numel()      >= MAX_BATCH_SIZE_CONST + 1, \
        f"qo_indptr too small (need >= {MAX_BATCH_SIZE_CONST + 1})"
    assert indptr.numel()         >= MAX_BATCH_SIZE_CONST + 1, \
        f"indptr too small (need >= {MAX_BATCH_SIZE_CONST + 1})"
    assert cum_kv_seq_len.numel() >= MAX_BATCH_SIZE_CONST + 1, \
        f"cum_kv_seq_len too small (need >= {MAX_BATCH_SIZE_CONST + 1})"

    # 0-index arrays extended up to MAX_BATCH_SIZE_CONST-1 → need len >= MAX_BATCH_SIZE_CONST
    assert last_page_len.numel()  >= MAX_BATCH_SIZE_CONST, \
        f"last_page_len too small (need >= {MAX_BATCH_SIZE_CONST})"

    # kv_seq_lengths writes only first BATCH_SIZE
    assert kv_seq_lengths.numel() >= batch_size, \
        f"kv_seq_lengths must be >= BATCH_SIZE ({batch_size})"

    # PF/DC partition metadata
    assert pf_qo_indptr.numel()     >= pf_target_size + 1, "pf_qo_indptr too small"
    assert pf_indptr.numel()        >= pf_target_size + 1, "pf_indptr too small"
    assert pf_last_page_len.numel() >= pf_target_size,     "pf_last_page_len too small"
    assert pf_cum_kv_seq_len.numel() >= pf_target_size + 1, "pf_cum_kv_seq_len too small"

    assert dc_qo_indptr.numel()     >= dc_target_size + 1, "dc_qo_indptr too small"
    assert dc_indptr.numel()        >= dc_target_size + 1, "dc_indptr too small"
    assert dc_last_page_len.numel() >= dc_target_size,     "dc_last_page_len too small"

    # Block tables
    _assert_2d(prefill_block_table, pf_target_size, max_num_blocks, "prefill_block_table")
    _assert_2d(decode_block_table, dc_target_size, max_num_blocks, "decode_block_table")
    _assert_2d(full_block_table,   batch_size,     max_num_blocks, "full_block_table")

    # kv_indices conservative bound
    assert kv_indices.numel() >= MAX_BATCH_SIZE_CONST * max_num_blocks, \
        f"kv_indices too small (need >= {MAX_BATCH_SIZE_CONST * max_num_blocks})"

    assert max_metadata.numel() == 2, "max_metadata must be 2 elements"
    assert device_decode_prefill.numel() == 2, "device_decode_prefill must be 2 elements"


@triton.jit
def _compute_layout_kernel(
    # --- Input Pointers ---
    REQUEST_QUERY_LENGTHS, KV_LENGTH_OFFSETS,
    # --- Full Output Pointers ---
    QO_INDPTR, LAST_PAGE_LEN, INDPTR, CUM_KV_SEQ_LEN, MAX_METADATA, KV_SEQ_LENGTHS_OUT,
    # --- Output Pointers ---
    PF_QO_INDPTR, PF_LAST_PAGE_LEN, PF_INDPTR, PF_CUM_KV_SEQ_LEN,
    DC_QO_INDPTR, DC_LAST_PAGE_LEN, DC_INDPTR,
    DEVICE_DECODE_PREFILL,
    # --- Tensor Metadata ---
    BATCH_SIZE,
    MAX_BATCH_SIZE_CONST: tl.constexpr,
    MAX_BATCH_SIZE_CONST_PW2: tl.constexpr,
    PF_COUNT,
    PF_TENSOR_SIZE: tl.constexpr,
    PF_TENSOR_SIZE_PW2: tl.constexpr,
    DC_COUNT,
    DC_TENSOR_SIZE: tl.constexpr,
    DC_TENSOR_SIZE_PW2: tl.constexpr,
    CHUNK_SIZE_TOKENS: tl.constexpr,
):
    """
    Triton kernel to compute paged attention layout metadata.
    This version uses the built-in tl.cumsum for a simpler and more efficient
    parallel scan.
    """

    # --- Phase 1: Load and Calculate in parallel across the block ---
    # Create indices for the entire block vector width (power-of-two for perf)
    batch_indices = tl.arange(0, MAX_BATCH_SIZE_CONST_PW2)
    batch_mask = batch_indices < BATCH_SIZE
    ext_mask = batch_indices < MAX_BATCH_SIZE_CONST

    # Load lengths for all requests into block-wide tensors
    q_lens = tl.load(REQUEST_QUERY_LENGTHS + batch_indices, mask=batch_mask, other=0)
    kv_offs = tl.load(KV_LENGTH_OFFSETS + batch_indices, mask=batch_mask, other=0)
    kv_lens = kv_offs + q_lens

    # Perform element-wise calculations on the block-wide tensors
    num_blocks = (kv_lens + CHUNK_SIZE_TOKENS - 1) // CHUNK_SIZE_TOKENS
    last_page_len = (kv_lens - 1) % CHUNK_SIZE_TOKENS + 1

    # --- Phase 2: Parallel Cumulative Sum using tl.cumsum ---
    # tl.cumsum performs a parallel scan across the specified axis of the block-wide tensor.
    qo_indptr_inclusive = tl.cumsum(q_lens, axis=0)
    indptr_inclusive = tl.cumsum(num_blocks, axis=0)
    cum_kv_inclusive = tl.cumsum(kv_lens, axis=0)

    # --- Phase 3: Save full ---
    tl.store(QO_INDPTR, 0)
    tl.store(INDPTR, 0)
    tl.store(CUM_KV_SEQ_LEN, 0)
    tl.store(QO_INDPTR + 1 + batch_indices, qo_indptr_inclusive, mask=batch_mask)
    tl.store(INDPTR + 1 + batch_indices, indptr_inclusive, mask=batch_mask)
    tl.store(LAST_PAGE_LEN + batch_indices, last_page_len, mask=batch_mask)
    tl.store(CUM_KV_SEQ_LEN + 1 + batch_indices, cum_kv_inclusive, mask=batch_mask)
    # also expose per-request kv sequence lengths
    tl.store(KV_SEQ_LENGTHS_OUT + batch_indices, kv_lens, mask=batch_mask)

    # Extend full outputs to MAX_BATCH_SIZE_CONST within provided buffer lengths
    # For arrays with +1 indexing (QO/INDPTR/CUMKV), cap extension at MAX-1 to avoid overflow
    full_ext_mask_qi = (batch_indices >= BATCH_SIZE) & (batch_indices < MAX_BATCH_SIZE_CONST)
    full_ext_mask_lp = (batch_indices >= BATCH_SIZE) & (batch_indices < MAX_BATCH_SIZE_CONST)
    last_qo_full = tl.load(QO_INDPTR + BATCH_SIZE)
    last_indptr_full = tl.load(INDPTR + BATCH_SIZE)
    last_cum_kv_full = tl.load(CUM_KV_SEQ_LEN + BATCH_SIZE)
    tl.store(QO_INDPTR + 1 + batch_indices, last_qo_full, mask=full_ext_mask_qi)
    tl.store(INDPTR + 1 + batch_indices, last_indptr_full, mask=full_ext_mask_qi)
    tl.store(LAST_PAGE_LEN + batch_indices, 0, mask=full_ext_mask_lp)
    tl.store(CUM_KV_SEQ_LEN + 1 + batch_indices, last_cum_kv_full, mask=full_ext_mask_qi)
    tl.store(KV_SEQ_LENGTHS_OUT + batch_indices, 0, mask=full_ext_mask_lp)

    # Max metadata: [0] max q len, [1] max k len
    max_q = tl.max(q_lens, axis=0)
    max_k = tl.max(kv_lens, axis=0)
    tl.store(MAX_METADATA + 0, max_q)
    tl.store(MAX_METADATA + 1, max_k)

    # --- Phase 4: Save decode ---
    tl.store(DC_QO_INDPTR, 0)
    tl.store(DC_INDPTR, 0)
    dc_range = tl.arange(0, DC_TENSOR_SIZE_PW2)
    dc_mask = dc_range < DC_COUNT
    ext_mask = (dc_range >= DC_COUNT) & (dc_range < DC_TENSOR_SIZE)
    # store decode qo_indptr
    decode_qo_indptr = tl.load(QO_INDPTR + 1 + dc_range, mask=dc_mask)
    last_qo = tl.load(QO_INDPTR + DC_COUNT)
    tl.store(DC_QO_INDPTR + 1 + dc_range, decode_qo_indptr, mask=dc_mask)
    tl.store(DC_QO_INDPTR + 1 + dc_range, last_qo, mask=ext_mask)
    # store decode indptr
    decode_indptr = tl.load(INDPTR + 1 + dc_range, mask=dc_mask)
    last_indptr = tl.load(INDPTR + DC_COUNT)
    tl.store(DC_INDPTR + 1 + dc_range, decode_indptr, mask=dc_mask)
    tl.store(DC_INDPTR + 1 + dc_range, last_indptr, mask=ext_mask)
    # store decode last_page_len
    decode_last_page_len = tl.load(LAST_PAGE_LEN + dc_range, mask=dc_mask)
    tl.store(DC_LAST_PAGE_LEN + dc_range, decode_last_page_len, mask=dc_mask)
    tl.store(DC_LAST_PAGE_LEN + dc_range, 0, mask=ext_mask)


    # --- Phase 5: Save prefill ---
    tl.store(PF_QO_INDPTR, 0)  # Start prefill qo_indptr from 0 for relative indexing
    tl.store(PF_INDPTR, last_indptr)
    tl.store(PF_CUM_KV_SEQ_LEN, 0)  # Start prefill cum_kv_seq_len from 0 for relative indexing
    pf_range = tl.arange(0, PF_TENSOR_SIZE_PW2)
    pf_mask = pf_range < PF_COUNT
    ext_mask = (pf_range >= PF_COUNT) & (pf_range < PF_TENSOR_SIZE)
    # store prefill qo_indptr
    prefill_qo_indptr = tl.load(QO_INDPTR + 1 + DC_COUNT + pf_range, mask=pf_mask)
    prefill_qo_indptr = prefill_qo_indptr - last_qo
    last_qo_end = tl.load(QO_INDPTR + DC_COUNT + PF_COUNT)
    tl.store(PF_QO_INDPTR + 1 + pf_range, prefill_qo_indptr, mask=pf_mask)
    tl.store(PF_QO_INDPTR + 1 + pf_range, last_qo_end - last_qo, mask=ext_mask)  # Use relative end value
    # store prefill indptr
    prefill_indptr = tl.load(INDPTR + 1 + DC_COUNT + pf_range, mask=pf_mask)
    last_indptr_end = tl.load(INDPTR + DC_COUNT + PF_COUNT)
    tl.store(PF_INDPTR + 1 + pf_range, prefill_indptr, mask=pf_mask)
    tl.store(PF_INDPTR + 1 + pf_range, last_indptr_end, mask=ext_mask)
    # store prefill last_page_len
    prefill_last_page_len = tl.load(LAST_PAGE_LEN + DC_COUNT + pf_range, mask=pf_mask)
    tl.store(PF_LAST_PAGE_LEN + pf_range, prefill_last_page_len, mask=pf_mask)
    tl.store(PF_LAST_PAGE_LEN + pf_range, 0, mask=ext_mask)
    # store prefill cum_kv_seq_len
    last_cum_kv_full = tl.load(CUM_KV_SEQ_LEN + DC_COUNT)
    prefill_cum_kv = tl.load(CUM_KV_SEQ_LEN + 1 + DC_COUNT + pf_range, mask=pf_mask)
    prefill_cum_kv = prefill_cum_kv - last_cum_kv_full
    last_cum_kv_end = tl.load(CUM_KV_SEQ_LEN + DC_COUNT + PF_COUNT)
    tl.store(PF_CUM_KV_SEQ_LEN + 1 + pf_range, prefill_cum_kv, mask=pf_mask)
    tl.store(PF_CUM_KV_SEQ_LEN + 1 + pf_range, last_cum_kv_end - last_cum_kv_full, mask=ext_mask)

    # --- Phase 6: Save device decode prefill ---
    tl.store(DEVICE_DECODE_PREFILL, DC_COUNT)
    tl.store(DEVICE_DECODE_PREFILL + 1, PF_COUNT)


@triton.jit
def _block_transform(
    # --- Input Pointers ---
    kv_indptr_ptr,
    block_table_ptr,
    # --- Output Pointers ---
    kv_indices_ptr,
    prefill_block_table_ptr,
    decode_block_table_ptr,
    full_block_table_ptr,
    # --- Tensor Metadata ---
    num_blocks: tl.constexpr,
    MAX_BLOCKS_PW2: tl.constexpr,
    PF_COUNT,
    DC_COUNT,
    batch_size,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    start = tl.load(kv_indptr_ptr + pid)
    end = tl.load(kv_indptr_ptr + pid + 1)
    block_table_range = tl.arange(0, MAX_BLOCKS_PW2)
    block_table = tl.load(block_table_ptr + num_blocks * pid + block_table_range, mask=block_table_range < num_blocks)
    kv_indptr_range = tl.arange(0, MAX_BLOCKS_PW2) + start
    tl.store(kv_indices_ptr + kv_indptr_range, block_table, mask=kv_indptr_range < end)
    # full copy of the input block table row
    tl.store(full_block_table_ptr + pid * num_blocks + block_table_range, block_table, mask=block_table_range < num_blocks)
    if pid < DC_COUNT:
        tl.store(decode_block_table_ptr + pid * num_blocks + block_table_range, block_table, mask=block_table_range < num_blocks)
    elif pid < PF_COUNT + DC_COUNT:
        offset = pid - DC_COUNT
        tl.store(prefill_block_table_ptr + offset * num_blocks + block_table_range, block_table, mask=block_table_range < num_blocks)


def compute_layout_triton(
    request_query_lengths_view: torch.Tensor,
    request_kv_length_offsets_view: torch.Tensor,
    block_table: torch.Tensor,
    chunk_size_tokens: int,
    # partitioning
    dc_count: int = 0,
    pf_count: int = 0,
    dc_target_size: int | None = None,
    pf_target_size: int | None = None,
    # preallocated outputs
    qo_indptr: torch.Tensor | None = None,
    last_page_len: torch.Tensor | None = None,
    indptr: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    pf_qo_indptr: torch.Tensor | None = None,
    pf_last_page_len: torch.Tensor | None = None,
    pf_indptr: torch.Tensor | None = None,
    pf_cum_kv_seq_len: torch.Tensor | None = None,
    dc_qo_indptr: torch.Tensor | None = None,
    dc_last_page_len: torch.Tensor | None = None,
    dc_indptr: torch.Tensor | None = None,
    prefill_block_table: torch.Tensor | None = None,
    decode_block_table: torch.Tensor | None = None,
    full_block_table: torch.Tensor | None = None,
    # new outputs
    cum_kv_seq_len: torch.Tensor | None = None,
    max_metadata: torch.Tensor | None = None,
    kv_seq_lengths: torch.Tensor | None = None,
    device_decode_prefill: torch.Tensor | None = None,
    MAX_BATCH_SIZE_CONST: int = 0,
    check_layout: bool = False,
):
    """
    Python wrapper for launching the simplified Triton kernel.

    Returns a tuple of 18 tensors:
      (qo_indptr, last_page_len, kv_indices, indptr,
       cum_kv_seq_len, max_metadata, kv_seq_lengths,
       pf_qo_indptr, pf_last_page_len, pf_indptr, pf_cum_kv_seq_len,
       dc_qo_indptr, dc_last_page_len, dc_indptr,
       prefill_block_table, decode_block_table, full_block_table,
       device_decode_prefill)
    """
    # 1. Prepare parameters and validate outputs

    if check_layout:
        _validate_preflight(
            device=request_query_lengths_view.device,
            batch_size=request_query_lengths_view.shape[0],
            max_num_blocks=block_table.shape[1],
            chunk_size_tokens=chunk_size_tokens,
            MAX_BATCH_SIZE_CONST=MAX_BATCH_SIZE_CONST,
            dc_target_size=dc_target_size,
            pf_target_size=pf_target_size,
            qo_indptr=qo_indptr,
            last_page_len=last_page_len,
            indptr=indptr,
            kv_indices=kv_indices,
            pf_qo_indptr=pf_qo_indptr,
            pf_last_page_len=pf_last_page_len,
            pf_indptr=pf_indptr,
            pf_cum_kv_seq_len=pf_cum_kv_seq_len,
            dc_qo_indptr=dc_qo_indptr,
            dc_last_page_len=dc_last_page_len,
            dc_indptr=dc_indptr,
            prefill_block_table=prefill_block_table,
            decode_block_table=decode_block_table,
            full_block_table=full_block_table,
            cum_kv_seq_len=cum_kv_seq_len,
            max_metadata=max_metadata,
            kv_seq_lengths=kv_seq_lengths,
            device_decode_prefill=device_decode_prefill,
        )
    with nvtx.annotate("paramter prepare"):
        batch_size = request_query_lengths_view.shape[0]
        max_num_blocks = block_table.shape[1]
        assert dc_count >= 0 and pf_count >= 0 and (dc_count + pf_count) == batch_size
        device = block_table.device
        if dc_target_size is None:
            dc_target_size = dc_count
        if pf_target_size is None:
            pf_target_size = pf_count

        # Ensure inputs are int32
        request_query_lengths_view = request_query_lengths_view.to(torch.int32)
        request_kv_length_offsets_view = request_kv_length_offsets_view.to(torch.int32)
        block_table = block_table.to(torch.int32)

        # Validate outputs presence
        assert qo_indptr is not None and last_page_len is not None and indptr is not None and kv_indices is not None
        assert pf_qo_indptr is not None and pf_last_page_len is not None and pf_indptr is not None and pf_cum_kv_seq_len is not None
        assert dc_qo_indptr is not None and dc_last_page_len is not None and dc_indptr is not None
        assert prefill_block_table is not None and decode_block_table is not None and full_block_table is not None
        assert cum_kv_seq_len is not None and max_metadata is not None and kv_seq_lengths is not None
        assert device_decode_prefill is not None

        # Validate dtype/device/shapes
        for t in (
            qo_indptr,
            last_page_len,
            indptr,
            kv_indices,
            pf_qo_indptr,
            pf_last_page_len,
            pf_indptr,
            pf_cum_kv_seq_len,
            dc_qo_indptr,
            dc_last_page_len,
            dc_indptr,
            prefill_block_table,
            decode_block_table,
            full_block_table,
            cum_kv_seq_len,
            max_metadata,
            device_decode_prefill,
        ):
            assert t.device == device and t.dtype == torch.int32

        dc_target_size_pw2 = triton.next_power_of_2(dc_target_size) if dc_target_size > 1 else 2
        pf_target_size_pw2 = triton.next_power_of_2(pf_target_size) if pf_target_size > 1 else 2

    with nvtx.annotate("kernel launch"):
        # 2. Define constants for the kernel.
        # MAX_BATCH_SIZE_CONST = dc_target_size + pf_target_size
        MAX_BATCH_SIZE_CONST_PW2 = triton.next_power_of_2(MAX_BATCH_SIZE_CONST) if MAX_BATCH_SIZE_CONST > 1 else 2

        # Per the request, use 8 warps.
        NUM_WARPS = 8

        # 4. Define grid and launch compute kernel (single program/block)
        grid = (1,)

        _compute_layout_kernel[grid](
            request_query_lengths_view,
            request_kv_length_offsets_view,
            qo_indptr,
            last_page_len,
            indptr,
            cum_kv_seq_len,
            max_metadata,
            kv_seq_lengths,
            # PF/DC outputs
            pf_qo_indptr,
            pf_last_page_len,
            pf_indptr,
            pf_cum_kv_seq_len,
            dc_qo_indptr,
            dc_last_page_len,
            dc_indptr,
            device_decode_prefill,
            # Meta
            batch_size,
            MAX_BATCH_SIZE_CONST=MAX_BATCH_SIZE_CONST,
            MAX_BATCH_SIZE_CONST_PW2=MAX_BATCH_SIZE_CONST_PW2,
            PF_COUNT=pf_count,
            PF_TENSOR_SIZE=pf_target_size,
            PF_TENSOR_SIZE_PW2=pf_target_size_pw2,
            DC_COUNT=dc_count,
            DC_TENSOR_SIZE=dc_target_size,
            DC_TENSOR_SIZE_PW2=dc_target_size_pw2,
            CHUNK_SIZE_TOKENS=chunk_size_tokens,
            num_warps=NUM_WARPS,
        )

    # 6. Build kv_indices via the block transform kernel (one program per batch)
    num_blocks = max_num_blocks
    max_blocks_pw2 = triton.next_power_of_2(num_blocks) if num_blocks > 1 else 2

    with nvtx.annotate("block_transform"):
        grid_bt = (batch_size,)
        _block_transform[grid_bt](
            indptr,
            block_table,
            kv_indices,
            prefill_block_table,
            decode_block_table,
            full_block_table,
            num_blocks=num_blocks,
            MAX_BLOCKS_PW2=max_blocks_pw2,
            PF_COUNT=pf_count,
            DC_COUNT=dc_count,
            batch_size=batch_size,
            num_warps=NUM_WARPS,
        )

    return (
        qo_indptr,
        last_page_len,
        kv_indices,
        indptr,
        cum_kv_seq_len,
        max_metadata,
        kv_seq_lengths,
        pf_qo_indptr,
        pf_last_page_len,
        pf_indptr,
        pf_cum_kv_seq_len,
        dc_qo_indptr,
        dc_last_page_len,
        dc_indptr,
        prefill_block_table,
        decode_block_table,
        full_block_table,
        device_decode_prefill,
    )
