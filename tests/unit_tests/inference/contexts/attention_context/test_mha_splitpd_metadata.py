# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import pytest

from megatron.core.inference.contexts.attention_context.triton import compute_layout_triton


@torch.no_grad()
def compute_layout_pytorch(
    request_query_lengths_view: torch.Tensor,
    request_kv_length_offsets_view: torch.Tensor,
    block_table: torch.Tensor,
    chunk_size_tokens: int,
    dc_count: int = 0,
    pf_count: int = 0,
    dc_target_size: int | None = None,
    pf_target_size: int | None = None,
):
    """
    Updated PyTorch reference implementation matching the Triton kernel semantics.

    - Always returns full outputs (qo_indptr [B+1], indptr [B+1]).
    - Returns last_page_len with extension when extend=True (length >= B+1; default B+1).
    - Returns PF/DC partitioned metadata tensors mirroring kernel behavior.
    - Returns PF/DC block tables re-ordered from the original `block_table`.
    """
    # Ensure inputs are int32 for consistency with the kernel
    request_query_lengths_view = request_query_lengths_view.to(torch.int32)
    request_kv_length_offsets_view = request_kv_length_offsets_view.to(torch.int32)
    block_table = block_table.to(torch.int32)

    device = block_table.device

    # Build qo_indptr and paged-kv metadata from inputs
    qo_indptr_all = torch.cumsum(request_query_lengths_view, dim=0)
    qo_indptr_all = torch.cat(
        [torch.tensor([0], device=qo_indptr_all.device, dtype=torch.int32), qo_indptr_all]
    )
    kv_seq_lengths_view = request_kv_length_offsets_view + request_query_lengths_view
    num_blocks_for_each_req = (kv_seq_lengths_view + chunk_size_tokens - 1) // chunk_size_tokens
    last_page_len_all = ((kv_seq_lengths_view - 1) % chunk_size_tokens + 1).to(torch.int32)
    batch_size = last_page_len_all.shape[0]
    qo_indptr_all = qo_indptr_all[: batch_size + 1]

    # kv_indices and indptr from block_table
    max_num_blocks = block_table.shape[1]
    col_indices = torch.arange(max_num_blocks, device=block_table.device)
    mask = col_indices < num_blocks_for_each_req.unsqueeze(1)
    kv_indices_all = block_table[: batch_size][mask]
    indptr_all = torch.cumsum(num_blocks_for_each_req, dim=0)
    indptr_all = torch.cat(
        [torch.tensor([0], device=indptr_all.device, dtype=torch.int32), indptr_all]
    )

    # Additional outputs: cumulative kv seq length (with leading 0) and max metadata
    cum_kv_seq_len = torch.cumsum(kv_seq_lengths_view, dim=0)
    cum_kv_seq_len = torch.cat(
        [torch.tensor([0], device=device, dtype=torch.int32), cum_kv_seq_len]
    )
    max_q = torch.max(request_query_lengths_view) if batch_size > 0 else torch.tensor(0, dtype=torch.int32, device=device)
    max_k = torch.max(kv_seq_lengths_view) if batch_size > 0 else torch.tensor(0, dtype=torch.int32, device=device)
    max_metadata = torch.stack([max_q.to(torch.int32), max_k.to(torch.int32)])

    # Defaults for partitioning
    assert dc_count >= 0 and pf_count >= 0 and (dc_count + pf_count) == batch_size
    if dc_target_size is None:
        dc_target_size = dc_count
    if pf_target_size is None:
        pf_target_size = pf_count

    # DC metadata outputs
    dc_qo_indptr = torch.zeros(dc_target_size + 1, dtype=torch.int32, device=device)
    dc_indptr = torch.zeros(dc_target_size + 1, dtype=torch.int32, device=device)
    dc_last_page_len = torch.empty(dc_target_size, dtype=torch.int32, device=device)

    if dc_count > 0:
        dc_qo_indptr[1 : 1 + dc_count] = qo_indptr_all[1 : 1 + dc_count]
        dc_indptr[1 : 1 + dc_count] = indptr_all[1 : 1 + dc_count]
    # pad with last value at index dc_count
    last_qo_dc = qo_indptr_all[dc_count]
    last_indptr_dc = indptr_all[dc_count]
    if dc_target_size > dc_count:
        dc_qo_indptr[1 + dc_count : 1 + dc_target_size] = last_qo_dc
        dc_indptr[1 + dc_count : 1 + dc_target_size] = last_indptr_dc
    if dc_count > 0:
        dc_last_page_len[: dc_count] = last_page_len_all[: dc_count]
    if dc_target_size - dc_count > 0:
        dc_last_page_len[dc_count : dc_target_size] = 0

    # PF metadata outputs
    pf_qo_indptr = torch.empty(pf_target_size + 1, dtype=torch.int32, device=device)
    pf_indptr = torch.empty(pf_target_size + 1, dtype=torch.int32, device=device)
    pf_last_page_len = torch.empty(pf_target_size, dtype=torch.int32, device=device)
    pf_cum_kv_seq_len = torch.empty(pf_target_size + 1, dtype=torch.int32, device=device)

    last_qo_prefix = qo_indptr_all[dc_count]
    last_qo_end = qo_indptr_all[dc_count + pf_count]
    last_indptr_prefix = indptr_all[dc_count]
    last_indptr_end = indptr_all[dc_count + pf_count]
    last_cum_kv_prefix = cum_kv_seq_len[dc_count]
    last_cum_kv_end = cum_kv_seq_len[dc_count + pf_count]

    pf_qo_indptr[0] = 0  # Start from 0 for relative indexing
    pf_indptr[0] = last_indptr_prefix
    pf_cum_kv_seq_len[0] = 0  # Start from 0 for relative indexing
    if pf_count > 0:
        # Make prefill qo_indptr relative by subtracting the decode boundary offset
        pf_qo_indptr[1 : 1 + pf_count] = qo_indptr_all[1 + dc_count : 1 + dc_count + pf_count] - last_qo_prefix
        pf_indptr[1 : 1 + pf_count] = indptr_all[1 + dc_count : 1 + dc_count + pf_count]
        pf_last_page_len[: pf_count] = last_page_len_all[dc_count : dc_count + pf_count]
        # Make prefill cum_kv_seq_len relative by subtracting the decode boundary offset
        pf_cum_kv_seq_len[1 : 1 + pf_count] = cum_kv_seq_len[1 + dc_count : 1 + dc_count + pf_count] - last_cum_kv_prefix
    # pad remainder up to pf_tensor_size with last values
    if pf_target_size - pf_count > 0:
        # For padding, use the relative end value
        pf_qo_indptr[1 + pf_count : 1 + pf_target_size] = last_qo_end - last_qo_prefix
        pf_indptr[1 + pf_count : 1 + pf_target_size] = last_indptr_end
        # Use the last valid value to fill padding region
        pf_last_page_len[pf_count : pf_target_size] = 0
        pf_cum_kv_seq_len[1 + pf_count : 1 + pf_target_size] = last_cum_kv_end - last_cum_kv_prefix

    # PF/DC block tables (re-ordered copies of rows)
    num_blocks = max_num_blocks
    prefill_block_table = torch.zeros((pf_target_size, num_blocks), dtype=torch.int32, device=device)
    decode_block_table = torch.zeros((dc_target_size, num_blocks), dtype=torch.int32, device=device)
    if dc_count > 0:
        decode_block_table[:dc_count, :] = block_table[:dc_count, :]
    if pf_count > 0:
        prefill_block_table[:pf_count, :] = block_table[dc_count : dc_count + pf_count, :]

    # Create device_decode_prefill tensor to match kernel output
    device_decode_prefill = torch.tensor([dc_count, pf_count], dtype=torch.int32, device=device)

    return (
        qo_indptr_all,
        last_page_len_all,
        kv_indices_all,
        indptr_all,
        cum_kv_seq_len,
        max_metadata,
        kv_seq_lengths_view.to(torch.int32),
        pf_qo_indptr,
        pf_last_page_len,
        pf_indptr,
        pf_cum_kv_seq_len,
        dc_qo_indptr,
        dc_last_page_len,
        dc_indptr,
        prefill_block_table,
        decode_block_table,
        device_decode_prefill,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMHASplitPDMetadata:
    """Test suite for MHA SplitPD Metadata Triton kernel."""

    def test_compute_layout_triton_basic(self):
        """Test basic functionality of the Triton layout computation kernel."""
        # Define test parameters
        BATCH_SIZE = 64
        MAX_SEQ_LEN = 2048
        MAX_NUM_BLOCKS = 128
        CHUNK_SIZE_TOKENS = 16
        TARGET_DC_SIZE = 48
        TARGET_PF_SIZE = 48
        PF_COUNT = 24
        DC_COUNT = BATCH_SIZE - PF_COUNT
        MAX_BATCH_SIZE = BATCH_SIZE + 16

        DEVICE = 'cuda'

        # Generate random input tensors
        request_query_lengths_view = torch.ones(BATCH_SIZE, dtype=torch.int32, device=DEVICE)
        request_kv_length_offsets_view = torch.randint(
            0, MAX_SEQ_LEN // 2, (BATCH_SIZE,), dtype=torch.int32, device=DEVICE
        )
        block_table = torch.randint(
            0, MAX_NUM_BLOCKS, (BATCH_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE
        )

        # Allocate output buffers for Triton
        qo_indptr_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        last_page_len_tr = torch.empty(MAX_BATCH_SIZE, dtype=torch.int32, device=DEVICE)
        indptr_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        kv_indices_tr = torch.empty(MAX_BATCH_SIZE * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE)
        cum_kv_seq_len_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        max_metadata_tr = torch.empty(2, dtype=torch.int32, device=DEVICE)
        kv_seq_lengths_tr = torch.empty(MAX_BATCH_SIZE, dtype=torch.int32, device=DEVICE)

        pf_qo_indptr_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)
        pf_last_page_len_tr = torch.empty(TARGET_PF_SIZE, dtype=torch.int32, device=DEVICE)
        pf_indptr_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)
        pf_cum_kv_seq_len_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)

        dc_qo_indptr_tr = torch.empty(TARGET_DC_SIZE + 1, dtype=torch.int32, device=DEVICE)
        dc_last_page_len_tr = torch.empty(TARGET_DC_SIZE, dtype=torch.int32, device=DEVICE)
        dc_indptr_tr = torch.empty(TARGET_DC_SIZE + 1, dtype=torch.int32, device=DEVICE)

        prefill_block_table_tr = torch.empty((TARGET_PF_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        decode_block_table_tr = torch.empty((TARGET_DC_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        full_block_table_tr = torch.empty((BATCH_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        device_decode_prefill_tr = torch.empty(2, dtype=torch.int32, device=DEVICE)

        # Run PyTorch reference
        (
            qo_indptr_pt,
            last_page_len_pt,
            kv_indices_pt,
            indptr_pt,
            cum_kv_seq_len_pt,
            max_metadata_pt,
            kv_seq_lengths_pt,
            pf_qo_indptr_pt,
            pf_last_page_len_pt,
            pf_indptr_pt,
            pf_cum_kv_seq_len_pt,
            dc_qo_indptr_pt,
            dc_last_page_len_pt,
            dc_indptr_pt,
            prefill_block_table_pt,
            decode_block_table_pt,
            device_decode_prefill_pt,
        ) = compute_layout_pytorch(
            request_query_lengths_view,
            request_kv_length_offsets_view,
            block_table,
            CHUNK_SIZE_TOKENS,
            dc_count=DC_COUNT,
            pf_count=PF_COUNT,
            dc_target_size=TARGET_DC_SIZE,
            pf_target_size=TARGET_PF_SIZE,
        )

        # Run Triton kernel
        (
            qo_indptr_tr,
            last_page_len_tr,
            kv_indices_tr,
            indptr_tr,
            cum_kv_seq_len_tr,
            max_metadata_tr,
            kv_seq_lengths_tr,
            pf_qo_indptr_tr,
            pf_last_page_len_tr,
            pf_indptr_tr,
            pf_cum_kv_seq_len_tr,
            dc_qo_indptr_tr,
            dc_last_page_len_tr,
            dc_indptr_tr,
            prefill_block_table_tr,
            decode_block_table_tr,
            full_block_table_tr,
            device_decode_prefill_tr,
        ) = compute_layout_triton(
            request_query_lengths_view,
            request_kv_length_offsets_view,
            block_table,
            CHUNK_SIZE_TOKENS,
            dc_count=DC_COUNT,
            pf_count=PF_COUNT,
            dc_target_size=TARGET_DC_SIZE,
            pf_target_size=TARGET_PF_SIZE,
            qo_indptr=qo_indptr_tr,
            last_page_len=last_page_len_tr,
            indptr=indptr_tr,
            kv_indices=kv_indices_tr,
            pf_qo_indptr=pf_qo_indptr_tr,
            pf_last_page_len=pf_last_page_len_tr,
            pf_indptr=pf_indptr_tr,
            pf_cum_kv_seq_len=pf_cum_kv_seq_len_tr,
            dc_qo_indptr=dc_qo_indptr_tr,
            dc_last_page_len=dc_last_page_len_tr,
            dc_indptr=dc_indptr_tr,
            prefill_block_table=prefill_block_table_tr,
            decode_block_table=decode_block_table_tr,
            full_block_table=full_block_table_tr,
            cum_kv_seq_len=cum_kv_seq_len_tr,
            max_metadata=max_metadata_tr,
            kv_seq_lengths=kv_seq_lengths_tr,
            device_decode_prefill=device_decode_prefill_tr,
            MAX_BATCH_SIZE_CONST=MAX_BATCH_SIZE,
            check_layout=True,
        )

        # Verify outputs match
        assert torch.equal(qo_indptr_pt[:BATCH_SIZE + 1], qo_indptr_tr[:BATCH_SIZE + 1]), "QO_INDPTR mismatch"
        assert torch.equal(last_page_len_pt[:BATCH_SIZE], last_page_len_tr[:BATCH_SIZE]), "LAST_PAGE_LEN mismatch"
        assert torch.equal(indptr_pt[:BATCH_SIZE + 1], indptr_tr[:BATCH_SIZE + 1]), "INDPTR mismatch"
        assert torch.equal(cum_kv_seq_len_pt[:BATCH_SIZE + 1], cum_kv_seq_len_tr[:BATCH_SIZE + 1]), "CUM_KV_SEQ_LEN mismatch"
        assert torch.equal(max_metadata_pt, max_metadata_tr), "MAX_METADATA mismatch"
        assert torch.equal(kv_seq_lengths_pt[:BATCH_SIZE], kv_seq_lengths_tr[:BATCH_SIZE]), "KV_SEQ_LENGTHS mismatch"
        assert torch.equal(kv_indices_pt[:indptr_pt[BATCH_SIZE]], kv_indices_tr[:indptr_tr[BATCH_SIZE]]), "KV_INDICES mismatch"
        assert torch.equal(pf_qo_indptr_pt, pf_qo_indptr_tr), "PF_QO_INDPTR mismatch"
        assert torch.equal(pf_last_page_len_pt, pf_last_page_len_tr), "PF_LAST_PAGE_LEN mismatch"
        assert torch.equal(pf_indptr_pt, pf_indptr_tr), "PF_INDPTR mismatch"
        assert torch.equal(pf_cum_kv_seq_len_pt, pf_cum_kv_seq_len_tr), "PF_CUM_KV_SEQ_LEN mismatch"
        assert torch.equal(dc_qo_indptr_pt, dc_qo_indptr_tr), "DC_QO_INDPTR mismatch"
        assert torch.equal(dc_last_page_len_pt, dc_last_page_len_tr), "DC_LAST_PAGE_LEN mismatch"
        assert torch.equal(dc_indptr_pt, dc_indptr_tr), "DC_INDPTR mismatch"
        assert torch.equal(prefill_block_table_pt[:PF_COUNT, :], prefill_block_table_tr[:PF_COUNT, :]), "PREFILL_BLOCK_TABLE mismatch"
        assert torch.equal(decode_block_table_pt[:DC_COUNT, :], decode_block_table_tr[:DC_COUNT, :]), "DECODE_BLOCK_TABLE mismatch"
        assert torch.equal(block_table, full_block_table_tr), "FULL_BLOCK_TABLE mismatch"
        assert torch.equal(device_decode_prefill_pt, device_decode_prefill_tr), "DEVICE_DECODE_PREFILL mismatch"

    def test_compute_layout_triton_large_batch(self):
        """Test with larger batch size."""
        # Define test parameters
        BATCH_SIZE = 1024
        MAX_SEQ_LEN = 4096
        MAX_NUM_BLOCKS = 256
        CHUNK_SIZE_TOKENS = 16
        TARGET_DC_SIZE = 768
        TARGET_PF_SIZE = 768
        PF_COUNT = 384
        DC_COUNT = BATCH_SIZE - PF_COUNT
        MAX_BATCH_SIZE = BATCH_SIZE + 256

        DEVICE = 'cuda'

        # Generate random input tensors
        request_query_lengths_view = torch.ones(BATCH_SIZE, dtype=torch.int32, device=DEVICE)
        request_kv_length_offsets_view = torch.randint(
            0, MAX_SEQ_LEN // 2, (BATCH_SIZE,), dtype=torch.int32, device=DEVICE
        )
        block_table = torch.randint(
            0, MAX_NUM_BLOCKS, (BATCH_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE
        )

        # Allocate output buffers for Triton
        qo_indptr_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        last_page_len_tr = torch.empty(MAX_BATCH_SIZE, dtype=torch.int32, device=DEVICE)
        indptr_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        kv_indices_tr = torch.empty(MAX_BATCH_SIZE * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE)
        cum_kv_seq_len_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        max_metadata_tr = torch.empty(2, dtype=torch.int32, device=DEVICE)
        kv_seq_lengths_tr = torch.empty(MAX_BATCH_SIZE, dtype=torch.int32, device=DEVICE)

        pf_qo_indptr_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)
        pf_last_page_len_tr = torch.empty(TARGET_PF_SIZE, dtype=torch.int32, device=DEVICE)
        pf_indptr_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)
        pf_cum_kv_seq_len_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)

        dc_qo_indptr_tr = torch.empty(TARGET_DC_SIZE + 1, dtype=torch.int32, device=DEVICE)
        dc_last_page_len_tr = torch.empty(TARGET_DC_SIZE, dtype=torch.int32, device=DEVICE)
        dc_indptr_tr = torch.empty(TARGET_DC_SIZE + 1, dtype=torch.int32, device=DEVICE)

        prefill_block_table_tr = torch.empty((TARGET_PF_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        decode_block_table_tr = torch.empty((TARGET_DC_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        full_block_table_tr = torch.empty((BATCH_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        device_decode_prefill_tr = torch.empty(2, dtype=torch.int32, device=DEVICE)

        # Run PyTorch reference
        (
            qo_indptr_pt,
            last_page_len_pt,
            kv_indices_pt,
            indptr_pt,
            cum_kv_seq_len_pt,
            max_metadata_pt,
            kv_seq_lengths_pt,
            pf_qo_indptr_pt,
            pf_last_page_len_pt,
            pf_indptr_pt,
            pf_cum_kv_seq_len_pt,
            dc_qo_indptr_pt,
            dc_last_page_len_pt,
            dc_indptr_pt,
            prefill_block_table_pt,
            decode_block_table_pt,
            device_decode_prefill_pt,
        ) = compute_layout_pytorch(
            request_query_lengths_view,
            request_kv_length_offsets_view,
            block_table,
            CHUNK_SIZE_TOKENS,
            dc_count=DC_COUNT,
            pf_count=PF_COUNT,
            dc_target_size=TARGET_DC_SIZE,
            pf_target_size=TARGET_PF_SIZE,
        )

        # Run Triton kernel
        (
            qo_indptr_tr,
            last_page_len_tr,
            kv_indices_tr,
            indptr_tr,
            cum_kv_seq_len_tr,
            max_metadata_tr,
            kv_seq_lengths_tr,
            pf_qo_indptr_tr,
            pf_last_page_len_tr,
            pf_indptr_tr,
            pf_cum_kv_seq_len_tr,
            dc_qo_indptr_tr,
            dc_last_page_len_tr,
            dc_indptr_tr,
            prefill_block_table_tr,
            decode_block_table_tr,
            full_block_table_tr,
            device_decode_prefill_tr,
        ) = compute_layout_triton(
            request_query_lengths_view,
            request_kv_length_offsets_view,
            block_table,
            CHUNK_SIZE_TOKENS,
            dc_count=DC_COUNT,
            pf_count=PF_COUNT,
            dc_target_size=TARGET_DC_SIZE,
            pf_target_size=TARGET_PF_SIZE,
            qo_indptr=qo_indptr_tr,
            last_page_len=last_page_len_tr,
            indptr=indptr_tr,
            kv_indices=kv_indices_tr,
            pf_qo_indptr=pf_qo_indptr_tr,
            pf_last_page_len=pf_last_page_len_tr,
            pf_indptr=pf_indptr_tr,
            pf_cum_kv_seq_len=pf_cum_kv_seq_len_tr,
            dc_qo_indptr=dc_qo_indptr_tr,
            dc_last_page_len=dc_last_page_len_tr,
            dc_indptr=dc_indptr_tr,
            prefill_block_table=prefill_block_table_tr,
            decode_block_table=decode_block_table_tr,
            full_block_table=full_block_table_tr,
            cum_kv_seq_len=cum_kv_seq_len_tr,
            max_metadata=max_metadata_tr,
            kv_seq_lengths=kv_seq_lengths_tr,
            device_decode_prefill=device_decode_prefill_tr,
            MAX_BATCH_SIZE_CONST=MAX_BATCH_SIZE,
            check_layout=True,
        )

        # Verify outputs match
        assert torch.equal(qo_indptr_pt[:BATCH_SIZE + 1], qo_indptr_tr[:BATCH_SIZE + 1]), "QO_INDPTR mismatch"
        assert torch.equal(last_page_len_pt[:BATCH_SIZE], last_page_len_tr[:BATCH_SIZE]), "LAST_PAGE_LEN mismatch"
        assert torch.equal(indptr_pt[:BATCH_SIZE + 1], indptr_tr[:BATCH_SIZE + 1]), "INDPTR mismatch"
        assert torch.equal(cum_kv_seq_len_pt[:BATCH_SIZE + 1], cum_kv_seq_len_tr[:BATCH_SIZE + 1]), "CUM_KV_SEQ_LEN mismatch"
        assert torch.equal(max_metadata_pt, max_metadata_tr), "MAX_METADATA mismatch"
        assert torch.equal(kv_seq_lengths_pt[:BATCH_SIZE], kv_seq_lengths_tr[:BATCH_SIZE]), "KV_SEQ_LENGTHS mismatch"
        assert torch.equal(kv_indices_pt[:indptr_pt[BATCH_SIZE]], kv_indices_tr[:indptr_tr[BATCH_SIZE]]), "KV_INDICES mismatch"
        assert torch.equal(pf_qo_indptr_pt, pf_qo_indptr_tr), "PF_QO_INDPTR mismatch"
        assert torch.equal(pf_last_page_len_pt, pf_last_page_len_tr), "PF_LAST_PAGE_LEN mismatch"
        assert torch.equal(pf_indptr_pt, pf_indptr_tr), "PF_INDPTR mismatch"
        assert torch.equal(pf_cum_kv_seq_len_pt, pf_cum_kv_seq_len_tr), "PF_CUM_KV_SEQ_LEN mismatch"
        assert torch.equal(dc_qo_indptr_pt, dc_qo_indptr_tr), "DC_QO_INDPTR mismatch"
        assert torch.equal(dc_last_page_len_pt, dc_last_page_len_tr), "DC_LAST_PAGE_LEN mismatch"
        assert torch.equal(dc_indptr_pt, dc_indptr_tr), "DC_INDPTR mismatch"
        assert torch.equal(prefill_block_table_pt[:PF_COUNT, :], prefill_block_table_tr[:PF_COUNT, :]), "PREFILL_BLOCK_TABLE mismatch"
        assert torch.equal(decode_block_table_pt[:DC_COUNT, :], decode_block_table_tr[:DC_COUNT, :]), "DECODE_BLOCK_TABLE mismatch"
        assert torch.equal(block_table, full_block_table_tr), "FULL_BLOCK_TABLE mismatch"
        assert torch.equal(device_decode_prefill_pt, device_decode_prefill_tr), "DEVICE_DECODE_PREFILL mismatch"


if __name__ == '__main__':
    import nvtx

    # --- Test Case ---
    print("Running test case to compare PyTorch and Triton implementations...")

    # Define test parameters
    BATCH_SIZE = 1024
    MAX_SEQ_LEN = 4096
    MAX_NUM_BLOCKS = 4096 // 16
    CHUNK_SIZE_TOKENS = 16
    TARGET_DC_SIZE = 768
    TARGET_PF_SIZE = 768
    PF_COUNT = 384
    DC_COUNT = BATCH_SIZE - PF_COUNT
    MAX_BATCH_SIZE = BATCH_SIZE + 256

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if DEVICE == 'cpu':
        print("CUDA device not found. Skipping Triton kernel execution.")
    else:
        # Generate random input tensors
        request_query_lengths_view = torch.ones(BATCH_SIZE, dtype=torch.int32, device=DEVICE)
        request_kv_length_offsets_view = torch.randint(
            0, MAX_SEQ_LEN // 2, (BATCH_SIZE,), dtype=torch.int32, device=DEVICE
        )
        block_table = torch.randint(
            0, MAX_NUM_BLOCKS, (BATCH_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE
        )

        qo_indptr_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        last_page_len_tr = torch.empty(MAX_BATCH_SIZE, dtype=torch.int32, device=DEVICE)
        indptr_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        kv_indices_tr = torch.empty(MAX_BATCH_SIZE * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE)
        cum_kv_seq_len_tr = torch.empty(MAX_BATCH_SIZE + 1, dtype=torch.int32, device=DEVICE)
        max_metadata_tr = torch.empty(2, dtype=torch.int32, device=DEVICE)
        kv_seq_lengths_tr = torch.empty(MAX_BATCH_SIZE, dtype=torch.int32, device=DEVICE)

        pf_qo_indptr_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)
        pf_last_page_len_tr = torch.empty(TARGET_PF_SIZE, dtype=torch.int32, device=DEVICE)
        pf_indptr_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)
        pf_cum_kv_seq_len_tr = torch.empty(TARGET_PF_SIZE + 1, dtype=torch.int32, device=DEVICE)

        dc_qo_indptr_tr = torch.empty(TARGET_DC_SIZE + 1, dtype=torch.int32, device=DEVICE)
        dc_last_page_len_tr = torch.empty(TARGET_DC_SIZE, dtype=torch.int32, device=DEVICE)
        dc_indptr_tr = torch.empty(TARGET_DC_SIZE + 1, dtype=torch.int32, device=DEVICE)

        prefill_block_table_tr = torch.empty((TARGET_PF_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        decode_block_table_tr = torch.empty((TARGET_DC_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        full_block_table_tr = torch.empty((BATCH_SIZE, MAX_NUM_BLOCKS), dtype=torch.int32, device=DEVICE)
        device_decode_prefill_tr = torch.empty(2, dtype=torch.int32, device=DEVICE)

        # Run the updated PyTorch reference
        (
            qo_indptr_pt,
            last_page_len_pt,
            kv_indices_pt,
            indptr_pt,
            cum_kv_seq_len_pt,
            max_metadata_pt,
            kv_seq_lengths_pt,
            pf_qo_indptr_pt,
            pf_last_page_len_pt,
            pf_indptr_pt,
            pf_cum_kv_seq_len_pt,
            dc_qo_indptr_pt,
            dc_last_page_len_pt,
            dc_indptr_pt,
            prefill_block_table_pt,
            decode_block_table_pt,
            device_decode_prefill_pt,
        ) = compute_layout_pytorch(
            request_query_lengths_view,
            request_kv_length_offsets_view,
            block_table,
            CHUNK_SIZE_TOKENS,
            dc_count=DC_COUNT,
            pf_count=PF_COUNT,
            dc_target_size=TARGET_DC_SIZE,
            pf_target_size=TARGET_PF_SIZE,
        )

        # Run the Triton kernel implementation
        (
            qo_indptr_tr,
            last_page_len_tr,
            kv_indices_tr,
            indptr_tr,
            cum_kv_seq_len_tr,
            max_metadata_tr,
            kv_seq_lengths_tr,
            pf_qo_indptr_tr,
            pf_last_page_len_tr,
            pf_indptr_tr,
            pf_cum_kv_seq_len_tr,
            dc_qo_indptr_tr,
            dc_last_page_len_tr,
            dc_indptr_tr,
            prefill_block_table_tr,
            decode_block_table_tr,
            full_block_table_tr,
            device_decode_prefill_tr,
        ) = compute_layout_triton(
            request_query_lengths_view,
            request_kv_length_offsets_view,
            block_table,
            CHUNK_SIZE_TOKENS,
            dc_count=DC_COUNT,
            pf_count=PF_COUNT,
            dc_target_size=TARGET_DC_SIZE,
            pf_target_size=TARGET_PF_SIZE,
            qo_indptr=qo_indptr_tr,
            last_page_len=last_page_len_tr,
            indptr=indptr_tr,
            kv_indices=kv_indices_tr,
            pf_qo_indptr=pf_qo_indptr_tr,
            pf_last_page_len=pf_last_page_len_tr,
            pf_indptr=pf_indptr_tr,
            pf_cum_kv_seq_len=pf_cum_kv_seq_len_tr,
            dc_qo_indptr=dc_qo_indptr_tr,
            dc_last_page_len=dc_last_page_len_tr,
            dc_indptr=dc_indptr_tr,
            prefill_block_table=prefill_block_table_tr,
            decode_block_table=decode_block_table_tr,
            full_block_table=full_block_table_tr,
            cum_kv_seq_len=cum_kv_seq_len_tr,
            max_metadata=max_metadata_tr,
            kv_seq_lengths=kv_seq_lengths_tr,
            device_decode_prefill=device_decode_prefill_tr,
            MAX_BATCH_SIZE_CONST=MAX_BATCH_SIZE,
            check_layout=True,
        )

        def _bench_call():
            compute_layout_triton(
                request_query_lengths_view,
                request_kv_length_offsets_view,
                block_table,
                CHUNK_SIZE_TOKENS,
                dc_count=DC_COUNT,
                pf_count=PF_COUNT,
                dc_target_size=TARGET_DC_SIZE,
                pf_target_size=TARGET_PF_SIZE,
                qo_indptr=qo_indptr_tr,
                last_page_len=last_page_len_tr,
                indptr=indptr_tr,
                kv_indices=kv_indices_tr,
                pf_qo_indptr=pf_qo_indptr_tr,
                pf_last_page_len=pf_last_page_len_tr,
                pf_indptr=pf_indptr_tr,
                dc_qo_indptr=dc_qo_indptr_tr,
                dc_last_page_len=dc_last_page_len_tr,
                dc_indptr=dc_indptr_tr,
                prefill_block_table=prefill_block_table_tr,
                decode_block_table=decode_block_table_tr,
                full_block_table=full_block_table_tr,
                cum_kv_seq_len=cum_kv_seq_len_tr,
                max_metadata=max_metadata_tr,
                kv_seq_lengths=kv_seq_lengths_tr,
                device_decode_prefill=device_decode_prefill_tr,
                MAX_BATCH_SIZE_CONST=MAX_BATCH_SIZE,
            )

        ms = triton.testing.do_bench(_bench_call, warmup=1000, rep=1000)
        print(f"Time: {ms:.5f} ms")

        # nsys capture
        with nvtx.annotate("compute_layout_triton"):
            for i in range(1000):
                _bench_call()

        # --- Verification ---
        print("Verification:")
        print(f"QO_INDPTR match: {torch.equal(qo_indptr_pt[:BATCH_SIZE + 1], qo_indptr_tr[:BATCH_SIZE + 1])}")
        print(f"LAST_PAGE_LEN match: {torch.equal(last_page_len_pt[:BATCH_SIZE], last_page_len_tr[:BATCH_SIZE])}")
        print(f"INDPTR match: {torch.equal(indptr_pt[:BATCH_SIZE + 1], indptr_tr[:BATCH_SIZE + 1])}")
        print(f"CUM_KV_SEQ_LEN match: {torch.equal(cum_kv_seq_len_pt[:BATCH_SIZE + 1], cum_kv_seq_len_tr[:BATCH_SIZE + 1])}")
        print(f"MAX_METADATA match: {torch.equal(max_metadata_pt, max_metadata_tr)}")
        print(f"KV_SEQ_LENGTHS match: {torch.equal(kv_seq_lengths_pt[:BATCH_SIZE], kv_seq_lengths_tr[:BATCH_SIZE])}")
        print(f"KV_INDICES match: {torch.equal(kv_indices_pt[:indptr_pt[BATCH_SIZE]], kv_indices_tr[:indptr_tr[BATCH_SIZE]])}")
        print(f"PF_QO_INDPTR match: {torch.equal(pf_qo_indptr_pt, pf_qo_indptr_tr)}")
        print(f"PF_LAST_PAGE_LEN match: {torch.equal(pf_last_page_len_pt, pf_last_page_len_tr)}")
        print(f"PF_INDPTR match: {torch.equal(pf_indptr_pt, pf_indptr_tr)}")
        print(f"PF_CUM_KV_SEQ_LEN match: {torch.equal(pf_cum_kv_seq_len_pt, pf_cum_kv_seq_len_tr)}")
        print(f"DC_QO_INDPTR match: {torch.equal(dc_qo_indptr_pt, dc_qo_indptr_tr)}")
        print(f"DC_LAST_PAGE_LEN match: {torch.equal(dc_last_page_len_pt, dc_last_page_len_tr)}")
        print(f"DC_INDPTR match: {torch.equal(dc_indptr_pt, dc_indptr_tr)}")
        print(f"PREFILL_BLOCK_TABLE match: {torch.equal(prefill_block_table_pt[:PF_COUNT, :], prefill_block_table_tr[:PF_COUNT, :])}")
        print(f"DECODE_BLOCK_TABLE match: {torch.equal(decode_block_table_pt[:DC_COUNT, :], decode_block_table_tr[:DC_COUNT, :])}")
        print(f"FULL_BLOCK_TABLE match: {torch.equal(block_table, full_block_table_tr)}")
        print(f"DEVICE_DECODE_PREFILL match: {torch.equal(device_decode_prefill_pt, device_decode_prefill_tr)}")
        print(f"DEVICE_DECODE_PREFILL values - PyTorch: {device_decode_prefill_pt}, Triton: {device_decode_prefill_tr}")

        # Assert that all outputs are identical
        assert torch.equal(qo_indptr_pt[:BATCH_SIZE + 1], qo_indptr_tr[:BATCH_SIZE + 1])
        assert torch.equal(last_page_len_pt[:BATCH_SIZE], last_page_len_tr[:BATCH_SIZE])
        assert torch.equal(indptr_pt[:BATCH_SIZE + 1], indptr_tr[:BATCH_SIZE + 1])
        assert torch.equal(cum_kv_seq_len_pt[:BATCH_SIZE + 1], cum_kv_seq_len_tr[:BATCH_SIZE + 1])
        assert torch.equal(max_metadata_pt, max_metadata_tr)
        assert torch.equal(kv_seq_lengths_pt[:BATCH_SIZE], kv_seq_lengths_tr[:BATCH_SIZE])
        assert torch.equal(kv_indices_pt[:indptr_pt[BATCH_SIZE]], kv_indices_tr[:indptr_tr[BATCH_SIZE]])
        assert torch.equal(pf_qo_indptr_pt, pf_qo_indptr_tr)
        assert torch.equal(pf_last_page_len_pt, pf_last_page_len_tr)
        assert torch.equal(pf_indptr_pt, pf_indptr_tr)
        assert torch.equal(pf_cum_kv_seq_len_pt, pf_cum_kv_seq_len_tr)
        assert torch.equal(dc_qo_indptr_pt, dc_qo_indptr_tr)
        assert torch.equal(dc_last_page_len_pt, dc_last_page_len_tr)
        assert torch.equal(dc_indptr_pt, dc_indptr_tr)
        assert torch.equal(prefill_block_table_pt[:PF_COUNT, :], prefill_block_table_tr[:PF_COUNT, :])
        assert torch.equal(decode_block_table_pt[:DC_COUNT, :], decode_block_table_tr[:DC_COUNT, :])
        assert torch.equal(block_table, full_block_table_tr)
        assert torch.equal(device_decode_prefill_pt, device_decode_prefill_tr)

        print("\n✅ All outputs match successfully!")
