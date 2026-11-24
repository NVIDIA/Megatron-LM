# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Host cost model for the two ratio-4 CSA indexer layouts."""

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import MLATransformerConfig


def _candidate_prefix(rows: int) -> int:
    """Count visible compressed blocks over the first `rows` queries of one document."""
    blocks = rows // 4
    # Block j (one-based) is visible to rows 4*j through rows, inclusive.
    return blocks * (rows + 1) - 2 * blocks * (blocks + 1)


def prefer_zigzag(
    lengths: list[int], cp_size: int, local_rows: int, config: "MLATransformerConfig"
) -> bool:
    """Compare the critical rank for the two eligible fused paths without enumerating tokens.

    The last length is the physical capacity tail. It travels with other rows but has no
    compressed keys. All other lengths are the padded segments actually scored by the kernel.
    """
    contiguous = [0] * cp_size
    retained = [0] * cp_size
    chunk_counts: Counter[int] = Counter()
    start = 0
    for doc, length in enumerate(lengths):
        if not length:
            continue
        end = start + length
        chunk = length // (2 * cp_size)
        has_keys = doc < len(lengths) - 1
        if has_keys:
            chunk_counts[chunk] += 1
        # Across all documents there are at most (documents + cp_size - 1) intersections.
        for rank in range(start // local_rows, (end - 1) // local_rows + 1):
            lo = max(0, rank * local_rows - start)
            hi = min(length, (rank + 1) * local_rows - start)
            if has_keys:
                contiguous[rank] += _candidate_prefix(hi) - _candidate_prefix(lo)
            for piece in (rank, 2 * cp_size - 1 - rank):
                left = piece * chunk
                retained[rank] += max(0, min(hi, left + chunk) - max(lo, left))
        start = end

    peak_contiguous = max(contiguous)
    if peak_contiguous * cp_size == sum(contiguous):
        return False

    zigzag = [0] * cp_size
    for chunk, count in chunk_counts.items():
        for rank in range(cp_size):
            for piece in (rank, 2 * cp_size - 1 - rank):
                lo = piece * chunk
                zigzag[rank] += count * (_candidate_prefix(lo + chunk) - _candidate_prefix(lo))

    heads = config.dsa_indexer_n_heads
    dim = config.dsa_indexer_head_dim
    topk = config.dsa_indexer_topk
    if heads is None or dim is None or topk is None:
        raise ValueError(
            "CSA auto layout requires the indexer head count, head dimension and top-k width."
        )
    flops_per_candidate = 2 * heads * dim
    saved_us = (
        (peak_contiguous - max(zigzag))
        * flops_per_candidate
        / (config.dsa_cp_indexer_auto_compute_tflops * 1e6)
    )

    # Dispatch the low-rank query and head weights. The inverse sends int32 IDs and,
    # for sparse-loss training, their paired fp32 predictions. Price that payload even
    # on a no-grad recomputation so the decision stays fixed throughout the microbatch.
    query_width = config.q_lora_rank or config.hidden_size
    return_prediction = (
        config.dsa_indexer_use_sparse_loss and (config.dsa_indexer_loss_coeff or 0) > 0
    )
    row_bytes = (query_width + heads) * config.params_dtype.itemsize
    row_bytes += topk * (8 if return_prediction else 4)
    # Equal-sized source/destination shards give each rank identical send/receive row
    # counts. Local permutations still process the full shard, including retained rows.
    remote_rows = local_rows - min(retained)
    route_us = remote_rows * row_bytes / (config.dsa_cp_indexer_auto_exchange_gbps * 1e3)
    copy_us = local_rows * row_bytes / (config.dsa_cp_indexer_auto_copy_gbps * 1e3)
    return saved_us > route_us + copy_us + config.dsa_cp_indexer_auto_overhead_us
