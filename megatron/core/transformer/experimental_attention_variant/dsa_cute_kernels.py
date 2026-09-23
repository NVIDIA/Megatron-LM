# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor adapter for the optional simplified sparse attention CuTe package."""

import math

import torch


def run_cute_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    loss_coeff: float,
    loss_denominator: int | torch.Tensor,
    row_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    query_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt B=1 local queries and globally ordered keys to the package interface.

    The package applies the auxiliary coefficient and denominator to both loss
    and gradients. Its result must not be normalized again by the caller.
    Explicit row bounds describe packed documents or context-parallel queries;
    their offsets address the gathered, globally ordered key tensor.
    Unsupported inputs raise; explicit CuTe selection never falls back.
    """
    try:
        from simplified_sparse_attention import RowMetadata, sparse_attention
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The CuTe backend requires the simplified-sparse-attention package (>=0.1.0.dev2) "
            "and its matching CuTe runtime."
        ) from exc

    if query.ndim != 4 or query.shape[1] != 1:
        raise ValueError("CuTe sparse attention requires ordinary SBHD with batch size 1.")
    sequence_length = query.shape[0]
    key_length = key.shape[0]
    if key.shape != (key_length, 1, 1, 256) or value.shape != key.shape:
        raise ValueError("CuTe sparse attention requires matching K/V [Sk,1,1,256].")
    if q_indexer.shape != (sequence_length, 1, 1, 128) or k_indexer.shape != (
        key_length,
        1,
        1,
        128,
    ):
        raise ValueError("CuTe sparse attention requires indexer Q [Sq,1,1,128], K [Sk,1,1,128].")
    if not math.isclose(softmax_scale, 256**-0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("CuTe sparse attention requires softmax scale 1/sqrt(256).")

    if row_bounds is None:
        if sequence_length != key_length:
            raise ValueError("Rectangular CuTe attention requires explicit causal row bounds.")
        starts = torch.zeros(sequence_length, dtype=torch.int64, device=query.device)
        ends = torch.arange(1, sequence_length + 1, dtype=torch.int64, device=query.device)
    else:
        starts, ends = row_bounds
        if starts.shape != (sequence_length,) or ends.shape != starts.shape:
            raise ValueError("CuTe row bounds must have one entry per local query.")
    if query_valid_rows is None:
        valid_rows = torch.ones(sequence_length, dtype=torch.bool, device=query.device)
    else:
        if query_valid_rows.shape != (sequence_length,):
            raise ValueError("CuTe query validity must have one entry per local query.")
        valid_rows = query_valid_rows
    layout = RowMetadata(
        starts,
        ends,
        valid_rows,
        torch.as_tensor(loss_denominator, dtype=torch.float32, device=query.device).reshape(1),
        triangular_scores=sequence_length == key_length,
        mask_invalid_rows=query_valid_rows is not None,
    )
    output, loss = sparse_attention(
        query[:, 0],
        key[:, 0],
        value[:, 0],
        q_indexer[:, 0, 0].contiguous(),
        k_indexer[:, 0, 0].contiguous(),
        layout,
        topk,
        loss_coeff=loss_coeff,
    )
    return output.reshape(sequence_length, 1, -1), loss
