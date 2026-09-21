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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt unpadded B=1 SBHD self-attention to the package tensor interface.

    The package applies the auxiliary coefficient and denominator to both loss
    and gradients. Its result must not be normalized again by the caller.
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
    if key.shape != (sequence_length, 1, 1, 256) or value.shape != key.shape:
        raise ValueError("CuTe sparse attention requires matching self-attention K/V [S,1,1,256].")
    if q_indexer.shape != (sequence_length, 1, 1, 128) or k_indexer.shape != q_indexer.shape:
        raise ValueError("CuTe sparse attention requires indexer Q/K [S,1,1,128].")
    if not math.isclose(softmax_scale, 256**-0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("CuTe sparse attention requires softmax scale 1/sqrt(256).")

    # These arrays describe one unpadded causal sequence.
    layout = RowMetadata(
        torch.zeros(sequence_length, dtype=torch.int64, device=query.device),
        torch.arange(1, sequence_length + 1, dtype=torch.int64, device=query.device),
        torch.ones(sequence_length, dtype=torch.bool, device=query.device),
        torch.as_tensor(loss_denominator, dtype=torch.float32, device=query.device).reshape(1),
        triangular_scores=True,
        mask_invalid_rows=False,
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
