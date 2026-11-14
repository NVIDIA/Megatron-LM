# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN frontend kernels for simplified sparse attention."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from megatron.core.ops._backends import require

_HEAD_DIM = 256
_INDEXER_DIM = 128


@dataclass(frozen=True)
class CudnnSparseAttentionKernels:
    """Kernel entry points bound once when the core attention is constructed.

    ``dense_attention`` (FlashAttention-4) and ``dense_indexer_kl`` (cuDNN frontend) are bound
    only for indexer-only training, where a frozen dense teacher replaces sparse attention.
    """

    sparse_attention: Callable
    row_metadata: type
    dense_attention: Optional[Callable] = None
    dense_indexer_kl: Optional[Callable] = None


def bind_cudnn_sparse_attention(*, dense_teacher: bool = False) -> CudnnSparseAttentionKernels:
    """Bind cuDNN frontend simplified sparse attention, failing early when it is absent.

    The kernels ship in ``nvidia-cudnn-frontend`` 1.33 and newer and require CuTe DSL 4.8.
    ``dense_teacher`` additionally binds the dense indexer objective and FlashAttention-4.
    """
    symbols = ["simplified_sparse_attention", "SimplifiedSparseAttentionMetadata"]
    if dense_teacher:
        symbols.append("dense_indexer_kl")
    ops = require(
        "cudnn.ops", *symbols, dist="nvidia-cudnn-frontend", needed_by="Simplified sparse attention"
    )
    dense_attention = None
    if dense_teacher:
        from megatron.core.transformer import attention

        if not attention.HAVE_FA4:
            raise ImportError(
                "Simplified sparse attention indexer-only training computes its dense teacher "
                f"with flash-attn-4 >= {attention._MIN_FA4_VERSION}, which is not installed."
            )
        dense_attention = attention.flash_attn4_varlen_func
    return CudnnSparseAttentionKernels(
        sparse_attention=ops.simplified_sparse_attention,
        row_metadata=ops.SimplifiedSparseAttentionMetadata,
        dense_attention=dense_attention,
        dense_indexer_kl=ops.dense_indexer_kl if dense_teacher else None,
    )


def _check_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
) -> int:
    if query.ndim != 4 or query.shape[1] != 1:
        raise ValueError("Simplified sparse attention requires SBHD input with batch size 1.")
    sequence_length = query.shape[0]
    if key.shape != (sequence_length, 1, 1, _HEAD_DIM) or value.shape != key.shape:
        raise ValueError("Simplified sparse attention requires self-attention K/V [S,1,1,256].")
    if (
        q_indexer.shape != (sequence_length, 1, 1, _INDEXER_DIM)
        or k_indexer.shape != q_indexer.shape
    ):
        raise ValueError("Simplified sparse attention requires indexer Q/K [S,1,1,128].")
    return sequence_length


def _causal_rows(
    sequence_length: int, device: torch.device, calculate_per_token_loss: bool
) -> dict[str, torch.Tensor]:
    """Row bounds of one unpadded causal sequence and the loss denominator."""
    return {
        "key_starts": torch.zeros(sequence_length, dtype=torch.int32, device=device),
        "key_ends": torch.arange(1, sequence_length + 1, dtype=torch.int32, device=device),
        "query_valid": torch.ones(sequence_length, dtype=torch.bool, device=device),
        "loss_denominator": torch.full(
            (1,),
            1.0 if calculate_per_token_loss else float(sequence_length),
            dtype=torch.float32,
            device=device,
        ),
    }


def run_sparse_attention(
    kernels: CudnnSparseAttentionKernels,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    *,
    topk: int,
    loss_coeff: float,
    calculate_per_token_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one unpadded causal sequence in SBHD layout with batch size one.

    The indexer selects ``topk`` keys per query and attention runs over that selection. The
    returned indexer loss is the selected-support KL from the mean attention distribution to the
    indexer distribution. cuDNN applies the coefficient and sum/mean normalization once, to both
    the loss and its gradients.
    """
    sequence_length = _check_shapes(query, key, value, q_indexer, k_indexer)
    metadata = kernels.row_metadata(
        **_causal_rows(sequence_length, query.device, calculate_per_token_loss),
        triangular_scores=True,
        mask_invalid_rows=False,
    )
    output, loss = kernels.sparse_attention(
        query[:, 0],
        key[:, 0],
        value[:, 0],
        q_indexer[:, 0, 0].contiguous(),
        k_indexer[:, 0, 0].contiguous(),
        metadata,
        topk,
        loss_coeff=loss_coeff,
    )
    return output.reshape(sequence_length, 1, -1), loss


def run_dense_teacher(
    kernels: CudnnSparseAttentionKernels,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    *,
    loss_coeff: float,
    calculate_per_token_loss: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Indexer-only training: frozen dense causal attention and the full-support indexer KL.

    FlashAttention-4 computes the dense attention output and its natural-log LSE without
    materializing probabilities; the teacher is detached because the backbone is frozen. With a
    positive ``loss_coeff``, cuDNN evaluates the KL from the mean dense attention distribution to
    the indexer distribution over all causal keys, normalized and scaled like the sparse loss.
    """
    sequence_length = _check_shapes(query, key, value, q_indexer, k_indexer)
    with torch.no_grad():
        query = query[:, 0].contiguous()
        key = key[:, 0].contiguous()
        value = value[:, 0].contiguous()
        cu_seqlens = torch.tensor((0, sequence_length), dtype=torch.int32, device=query.device)
        output, lse = kernels.dense_attention(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=sequence_length,
            max_seqlen_k=sequence_length,
            softmax_scale=_HEAD_DIM**-0.5,
            causal=True,
            return_lse=True,
        )
        # FlashAttention-4 returns the varlen LSE heads-first, [Hq, S]; cuDNN takes [S, Hq].
        lse = lse.transpose(0, 1).contiguous()
    loss = None
    if loss_coeff > 0:
        loss = kernels.dense_indexer_kl(
            query,
            key,
            q_indexer[:, 0, 0].contiguous(),
            k_indexer[:, 0, 0].contiguous(),
            **_causal_rows(sequence_length, query.device, calculate_per_token_loss),
            teacher_softmax_scale=_HEAD_DIM**-0.5,
            student_softmax_scale=_INDEXER_DIM**-0.5,
            loss_coeff=loss_coeff,
            teacher_lse=lse,
            use_kf=True,
        )
    return output.reshape(sequence_length, 1, -1), loss
