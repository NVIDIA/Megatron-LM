# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compact CSA indexer adapter, selected once during attention construction.

Q/K enter as BF16 BSHD or packed THD tensors. The MXFP8 path quantizes only
indexer scoring; the attention and indexer backward retain BF16 inputs.
Persistent workspaces contain scratch, never autograd-saved outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from cudnn import DSA as _DSA
from torch import Tensor

from megatron.core.ops.attention.csa.indexer_quantization import (
    IndexerMXFP8QuantizationBuffers,
    create_indexer_mxfp8_quantization_buffers,
    indexer_mxfp8_scale_shape,
    indexer_mxfp8_thd_scale_capacity,
    indexer_mxfp8_thd_scale_shape,
    make_indexer_mxfp8_scale_cu_seqlens,
    quantize_indexer_mxfp8,
    refresh_indexer_mxfp8_scale_cu_seqlens,
)


@dataclass(frozen=True)
class _BSHDCompactIndexerGeometry:
    """Static BSHD geometry for a caller-owned compact workspace."""

    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    topk: int
    ratio: int
    return_softmax: bool
    precision: str = "bf16"


@dataclass(frozen=True)
class _THDCompactIndexerGeometry:
    """Static THD geometry for a caller-owned compact workspace."""

    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    batch_size: int
    has_q_causal_offsets: bool
    topk: int
    ratio: int
    max_seqlen_q: int
    max_seqlen_k: int
    return_softmax: bool
    precision: str = "bf16"


@dataclass
class MXFP8IndexerWorkspace:
    """Caller-owned TE quantization destinations and packed Indexer scales."""

    q_buffers: IndexerMXFP8QuantizationBuffers
    k_buffers: IndexerMXFP8QuantizationBuffers
    q_scale: Tensor
    k_scale: Tensor
    cu_seqlens_q_scale_padded: Tensor | None = None
    cu_seqlens_k_scale_padded: Tensor | None = None


@dataclass
class BSHDCompactIndexerWorkspace:
    """Persistent compact BSHD capture metadata and MXFP8 storage."""

    cand_buffer_numel: int
    geometry: _BSHDCompactIndexerGeometry
    mxfp8: MXFP8IndexerWorkspace | None = None

    def matches_shape(
        self,
        *,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        device: torch.device,
        topk: int,
        ratio: int,
        return_softmax: bool,
        precision: str = "bf16",
    ) -> bool:
        """Return whether static BSHD metadata matches this workspace."""
        if len(q_shape) != 4 or len(k_shape) != 3:
            return False
        batch_size, seqlen_q, num_heads, head_dim = q_shape
        k_batch, seqlen_k, k_head_dim = k_shape
        mxfp8_valid = self.mxfp8 is None
        if precision == "mxfp8":
            mxfp8 = self.mxfp8
            q_scale_shape = indexer_mxfp8_scale_shape(batch_size, seqlen_q, num_heads, head_dim)
            k_scale_shape = indexer_mxfp8_scale_shape(batch_size, seqlen_k, 1, k_head_dim)
            mxfp8_valid = mxfp8 is not None and all(
                (
                    tuple(mxfp8.q_buffers.input_shape) == q_shape,
                    mxfp8.q_buffers.data.device == device,
                    tuple(mxfp8.k_buffers.input_shape) == k_shape,
                    mxfp8.k_buffers.data.device == device,
                    mxfp8.q_scale.device == device,
                    mxfp8.q_scale.dtype == torch.float8_e8m0fnu,
                    tuple(mxfp8.q_scale.shape) == q_scale_shape,
                    mxfp8.q_scale.is_contiguous(),
                    mxfp8.k_scale.device == device,
                    mxfp8.k_scale.dtype == torch.float8_e8m0fnu,
                    tuple(mxfp8.k_scale.shape) == k_scale_shape,
                    mxfp8.k_scale.is_contiguous(),
                    mxfp8.cu_seqlens_q_scale_padded is None,
                    mxfp8.cu_seqlens_k_scale_padded is None,
                )
            )
        return all(
            (
                precision in ("bf16", "mxfp8"),
                self.geometry.precision == precision,
                self.geometry.q_shape == q_shape,
                self.geometry.k_shape == k_shape,
                self.geometry.topk == topk,
                self.geometry.ratio == ratio,
                self.geometry.return_softmax == return_softmax,
                batch_size == k_batch,
                head_dim == k_head_dim,
                isinstance(self.cand_buffer_numel, int) and self.cand_buffer_numel >= 0,
                mxfp8_valid,
            )
        )

    def matches(
        self,
        *,
        q: Tensor,
        k: Tensor,
        topk: int,
        ratio: int,
        return_softmax: bool,
        precision: str = "bf16",
    ) -> bool:
        """Return whether this workspace can serve the current BSHD call."""
        inputs_valid = all(
            (
                q.dtype == torch.bfloat16,
                k.dtype == torch.bfloat16,
                q.device == k.device,
                q.is_contiguous(),
                k.is_contiguous(),
            )
        )
        if not inputs_valid or not self.matches_shape(
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            device=q.device,
            topk=topk,
            ratio=ratio,
            return_softmax=return_softmax,
            precision=precision,
        ):
            return False
        if precision == "mxfp8":
            assert self.mxfp8 is not None
            return self.mxfp8.q_buffers.matches(q) and self.mxfp8.k_buffers.matches(k)
        return True

    def validate(
        self,
        *,
        q: Tensor,
        k: Tensor,
        topk: int,
        ratio: int,
        return_softmax: bool,
        precision: str = "bf16",
    ) -> None:
        """Validate caller-owned BSHD state before dispatch."""
        if not self.matches(
            q=q, k=k, topk=topk, ratio=ratio, return_softmax=return_softmax, precision=precision
        ):
            raise ValueError(
                "BSHD compact_workspace does not match the current buffers or static geometry; "
                "prepare a workspace for this exact call during eager warmup."
            )


@dataclass
class THDCompactIndexerWorkspace:
    """Persistent compact THD offsets, capture metadata, and MXFP8 storage."""

    cand_batch_offsets: Tensor
    cand_buffer_numel: int
    geometry: _THDCompactIndexerGeometry
    mxfp8: MXFP8IndexerWorkspace | None = None

    def matches(
        self,
        *,
        q: Tensor,
        k: Tensor,
        topk: int,
        ratio: int,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        q_causal_offsets: Tensor | None,
        return_softmax: bool,
        precision: str = "bf16",
    ) -> bool:
        """Return whether this workspace can serve the current static indexer geometry."""
        geometry = self.geometry
        batch_size = geometry.batch_size
        inputs_valid = all(
            (
                q.dtype == torch.bfloat16,
                k.dtype == torch.bfloat16,
                tuple(q.shape) == geometry.q_shape,
                tuple(k.shape) == geometry.k_shape,
                q.device == k.device,
                q.is_contiguous(),
                k.is_contiguous(),
            )
        )
        cu_seqlens_valid = all(
            tensor.device == q.device
            and tensor.dtype == torch.int32
            and tensor.ndim == 1
            and tensor.numel() == batch_size + 1
            and tensor.is_contiguous()
            for tensor in (cu_seqlens_q, cu_seqlens_k)
        )
        q_causal_offsets_valid = (
            q_causal_offsets is None
            if not geometry.has_q_causal_offsets
            else (
                q_causal_offsets is not None
                and q_causal_offsets.device == q.device
                and q_causal_offsets.dtype == torch.int32
                and q_causal_offsets.ndim == 1
                and q_causal_offsets.numel() == batch_size
                and q_causal_offsets.is_contiguous()
            )
        )
        mxfp8_valid = self.mxfp8 is None
        if precision == "mxfp8":
            mxfp8 = self.mxfp8
            q_scale_shape = indexer_mxfp8_thd_scale_shape(
                indexer_mxfp8_thd_scale_capacity(q.shape[0], batch_size, q.shape[1]),
                q.shape[1],
                q.shape[2],
            )
            k_scale_shape = indexer_mxfp8_thd_scale_shape(
                indexer_mxfp8_thd_scale_capacity(k.shape[0], batch_size, 1), 1, k.shape[1]
            )
            scale_prefixes_valid = mxfp8 is not None and all(
                prefix is not None
                and prefix.device == q.device
                and prefix.dtype == torch.int32
                and prefix.ndim == 1
                and prefix.numel() == batch_size + 1
                and prefix.is_contiguous()
                for prefix in (mxfp8.cu_seqlens_q_scale_padded, mxfp8.cu_seqlens_k_scale_padded)
            )
            mxfp8_valid = (
                mxfp8 is not None
                and scale_prefixes_valid
                and all(
                    (
                        mxfp8.q_scale.device == q.device,
                        mxfp8.q_scale.dtype == torch.float8_e8m0fnu,
                        tuple(mxfp8.q_scale.shape) == q_scale_shape,
                        mxfp8.q_scale.is_contiguous(),
                        mxfp8.k_scale.device == q.device,
                        mxfp8.k_scale.dtype == torch.float8_e8m0fnu,
                        tuple(mxfp8.k_scale.shape) == k_scale_shape,
                        mxfp8.k_scale.is_contiguous(),
                        mxfp8.q_buffers.matches(q),
                        mxfp8.k_buffers.matches(k),
                    )
                )
            )
        static_valid = (
            precision in ("bf16", "mxfp8")
            and inputs_valid
            and geometry.precision == precision
            and geometry.topk == topk
            and geometry.ratio == ratio
            and geometry.max_seqlen_q == max_seqlen_q
            and geometry.max_seqlen_k == max_seqlen_k
            and geometry.return_softmax == return_softmax
            and cu_seqlens_valid
            and q_causal_offsets_valid
            and self.cand_batch_offsets.device == q.device
            and self.cand_batch_offsets.dtype == torch.int64
            and self.cand_batch_offsets.ndim == 1
            and self.cand_batch_offsets.numel() == batch_size + 1
            and self.cand_batch_offsets.is_contiguous()
            and isinstance(self.cand_buffer_numel, int)
            and self.cand_buffer_numel >= 0
            and mxfp8_valid
        )
        return static_valid

    def validate(
        self,
        *,
        q: Tensor,
        k: Tensor,
        topk: int,
        ratio: int,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        q_causal_offsets: Tensor | None,
        return_softmax: bool,
        precision: str = "bf16",
    ) -> None:
        """Validate static workspace metadata without synchronizing."""
        if not self.matches(
            q=q,
            k=k,
            topk=topk,
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            return_softmax=return_softmax,
            precision=precision,
        ):
            raise ValueError(
                "THD compact_workspace does not match the current buffers or static geometry; "
                "prepare a compatible workspace during eager warmup."
            )


def _allocate_compact_candidate_buffer(numel: int, device: torch.device) -> Tensor:
    """Allocate candidate scratch in the active eager or CUDA-graph memory pool.

    The compact wrapper consumes this stage-1 output before returning, so the
    tensor must not be retained in the persistent workspace. During capture,
    its short lifetime lets serialized graphs reuse storage from their shared
    graph pool.
    """
    return torch.empty(numel, dtype=torch.float32, device=device)


def _allocate_compact_output_buffers(
    shape: tuple[int, ...], device: torch.device, return_softmax: bool
) -> dict[str, Tensor]:
    """Allocate compact outputs in the active eager or CUDA-graph memory pool.

    These tensors follow the captured forward's dataflow instead of being
    retained by the module. Keeping their ownership dispatch-local lets
    serialized CUDA graphs reuse their storage from a shared graph pool.
    """
    outputs = {
        "out_indices": torch.empty(shape, dtype=torch.int32, device=device),
        "out_logits": torch.empty(shape, dtype=torch.float32, device=device),
    }
    if return_softmax:
        outputs["softmax_out"] = torch.empty(shape, dtype=torch.float32, device=device)
    return outputs


def prepare_bshd_compact_indexer_workspace(
    q: Tensor,
    k: Tensor,
    *,
    topk: int,
    ratio: int,
    return_softmax: bool = False,
    precision: str = "bf16",
) -> BSHDCompactIndexerWorkspace | None:
    """Prepare persistent compact BSHD state before CUDA graph capture.

    ``q`` and ``k`` use the BSHD/BSD layouts consumed by
    :func:`_indexer_topk_core`. The workspace forces the upstream compact
    wrapper's single-launch mode so its candidate sizing is identical for
    BF16 and MXFP8. Candidate scratch and compact outputs are allocated by each
    dispatch, allowing a CUDA-graph capture pool to reuse them across serialized
    graphs.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("BSHD compact indexer workspace must be prepared before CUDA capture")
    if q.ndim != 4 or k.ndim != 3:
        raise ValueError(
            f"BSHD workspace expects q (B,S,H,D) and k (B,S,D), got {q.shape}, {k.shape}"
        )
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("BSHD workspace inputs must be contiguous")

    size_helper = _DSA.compress_topk_cand_buffer_size

    batch_size, seqlen_q, num_heads, head_dim = q.shape
    k_batch, seqlen_k, k_head_dim = k.shape
    if batch_size != k_batch or head_dim != k_head_dim:
        raise ValueError("BSHD workspace q/k batch size and head dimension must match")
    cand_floats = size_helper(batch_size, seqlen_q, seqlen_k, ratio, microbatch_rows=0)
    device = q.device
    mxfp8_workspace = None
    if precision == "mxfp8":
        mxfp8_workspace = MXFP8IndexerWorkspace(
            q_buffers=create_indexer_mxfp8_quantization_buffers(q),
            k_buffers=create_indexer_mxfp8_quantization_buffers(k),
            q_scale=torch.empty(
                indexer_mxfp8_scale_shape(batch_size, seqlen_q, num_heads, head_dim),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            k_scale=torch.empty(
                indexer_mxfp8_scale_shape(batch_size, seqlen_k, 1, k_head_dim),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
        )
    return BSHDCompactIndexerWorkspace(
        cand_buffer_numel=int(cand_floats),
        geometry=_BSHDCompactIndexerGeometry(
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            topk=topk,
            ratio=ratio,
            return_softmax=return_softmax,
            precision=precision,
        ),
        mxfp8=mxfp8_workspace,
    )


@torch.compile
def _refresh_thd_compact_cand_batch_offsets(
    destination: Tensor, cu_seqlens_q: Tensor, ratio: int, q_causal_offsets: Tensor | None
) -> None:
    """Refresh caller-owned THD candidate offsets from graph inputs."""
    cu_q = cu_seqlens_q.to(torch.int64)
    q_lengths = cu_q[1:] - cu_q[:-1]
    if q_causal_offsets is None:
        q_starts = torch.zeros_like(q_lengths)
    else:
        q_starts = q_causal_offsets.to(torch.int64)

    def prefix_candidate_count(length: Tensor) -> Tensor:
        quotient = torch.div(length, ratio, rounding_mode="floor")
        remainder = length - quotient * ratio
        return ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)

    per_sequence = prefix_candidate_count(q_starts + q_lengths) - prefix_candidate_count(q_starts)
    destination.zero_()
    torch.cumsum(per_sequence, dim=0, out=destination[1:])


def prepare_thd_compact_indexer_workspace(
    q: Tensor,
    k: Tensor,
    *,
    topk: int,
    ratio: int,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    q_causal_offsets: Tensor | None = None,
    return_softmax: bool = False,
    precision: str = "bf16",
) -> THDCompactIndexerWorkspace | None:
    """Prepare persistent compact THD state before CUDA graph capture.

    The sizing helper performs a GPU-to-host sync and therefore this function
    must never be called from inside graph capture.
    Eager warmup must prepare this workspace before any THD compact capture.
    Candidate scratch and compact outputs are allocated by each dispatch so a
    CUDA-graph capture pool can reuse that temporary storage across serialized
    graphs.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("THD compact indexer workspace must be prepared before CUDA capture")

    size_helper = _DSA.compress_topk_cand_buffer_size_thd

    cand_batch_offsets, cand_floats = size_helper(
        cu_seqlens_q, cu_seqlens_k, ratio, q_causal_offsets=q_causal_offsets
    )
    device = q.device
    # Sequence boundaries may change between graph capture and replay while
    # tensor shapes and maxima remain static. Reserve the geometry-wide upper
    # bound so refreshed candidate offsets always address valid storage.
    cand_floats = max(cand_floats, q.shape[0] * int(max_seqlen_k))
    mxfp8_workspace = None
    if precision == "mxfp8":
        q_buffers = create_indexer_mxfp8_quantization_buffers(q)
        k_buffers = create_indexer_mxfp8_quantization_buffers(k)
        cu_seqlens_q_scale_padded = make_indexer_mxfp8_scale_cu_seqlens(cu_seqlens_q, q.shape[1])
        cu_seqlens_k_scale_padded = make_indexer_mxfp8_scale_cu_seqlens(cu_seqlens_k, 1)
        batch_size = cu_seqlens_q.numel() - 1
        q_scale_capacity = indexer_mxfp8_thd_scale_capacity(q.shape[0], batch_size, q.shape[1])
        k_scale_capacity = indexer_mxfp8_thd_scale_capacity(k.shape[0], batch_size, 1)
        mxfp8_workspace = MXFP8IndexerWorkspace(
            q_buffers=q_buffers,
            k_buffers=k_buffers,
            q_scale=torch.empty(
                indexer_mxfp8_thd_scale_shape(q_scale_capacity, q.shape[1], q.shape[2]),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            k_scale=torch.empty(
                indexer_mxfp8_thd_scale_shape(k_scale_capacity, 1, k.shape[1]),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        )
    return THDCompactIndexerWorkspace(
        cand_batch_offsets=cand_batch_offsets,
        cand_buffer_numel=int(cand_floats),
        geometry=_THDCompactIndexerGeometry(
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            batch_size=cu_seqlens_q.numel() - 1,
            has_q_causal_offsets=q_causal_offsets is not None,
            topk=topk,
            ratio=ratio,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            return_softmax=return_softmax,
            precision=precision,
        ),
        mxfp8=mxfp8_workspace,
    )


@torch.compile
def build_thd_compact_k_layout(
    cu_seqlens_q: Tensor, cu_seqlens_k: Tensor, total_k_rows: int, ratio: int
) -> Tuple[Tensor, Tensor]:
    """Build compact-THD K metadata and a physical-to-logical row map.

    The compact wrapper requires ``cu_seqlens_k[-1] == k.shape[0]`` and
    ``seqlen_q[b] <= seqlen_k[b] * ratio`` for every packed segment. A
    floor-compressed sequence violates the latter whenever it has a tail.
    Insert one zero-valued row after every real K segment and distribute any
    fixed-capacity K tail across those segments. The compact causal mask can
    never expose these appended rows, so returned local K ids are unchanged.

    ``cu_seqlens_q`` may contain one additional synthetic padding segment, as
    used by the CP path. Its K rows consume the corresponding fixed-capacity
    tail instead of being assigned to a real sequence.
    """
    num_sequences = cu_seqlens_k.shape[0] - 1
    zero = torch.zeros((1,), dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device)
    valid_k_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    padding_q = (cu_seqlens_q[-1] - cu_seqlens_q[num_sequences]).clamp_min(0)
    padding_k = torch.div(padding_q + int(ratio) - 1, int(ratio), rounding_mode="floor")
    capacity_gap = (int(total_k_rows) - cu_seqlens_k[-1]).clamp_min(0)
    # The static compressed capacity reserves enough rows for a synthetic Q
    # segment. All remaining capacity can be spread over real segments.
    remaining_extra = (capacity_gap - padding_k).clamp_min(0)
    sequence_ids = torch.arange(num_sequences, dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device)
    base_extra = torch.div(remaining_extra, num_sequences, rounding_mode="floor")
    extra_remainder = remaining_extra - base_extra * num_sequences
    extra_per_sequence = base_extra + (sequence_ids < extra_remainder).to(cu_seqlens_k.dtype)

    kernel_k_lens = valid_k_lens + 1 + extra_per_sequence
    kernel_k_prefix = torch.cumsum(kernel_k_lens, dim=0, dtype=torch.int32)
    if cu_seqlens_q.shape[0] == cu_seqlens_k.shape[0]:
        compact_cu_seqlens_k = torch.cat((zero, kernel_k_prefix))
    else:
        compact_cu_seqlens_k = torch.cat(
            (zero, kernel_k_prefix, (kernel_k_prefix[-1] + padding_k).view(1))
        )

    kernel_rows = torch.arange(
        int(total_k_rows) + num_sequences, dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device
    )
    # Avoid ``torch.bucketize`` here: ``torch.compile`` can miscompile CUDA
    # bucketize when the boundaries are produced inside the same graph.
    kernel_sequence_ids = (
        (kernel_rows.unsqueeze(1) >= compact_cu_seqlens_k[1:].unsqueeze(0))
        .sum(dim=1, dtype=torch.int64)
        .clamp_max(num_sequences)
    )
    safe_sequence_ids = kernel_sequence_ids.clamp_max(num_sequences - 1)
    kernel_positions = kernel_rows - compact_cu_seqlens_k[kernel_sequence_ids]
    valid_source = (kernel_sequence_ids < num_sequences) & (
        kernel_positions < valid_k_lens[safe_sequence_ids]
    )
    source_rows = cu_seqlens_k[safe_sequence_ids] + kernel_positions
    source_row_map = torch.where(
        valid_source, source_rows, torch.full_like(source_rows, int(total_k_rows))
    ).to(torch.int64)
    return compact_cu_seqlens_k, source_row_map


@torch.compile
def pack_thd_compact_k(k: Tensor, source_row_map: Tensor) -> Tensor:
    """Pack THD K with zero-valued rows that remain causally unreachable."""
    k_with_padding = torch.cat((k, torch.zeros_like(k[:1])))
    return torch.index_select(k_with_padding, 0, source_row_map)


def _canonicalize_compact_topk(
    topk_indices: Tensor, compact_logits: Tensor, compact_softmax: Optional[Tensor]
) -> Tuple[Tensor, Optional[Tensor]]:
    """Put each row's selected ids in ascending order (``-1`` padding last) and recompute the
    compact softmax over them in that fixed order.

    FlashMLA's online softmax and the indexer backward consume the ids in slot order, so with
    the kernel's run-dependent order the attention output and the indexer gradients change from
    run to run even though the selected set does not. After this the outputs are a pure function
    of the selected set. The recomputed softmax equals the kernel's up to fp32 rounding order.
    """
    order_key = topk_indices.masked_fill(topk_indices < 0, torch.iinfo(topk_indices.dtype).max)
    _, perm = torch.sort(order_key, dim=-1, stable=True)
    topk_indices = torch.gather(topk_indices, -1, perm)
    if compact_softmax is not None:
        valid = topk_indices >= 0
        logits = torch.gather(compact_logits.float(), -1, perm)
        logits = logits.masked_fill(~valid, float("-inf"))
        softmax = torch.softmax(logits, dim=-1)
        # Rows without any valid id (padding rows) have an all -inf input; keep them at 0.
        softmax = torch.where(valid.any(dim=-1, keepdim=True), softmax, torch.zeros_like(softmax))
        compact_softmax = softmax.to(compact_softmax.dtype)
    return topk_indices, compact_softmax


def compact_indexer_topk(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    topk: int,
    ratio: int = 4,
    *,
    cu_seqlens_q: Tensor | None = None,
    cu_seqlens_kv: Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_kv: int | None = None,
    q_causal_offsets: Tensor | None = None,
    return_softmax: bool = False,
    compact_workspace: BSHDCompactIndexerWorkspace | THDCompactIndexerWorkspace | None = None,
    precision: str = "mxfp8",
    deterministic: bool = False,
) -> Tuple[Tensor, Tensor, None, Optional[Tensor]]:
    """Run compact forward + Top-K without materializing the dense score matrix."""
    if precision != "mxfp8":
        raise ValueError("This compact adapter supports precision='mxfp8' only")
    if not all(t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous() for t in (q, k, w)):
        raise ValueError("MXFP8 compact indexer expects contiguous CUDA BF16 source tensors")
    if q.shape[-2:] != (64, 128) or k.shape[-1] != 128 or w.shape[-1] != 64:
        raise ValueError("MXFP8 compact indexer requires 64 Q heads and head_dim=128")
    is_thd = cu_seqlens_q is not None
    device = q.device
    compact_wrapper = _DSA.indexer_forward_top_k_wrapper
    if torch.cuda.is_current_stream_capturing() and compact_workspace is None:
        raise RuntimeError("Compact indexer CUDA graph capture requires eager workspace warmup")
    compact_kwargs = dict(
        ratio=ratio,
        precision=precision,
        return_softmax=return_softmax,
        topk_indices_global=False,
        deterministic=deterministic,
    )
    if is_thd:
        if compact_workspace is not None:
            if not isinstance(compact_workspace, THDCompactIndexerWorkspace):
                raise ValueError("THD compact dispatch requires a THDCompactIndexerWorkspace")
            compact_workspace.validate(
                q=q,
                k=k,
                topk=topk,
                ratio=ratio,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=int(max_seqlen_kv),
                q_causal_offsets=q_causal_offsets,
                return_softmax=return_softmax,
                precision=precision,
            )
            _refresh_thd_compact_cand_batch_offsets(
                compact_workspace.cand_batch_offsets, cu_seqlens_q, ratio, q_causal_offsets
            )
        compact_kwargs.update(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_kv),
        )
        if compact_workspace is not None:
            compact_kwargs.update(
                cand_buffer=_allocate_compact_candidate_buffer(
                    compact_workspace.cand_buffer_numel, device
                ),
                cand_batch_offsets=compact_workspace.cand_batch_offsets,
            )
            compact_kwargs.update(
                _allocate_compact_output_buffers((q.shape[0], topk), device, return_softmax)
            )
        if q_causal_offsets is not None:
            compact_kwargs["q_causal_offsets"] = q_causal_offsets
    elif compact_workspace is not None:
        if not isinstance(compact_workspace, BSHDCompactIndexerWorkspace):
            raise ValueError("BSHD compact dispatch requires a BSHDCompactIndexerWorkspace")
        compact_workspace.validate(
            q=q, k=k, topk=topk, ratio=ratio, return_softmax=return_softmax, precision=precision
        )
        compact_kwargs.update(
            cand_buffer=_allocate_compact_candidate_buffer(
                compact_workspace.cand_buffer_numel, device
            ),
            microbatch_rows=0,
        )
        compact_kwargs.update(
            _allocate_compact_output_buffers((q.shape[0], q.shape[1], topk), device, return_softmax)
        )

    kernel_q, kernel_k = q, k
    if precision == "mxfp8":
        mxfp8_workspace = compact_workspace.mxfp8 if compact_workspace is not None else None
        cu_seqlens_q_scale_padded = None
        cu_seqlens_k_scale_padded = None
        if is_thd:
            if mxfp8_workspace is not None:
                cu_seqlens_q_scale_padded = mxfp8_workspace.cu_seqlens_q_scale_padded
                cu_seqlens_k_scale_padded = mxfp8_workspace.cu_seqlens_k_scale_padded
                assert cu_seqlens_q_scale_padded is not None
                assert cu_seqlens_k_scale_padded is not None
                refresh_indexer_mxfp8_scale_cu_seqlens(
                    cu_seqlens_q_scale_padded, cu_seqlens_q, q.shape[1]
                )
                refresh_indexer_mxfp8_scale_cu_seqlens(cu_seqlens_k_scale_padded, cu_seqlens_kv, 1)
            else:
                cu_seqlens_q_scale_padded = make_indexer_mxfp8_scale_cu_seqlens(
                    cu_seqlens_q, q.shape[1]
                )
                cu_seqlens_k_scale_padded = make_indexer_mxfp8_scale_cu_seqlens(cu_seqlens_kv, 1)

        kernel_q, q_scale = quantize_indexer_mxfp8(
            q,
            cu_seqlens=cu_seqlens_q,
            cu_seqlens_scale_padded=cu_seqlens_q_scale_padded,
            buffers=(mxfp8_workspace.q_buffers if mxfp8_workspace is not None else None),
            out_scale=(mxfp8_workspace.q_scale if mxfp8_workspace is not None else None),
        )
        kernel_k, k_scale = quantize_indexer_mxfp8(
            k,
            cu_seqlens=cu_seqlens_kv,
            cu_seqlens_scale_padded=cu_seqlens_k_scale_padded,
            buffers=(mxfp8_workspace.k_buffers if mxfp8_workspace is not None else None),
            out_scale=(mxfp8_workspace.k_scale if mxfp8_workspace is not None else None),
        )
        compact_kwargs.update(q_scale=q_scale, k_scale=k_scale, sf_vec_size=32)
        if is_thd:
            compact_kwargs.update(
                cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
                cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            )

    if is_thd:
        compact_result = compact_wrapper(
            kernel_q, kernel_k.unsqueeze(1), w, top_k=topk, **compact_kwargs
        )
    else:
        compact_result = compact_wrapper(
            kernel_q, kernel_k.unsqueeze(2), w, top_k=topk, **compact_kwargs
        )
    if compact_workspace is not None:
        del compact_kwargs["cand_buffer"]

    topk_indices = compact_result["indices"]
    compact_logits = compact_result["logits"]
    compact_softmax = compact_result["softmax"] if return_softmax else None
    if compact_workspace is not None:
        returned_buffers = (
            ("indices", topk_indices, compact_kwargs["out_indices"]),
            ("logits", compact_logits, compact_kwargs["out_logits"]),
        )
        if return_softmax:
            returned_buffers += (("softmax", compact_softmax, compact_kwargs["softmax_out"]),)
        for name, actual, expected in returned_buffers:
            if actual.data_ptr() != expected.data_ptr():
                layout = "THD" if is_thd else "BSHD"
                raise RuntimeError(
                    f"cuDNN compact {layout} {name} did not alias "
                    "the dispatch-local output buffer"
                )
        for output_name in ("out_indices", "out_logits", "softmax_out"):
            compact_kwargs.pop(output_name, None)
        del returned_buffers, actual, expected
    if deterministic:
        # The kernel's ``deterministic`` flag fixes *which* ids are selected but not their
        # slot order; make the outputs a pure function of the selected set.
        topk_indices, compact_softmax = _canonicalize_compact_topk(
            topk_indices, compact_logits, compact_softmax
        )
    del compact_logits, compact_result

    topk_indices = topk_indices.int()
    topk_length = (topk_indices >= 0).sum(dim=-1).int()
    if is_thd:
        return topk_indices, topk_length, None, compact_softmax
    b, sq = q.shape[:2]
    return (topk_indices.view(b, sq, topk), topk_length.view(b, sq), None, compact_softmax)


class CompactIndexer:
    """Bind compact dispatch and reuse graph workspaces across serial CSA layers.

    Every result and candidate buffer is dispatch-local. Only quantization
    buffers and packed scale metadata are shared, so backward never observes
    another layer's overwritten values.
    """

    def __init__(self, *, workspaces: dict, cuda_graphs: bool, deterministic: bool) -> None:
        self.workspaces = workspaces
        self.cuda_graphs = cuda_graphs
        self.deterministic = deterministic

    @torch.no_grad()
    def __call__(self, q, k, weights, topk, ratio, *, return_softmax=False, **layout):
        kwargs = dict(topk=topk, ratio=ratio, return_softmax=return_softmax, precision="mxfp8")
        workspace = None
        if self.cuda_graphs:
            is_thd = layout.get("cu_seqlens_q") is not None
            key = (
                q.device,
                tuple(q.shape),
                tuple(k.shape),
                topk,
                ratio,
                return_softmax,
                len(layout["cu_seqlens_q"]) if is_thd else None,
                layout.get("max_seqlen_q"),
                layout.get("max_seqlen_kv"),
                layout.get("q_causal_offsets") is not None,
            )
            workspace = self.workspaces.get(key)
            if workspace is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Compact indexer graph geometry was not warmed up eagerly")
                if is_thd:
                    workspace = prepare_thd_compact_indexer_workspace(
                        q,
                        k,
                        **kwargs,
                        cu_seqlens_q=layout["cu_seqlens_q"],
                        cu_seqlens_k=layout["cu_seqlens_kv"],
                        max_seqlen_q=layout["max_seqlen_q"],
                        max_seqlen_k=layout["max_seqlen_kv"],
                        q_causal_offsets=layout.get("q_causal_offsets"),
                    )
                else:
                    workspace = prepare_bshd_compact_indexer_workspace(q, k, **kwargs)
                self.workspaces[key] = workspace
        return compact_indexer_topk(
            q,
            k,
            weights,
            **kwargs,
            **layout,
            compact_workspace=workspace,
            deterministic=self.deterministic,
        )

    @torch.no_grad()
    def packed(
        self,
        q,
        k,
        weights,
        topk,
        ratio,
        *,
        cu_seqlens_compressed,
        indexer_layout,
        max_seqlen_q,
        indexer_softmax_scale=1.0,
        return_softmax=False,
    ):
        """Pack zero K padding while retaining logical key ids and softmax slot order."""
        cu_q, _cu_k, q_offsets = indexer_layout
        compact_cu_k, source_rows = build_thd_compact_k_layout(
            cu_q, cu_seqlens_compressed, k.shape[0], ratio
        )
        packed_k = pack_thd_compact_k(k, source_rows)
        # Underfilled static buckets can leave more than one padding row per
        # document. The physical row bound stays valid across graph replays;
        # max_seqlen_q // ratio + 2 does not bound those padded K segments.
        scaled_weights = (weights.float() * indexer_softmax_scale).to(weights.dtype)
        topk_ids, lengths, _, predict = self(
            q,
            packed_k,
            scaled_weights,
            topk,
            ratio,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=compact_cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=packed_k.shape[0],
            q_causal_offsets=q_offsets,
            return_softmax=return_softmax,
        )
        return topk_ids, lengths, predict
