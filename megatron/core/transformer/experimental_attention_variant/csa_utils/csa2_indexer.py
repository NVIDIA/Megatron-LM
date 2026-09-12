# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Candidate-aware CSA2 selection and indexer supervision.

Candidate IDs are read directly by the kernels. Selection uses TE MXFP8 data
and E8M0 scales or BF16, followed by DSv4 radix Top-K. Supervision uses the
original projections. Forward consumes bounded score/target tiles once for loss
and unit-loss gradients; backward only scales those gradients, as in V4. Shared
K gradients still reach the owning Full layer.
"""

from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from megatron.core.quantization.indexer_quantization import quantize_indexer_mxfp8_logical

from . import fused_sparse_attention
from .csa2_candidates import CSA2CandidateBlocks, select_compact_positions
from .csa_teacher_lse import fused_csa_window_lse
from .thd_utils import CSA2THDCompressionLayout, CSA2THDLayout

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False


_QUERY_CHUNK_SIZE = 128
_SCORE_CHUNK_MAX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class CSA2IndexerInputs:
    """Prepared BF16 projections and bounds for one indexer invocation.

    Q/K/weights are contiguous, batch-major flat rows for SBHD, or physical
    packed rows for THD. Weights remain unscaled. Prepare these tensors with
    autograd enabled when supervising the indexer; selection only reads them
    under no_grad. Q/weights belong to this layer, while K may share its Full
    owner's storage and autograd edge across Reindex layers in the same forward.
    No original-layout copies or selection-only padding are retained here.
    """

    q: Tensor
    k: Tensor
    weights: Tensor
    ratio: int
    starts: Tensor
    visible: Tensor
    valid: Tensor
    max_keys: int
    key_capacity: int
    output_shape: tuple[int, ...]
    candidates: CSA2CandidateBlocks | None = None
    thd_layout: CSA2THDLayout | None = None
    compressed_layout: CSA2THDCompressionLayout | None = None

    @cached_property
    def packed_metadata(self) -> tuple[Tensor, Tensor, int, int]:
        """Prepare cuDNN prefixes with host-known bounds, without reading device values."""
        if self.thd_layout is None:
            batch, seq_len = self.output_shape
            prefix = torch.arange(batch + 1, dtype=torch.int32, device=self.q.device)
            return prefix * seq_len, prefix * self.key_capacity, seq_len, self.max_keys
        cu_q = self.thd_layout.cu_seqlens_padded.to(torch.int32).contiguous()
        cu_k = self.compressed_layout.cu_seqlens_padded.to(torch.int32).contiguous()
        # Compact Top-K scans every physical row, including unassigned capacity.
        # Split that tail into bounded padding segments on the device. Like V4,
        # launch bounds come from host metadata, not a device-to-host length read;
        # they stay per-sequence instead of growing with the whole packed batch.
        max_q, max_k = max(self.thd_layout.max_seqlen, 1), self.max_keys
        tail_segments = max(
            (self.q.shape[0] + max_q - 1) // max_q,
            (self.k.shape[0] + max(max_k, 1) - 1) // max(max_k, 1),
            1,
        )
        steps = torch.arange(1, tail_segments + 1, device=cu_q.device, dtype=torch.int64)
        tail_q = (cu_q[-1] + steps * max_q).clamp_max(self.q.shape[0]).int()
        tail_k = (cu_k[-1] + steps * max_k).clamp_max(self.k.shape[0]).int()
        return torch.cat((cu_q, tail_q)), torch.cat((cu_k, tail_k)), max_q, max_k


def prepare_csa2_indexer_inputs(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    ratio: int,
    candidates: CSA2CandidateBlocks | None = None,
    thd_layout: CSA2THDLayout | None = None,
    compressed_layout: CSA2THDCompressionLayout | None = None,
    *,
    k_flat: Tensor | None = None,
) -> CSA2IndexerInputs:
    """Pack projections once, using the Full owner's prepared K when supplied.

    SBHD inputs have shapes [S, B, H, D], [K, B, D], and [S, B, H]. THD
    inputs have shapes [T, H, D], [C, D], and [T, H]. Shared k_flat must have
    the same values and autograd provenance as K, in contiguous flat layout.
    """
    key_capacity = k.shape[0]
    if thd_layout is None:
        seq_len, batch, heads, dim = q.shape
        q = q.permute(1, 0, 2, 3).reshape(-1, heads, dim).contiguous()
        expected_k_shape = (batch * key_capacity, dim)
        k = k.transpose(0, 1).reshape(-1, dim).contiguous() if k_flat is None else k
        weights = weights.transpose(0, 1).reshape(-1, heads).contiguous()
        starts = torch.arange(batch, device=q.device, dtype=torch.int32).repeat_interleave(seq_len)
        starts = starts * key_capacity
        visible = ((torch.arange(seq_len, device=q.device) + 1) // ratio).clamp(max=key_capacity)
        visible = visible.repeat(batch).int()
        valid = torch.ones(batch * seq_len, device=q.device, dtype=torch.bool)
        max_keys, shape = key_capacity, (batch, seq_len)
    else:
        if compressed_layout is None or compressed_layout.ratio != ratio:
            raise ValueError("CSA2 indexer requires a matching THD compression layout")
        expected_k_shape = tuple(k.shape)
        starts = compressed_layout.cu_seqlens_padded[thd_layout.sequence_ids.clamp_min(0)].int()
        visible = ((thd_layout.position_ids + 1) // ratio).masked_fill(~thd_layout.valid_tokens, 0)
        visible = visible.int()
        valid, max_keys, shape = (
            thd_layout.valid_tokens,
            compressed_layout.max_seqlen,
            (q.shape[0],),
        )
    if k_flat is not None:
        if (
            k_flat.shape != expected_k_shape
            or k_flat.dtype != k.dtype
            or k_flat.device != k.device
            or not k_flat.is_contiguous()
        ):
            raise ValueError("CSA2 shared flat indexer K must match its projected keys")
        k = k_flat
    return CSA2IndexerInputs(
        q=q.contiguous(),
        k=k.contiguous(),
        weights=weights.contiguous(),
        ratio=ratio,
        starts=starts,
        visible=visible,
        valid=valid,
        max_keys=max_keys,
        key_capacity=key_capacity,
        output_shape=shape,
        candidates=candidates,
        thd_layout=thd_layout,
        compressed_layout=compressed_layout,
    )


def _chunk_rows(width: int) -> int:
    # A loss chunk has scores, teacher mass, and (in backward) score gradients.
    return min(_QUERY_CHUNK_SIZE, max(1, _SCORE_CHUNK_MAX_BYTES // (max(width, 1) * 12)))


if HAVE_TRITON:

    @triton.jit
    def _key_ids(
        ids,
        counts,
        starts,
        visible,
        row,
        cols,
        WIDTH: tl.constexpr,
        ID_WIDTH: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """MODE 0: causal keys; 1: local blocks; 2/3: physical/local sparse Top-K."""
        start, length = tl.load(starts + row), tl.load(visible + row)
        if MODE == 1:
            slots = cols // BLOCK_SIZE
            count = tl.load(counts + row)
            block = tl.load(
                ids + row * ID_WIDTH + slots, (cols < WIDTH) & (slots < count), other=-1
            )
            local = block * BLOCK_SIZE + cols % BLOCK_SIZE
            valid = (cols < WIDTH) & (slots < count) & (block >= 0)
            valid = valid & (local < length)
            physical = start + local
        elif MODE == 2 or MODE == 3:
            physical = tl.load(ids + row * ID_WIDTH + cols, cols < WIDTH, other=-1)
            if MODE == 3:
                physical += start
            valid = (cols < WIDTH) & (physical >= start) & (physical < start + length)
        else:
            physical = start + cols
            valid = (cols < WIDTH) & (cols < length)
        return physical, valid

    @triton.jit
    def _score_kernel(
        q,
        k,
        weights,
        q_scale,
        k_scale,
        ids,
        counts,
        starts,
        visible,
        scores,
        Q_SCALE_STRIDE: tl.constexpr,
        K_SCALE_STRIDE: tl.constexpr,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        ID_WIDTH: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MXFP8: tl.constexpr,
        SCALE: tl.constexpr,
        ROUND_WEIGHTS: tl.constexpr,
        TILE_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * TILE_K + tl.arange(0, TILE_K)
        keys, valid = _key_ids(
            ids, counts, starts, visible, row, cols, WIDTH, ID_WIDTH, MODE, BLOCK_SIZE
        )
        h, d = tl.arange(0, HEADS), tl.arange(0, DIM)
        qv = tl.load(
            q + (row * HEADS + h[:, None]) * DIM + d[None, :], h[:, None] < HEADS, other=0.0
        )
        kv = tl.load(k + keys[None, :] * DIM + d[:, None], valid[None, :], other=0.0)
        if MXFP8:
            groups = tl.arange(0, DIM // 32)
            qs = tl.load(
                q_scale + (row * HEADS + h[:, None]) * Q_SCALE_STRIDE + groups[None, :],
                h[:, None] < HEADS,
                other=127,
            )
            ks = tl.load(
                k_scale + keys[:, None] * K_SCALE_STRIDE + groups[None, :],
                valid[:, None],
                other=127,
            )
            # Triton's SM100 scaled MMA lowering requires M >= 128. Use the
            # candidate-key axis as M and real H32/H64 as N, then transpose.
            # This avoids padding Q or silently emulating MXFP8 through BF16.
            dots = tl.trans(tl.dot_scaled(tl.trans(kv), ks, "e4m3", tl.trans(qv), qs, "e4m3"))
        else:
            dots = tl.dot(qv, kv)
        w = tl.load(weights + row * HEADS + h, h < HEADS, other=0.0).to(tl.float32) * SCALE
        if ROUND_WEIGHTS:
            w = w.to(tl.bfloat16).to(tl.float32)
        result = tl.sum(tl.maximum(dots, 0.0) * w[:, None], 0)
        tl.store(
            scores + row * STRIDE + cols, tl.where(valid, result, -float("inf")), cols < STRIDE
        )

    @triton.jit
    def _map_selected_kernel(
        slots,
        ids,
        counts,
        starts,
        visible,
        output,
        WIDTH: tl.constexpr,
        ID_WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TOPK: tl.constexpr,
        PHYSICAL: tl.constexpr,
        TILE: tl.constexpr,
    ):
        row, cols = tl.program_id(0), tl.arange(0, TILE)
        slot = tl.load(slots + row * TOPK + cols, cols < TOPK, other=-1)
        keys, valid = _key_ids(
            ids, counts, starts, visible, row, tl.maximum(slot, 0), WIDTH, ID_WIDTH, 1, BLOCK_SIZE
        )
        if not PHYSICAL:
            keys -= tl.load(starts + row)
        tl.store(output + row * TOPK + cols, tl.where((slot >= 0) & valid, keys, -1), cols < TOPK)

    @triton.jit
    def _teacher_lse_kernel(
        q,
        k,
        non_compressed_lse,
        ids,
        counts,
        starts,
        visible,
        lse,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        WIDTH: tl.constexpr,
        ID_WIDTH: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
        TILE_H: tl.constexpr,
        TILE_K: tl.constexpr,
    ):
        # This launch spans the full flattened batch, before loss query chunking.
        row = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1) * TILE_H + tl.arange(0, TILE_H)
        d = tl.arange(0, DIM)
        qv = tl.load(q + (row * HEADS + h[:, None]) * DIM + d[None, :], h[:, None] < HEADS, other=0)
        maximum = tl.load(non_compressed_lse + row * HEADS + h, h < HEADS, other=-float("inf"))
        denom = tl.where(maximum > -float("inf"), 1.0, 0.0)
        for start in range(0, tl.cdiv(WIDTH, TILE_K)):
            cols = start * TILE_K + tl.arange(0, TILE_K)
            keys, valid = _key_ids(
                ids, counts, starts, visible, row, cols, WIDTH, ID_WIDTH, MODE, BLOCK_SIZE
            )
            kv = tl.load(k + keys[None, :].to(tl.int64) * DIM + d[:, None], valid[None, :], other=0)
            logits = tl.where(valid[None, :], tl.dot(qv, kv) * SCALE, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(logits, 1))
            old_scale = tl.where(maximum > -float("inf"), tl.exp(maximum - new_max), 0.0)
            denom = denom * old_scale + tl.sum(
                tl.where(valid[None, :], tl.exp(logits - new_max[:, None]), 0.0), 1
            )
            maximum = new_max
        tl.store(
            lse + row * HEADS + h,
            tl.where(denom > 0, maximum + tl.log(denom), -float("inf")),
            h < HEADS,
        )

    @triton.jit
    def _teacher_target_kernel(
        q,
        k,
        lse,
        ids,
        counts,
        starts,
        visible,
        target,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        ID_WIDTH: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
        TILE_H: tl.constexpr,
        TILE_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * TILE_K + tl.arange(0, TILE_K)
        keys, valid = _key_ids(
            ids, counts, starts, visible, row, cols, WIDTH, ID_WIDTH, MODE, BLOCK_SIZE
        )
        d = tl.arange(0, DIM)
        kv = tl.load(k + keys[None, :] * DIM + d[:, None], valid[None, :], other=0)
        result = tl.full((TILE_K,), 0.0, tl.float32)
        for head_start in range(0, tl.cdiv(HEADS, TILE_H)):
            h = head_start * TILE_H + tl.arange(0, TILE_H)
            qv = tl.load(
                q + (row * HEADS + h[:, None]) * DIM + d[None, :], h[:, None] < HEADS, other=0
            )
            denom = tl.load(lse + row * HEADS + h, h < HEADS, other=0.0)
            probs = tl.exp(tl.dot(qv, kv) * SCALE - denom[:, None])
            result += tl.sum(tl.where((h[:, None] < HEADS) & valid[None, :], probs, 0.0), 0)
        tl.store(target + row * STRIDE + cols, result, cols < STRIDE)

    @triton.jit
    def _loss_kernel(
        scores,
        target,
        stats,
        losses,
        divisor,
        STRIDE: tl.constexpr,
        COEFFICIENT: tl.constexpr,
        TILE: tl.constexpr,
    ):
        """Stable log-softmax and the same unclamped-target KL as native DSA."""
        row = tl.program_id(0)
        offsets = tl.arange(0, TILE)
        maximum = tl.full((), -float("inf"), tl.float32)
        denom, mass = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
        for start in range(0, tl.cdiv(STRIDE, TILE)):
            cols = start * TILE + offsets
            score = tl.load(scores + row * STRIDE + cols, cols < STRIDE, other=-float("inf"))
            valid = score > -float("inf")
            next_max = tl.maximum(maximum, tl.max(score, 0))
            denom = denom * tl.where(maximum > -float("inf"), tl.exp(maximum - next_max), 0.0)
            denom += tl.sum(tl.where(valid, tl.exp(score - next_max), 0.0), 0)
            mass += tl.sum(tl.load(target + row * STRIDE + cols, cols < STRIDE, other=0.0), 0)
            maximum = next_max
        logsum = tl.where(denom > 0, maximum + tl.log(denom), 0.0)
        safe_mass = tl.maximum(mass, 1.1754943508222875e-38)
        loss = tl.full((), 0.0, tl.float32)
        for start in range(0, tl.cdiv(STRIDE, TILE)):
            cols = start * TILE + offsets
            score = tl.load(scores + row * STRIDE + cols, cols < STRIDE, other=-float("inf"))
            t = tl.load(target + row * STRIDE + cols, cols < STRIDE, other=0.0) / safe_mass
            term = t * (tl.log(tl.maximum(t, 1.0e-10)) - (score - logsum))
            loss += tl.sum(tl.where(score > -float("inf"), term, 0.0), 0)
        tl.store(stats + row * 3, logsum)
        tl.store(stats + row * 3 + 1, safe_mass)
        tl.store(stats + row * 3 + 2, mass / safe_mass)
        tl.store(losses + row, loss * COEFFICIENT / tl.maximum(tl.load(divisor), 1.0))

    @triton.jit
    def _loss_grad_kernel(
        scores,
        target,
        stats,
        divisor,
        grad_loss,
        grad_scores,
        STRIDE: tl.constexpr,
        COEFFICIENT: tl.constexpr,
        TILE: tl.constexpr,
    ):
        row, cols = tl.program_id(0), tl.program_id(1) * TILE + tl.arange(0, TILE)
        score = tl.load(scores + row * STRIDE + cols, cols < STRIDE, other=-float("inf"))
        t = tl.load(target + row * STRIDE + cols, cols < STRIDE, other=0.0)
        logsum, mass = tl.load(stats + row * 3), tl.load(stats + row * 3 + 1)
        target_sum = tl.load(stats + row * 3 + 2)
        scale = COEFFICIENT * tl.load(grad_loss) / tl.maximum(tl.load(divisor), 1.0)
        grad = (tl.exp(score - logsum) * target_sum - t / mass) * scale
        tl.store(grad_scores + row * STRIDE + cols, grad, cols < STRIDE)

    @triton.jit
    def _score_backward_kernel(
        q,
        k,
        weights,
        ids,
        counts,
        starts,
        visible,
        grad_scores,
        dq,
        dk,
        dw,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        ID_WIDTH: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
        TILE_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * TILE_K + tl.arange(0, TILE_K)
        keys, valid = _key_ids(
            ids, counts, starts, visible, row, cols, WIDTH, ID_WIDTH, MODE, BLOCK_SIZE
        )
        h, d = tl.arange(0, HEADS), tl.arange(0, DIM)
        qv = tl.load(q + (row * HEADS + h[:, None]) * DIM + d[None, :])
        kv = tl.load(k + keys[None, :] * DIM + d[:, None], valid[None, :], other=0)
        dots = tl.dot(qv, kv)
        ds = tl.load(grad_scores + row * STRIDE + cols, valid, other=0.0)
        w = tl.load(weights + row * HEADS + h).to(tl.float32) * SCALE
        ddot = tl.where(dots > 0.0, w[:, None] * ds[None, :], 0.0)
        # Keep score derivatives and repeated K contributions in FP32. Casting
        # ddot to BF16 would change the auxiliary objective's backward math.
        grad_q = tl.dot(ddot, tl.trans(kv).to(tl.float32), input_precision="tf32x3")
        grad_k = tl.dot(tl.trans(ddot), qv.to(tl.float32), input_precision="tf32x3")
        grad_w = tl.sum(tl.maximum(dots, 0.0) * ds[None, :], 1) * SCALE
        tl.atomic_add(dq + (row * HEADS + h[:, None]) * DIM + d[None, :], grad_q, sem="relaxed")
        tl.atomic_add(dk + keys[:, None] * DIM + d[None, :], grad_k, valid[:, None], sem="relaxed")
        tl.atomic_add(dw + row * HEADS + h, grad_w, sem="relaxed")


def _score_chunk(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    ids: Tensor,
    counts: Tensor,
    starts: Tensor,
    visible: Tensor,
    *,
    width: int,
    stride: int,
    mode: int,
    block_size: int,
    selection: bool = False,
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
) -> Tensor:
    scores = torch.empty((q.shape[0], stride), dtype=torch.float32, device=q.device)
    tile_k = 128 if q_scale is not None else 64
    _score_kernel[(q.shape[0], triton.cdiv(stride, tile_k))](
        q,
        k,
        weights,
        q_scale,
        k_scale,
        ids,
        counts,
        starts,
        visible,
        scores,
        q_scale.stride(0) if q_scale is not None else 0,
        k_scale.stride(0) if k_scale is not None else 0,
        q.shape[-2],
        q.shape[-1],
        stride,
        width,
        ids.shape[-1],
        mode,
        block_size,
        q_scale is not None,
        q.shape[-1] ** -0.5 * q.shape[-2] ** -0.5,
        selection,
        tile_k,
    )
    return scores


@torch.no_grad()
def fused_candidate_topk(inputs: CSA2IndexerInputs, topk: int, precision: str) -> Tensor:
    """Score only candidate blocks and select sorted local/physical key positions."""
    if not HAVE_TRITON or not inputs.q.is_cuda:
        raise RuntimeError("Fused CSA2 Reindex requires CUDA and Triton")
    if precision not in ("bf16", "mxfp8"):
        raise ValueError("CSA2 Reindex precision must be bf16 or mxfp8")
    if inputs.candidates is None:
        raise ValueError("Candidate-aware Reindex requires compact candidate blocks")
    q, k, w = inputs.q, inputs.k, inputs.weights
    starts, visible, shape = inputs.starts, inputs.visible, inputs.output_shape
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or w.dtype != q.dtype:
        raise ValueError("CSA2 Reindex projections must be BF16")
    if q.shape[-2] not in (32, 64) or q.shape[-1] != 128:
        raise ValueError("Fused CSA2 Reindex requires H32/H64 and D128")
    topk = min(topk, inputs.key_capacity)
    output = torch.full((*shape, topk), -1, dtype=torch.int32, device=q.device)
    candidate = inputs.candidates
    width = candidate.indices.shape[-1] * candidate.block_size
    if q.shape[0] == 0 or k.shape[0] == 0 or width == 0 or topk == 0:
        return output
    ids = candidate.indices.reshape(q.shape[0], -1).contiguous()
    counts = candidate.lengths.reshape(-1).contiguous()
    stride = triton.cdiv(max(width, topk), 8) * 8
    flat_output = output.reshape(q.shape[0], topk)
    kernel_k, k_scale = quantize_indexer_mxfp8_logical(k) if precision == "mxfp8" else (k, None)
    for start in range(0, q.shape[0], _chunk_rows(stride)):
        end = min(start + _chunk_rows(stride), q.shape[0])
        kernel_q, q_scale = (
            quantize_indexer_mxfp8_logical(q[start:end])
            if precision == "mxfp8"
            else (q[start:end], None)
        )
        scores = _score_chunk(
            kernel_q,
            kernel_k,
            w[start:end],
            ids[start:end],
            counts[start:end],
            starts[start:end],
            visible[start:end],
            width=width,
            stride=stride,
            mode=1,
            block_size=candidate.block_size,
            selection=True,
            q_scale=q_scale,
            k_scale=k_scale,
        )
        slots, _ = select_compact_positions(
            scores, torch.full_like(counts[start:end], stride), topk
        )
        _map_selected_kernel[(end - start,)](
            slots,
            ids[start:end],
            counts[start:end],
            starts[start:end],
            visible[start:end],
            flat_output[start:end],
            width,
            ids.shape[-1],
            candidate.block_size,
            topk,
            inputs.thd_layout is not None,
            triton.next_power_of_2(topk),
        )
        del scores, slots
    return output


def _target_chunk(
    q: Tensor,
    k: Tensor,
    lse: Tensor,
    ids: Tensor,
    counts: Tensor,
    starts: Tensor,
    visible: Tensor,
    *,
    width: int,
    stride: int,
    mode: int,
    block_size: int,
    scale: float,
) -> Tensor:
    target = torch.empty((q.shape[0], stride), dtype=torch.float32, device=q.device)
    _teacher_target_kernel[(q.shape[0], triton.cdiv(stride, 64))](
        q,
        k,
        lse,
        ids,
        counts,
        starts,
        visible,
        target,
        q.shape[1],
        q.shape[2],
        stride,
        width,
        ids.shape[-1],
        mode,
        block_size,
        scale,
        32,
        64,
        num_stages=1,
    )
    return target


def _precompute_indexer_loss(
    q,
    k,
    weights,
    teacher_q,
    teacher_k,
    lse,
    ids,
    counts,
    starts,
    visible,
    divisor,
    options,
    *,
    full_lse=False,
):
    """Consume each score/target tile once for KL and unit-loss gradients.

    Only dK needs a full FP32 accumulator. dQ/dW accumulate within one query
    chunk, then round to their input dtype. Retain gradients instead of teacher
    tensors, projections, scores or targets until the main backward.
    """
    width, mode, block_size, scale, coefficient = options
    stride = triton.cdiv(width, 8) * 8
    rows = q.shape[0]
    if rows == 0 or k.shape[0] == 0 or width == 0:
        return q.new_zeros((), dtype=torch.float32), tuple(
            torch.zeros_like(t) for t in (q, k, weights)
        )
    if not full_lse:
        full = torch.empty_like(lse)
        _teacher_lse_kernel[(rows, triton.cdiv(teacher_q.shape[1], 16))](
            teacher_q,
            teacher_k,
            lse,
            ids,
            counts,
            starts,
            visible,
            full,
            teacher_q.shape[1],
            teacher_q.shape[2],
            width,
            ids.shape[-1],
            mode,
            block_size,
            scale,
            16,
            64,
            num_stages=1,
        )
        lse = full
    losses = torch.empty((rows,), device=q.device, dtype=torch.float32)
    dq, dw = torch.empty_like(q), torch.empty_like(weights)
    dk = torch.zeros_like(k, dtype=torch.float32)
    unit = q.new_ones((), dtype=torch.float32)
    for start in range(0, rows, _chunk_rows(stride)):
        end = min(start + _chunk_rows(stride), rows)
        bounds = (ids[start:end], counts[start:end], starts[start:end], visible[start:end])
        scores = _score_chunk(
            q[start:end],
            k,
            weights[start:end],
            *bounds,
            width=width,
            stride=stride,
            mode=mode,
            block_size=block_size,
        )
        target = _target_chunk(
            teacher_q[start:end],
            teacher_k,
            lse[start:end],
            *bounds,
            width=width,
            stride=stride,
            mode=mode,
            block_size=block_size,
            scale=scale,
        )
        stats = torch.empty((end - start, 3), device=q.device, dtype=torch.float32)
        _loss_kernel[(end - start,)](
            scores, target, stats, losses[start:end], divisor, stride, coefficient, 1024
        )
        # Turn the score tile into dScore while its target and row stats are live.
        _loss_grad_kernel[(end - start, triton.cdiv(stride, 1024))](
            scores, target, stats, divisor, unit, scores, stride, coefficient, 1024
        )
        del target, stats
        dq_chunk = torch.zeros_like(q[start:end], dtype=torch.float32)
        dw_chunk = torch.zeros_like(weights[start:end], dtype=torch.float32)
        _score_backward_kernel[(end - start, triton.cdiv(width, 64))](
            q[start:end],
            k,
            weights[start:end],
            *bounds,
            scores,
            dq_chunk,
            dk,
            dw_chunk,
            q.shape[1],
            q.shape[2],
            stride,
            width,
            ids.shape[-1],
            mode,
            block_size,
            q.shape[-1] ** -0.5 * q.shape[-2] ** -0.5,
            64,
        )
        dq[start:end].copy_(dq_chunk)
        dw[start:end].copy_(dw_chunk)
        del scores, dq_chunk, dw_chunk
    return losses.sum(), (dq, dk.to(k.dtype), dw)


class _CSA2IndexerLoss(torch.autograd.Function):
    """V4-style eager indexer backward, retaining only unit-loss gradients."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        weights,
        teacher_q,
        teacher_k,
        lse,
        ids,
        counts,
        starts,
        visible,
        divisor,
        options,
    ):
        loss, gradients = _precompute_indexer_loss(
            q, k, weights, teacher_q, teacher_k, lse, ids, counts, starts, visible, divisor, options
        )
        ctx.save_for_backward(*gradients)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        return *(g * grad_loss for g in ctx.saved_tensors), *([None] * 9)


class _CSA2IndexerSparseAttn(torch.autograd.Function):
    """Joint attention/KL forward with V4-style eager indexer gradients.

    Keep the producer's local/global KV tensors, not a concatenated KV allocation
    per consumer. The same shared global storage is reused by every layer and
    concatenated only while a forward or backward kernel is running.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        local,
        global_kv,
        sink,
        iq,
        ik,
        weights,
        window,
        indices,
        topk_length,
        q_padding_mask,
        starts,
        visible,
        divisor,
        ids,
        counts,
        options,
        query_shape,
    ):
        width, mode, block_size, scale, coefficient = options
        kv = torch.cat((local, global_kv))
        out, lse, _ = fused_sparse_attention._csa_fwd_flash_mla(
            q, kv, indices, scale, attn_sink=sink, topk_length=topk_length
        )
        del kv
        # Sparse supervision has exactly the main attention's support. FlashMLA
        # returns natural-log LSE excluding sink, matching V4's Path B contract.
        sparse = mode == 2
        teacher_q, teacher_lse = q, lse
        if query_shape is not None:
            seq_len, batch = query_shape
            teacher_q = (
                q.reshape(seq_len, batch, q.shape[1], q.shape[2])
                .transpose(0, 1)
                .reshape(-1, q.shape[1], q.shape[2])
                .contiguous()
            )
            teacher_lse = (
                lse.reshape(seq_len, batch, -1)
                .transpose(0, 1)
                .reshape(-1, lse.shape[-1])
                .contiguous()
            )
        teacher_lse = (
            torch.logaddexp(teacher_lse, sink.unsqueeze(0))
            if sparse
            else fused_csa_window_lse(teacher_q, local, sink, window, scale)
        )
        loss, gradients = _precompute_indexer_loss(
            iq,
            ik,
            weights,
            teacher_q,
            global_kv,
            teacher_lse,
            ids,
            counts,
            starts,
            visible,
            divisor,
            options,
            full_lse=sparse,
        )
        ctx.save_for_backward(
            q, local, global_kv, sink, indices, topk_length, q_padding_mask, out, lse, *gradients
        )
        ctx.scale = scale
        ctx.set_materialize_grads(False)
        return out, loss

    @staticmethod
    def backward(ctx, grad_out, grad_loss):
        q, local, global_kv, sink, indices, topk_length, q_padding_mask, out, lse, diq, dik, dw = (
            ctx.saved_tensors
        )
        dq = dlocal = dglobal = dsink = None
        if grad_out is not None:
            kv = torch.cat((local, global_kv))
            dq, dkv, dsink = fused_sparse_attention._csa_bwd_cudnn(
                q,
                kv,
                out,
                grad_out,
                lse,
                sink,
                indices,
                ctx.scale,
                topk_length=topk_length,
                q_padding_mask=q_padding_mask,
            )
            dlocal, dglobal = dkv.split((local.shape[0], global_kv.shape[0]))
        gradients = (
            (None, None, None)
            if grad_loss is None
            else tuple(g * grad_loss for g in (diq, dik, dw))
        )
        return dq, dlocal, dglobal, dsink, *gradients, *([None] * 11)


def fused_csa2_indexer_sparse_attn(
    inputs: CSA2IndexerInputs,
    query: Tensor,
    local_kv: Tensor,
    global_kv: Tensor,
    attn_sink: Tensor,
    window_indices: Tensor | None,
    global_indices: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    calculate_per_token_loss: bool = False,
    *,
    global_kv_flat: Tensor | None = None,
    attention_indices: Tensor | None = None,
    attention_topk_length: Tensor | None = None,
    q_padding_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Keep V4 attention layout; pack the detached teacher only for the loss work.

    Returning a view of the kernel output avoids retaining a second full output
    allocation for the following projection, especially at SBHD batch > 1.
    THD sparse loss reuses the main attention LSE and can omit ``window_indices``
    when final ``attention_indices`` have already been built.
    """
    if window_indices is None and (
        inputs.thd_layout is None or not sparse_loss or attention_indices is None
    ):
        raise ValueError("Only THD sparse loss with final attention indices can omit the window.")
    iq, ik, w = inputs.q, inputs.k, inputs.weights
    starts, visible, valid = inputs.starts, inputs.visible, inputs.valid
    max_keys = inputs.max_keys
    if inputs.thd_layout is None:
        seq_len, batch, heads, dim = query.shape
        q = query.reshape(-1, heads, dim)
        local = local_kv.transpose(0, 1).reshape(-1, dim).contiguous()
        global_flat = (
            global_kv.transpose(0, 1).reshape(-1, dim).contiguous()
            if global_kv_flat is None
            else global_kv_flat
        )
        window = window_indices.reshape(iq.shape[0], -1)
        offset = torch.arange(batch, device=q.device).repeat_interleave(seq_len) * seq_len
        window = torch.where(window >= 0, window + offset[:, None], -1).int()
        topk = global_indices.reshape(iq.shape[0], -1)
        topk = torch.where(topk >= 0, topk + starts[:, None], -1).int()
    else:
        q, local = query.contiguous(), local_kv.contiguous()
        global_flat = global_kv.squeeze(1) if global_kv_flat is None else global_kv_flat
        window = None if window_indices is None else window_indices.contiguous()
        topk = global_indices.contiguous()
    counts = visible
    if sparse_loss:
        ids, width, mode, block_size = topk, topk.shape[-1], 2, 1
    elif inputs.candidates is not None:
        candidate = inputs.candidates
        ids = candidate.indices.reshape(iq.shape[0], -1).contiguous()
        counts = candidate.lengths.reshape(-1).contiguous()
        width, mode, block_size = ids.shape[-1] * candidate.block_size, 1, candidate.block_size
    else:
        ids = torch.empty((iq.shape[0], 0), dtype=torch.int32, device=q.device)
        width, mode, block_size = max_keys, 0, 1
    divisor = (
        iq.new_ones((), dtype=torch.float32)
        if calculate_per_token_loss
        else valid.sum(dtype=torch.float32).clamp_min(1)
    )
    if attention_indices is None:
        attention_indices = torch.cat(
            (window, torch.where(topk >= 0, topk + local.shape[0], -1)), -1
        )
        if inputs.thd_layout is None:
            attention_indices = (
                attention_indices.reshape(batch, seq_len, -1)
                .transpose(0, 1)
                .reshape(batch * seq_len, -1)
                .contiguous()
            )
    if attention_topk_length is None:
        attention_indices, attention_topk_length = fused_sparse_attention._compact_flat_topk_idxs(
            attention_indices
        )
    if q_padding_mask is None and inputs.thd_layout is not None:
        q_padding_mask = attention_topk_length == 0
    output, loss = _CSA2IndexerSparseAttn.apply(
        q,
        local,
        global_flat,
        attn_sink,
        iq,
        ik,
        w,
        window,
        attention_indices,
        attention_topk_length,
        q_padding_mask,
        starts,
        visible,
        divisor,
        ids,
        counts,
        (width, mode, block_size, softmax_scale, loss_coeff),
        (seq_len, batch) if inputs.thd_layout is None else None,
    )
    if inputs.thd_layout is None:
        output = output.reshape(seq_len, batch, -1)
    else:
        output = output.flatten(1)
    return output, loss


def fused_csa2_indexer_loss(
    inputs: CSA2IndexerInputs,
    query: Tensor,
    local_kv: Tensor,
    global_kv: Tensor,
    attn_sink: Tensor,
    window_indices: Tensor,
    global_indices: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """Apply the DSv4 KL objective over causal/candidate keys or this layer's Top-K.

    The teacher normalizes with the sliding window and sink before summing
    heads, then renormalizes over the supervised global keys. Padding does
    not enter the token divisor. Only indexer Q/K/W receive gradients.
    """
    if not HAVE_TRITON or not inputs.q.is_cuda:
        raise RuntimeError("Fused CSA2 indexer loss requires CUDA and Triton")
    q, k, w = inputs.q, inputs.k, inputs.weights
    starts, visible, valid = inputs.starts, inputs.visible, inputs.valid
    max_keys = inputs.max_keys
    if q.shape[0] == 0 or k.shape[0] == 0 or max_keys == 0:
        return (q.float().sum() + k.float().sum() + w.float().sum()) * 0.0
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or w.dtype != q.dtype:
        raise ValueError("Fused CSA2 supervision requires original BF16 projections")
    if q.shape[-2] not in (32, 64) or q.shape[-1] != 128:
        raise ValueError("Fused CSA2 supervision requires indexer H32/H64 and D128")
    if query.shape[-1] not in (16, 32, 64, 128, 256, 512):
        raise ValueError("Fused CSA2 teacher head dimension must be a power of two from 16 to 512")
    if inputs.thd_layout is None:
        seq_len, batch, heads, dim = query.shape
        teacher_q = query.detach().permute(1, 0, 2, 3).reshape(-1, heads, dim).contiguous()
        teacher_k = global_kv.detach().transpose(0, 1).reshape(-1, dim).contiguous()
        local = local_kv.detach().transpose(0, 1).reshape(-1, dim).contiguous()
        window = window_indices.reshape(q.shape[0], -1)
        offset = torch.arange(batch, device=q.device).repeat_interleave(seq_len) * local_kv.shape[0]
        window = torch.where(window >= 0, window + offset[:, None], -1).int()
        topk = global_indices.reshape(q.shape[0], -1)
    else:
        teacher_q = query.detach().contiguous()
        teacher_k = global_kv.detach().squeeze(1).contiguous()
        local = local_kv.detach().contiguous()
        window, topk = window_indices.contiguous(), global_indices.contiguous()
    counts = visible
    if sparse_loss:
        ids, width, mode, block_size = (
            topk,
            topk.shape[-1],
            (3 if inputs.thd_layout is None else 2),
            1,
        )
    elif inputs.candidates is not None:
        candidate = inputs.candidates
        ids = candidate.indices.reshape(q.shape[0], -1).contiguous()
        counts = candidate.lengths.reshape(-1).contiguous()
        width, mode, block_size = ids.shape[-1] * candidate.block_size, 1, candidate.block_size
    else:
        ids = torch.empty((q.shape[0], 0), dtype=torch.int32, device=q.device)
        width, mode, block_size = max_keys, 0, 1
    if width == 0:
        return (q.float().sum() + k.float().sum() + w.float().sum()) * 0.0
    non_compressed_lse = fused_csa_window_lse(teacher_q, local, attn_sink, window, softmax_scale)
    divisor = (
        q.new_ones((), dtype=torch.float32)
        if calculate_per_token_loss
        else valid.sum(dtype=torch.float32).clamp_min(1)
    )
    return _CSA2IndexerLoss.apply(
        q,
        k,
        w,
        teacher_q,
        teacher_k,
        non_compressed_lse,
        ids,
        counts,
        starts,
        visible,
        divisor,
        (width, mode, block_size, softmax_scale, loss_coeff),
    )
