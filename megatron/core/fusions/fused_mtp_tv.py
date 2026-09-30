# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Full-vocabulary TV distance with fused CUDA dispatch and a PyTorch fallback."""

# Triton kernel signatures and programs naturally exceed the host-code lint limits.
# pylint: disable=invalid-name,too-many-arguments,too-many-locals

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from megatron.core import parallel_state

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False


_BLOCK_SIZE = 1024


if HAVE_TRITON:

    @triton.jit
    def _load_tv_target_values(
        target,
        target_halo,
        target_row_indices,
        target_valid_rows,
        row,
        cols,
        vocab_mask,
        vocab_size,
        local_target_rows,
        target_halo_rows,
        HAS_TARGET_ROW_MAP: tl.constexpr,
    ):
        """Load one target row from local storage, a compact halo, or logical zeros."""
        if HAS_TARGET_ROW_MAP:
            target_row = tl.load(target_row_indices + row).to(tl.int64)
            target_is_valid = tl.load(target_valid_rows + row)
            target_is_valid = (
                target_is_valid
                & (target_row >= 0)
                & (target_row < local_target_rows + target_halo_rows)
            )
            safe_target_row = tl.where(target_is_valid, target_row, 0)
            target_row_ptr = tl.where(
                safe_target_row < local_target_rows,
                target + safe_target_row * vocab_size,
                target_halo + (safe_target_row - local_target_rows) * vocab_size,
            )
            target_values = tl.load(
                target_row_ptr + cols, mask=vocab_mask & target_is_valid, other=-float("inf")
            )
            return tl.where(target_is_valid, target_values.to(tl.float32), 0.0)

        return tl.load(target + row * vocab_size + cols, mask=vocab_mask, other=-float("inf")).to(
            tl.float32
        )

    @triton.jit
    def _tv_row_stats_kernel(
        draft,
        target,
        target_halo,
        target_row_indices,
        target_valid_rows,
        maxima,
        denominators,
        vocab_size,
        local_target_rows,
        target_halo_rows,
        HAS_TARGET_ROW_MAP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute stable local softmax maxima and denominators for one row."""
        row = tl.program_id(0).to(tl.int64)
        row_start = row * vocab_size
        draft_max = -float("inf")
        target_max = -float("inf")
        draft_sum = 0.0
        target_sum = 0.0

        for col_start in range(0, vocab_size, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            draft_values = tl.load(draft + row_start + cols, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            target_values = _load_tv_target_values(
                target,
                target_halo,
                target_row_indices,
                target_valid_rows,
                row,
                cols,
                mask,
                vocab_size,
                local_target_rows,
                target_halo_rows,
                HAS_TARGET_ROW_MAP,
            )

            draft_tile_max = tl.max(draft_values, axis=0)
            target_tile_max = tl.max(target_values, axis=0)
            draft_new_max = tl.maximum(draft_max, draft_tile_max)
            target_new_max = tl.maximum(target_max, target_tile_max)
            draft_old_scale = tl.where(
                draft_max == -float("inf"), 0.0, tl.exp(draft_max - draft_new_max)
            )
            target_old_scale = tl.where(
                target_max == -float("inf"), 0.0, tl.exp(target_max - target_new_max)
            )
            draft_tile_sum = tl.sum(
                tl.where(mask, tl.exp(draft_values - draft_new_max), 0.0), axis=0
            )
            target_tile_sum = tl.sum(
                tl.where(mask, tl.exp(target_values - target_new_max), 0.0), axis=0
            )
            draft_sum = draft_sum * draft_old_scale + draft_tile_sum
            target_sum = target_sum * target_old_scale + target_tile_sum
            draft_max = draft_new_max
            target_max = target_new_max

        num_rows = tl.num_programs(0)
        tl.store(maxima + row, draft_max)
        tl.store(maxima + num_rows + row, target_max)
        tl.store(denominators + row, draft_sum)
        tl.store(denominators + num_rows + row, target_sum)

    @triton.jit
    def _tv_rescale_denominators_kernel(
        local_maxima, global_maxima, denominators, num_values, BLOCK_SIZE: tl.constexpr
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_values
        local_max = tl.load(local_maxima + offsets, mask=mask)
        global_max = tl.load(global_maxima + offsets, mask=mask)
        denominator = tl.load(denominators + offsets, mask=mask)
        denominator *= tl.exp(local_max - global_max)
        tl.store(denominators + offsets, denominator, mask=mask)

    @triton.jit
    def _tv_overlap_kernel(
        draft,
        target,
        target_halo,
        target_row_indices,
        target_valid_rows,
        maxima,
        denominators,
        overlap_and_s,
        draft_above_target_bits,
        vocab_size,
        packed_vocab_size,
        local_target_rows,
        target_halo_rows,
        HAS_TARGET_ROW_MAP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute local overlap and the draft mass at or below the target."""
        row = tl.program_id(0).to(tl.int64)
        num_rows = tl.num_programs(0)
        row_start = row * vocab_size
        draft_max = tl.load(maxima + row)
        target_max = tl.load(maxima + num_rows + row)
        draft_denominator = tl.load(denominators + row)
        target_denominator = tl.load(denominators + num_rows + row)
        overlap = 0.0
        draft_mass_below_target = 0.0

        bit_offsets = tl.arange(0, 8)
        for col_start in range(0, vocab_size, BLOCK_SIZE):
            byte_offsets = tl.arange(0, BLOCK_SIZE // 8)
            cols = col_start + byte_offsets[:, None] * 8 + bit_offsets[None, :]
            mask = cols < vocab_size
            draft_values = tl.load(draft + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            target_values = _load_tv_target_values(
                target,
                target_halo,
                target_row_indices,
                target_valid_rows,
                row,
                cols,
                mask,
                vocab_size,
                local_target_rows,
                target_halo_rows,
                HAS_TARGET_ROW_MAP,
            )
            draft_prob = tl.exp(draft_values - draft_max) / draft_denominator
            target_prob = tl.exp(target_values - target_max) / target_denominator
            overlap += tl.sum(
                tl.sum(tl.where(mask, tl.minimum(draft_prob, target_prob), 0.0), axis=1), axis=0
            )
            draft_mass_below_target += tl.sum(
                tl.sum(tl.where(mask & (draft_prob <= target_prob), draft_prob, 0.0), axis=1),
                axis=0,
            )
            packed_above_target = tl.sum(
                tl.where(mask & (draft_prob > target_prob), 1 << bit_offsets[None, :], 0), axis=1
            )
            packed_offsets = col_start // 8 + byte_offsets
            tl.store(
                draft_above_target_bits + row * packed_vocab_size + packed_offsets,
                packed_above_target,
                mask=packed_offsets < packed_vocab_size,
            )

        tl.store(overlap_and_s + row, overlap)
        tl.store(overlap_and_s + num_rows + row, draft_mass_below_target)

    @triton.jit
    def _tv_finalize_kernel(
        overlap, output, clamp_gradient_mask, num_rows, BLOCK_SIZE: tl.constexpr
    ):
        rows = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = rows < num_rows
        raw_distance = 1.0 - tl.load(overlap + rows, mask=mask)
        distance = tl.maximum(0.0, tl.minimum(1.0, raw_distance))
        tl.store(output + rows, distance, mask=mask)
        tl.store(
            clamp_gradient_mask + rows, (raw_distance >= 0.0) & (raw_distance <= 1.0), mask=mask
        )

    @triton.jit
    def _tv_backward_kernel(
        draft,
        draft_maxima,
        draft_denominators,
        draft_mass_below_target,
        draft_above_target_bits,
        clamp_gradient_mask,
        grad_output,
        grad_draft,
        vocab_size,
        packed_vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        row_start = row * vocab_size
        draft_max = tl.load(draft_maxima + row)
        draft_denominator = tl.load(draft_denominators + row)
        draft_mass = tl.load(draft_mass_below_target + row)
        row_gradient = tl.load(grad_output + row).to(tl.float32)
        row_gradient *= tl.load(clamp_gradient_mask + row).to(tl.float32)

        for col_start in range(0, vocab_size, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            draft_values = tl.load(draft + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            draft_prob = tl.exp(draft_values - draft_max) / draft_denominator
            packed_above_target = tl.load(
                draft_above_target_bits + row * packed_vocab_size + cols // 8, mask=mask, other=0
            ).to(tl.int32)
            draft_above_target = (packed_above_target >> (cols % 8)) & 1
            gradient = draft_prob * (draft_mass - 1.0 + draft_above_target.to(tl.float32))
            gradient *= row_gradient
            tl.store(grad_draft + row_start + cols, gradient, mask=mask)


def fused_mtp_tv_unavailable_reason(  # pylint: disable=too-many-return-statements
    draft_logits: Tensor, target_logits: Tensor
) -> Optional[str]:
    """Return why the fused kernel cannot consume the two logits tensors."""
    if not HAVE_TRITON:
        return "Triton is not available"
    if draft_logits.shape != target_logits.shape:
        return "draft and target logits have different shapes"
    if draft_logits.device != target_logits.device:
        return "draft and target logits are on different devices"
    if not draft_logits.is_cuda:
        return "logits are not CUDA tensors"
    if draft_logits.dtype not in (torch.bfloat16, torch.float32):
        return f"draft dtype {draft_logits.dtype} is not supported"
    if target_logits.dtype != draft_logits.dtype:
        return "draft and target logits have different dtypes"
    if draft_logits.ndim == 0 or draft_logits.size(-1) == 0:
        return "logits must have a non-empty vocabulary dimension"
    if draft_logits.numel() == 0:
        return "logits must have at least one row"
    if not draft_logits.is_contiguous() or not target_logits.is_contiguous():
        return "logits are not contiguous"
    return None


def _validate_target_row_addressing(
    draft_logits: Tensor,
    target_logits: Tensor,
    target_row_indices: Optional[Tensor],
    target_valid_rows: Optional[Tensor],
    target_halo_logits: Optional[Tensor],
) -> bool:
    """Validate optional direct target-row addressing and return whether it is active."""
    has_target_row_map = target_row_indices is not None
    if has_target_row_map != (target_valid_rows is not None):
        raise ValueError("Target row indices and validity must be provided together.")
    if target_halo_logits is not None and not has_target_row_map:
        raise ValueError("Target halo logits require target row indices and validity.")
    if not has_target_row_map:
        return False

    assert target_row_indices is not None
    assert target_valid_rows is not None
    if target_row_indices.shape != draft_logits.shape[:-1]:
        raise ValueError(
            "Target row indices must match the draft leading dimensions, got "
            f"{tuple(target_row_indices.shape)} and {tuple(draft_logits.shape[:-1])}."
        )
    if target_valid_rows.shape != target_row_indices.shape:
        raise ValueError(
            "Target row validity must match target row indices, got "
            f"{tuple(target_valid_rows.shape)} and {tuple(target_row_indices.shape)}."
        )
    if target_row_indices.device != draft_logits.device:
        raise ValueError("Target row indices must be on the logits device.")
    if target_valid_rows.device != draft_logits.device:
        raise ValueError("Target row validity must be on the logits device.")
    if target_row_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Target row indices must use torch.int32 or torch.int64.")
    if target_valid_rows.dtype != torch.bool:
        raise ValueError("Target row validity must use torch.bool.")
    if not target_row_indices.is_contiguous() or not target_valid_rows.is_contiguous():
        raise ValueError("Target row metadata must be contiguous.")

    if target_halo_logits is not None:
        if target_halo_logits.device != target_logits.device:
            raise ValueError("Target halo logits must be on the target-logits device.")
        if target_halo_logits.dtype != target_logits.dtype:
            raise ValueError("Target halo logits must use the target-logits dtype.")
        if target_halo_logits.ndim != target_logits.ndim:
            raise ValueError("Target halo logits must have the target-logits rank.")
        if target_halo_logits.shape[1:] != target_logits.shape[1:]:
            raise ValueError("Target halo logits must match non-sequence target dimensions.")
        if not target_halo_logits.is_contiguous():
            raise ValueError("Target halo logits must be contiguous.")
    return True


class _FusedVocabParallelTVDistance(torch.autograd.Function):
    """Triton TV distance with compact analytical-backward state."""

    @staticmethod
    def forward(
        ctx,
        draft_logits: Tensor,
        target_logits: Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        logits_are_vocab_sharded: bool,
        target_row_indices: Optional[Tensor],
        target_valid_rows: Optional[Tensor],
        target_halo_logits: Optional[Tensor],
    ) -> Tensor:
        """Run fused forward passes and preserve compact backward state."""
        unavailable_reason = fused_mtp_tv_unavailable_reason(draft_logits, target_logits)
        if unavailable_reason is not None:
            raise RuntimeError(f"Fused MTP TV distance is unavailable: {unavailable_reason}.")

        has_target_row_map = _validate_target_row_addressing(
            draft_logits, target_logits, target_row_indices, target_valid_rows, target_halo_logits
        )
        vocab_size = draft_logits.size(-1)
        output_shape = draft_logits.shape[:-1]
        num_rows = draft_logits.numel() // vocab_size
        maxima = torch.empty((2, num_rows), dtype=torch.float32, device=draft_logits.device)
        denominators = torch.empty_like(maxima)
        target_row_indices_arg = target_row_indices if has_target_row_map else draft_logits
        target_valid_rows_arg = target_valid_rows if has_target_row_map else draft_logits
        target_halo_logits_arg = (
            target_halo_logits if target_halo_logits is not None else target_logits
        )
        local_target_rows = target_logits.numel() // vocab_size
        target_halo_rows = (
            target_halo_logits.numel() // vocab_size if target_halo_logits is not None else 0
        )
        _tv_row_stats_kernel[(num_rows,)](
            draft_logits,
            target_logits,
            target_halo_logits_arg,
            target_row_indices_arg,
            target_valid_rows_arg,
            maxima,
            denominators,
            vocab_size=vocab_size,
            local_target_rows=local_target_rows,
            target_halo_rows=target_halo_rows,
            HAS_TARGET_ROW_MAP=has_target_row_map,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
        )

        tp_size = torch.distributed.get_world_size(group=tp_group) if tp_group is not None else 1
        if logits_are_vocab_sharded and tp_size > 1:
            local_maxima = maxima
            maxima = local_maxima.clone()
            torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX, group=tp_group)
            num_stats = maxima.numel()
            _tv_rescale_denominators_kernel[(triton.cdiv(num_stats, 256),)](
                local_maxima,
                maxima,
                denominators,
                num_values=num_stats,
                BLOCK_SIZE=256,
                num_warps=4,
            )
            torch.distributed.all_reduce(
                denominators, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )

        overlap_and_s = torch.empty_like(maxima)
        packed_vocab_size = (vocab_size + 7) // 8
        draft_above_target_bits = torch.empty(
            (num_rows, packed_vocab_size), dtype=torch.uint8, device=draft_logits.device
        )
        _tv_overlap_kernel[(num_rows,)](
            draft_logits,
            target_logits,
            target_halo_logits_arg,
            target_row_indices_arg,
            target_valid_rows_arg,
            maxima,
            denominators,
            overlap_and_s,
            draft_above_target_bits,
            vocab_size=vocab_size,
            packed_vocab_size=packed_vocab_size,
            local_target_rows=local_target_rows,
            target_halo_rows=target_halo_rows,
            HAS_TARGET_ROW_MAP=has_target_row_map,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
        )
        if logits_are_vocab_sharded and tp_size > 1:
            torch.distributed.all_reduce(
                overlap_and_s, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )

        output = torch.empty(num_rows, dtype=torch.float32, device=draft_logits.device)
        clamp_gradient_mask = torch.empty(num_rows, dtype=torch.bool, device=draft_logits.device)
        _tv_finalize_kernel[(triton.cdiv(num_rows, 256),)](
            overlap_and_s,
            output,
            clamp_gradient_mask,
            num_rows=num_rows,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        ctx.save_for_backward(
            draft_logits,
            maxima[0],
            denominators[0],
            overlap_and_s[1],
            draft_above_target_bits,
            clamp_gradient_mask,
        )
        ctx.vocab_size = vocab_size
        ctx.packed_vocab_size = packed_vocab_size
        return output.view(output_shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Apply the analytical draft-logit gradient from packed metadata."""
        (
            draft_logits,
            draft_maxima,
            draft_denominators,
            draft_mass_below_target,
            draft_above_target_bits,
            clamp_gradient_mask,
        ) = ctx.saved_tensors
        num_rows = draft_logits.numel() // ctx.vocab_size
        grad_draft = torch.empty_like(draft_logits)
        _tv_backward_kernel[(num_rows,)](
            draft_logits,
            draft_maxima,
            draft_denominators,
            draft_mass_below_target,
            draft_above_target_bits,
            clamp_gradient_mask,
            grad_output.contiguous(),
            grad_draft,
            vocab_size=ctx.vocab_size,
            packed_vocab_size=ctx.packed_vocab_size,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
        )
        return grad_draft, None, None, None, None, None, None


def _fused_vocab_parallel_tv_distance(
    draft_logits: Tensor,
    target_logits: Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup],
    logits_are_vocab_sharded: bool,
    *,
    target_row_indices: Optional[Tensor] = None,
    target_valid_rows: Optional[Tensor] = None,
    target_halo_logits: Optional[Tensor] = None,
) -> Tensor:
    """Compute TV distance using Triton, optional target-row addressing, and TP collectives."""
    return _FusedVocabParallelTVDistance.apply(
        draft_logits,
        target_logits.detach(),
        tp_group,
        logits_are_vocab_sharded,
        target_row_indices,
        target_valid_rows,
        target_halo_logits.detach() if target_halo_logits is not None else None,
    )


def _validate_vocab_parallel_tv_group(
    tp_group: Optional[torch.distributed.ProcessGroup], logits_are_vocab_sharded: bool
) -> None:
    """Reject vocab-sharded TV before a missing TP group can change its normalization."""
    # Compatibility guard retained from the original MTP-local dispatcher. New
    # callers should pass the explicit TP group instead of relying on MPU state.
    if logits_are_vocab_sharded and tp_group is None and parallel_state.is_initialized():
        initialized_tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if initialized_tp_size > 1:
            raise ValueError(
                "tp_group must be provided for TV distance over vocab-sharded logits "
                f"when tensor parallel size is {initialized_tp_size}."
            )


class _VocabParallelTVDistance(torch.autograd.Function):
    """Full-vocabulary TV distance with a TP-aware analytical backward.

    The implementation follows Algorithms 1 and 2 in the Bebop paper
    (https://arxiv.org/abs/2606.12370). It intentionally keeps the target
    distribution detached and returns gradients for draft logits only.

    This PyTorch implementation establishes the mathematical contract and is
    retained as the oracle and fallback for unsupported fused-kernel inputs.
    """

    @staticmethod
    def forward(
        ctx,
        draft_logits: Tensor,
        target_logits: Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        logits_are_vocab_sharded: bool,
    ) -> Tensor:
        """Compute the TV distance and save the draft-side backward state."""
        if draft_logits.shape != target_logits.shape:
            raise ValueError(
                "Draft and target logits must have identical shapes, got "
                f"{tuple(draft_logits.shape)} and {tuple(target_logits.shape)}."
            )
        if draft_logits.device != target_logits.device:
            raise ValueError("Draft and target logits must be on the same device.")

        tp_size = torch.distributed.get_world_size(group=tp_group) if tp_group is not None else 1
        _validate_vocab_parallel_tv_group(tp_group, logits_are_vocab_sharded)

        draft_logits_fp32 = draft_logits.float()
        target_logits_fp32 = target_logits.float()
        maxima = torch.stack(
            (draft_logits_fp32.max(dim=-1).values, target_logits_fp32.max(dim=-1).values)
        )
        if logits_are_vocab_sharded and tp_size > 1:
            torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        draft_max, target_max = maxima.unbind(dim=0)

        draft_exp = torch.exp(draft_logits_fp32 - draft_max.unsqueeze(-1))
        target_exp = torch.exp(target_logits_fp32 - target_max.unsqueeze(-1))
        denominators = torch.stack((draft_exp.sum(dim=-1), target_exp.sum(dim=-1)))
        if logits_are_vocab_sharded and tp_size > 1:
            torch.distributed.all_reduce(
                denominators, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
        draft_denominator, target_denominator = denominators.unbind(dim=0)

        draft_prob = draft_exp / draft_denominator.unsqueeze(-1)
        target_prob = target_exp / target_denominator.unsqueeze(-1)
        draft_below_target = draft_prob <= target_prob
        overlap_and_s = torch.stack(
            (
                torch.minimum(draft_prob, target_prob).sum(dim=-1),
                (draft_prob * draft_below_target).sum(dim=-1),
            )
        )
        if logits_are_vocab_sharded and tp_size > 1:
            torch.distributed.all_reduce(
                overlap_and_s, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
        overlap, s = overlap_and_s.unbind(dim=0)

        raw_tv_distance = 1.0 - overlap
        tv_distance = raw_tv_distance.clamp(min=0.0, max=1.0)
        clamp_gradient_mask = (raw_tv_distance >= 0.0) & (raw_tv_distance <= 1.0)
        ctx.save_for_backward(
            draft_logits,
            draft_max,
            draft_denominator,
            s,
            draft_prob > target_prob,
            clamp_gradient_mask,
        )
        return tv_distance

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Apply the analytical Bebop gradient to the draft logits only."""
        draft_logits, draft_max, draft_denominator, s, draft_above_target, clamp_gradient_mask = (
            ctx.saved_tensors
        )

        draft_prob = torch.exp(draft_logits.float() - draft_max.unsqueeze(-1))
        draft_prob = draft_prob / draft_denominator.unsqueeze(-1)
        grad_draft = draft_prob * (s.unsqueeze(-1) - 1.0 + draft_above_target.to(draft_prob.dtype))
        grad_draft *= (grad_output * clamp_gradient_mask).unsqueeze(-1)
        return grad_draft.to(draft_logits.dtype), None, None, None


def vocab_parallel_tv_distance(
    draft_logits: Tensor,
    target_logits: Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    logits_are_vocab_sharded: bool = True,
    *,
    target_row_indices: Optional[Tensor] = None,
    target_valid_rows: Optional[Tensor] = None,
    target_halo_logits: Optional[Tensor] = None,
) -> Tensor:
    """Compute full-vocabulary TV distance with automatic fused/reference dispatch.

    Args:
        draft_logits: Trainable draft logits with vocabulary in the last dimension.
        target_logits: Detached local target-logit storage with the same vocabulary layout.
        tp_group: Tensor-parallel group owning vocabulary shards.
        logits_are_vocab_sharded: Whether the last dimension is sharded across ``tp_group``.
        target_row_indices: Optional flattened row in ``target_logits`` or its halo for each
            draft row.
        target_valid_rows: Validity mask paired with ``target_row_indices``. Invalid and
            out-of-bounds rows select a safe all-zero target-logit vector.
        target_halo_logits: Optional compact rows logically appended to ``target_logits``.

    Returns:
        A FP32 tensor with the vocabulary dimension removed. Gradients flow only to
        ``draft_logits``.
    """
    if draft_logits.shape != target_logits.shape:
        raise ValueError(
            "Draft and target logits must have identical shapes, got "
            f"{tuple(draft_logits.shape)} and {tuple(target_logits.shape)}."
        )
    if draft_logits.device != target_logits.device:
        raise ValueError("Draft and target logits must be on the same device.")
    _validate_vocab_parallel_tv_group(tp_group, logits_are_vocab_sharded)
    has_target_row_map = _validate_target_row_addressing(
        draft_logits, target_logits, target_row_indices, target_valid_rows, target_halo_logits
    )

    # Materialized sequence rolls produce a noncontiguous target view. Pack only
    # that detached input when the trainable draft otherwise supports Triton.
    if (
        not has_target_row_map
        and not target_logits.is_contiguous()
        and fused_mtp_tv_unavailable_reason(draft_logits, draft_logits) is None
    ):
        target_logits = target_logits.detach().contiguous()

    if fused_mtp_tv_unavailable_reason(draft_logits, target_logits) is None:
        return _fused_vocab_parallel_tv_distance(
            draft_logits,
            target_logits,
            tp_group,
            logits_are_vocab_sharded,
            target_row_indices=target_row_indices,
            target_valid_rows=target_valid_rows,
            target_halo_logits=target_halo_logits,
        )

    if has_target_row_map:
        assert target_row_indices is not None
        assert target_valid_rows is not None
        vocab_size = draft_logits.size(-1)
        local_targets = target_logits.detach().reshape(-1, vocab_size)
        flat_indices = target_row_indices.reshape(-1).long()
        flat_valid = target_valid_rows.reshape(-1)
        num_local_rows = local_targets.size(0)
        halo_targets = None
        num_halo_rows = 0
        if target_halo_logits is not None:
            halo_targets = target_halo_logits.detach().reshape(-1, vocab_size)
            num_halo_rows = halo_targets.size(0)

        num_addressable_rows = num_local_rows + num_halo_rows
        flat_valid = flat_valid & (flat_indices >= 0) & (flat_indices < num_addressable_rows)
        safe_local_indices = torch.where(
            flat_valid & (flat_indices < num_local_rows), flat_indices, 0
        )
        materialized_target = local_targets.index_select(0, safe_local_indices)
        if halo_targets is not None and num_halo_rows > 0:
            selects_halo = flat_valid & (flat_indices >= num_local_rows)
            safe_halo_indices = torch.where(selects_halo, flat_indices - num_local_rows, 0)
            selected_halo = halo_targets.index_select(0, safe_halo_indices)
            materialized_target = torch.where(
                selects_halo.unsqueeze(1), selected_halo, materialized_target
            )
        materialized_target.masked_fill_(~flat_valid.unsqueeze(1), 0)
        target_logits = materialized_target.view_as(draft_logits)

    return _VocabParallelTVDistance.apply(
        draft_logits, target_logits.detach(), tp_group, logits_are_vocab_sharded
    )
