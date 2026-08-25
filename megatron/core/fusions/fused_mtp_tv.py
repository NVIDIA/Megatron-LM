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
    def _tv_row_stats_kernel(
        draft, target, maxima, denominators, vocab_size, BLOCK_SIZE: tl.constexpr
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
            target_values = tl.load(target + row_start + cols, mask=mask, other=-float("inf")).to(
                tl.float32
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
        maxima,
        denominators,
        overlap_and_s,
        draft_above_target_bits,
        vocab_size,
        packed_vocab_size,
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
            target_values = tl.load(target + row_start + cols, mask=mask, other=0.0).to(tl.float32)
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


class _FusedVocabParallelTVDistance(torch.autograd.Function):
    """Triton TV distance with compact analytical-backward state."""

    @staticmethod
    def forward(
        ctx,
        draft_logits: Tensor,
        target_logits: Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        logits_are_vocab_sharded: bool,
    ) -> Tensor:
        """Run fused forward passes and preserve compact backward state."""
        unavailable_reason = fused_mtp_tv_unavailable_reason(draft_logits, target_logits)
        if unavailable_reason is not None:
            raise RuntimeError(f"Fused MTP TV distance is unavailable: {unavailable_reason}.")

        vocab_size = draft_logits.size(-1)
        output_shape = draft_logits.shape[:-1]
        num_rows = draft_logits.numel() // vocab_size
        maxima = torch.empty((2, num_rows), dtype=torch.float32, device=draft_logits.device)
        denominators = torch.empty_like(maxima)
        _tv_row_stats_kernel[(num_rows,)](
            draft_logits,
            target_logits,
            maxima,
            denominators,
            vocab_size=vocab_size,
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
            maxima,
            denominators,
            overlap_and_s,
            draft_above_target_bits,
            vocab_size=vocab_size,
            packed_vocab_size=packed_vocab_size,
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
        return grad_draft, None, None, None


def _fused_vocab_parallel_tv_distance(
    draft_logits: Tensor,
    target_logits: Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup],
    logits_are_vocab_sharded: bool,
) -> Tensor:
    """Compute TV distance using Triton and TP collectives."""
    return _FusedVocabParallelTVDistance.apply(
        draft_logits, target_logits.detach(), tp_group, logits_are_vocab_sharded
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
) -> Tensor:
    """Compute full-vocabulary TV distance with automatic fused/reference dispatch.

    Args:
        draft_logits: Trainable draft logits with vocabulary in the last dimension.
        target_logits: Detached target logits with the same local or global vocabulary layout.
        tp_group: Tensor-parallel group owning vocabulary shards.
        logits_are_vocab_sharded: Whether the last dimension is sharded across ``tp_group``.

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

    # Materialized sequence rolls produce a noncontiguous target view. Pack only
    # that detached input when the trainable draft otherwise supports Triton.
    if (
        not target_logits.is_contiguous()
        and fused_mtp_tv_unavailable_reason(draft_logits, draft_logits) is None
    ):
        target_logits = target_logits.detach().contiguous()

    if fused_mtp_tv_unavailable_reason(draft_logits, target_logits) is None:
        return _fused_vocab_parallel_tv_distance(
            draft_logits, target_logits, tp_group, logits_are_vocab_sharded
        )
    return _VocabParallelTVDistance.apply(
        draft_logits, target_logits.detach(), tp_group, logits_are_vocab_sharded
    )
