# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-facing fused shortcut boundaries; GPU imports are lazy.

Training covers the shortcut graph and registered native streamwise consumers,
including pruned residual/reader gradients, on a qualified GB300.
No-grad calls bypass autograd and omit all tensors saved for backward.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from .residual_replay import ResidualSlot, current_manager, retained_slot


def validate_input(residual, hidden_size, num_streams, *, inference=False):
    """Validate the local flattened token layout before launching fused work."""
    if type(num_streams) is not int or not 1 <= num_streams <= 6:
        raise ValueError("shortcut fusion requires an integer stream count in [1, 6]")
    if type(hidden_size) is not int or not 32 <= hidden_size <= 16384 or hidden_size % 32:
        raise ValueError("shortcut fusion requires H in [32, 16384], divisible by 32")
    if residual.ndim < 2 or residual.shape[-1] != num_streams * hidden_size:
        raise ValueError("shortcut fusion expects residual shape [..., S * H]")
    tokens = residual.numel() // residual.shape[-1]
    maximum = 65536 if hidden_size <= 4096 else 16384
    granularity = 1 if inference else 1024
    if tokens <= 0 or tokens % granularity or tokens > maximum:
        raise ValueError(
            "shortcut fusion requires local flattened T to be a positive "
            f"multiple of {granularity} "
            f"and at most {maximum} for H={hidden_size}; got T={tokens}"
        )
    if residual.dtype not in (torch.float32, torch.bfloat16) or not residual.is_contiguous():
        raise ValueError("shortcut fusion requires contiguous FP32 or BF16 residuals")
    if not residual.is_cuda:
        raise ValueError("shortcut fusion requires CUDA tensors")
    return tokens


class _Boundary(torch.autograd.Function):
    """Own one boundary's tensors; no workspace or handoff is stored on a module."""

    @staticmethod
    def forward(ctx, options, consumer_weight, residual, *values):
        streams, replay_manager, mxfp8_handoff, mxfp8_consumer, epsilon, update_count = options
        from .forward_kernel import forward

        # values: updates, write, retain, then one read and one gamma per reader.
        count = (len(values) - 2 - update_count) // 2
        updates = values[:update_count]
        write, retain = values[update_count : update_count + 2]
        reads = values[update_count + 2 : update_count + 2 + count]
        gammas = values[update_count + 2 + count :]
        hidden = updates[0].shape[-1]
        validate_input(residual, hidden, streams, inference=ctx is None)
        leading = residual.shape[:-1]
        if any(t.shape != (*leading, hidden) for t in updates):
            raise ValueError("shortcut updates must have the residual's leading dimensions")
        current, saved, outputs = forward(
            residual.view(-1, streams * hidden),
            tuple(t.contiguous().view(-1, hidden) for t in updates),
            write,
            retain,
            reads,
            gammas,
            mxfp8_handoff=mxfp8_handoff,
            num_streams=streams,
            epsilon=epsilon,
            inference=ctx is None,
        )
        outputs = tuple(q.view(*leading, hidden) for q in outputs)
        result = (current.view(*leading, streams * hidden), *outputs)
        if ctx is None:
            return result
        ctx.replay_link = None
        if replay_manager is not None:
            ctx.replay_link = replay_manager.register(
                residual, current, saved, prepare_columns=mxfp8_handoff
            )
        ctx.previous = (
            ResidualSlot(residual) if ctx.replay_link is None else ctx.replay_link.previous_slot
        )
        # Save the predecessor even without recomputation. The backward never
        # reconstructs it from a rounded successor.
        ctx.save_for_backward(
            residual, saved.update, saved.coefficients, *saved.gammas, *saved.rstd
        )
        ctx.boundary_spec = (
            saved.controller_sizes,
            saved.has_retention,
            streams,
            leading,
            update_count,
        )
        ctx.mxfp8_reader = None
        if mxfp8_handoff:
            from .mxfp8_handoff import Reader

            ctx.mxfp8_reader = Reader(outputs[0], saved, mxfp8_consumer)
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, carry, *dys):
        from .backward_kernel import backward
        from .kernel_utils import SavedWrite

        sizes, has_retention, streams, leading, update_count = ctx.boundary_spec
        previous, update, coefficients, *readers = ctx.saved_tensors
        ctx.previous.ensure()
        count = len(dys)
        saved = SavedWrite(
            update,
            coefficients,
            tuple(readers[:count]),
            tuple(readers[count:]),
            sizes,
            num_streams=streams,
            has_retention=has_retention,
            num_updates=update_count,
        )
        hidden = update.shape[1]
        previous = previous.view(-1, streams * hidden)
        dy = tuple(None if t is None else t.contiguous().view(-1, hidden) for t in dys)
        dc = None if carry is None else carry.contiguous().view_as(previous)
        reader = ctx.mxfp8_reader
        needs_columns = reader is not None and bool(reader.pending)
        column = (
            ctx.replay_link.columns() if needs_columns and ctx.replay_link is not None else None
        )
        columns = (column,) + (None,) * (count - 1)
        grads, _ = backward(
            saved,
            previous,
            dy,
            dc,
            save_columns=needs_columns,
            columns=columns,
            on_columns_ready=(
                (lambda columns: reader.compute_wgrad(previous, columns[0]))
                if needs_columns
                else None
            ),
        )
        if reader is not None:
            reader.finish(previous)
        # Preserve allocator publication for retained backwards while releasing
        # interval tapes/columns once their final autograd consumer has returned.
        ctx.previous = retained_slot(ctx.previous)
        ctx.replay_link = None
        grads = list(grads)
        # A missing reader must not create gradients for its private controller
        # or norm weight, even when the internal kernel materializes zero dy.
        for index, grad in enumerate(dys):
            if grad is None:
                grads[update_count + 3 + index] = None
                grads[update_count + 3 + count + index] = None
        # Memoize views so shared update gradients preserve Tensor identity.
        views = {}
        for i, grad in enumerate(grads[: update_count + 1]):
            if grad is not None:
                if id(grad) not in views:
                    views[id(grad)] = grad.view(*leading, streams * hidden if i == 0 else hidden)
                grads[i] = views[id(grad)]
        return (None, None) + tuple(
            g if needed else None for g, needed in zip(grads, ctx.needs_input_grad[2:])
        )


@lru_cache
def _trains_on(device_index: int) -> bool:
    from .kernel_utils import qualified_device

    return qualified_device(torch.cuda.get_device_properties(device_index))


def _apply_boundary(
    inputs, updates, *, num_streams, mxfp8_handoff=False, mxfp8_consumer=None, epsilon=1.0e-5
):
    mxfp8 = bool(mxfp8_handoff or mxfp8_consumer is not None)
    if not torch.is_grad_enabled():
        options = (num_streams, None, mxfp8, mxfp8_consumer, epsilon, updates)
        return _Boundary.forward(None, options, None, *inputs)
    if mxfp8_handoff and mxfp8_consumer is None:
        raise ValueError("training shortcut MXFP8 requires an explicit deferred consumer")
    residual = inputs[0]
    if residual.is_cuda and not _trains_on(residual.device.index):
        # Fail at the first boundary rather than after a whole forward pass.
        raise ValueError(
            "training through the fused shortcut seams requires a qualified GB300 (the fused "
            f"backward is GB300-only); got {torch.cuda.get_device_name(residual.device)}"
        )
    # The explicit weight edge orders AccumulateGrad after the deferred TE wgrad.
    options = (num_streams, current_manager(), mxfp8, mxfp8_consumer, epsilon, updates)
    weight = None if mxfp8_consumer is None else mxfp8_consumer.weight
    return _Boundary.apply(options, weight, *inputs)


def wide_residual_write_read_rmsnorm_quant(residual, update, write, retain, read, gamma, **options):
    """Fuse a compute-layer write and the shared expert's read/RMSNorm/quantization.

    Keyword options require ``num_streams``. ``mxfp8_handoff`` defaults to False
    and ``mxfp8_consumer`` to None.
    ``epsilon`` is the RMSNorm epsilon and defaults to 1e-5.

    ``mxfp8_handoff`` emits reader zero directly from FP32 into rowwise MXFP8;
    TELinear automatically installs the saved-activation handoff after forward.
    Its columnwise weight-gradient input is prepared during checkpoint replay,
    or from the saved predecessor when checkpoint replay is disabled.

    ``mxfp8_consumer`` also enables this format and declares that reader zero
    is consumed exclusively by that Linear, with no other gradient-producing
    uses. Its weight gradient must support deferral so the prepared column
    operand can be handed to that consumer before parameter reductions.
    Views passed to that same consumer are allowed.
    Training MXFP8 calls require this consumer; standalone column reconstruction
    is not supported. Unused residual and reader outputs are supported.
    """
    return _apply_boundary((residual, update, write, retain, read, gamma), 1, **options)


@dataclass(frozen=True)
class PendingShortcutAdd:
    """A branch output not yet written to the wide residual.

    The next layer's seam kernel writes it together with that layer's read/RMSNorm(s):
    ``shortcut_add_wide_residual_write_two_reader_rmsnorm_quant`` for a fused shortcut block,
    ``shortcut_add_wide_residual_write_read_rmsnorm_quant`` for a lone layer. A MoE output is a
    shortcut add of ``routed`` and ``shared``; a single branch update (a lone layer's output)
    leaves ``shared`` None. ``write``/``retain`` are the writing connection's logits
    (``retain`` is None for fixed identity retention).
    """

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor | None
    write: torch.Tensor
    retain: torch.Tensor | None


def shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(pending, reads, gammas, **options):
    """Fuse a pending write and the next pair's two independent read/RMSNorms.

    The write is a shortcut add (BF16 ``routed + shared``, as a MoE write) or, with
    ``pending.shared`` None, a single update. MXFP8 options have the same exclusivity contract
    as ``wide_residual_write_read_rmsnorm_quant``. Only reader zero uses MXFP8; the router
    (reader one) remains BF16.
    """
    if len(reads) != 2 or len(gammas) != 2:
        raise ValueError(
            "shortcut_add_wide_residual_write_two_reader_rmsnorm_quant requires two read "
            "controllers and two RMSNorm weights"
        )
    return _apply_pending(pending, reads, gammas, **options)


def shortcut_add_wide_residual_write_read_rmsnorm_quant(pending, read, gamma, **options):
    """Fuse a pending write and one following read/RMSNorm.

    This is the seam into a lone layer: a fused block's MoE write (a shortcut add of
    ``routed + shared``) or a lone layer's single update (``pending.shared`` None) is written
    together with the next layer's input read and RMSNorm. Options and the MXFP8 contract
    match ``wide_residual_write_read_rmsnorm_quant``.
    """
    return _apply_pending(pending, (read,), (gamma,), **options)


def _apply_pending(pending, reads, gammas, **options):
    p = pending
    updates = (p.routed,) if p.shared is None else (p.routed, p.shared)
    return _apply_boundary(
        (p.residual, *updates, p.write, p.retain, *reads, *gammas), len(updates), **options
    )
