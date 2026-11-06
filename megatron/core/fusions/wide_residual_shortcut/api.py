# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-facing fused shortcut seams; GPU imports are lazy.

Training (a qualified GB300) differentiates the seam, including pruned residual/reader
gradients; no-grad calls bypass autograd and save nothing for backward.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from .mxfp8_handoff import Reader
from .residual_replay import graph_retained


def validate_input(residual, hidden_size, num_streams, *, inference=False):
    """Validate the local flattened token layout before launching fused work."""
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
    return tokens


class _Boundary(torch.autograd.Function):
    """One seam; its tensors and MXFP8 handoff are call-local, never stored on a module."""

    @staticmethod
    def forward(ctx, options, consumer_weight, residual, *values):
        """Run the seam; ``ctx`` None is an inference call without autograd state."""
        streams, replay, mxfp8_handoff, mxfp8_consumer, epsilon, update_count = options
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
        # Under checkpoint replay the backward takes the input residual (and reader zero's
        # MXFP8 columns) from the replayed interval instead of keeping it alive; it never
        # reconstructs the predecessor from a rounded successor.
        ctx.replay = (
            None
            if replay is None
            else replay.record(residual, current, saved, prepare_columns=mxfp8_handoff)
        )
        ctx.save_for_backward(
            residual if ctx.replay is None else None,
            saved.update,
            saved.coefficients,
            *saved.gammas,
            *saved.rstd,
        )
        ctx.spec = replace(saved, update=None, coefficients=None, gammas=(), rstd=())
        ctx.leading = leading
        ctx.mxfp8_reader = Reader(outputs[0], saved, mxfp8_consumer) if mxfp8_handoff else None
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, carry, *dys):
        """Fused seam backward; replay supplies the predecessor and any MXFP8 columns."""
        from .backward_kernel import backward

        streams, update_count, leading = ctx.spec.num_streams, ctx.spec.update_count, ctx.leading
        previous, update, coefficients, *readers = ctx.saved_tensors
        column = None
        if ctx.replay is not None:
            previous, column = ctx.replay.take()
            # Autograd frees saved tensors after backward but not ctx attributes, and the graph's
            # nodes may outlive it (until the next forward): release the interval explicitly.
            if not graph_retained():
                ctx.replay = None
        count = len(dys)
        saved = replace(
            ctx.spec,
            update=update,
            coefficients=coefficients,
            gammas=tuple(readers[:count]),
            rstd=tuple(readers[count:]),
        )
        hidden = update.shape[1]
        previous = previous.view(-1, streams * hidden)
        dy = tuple(None if t is None else t.contiguous().view(-1, hidden) for t in dys)
        dc = None if carry is None else carry.contiguous().view_as(previous)
        reader = ctx.mxfp8_reader
        needs = reader is not None and bool(reader.pending)
        grads, _ = backward(
            saved,
            previous,
            dy,
            dc,
            save_columns=needs,
            columns=(column if needs else None,) + (None,) * (count - 1),
            on_columns_ready=(lambda c: reader.compute_wgrad(previous, c[0])) if needs else None,
        )
        if reader is not None:
            reader.finish(previous)
        grads = list(grads)
        # A missing reader gets no controller or norm-weight gradient, although the kernel
        # materializes a zero dy for it.
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
    p,
    reads,
    gammas,
    *,
    num_streams,
    mxfp8_handoff=False,
    mxfp8_consumer=None,
    epsilon=1.0e-5,
    replay=None,
):
    """``replay`` (a ReplayInterval, training only) records this write for checkpoint replay."""
    updates = (p.routed,) if p.shared is None else (p.routed, p.shared)
    inputs = (p.residual, *updates, p.write, p.retain, *reads, *gammas)
    mxfp8 = bool(mxfp8_handoff or mxfp8_consumer is not None)
    if not torch.is_grad_enabled():
        options = (num_streams, None, mxfp8, mxfp8_consumer, epsilon, len(updates))
        return _Boundary.forward(None, options, None, *inputs)
    if mxfp8_handoff and mxfp8_consumer is None:
        raise ValueError("training shortcut MXFP8 requires an explicit deferred consumer")
    if p.residual.is_cuda and not _trains_on(p.residual.device.index):
        # Fail at the first seam rather than after a whole forward pass.
        raise ValueError(
            "training through the fused shortcut seams requires a qualified GB300 (the fused "
            f"backward is GB300-only); got {torch.cuda.get_device_name(p.residual.device)}"
        )
    # The explicit weight edge orders AccumulateGrad after the deferred TE wgrad.
    options = (num_streams, replay, mxfp8, mxfp8_consumer, epsilon, len(updates))
    weight = None if mxfp8_consumer is None else mxfp8_consumer.weight
    return _Boundary.apply(options, weight, *inputs)


@dataclass(frozen=True)
class PendingWideResidualWrites:
    """A branch output not yet written to the wide residual.

    The next seam kernel writes it together with that layer's read/RMSNorm(s). A MoE output is
    a shortcut add of ``routed`` and ``shared``; a single branch update (a lone layer's output)
    leaves ``shared`` None. ``write``/``retain`` are the writing connection's logits
    (``retain`` None is fixed identity retention).
    """

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor | None
    write: torch.Tensor
    retain: torch.Tensor | None


def shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(pending, reads, gammas, **options):
    """Write a pending update and emit the next block's two independent read/RMSNorms.

    Options match ``shortcut_add_wide_residual_write_read_rmsnorm_quant``; only reader zero may
    use MXFP8, the router input (reader one) stays BF16.
    """
    if len(reads) != 2 or len(gammas) != 2:
        raise ValueError(
            "the two-reader seam requires two read controllers and two RMSNorm weights"
        )
    return _apply_boundary(pending, reads, gammas, **options)


def shortcut_add_wide_residual_write_read_rmsnorm_quant(pending, read, gamma, **options):
    """Write a pending update and emit one following read/RMSNorm.

    Keyword options: ``num_streams`` (required), ``epsilon`` (RMSNorm, default 1e-5),
    ``mxfp8_handoff`` and ``mxfp8_consumer``. Either MXFP8 option emits reader zero directly
    from FP32 as rowwise MXFP8. ``mxfp8_consumer`` (required in training) declares that a
    TELinear with deferrable weight gradients is that reader's exclusive gradient-producing
    consumer (views of it are allowed): TELinear installs the saved-activation handoff, and
    the columnwise wgrad operand comes from checkpoint replay or from the saved predecessor.
    """
    return _apply_boundary(pending, (read,), (gamma,), **options)
