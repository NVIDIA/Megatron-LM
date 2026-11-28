# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fused seam op (``fused_seam``); GPU imports are lazy.

Training (a qualified GB300) differentiates the seam, including pruned residual/reader
gradients, and records it into a ReplayInterval; no-grad calls bypass autograd and save nothing
for backward. The op knows tensors, formats and kernel limits, not the modules calling it.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from .residual_replay import ReplayInterval

# MXFP8 handoff: a seam writes a reader as rowwise MXFP8 for its consumer Linears' forward GEMMs,
# as BF16 for other modules, or both. A reader with two consumers returns one output per
# consumer, in its format, so each consumer's gradient reaches the backward kernel separately
# (summed there). In training an MXFP8 reader's ColumnSlot receives its columnwise operand from
# the replay, and its consumers' weight-gradient GEMMs unpack it from TE's saved slot
# (residual_replay.install_columns).


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
    def forward(ctx, options, residual, routed, shared, write, retain, *reader_params):
        """Run the seam; ``ctx`` None is an inference call without autograd state."""
        (
            streams,
            replay,
            uses,
            slots,
            epsilon,
            batch_invariant,
            fuse,
            update_formats,
            compact_scales_for_tp,
            compact_grad_scales_for_tp,
        ) = options
        formats = tuple(use[0] if use[0] == use[-1] else "both" for use in uses)
        from .forward_kernel import forward

        count = len(uses)
        reads, gammas = reader_params[:count], reader_params[count:]
        updates = (routed,) if shared is None else (routed, shared)
        hidden = routed.shape[-1]
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
            outputs=formats,
            compact_scales_for_tp=compact_scales_for_tp,
            num_streams=streams,
            epsilon=epsilon,
            inference=ctx is None,
            batch_invariant=batch_invariant,
        )
        result = [current.view_as(residual)]
        for use, output, slot in zip(uses, outputs, slots):
            mx, plain = output if isinstance(output, tuple) else (output, output)
            if ctx is not None and slot is not None:
                # Detached views survive checkpointed GEMM recomputation. Pin the slot
                # to the underlying rowwise data they all share.
                data = mx._rowwise_data
                (data if data._base is None else data._base)._shortcut_columns = slot
            for index, fmt in enumerate(use):
                if index and fmt == use[0]:
                    # A distinct autograd output for a second consumer of the same tensor.
                    result.append(result[-1].detach())
                else:
                    tensor = mx if fmt == "mxfp8" else plain
                    result.append(tensor.view(*leading, hidden))
        result = tuple(result)
        if ctx is None:
            return result
        # The interval keeps or replays this seam's input residual (and fills its MXFP8 readers'
        # column slots): the backward never reconstructs it from a rounded successor.
        ctx.replay, ctx.index = replay, replay.record(residual, current, saved, slots)
        ctx.save_for_backward(saved.update, saved.coefficients, *saved.gammas, *saved.rstd)
        # Autograd owns the saved tensors. Keep only the other SavedWrite fields here.
        ctx.write_metadata = (
            saved.controller_sizes,
            streams,
            saved.has_retention,
            len(updates),
            saved.compact_scales_for_tp,
        )
        ctx.leading, ctx.uses, ctx.update_formats = leading, uses, update_formats
        ctx.compact_grad_scales_for_tp = compact_grad_scales_for_tp
        # Controller and norm weight parameters whose gradients may accumulate in the kernel.
        ctx.parameters = (
            (write, retain, *reader_params) if fuse else (None,) * (2 + len(reader_params))
        )
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, carry, *flat):
        """Fused seam backward; the replay interval supplies the predecessor."""
        from .backward_kernel import backward
        from .kernel_utils import SavedWrite

        update, coefficients, *readers = ctx.saved_tensors
        count = len(ctx.uses)
        saved = SavedWrite(
            update,
            coefficients,
            tuple(readers[:count]),
            tuple(readers[count:]),
            *ctx.write_metadata,
        )
        hidden = update.shape[1]
        streams, update_count, leading = saved.num_streams, saved.update_count, ctx.leading
        previous = ctx.replay.take(ctx.index).view(-1, streams * hidden)
        # Each reader contributes zero, one or two gradients; the kernel adds a pair in FP32.
        dy = []
        for group in _reader_groups(ctx.uses, flat):
            live = tuple(g.contiguous().view(-1, hidden) for g in group if g is not None)
            dy.append(live[0] if len(live) == 1 else live or None)
        dc = None if carry is None else carry.contiguous().view_as(previous)
        # A missing reader gets no read or norm weight gradient (its parameters' main_grad stays
        # untouched), although the kernel materializes a zero dy for it.
        active = (True, True) + tuple(g is not None for g in dy) * 2
        targets = tuple(
            _main_grad_target(p) if used else None for p, used in zip(ctx.parameters, active)
        )
        grads, _ = backward(
            saved,
            previous,
            dy,
            dc,
            main_grads=targets,
            update_formats=ctx.update_formats,
            compact_scales_for_tp=ctx.compact_grad_scales_for_tp,
        )
        if ctx.index == 0 and ctx.replay.previous is not None:
            # This seam applies the previous interval's last write: replay that interval now,
            # before any of its backward GEMMs (which need gradients this seam produces).
            ctx.replay.previous.replay()
            ctx.replay.previous = None
        grads = list(grads)
        for j, (parameter, target, used) in enumerate(zip(ctx.parameters, targets, active)):
            if not used:
                grads[update_count + 1 + j] = None
            elif target is not None:
                grads[update_count + 1 + j] = _accumulated(parameter)
        # Memoize views so shared update gradients preserve Tensor identity.
        views = {}
        for i, grad in enumerate(grads[: update_count + 1]):
            if grad is not None:
                if id(grad) not in views:
                    views[id(grad)] = grad.view(*leading, streams * hidden if i == 0 else hidden)
                grads[i] = views[id(grad)]
        if update_count == 1:
            grads.insert(2, None)  # shared is an explicit autograd input even for a single update.
        return (None,) + tuple(
            g if needed else None for g, needed in zip(grads, ctx.needs_input_grad[1:])
        )


def _main_grad_target(parameter):
    """``(main_grad, dtype)`` when the kernel may accumulate ``parameter``'s gradient itself: the
    Megatron DDP contract Transformer Engine's fused weight-gradient accumulation follows (a
    ``main_grad`` and the ``grad_added_to_main_grad`` flag, without overwrite or FSDP), else
    None."""
    if (
        parameter is None
        or not parameter.requires_grad
        or getattr(parameter, "main_grad", None) is None
        or not hasattr(parameter, "grad_added_to_main_grad")
        or getattr(parameter, "overwrite_main_grad", False)
        or hasattr(parameter, "__fsdp_param__")
    ):
        return None
    return parameter.main_grad, parameter.dtype


def _accumulated(parameter):
    """Mark ``parameter``'s gradient as accumulated and return TE's placeholder, which DDP's hook
    drops (zeros when it adds it anyway, ``zero_out_wgrad``)."""
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    parameter.grad_added_to_main_grad = True
    zero = getattr(parameter, "zero_out_wgrad", False)
    return get_dummy_wgrad(list(parameter.shape), parameter.dtype, zero=zero)


@lru_cache
def _trains_on(device_index: int) -> bool:
    from .configs import qualified_device

    return qualified_device(torch.cuda.get_device_properties(device_index))


def check_mxfp8_call(tensor) -> None:
    """An MXFP8 handoff needs a tuned training shape or an inference row count divisible by 32."""
    hidden = tensor.shape[-1]
    rows = tensor.numel() // hidden
    if torch.is_grad_enabled():
        max_rows = 65536 if hidden <= 4096 else 16384
        if hidden in range(2048, 16385, 1024) and rows in range(2048, max_rows + 1, 1024):
            return
        reason = f"training (widths 2048–16384 and rows 2048–{max_rows}, both in steps of 1024)"
    elif rows % 32:
        reason = "inference (rows must be divisible by 32)"
    else:
        return
    raise ValueError(
        f"the MXFP8 seam handoff does not support {rows} tokens at hidden size {hidden} in {reason}"
    )


def fused_seam(
    pending,
    reads,
    gammas,
    *,
    num_streams,
    outputs=None,
    compact_scales_for_tp=None,
    columns=None,
    interval=None,
    epsilon=1.0e-5,
    batch_invariant=False,
    fuse_wgrad_accumulation=False,
):
    """Write ``pending`` and emit one or two following read/RMSNorms in one kernel; returns
    ``(residual, *readers)``.

    ``reads``/``gammas`` hold each reader's read logits and RMSNorm weight. ``outputs`` gives
    each reader's format (default BF16): "mxfp8" emits it directly from FP32 as rowwise MXFP8, or
    a tuple of two consumers' formats (e.g. ("bf16", "mxfp8"), "both" for short) returns one
    output per consumer in that order. A true entry in ``compact_scales_for_tp`` selects all-gather
    order for that reader's forward scales and replayed column scales. Consumer gradients
    are summed by the backward kernel. With gradients the write records into ``interval`` (a
    ReplayInterval), and each MXFP8 reader's ``columns`` entry is a ColumnSlot that the interval's
    replay fills with its columnwise operand, for its consumers' weight-gradient GEMMs; the
    interval must replay before them. ``batch_invariant`` (inference) fixes the schedule across
    batch sizes. ``fuse_wgrad_accumulation`` lets the backward accumulate the controller and norm
    weight gradients into their parameters' DDP ``main_grad`` (Transformer Engine's contract).
    """
    if not 1 <= len(reads) == len(gammas) <= 2:
        raise ValueError("a seam has one or two readers, each with read logits and a norm weight")
    p = pending
    inputs = (p.residual, p.routed, p.shared, p.write, p.retain, *reads, *gammas)
    uses = tuple(
        _consumer_formats(fmt) for fmt in (("bf16",) * len(reads) if outputs is None else outputs)
    )
    slots = (None,) * len(reads) if columns is None else tuple(columns)
    if len(uses) != len(reads) or len(slots) != len(reads):
        raise ValueError("a seam takes one output format and column slot per reader")
    options = (
        num_streams,
        interval,
        uses,
        slots,
        epsilon,
        batch_invariant,
        fuse_wgrad_accumulation,
        p.update_formats,
        compact_scales_for_tp,
        p.compact_grad_scales_for_tp,
    )
    if not torch.is_grad_enabled():
        return _regroup(uses, _Boundary.forward(None, options, *inputs))
    if any(("mxfp8" in use) != (slot is not None) for use, slot in zip(uses, slots)):
        raise ValueError("in training an MXFP8 seam reader (and only one) requires a ColumnSlot")
    if not isinstance(interval, ReplayInterval):
        raise ValueError("a seam with gradients records into a ReplayInterval")
    if p.residual.is_cuda and not _trains_on(p.residual.device.index):
        # Fail at the first seam rather than after a whole forward pass.
        raise ValueError(
            "training through the fused shortcut seams requires a qualified GB300 (the fused "
            f"backward is GB300-only); got {torch.cuda.get_device_name(p.residual.device)}"
        )
    flat = _Boundary.apply(options, *inputs)
    for fmt, tensor in zip((fmt for use in uses for fmt in use), flat[1:]):
        if fmt == "mxfp8":
            # TE caches requires_grad when it creates the quantized reader, before autograd
            # attaches this seam: refresh it, or the consumer would skip its input gradient.
            tensor.requires_grad_(True)
    return _regroup(uses, flat)


def _consumer_formats(fmt):
    """A reader's per-consumer formats: a tuple of one or two, or a single format ("both" is a
    BF16 consumer and an MXFP8 one)."""
    use = ("bf16", "mxfp8") if fmt == "both" else (fmt,) if isinstance(fmt, str) else tuple(fmt)
    if not 1 <= len(use) <= 2 or any(f not in ("bf16", "mxfp8") for f in use):
        raise ValueError("a seam reader feeds one or two consumers, each BF16 or MXFP8")
    return use


def _reader_groups(uses, tensors):
    """Partition flat autograd inputs or outputs in consumer order."""
    begin = 0
    for use in uses:
        end = begin + len(use)
        yield tensors[begin:end]
        begin = end


def _regroup(uses, flat):
    readers = (g[0] if len(g) == 1 else g for g in _reader_groups(uses, flat[1:]))
    return (flat[0], *readers)


@dataclass(frozen=True)
class PendingWideResidualWrites:
    """A branch output not yet written to the wide residual.

    The next seam kernel writes it together with that layer's read/RMSNorm(s). A MoE output is
    a shortcut add of ``routed`` and ``shared``; a single branch update (a lone layer's output)
    leaves ``shared`` None. ``write``/``retain`` are the writing connection's logits
    (``retain`` None is fixed identity retention). ``update_formats`` may mark each update
    (routed, then shared) "mxfp8" when its producer is an MXFP8 Linear taking its grad_output
    directly, which the seam backward then emits quantized; else BF16 (default).
    """

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor | None
    write: torch.Tensor
    retain: torch.Tensor | None
    update_formats: tuple | None = None
    compact_grad_scales_for_tp: bool = False
