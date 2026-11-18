# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-facing fused shortcut seams; GPU imports are lazy.

Training (a qualified GB300) differentiates the seam, including pruned residual/reader
gradients; no-grad calls bypass autograd and save nothing for backward.
"""

from dataclasses import dataclass, replace
from functools import lru_cache, partial
from weakref import WeakValueDictionary, ref

import torch
from torch.autograd.function import once_differentiable
from torch.multiprocessing.reductions import StorageWeakRef

from .residual_replay import ReplayInterval, graph_retained

# MXFP8 handoff: a seam writes a reader as rowwise MXFP8 for its consumer Linears' forward GEMMs,
# as BF16 for other modules, or both. A reader with two consumers returns one output per
# consumer, in its format, so each consumer's gradient reaches the backward kernel separately
# (summed there). The consumers' weight-gradient GEMMs share one columnwise operand replayed
# from the seam (wide_residual.mxfp8_consumer swaps it into TE's saved slots at unpack).
_READERS = WeakValueDictionary()


def reader_for(tensor):
    """Find a producer through views/detaches without retaining its FP8 allocation."""
    data = getattr(tensor, "_rowwise_data", None)
    if data is None:
        return None
    return _READERS.get(StorageWeakRef(data.untyped_storage()))


class Reader:
    """Where one seam call's MXFP8 reader gets its columnwise operand from: taken from the replay
    once per backward and shared by its declared consumers' weight-gradient GEMMs."""

    def __init__(self, tensor, columns, consumers):
        self._take = columns  # () -> the columnwise MXFP8 operand (one replay hand-out)
        self.consumers = tuple(ref(linear) for linear in consumers)
        self.hooked = 0  # consumer saved slots unpacking these columns in the current graph
        self._columns, self._pending = None, 0
        # The weak reference prevents StorageImpl address reuse without retaining bytes.
        _READERS[StorageWeakRef(tensor._rowwise_data.untyped_storage())] = self
        # Pin the Reader on the rowwise data tensor, which detached aliases share, rather than
        # only on this wrapper: a checkpointed consumer (e.g. GatedDeltaProduct's gdp_in_proj
        # recompute) drops this wrapper after its no-grad first pass and, in backward, re-runs
        # on a detached alias, whose lookup must still find the Reader.
        tensor._rowwise_data._shortcut_mxfp8_reader = self
        tensor._shortcut_mxfp8_reader = self

    def declares(self, linear):
        """Whether ``linear`` is one of this reader's declared MXFP8 consumers."""
        return any(consumer() is linear for consumer in self.consumers)

    def hook(self):
        """Count one consumer slot that will unpack the columns; each declared consumer runs
        once per graph (a recompute's no-grad pass installs nothing)."""
        if self.hooked == len(self.consumers):
            raise RuntimeError("a seam's MXFP8 reader was consumed more often than declared")
        self.hooked += 1

    def columns(self):
        """The columnwise operand for one consumer's unpack: the first unpack takes it from the
        replay, the declared consumers share it, and the last releases it (a retained graph's
        next backward, whose recomputed consumers install again, takes it anew)."""
        if self._columns is None:
            self._columns, self._pending = self._take(), len(self.consumers)
        columns, self._pending = self._columns, self._pending - 1
        if self._pending == 0:
            self._columns, self.hooked = None, 0
        return columns


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
    def forward(ctx, options, residual, *values):
        """Run the seam; ``ctx`` None is an inference call without autograd state."""
        streams, replay, uses, consumers, epsilon, update_count = options
        formats = tuple(_kernel_format(use) for use in uses)
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
            outputs=formats,
            num_streams=streams,
            epsilon=epsilon,
            inference=ctx is None,
        )
        outputs = tuple(
            (
                tuple(t.view(*leading, hidden) for t in q)
                if isinstance(q, tuple)
                else q.view(*leading, hidden)
            )
            for q in outputs
        )
        # One autograd output per consumer: its format's tensor, or a detached alias of it for a
        # second consumer of the same format (each consumer's gradient arrives separately).
        quantized = tuple(q[0] if isinstance(q, tuple) else q for q in outputs)
        plain = tuple(q[1] if isinstance(q, tuple) else q for q in outputs)
        flat = []
        for use, mx, bf16 in zip(uses, quantized, plain):
            handed = set()
            for fmt in use:
                tensor = mx if fmt == "mxfp8" else bf16
                flat.append(tensor.detach() if fmt in handed else tensor)
                handed.add(fmt)
        result = (current.view(*leading, streams * hidden), *flat)
        if ctx is None:
            return result
        # The interval keeps or replays this seam's input residual (and its MXFP8 readers'
        # columns): the backward never reconstructs it from a rounded successor.
        mxfp8 = tuple(fmt != "bf16" for fmt in formats)
        ctx.replay = replay.record(residual, current, saved, prepare_columns=mxfp8)
        ctx.save_for_backward(saved.update, saved.coefficients, *saved.gammas, *saved.rstd)
        ctx.spec = replace(saved, update=None, coefficients=None, gammas=(), rstd=())
        ctx.leading, ctx.uses = leading, uses
        for reader, (needs, operand) in enumerate(zip(mxfp8, quantized)):
            if needs:  # the consumers' weight-gradient GEMMs unpack replayed columns
                Reader(operand, partial(ctx.replay.columns, reader), consumers[reader])
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, carry, *flat):
        """Fused seam backward; the replay interval supplies the predecessor."""
        from .backward_kernel import backward

        # Per reader, its consumers' gradients: one, or a pair the kernel sums.
        dys, position = [], 0
        for use in ctx.uses:
            grads = [g for g in flat[position : position + len(use)] if g is not None]
            position += len(use)
            dys.append(tuple(grads) if len(grads) == 2 else grads[0] if grads else None)

        streams, update_count, leading = ctx.spec.num_streams, ctx.spec.update_count, ctx.leading
        update, coefficients, *readers = ctx.saved_tensors
        replay = ctx.replay
        previous = replay.previous()
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
        flat2d = lambda t: t.contiguous().view(-1, hidden)
        dy = tuple(
            None if t is None else tuple(map(flat2d, t)) if isinstance(t, tuple) else flat2d(t)
            for t in dys
        )
        dc = None if carry is None else carry.contiguous().view_as(previous)
        grads, _ = backward(saved, previous, dy, dc)
        if replay.index == 0:
            # This seam applies the previous interval's last write: replay that interval now,
            # before any of its backward GEMMs (which need gradients this seam produces).
            replay.interval.replay_previous()
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
        return (None,) + tuple(
            g if needed else None for g, needed in zip(grads, ctx.needs_input_grad[1:])
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
    outputs=None,
    mxfp8_consumers=None,
    epsilon=1.0e-5,
    replay=None,
):
    """In training the write records into ``replay`` (a ReplayInterval; None: its own one-write
    interval), which keeps or replays its input residual and its MXFP8 readers' columns."""
    updates = (p.routed,) if p.shared is None else (p.routed, p.shared)
    inputs = (p.residual, *updates, p.write, p.retain, *reads, *gammas)
    uses = tuple(
        _consumer_formats(fmt) for fmt in (("bf16",) * len(reads) if outputs is None else outputs)
    )
    formats = tuple(_kernel_format(use) for use in uses)
    consumers = (
        ((),) * len(reads) if mxfp8_consumers is None else tuple(map(tuple, mxfp8_consumers))
    )
    if not len(formats) == len(consumers) == len(reads):
        raise ValueError("a seam takes one output format and consumer set per reader")
    if not torch.is_grad_enabled():
        options = (num_streams, None, uses, consumers, epsilon, len(updates))
        return _regroup(uses, _Boundary.forward(None, options, *inputs))
    if any((fmt == "bf16") != (not linears) for fmt, linears in zip(formats, consumers)):
        raise ValueError(
            "in training an MXFP8 seam reader (and only one) requires its consumer Linears"
        )
    if p.residual.is_cuda and not _trains_on(p.residual.device.index):
        # Fail at the first seam rather than after a whole forward pass.
        raise ValueError(
            "training through the fused shortcut seams requires a qualified GB300 (the fused "
            f"backward is GB300-only); got {torch.cuda.get_device_name(p.residual.device)}"
        )
    # Without checkpoint replay, the seam is a one-write interval keeping its input residual.
    replay = ReplayInterval() if replay is None else replay
    options = (num_streams, replay, uses, consumers, epsilon, len(updates))
    flat = _Boundary.apply(options, *inputs)
    position = 1
    for use in uses:
        for fmt in use:
            if fmt == "mxfp8":
                # TE caches requires_grad when it creates the quantized reader, before autograd
                # attaches this seam: refresh it, or the consumer would skip its input gradient.
                flat[position].requires_grad_(True)
            position += 1
    return _regroup(uses, flat)


def _consumer_formats(fmt):
    """A reader's per-consumer formats: a tuple of one or two, or a single format ("both" is a
    BF16 consumer and an MXFP8 one)."""
    use = ("bf16", "mxfp8") if fmt == "both" else (fmt,) if isinstance(fmt, str) else tuple(fmt)
    if not 1 <= len(use) <= 2 or any(f not in ("bf16", "mxfp8") for f in use):
        raise ValueError("a seam reader feeds one or two consumers, each BF16 or MXFP8")
    return use


def _kernel_format(use):
    """The forward kernel's format for a reader's consumers: MXFP8, BF16 or both."""
    if all(f == "mxfp8" for f in use):
        return "mxfp8"
    return "bf16" if all(f == "bf16" for f in use) else "both"


def _regroup(uses, flat):
    """``(residual, *readers)``: a reader with two consumers as a pair, in consumer order."""
    result, position = [flat[0]], 1
    for use in uses:
        width = len(use)
        result.append(flat[position] if width == 1 else tuple(flat[position : position + width]))
        position += width
    return tuple(result)


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

    Options match ``shortcut_add_wide_residual_write_read_rmsnorm_quant``, per reader.
    """
    if len(reads) != 2 or len(gammas) != 2:
        raise ValueError(
            "the two-reader seam requires two read controllers and two RMSNorm weights"
        )
    return _apply_boundary(pending, reads, gammas, **options)


def shortcut_add_wide_residual_write_read_rmsnorm_quant(pending, read, gamma, **options):
    """Write a pending update and emit one following read/RMSNorm.

    Keyword options: ``num_streams`` (required), ``epsilon`` (RMSNorm, default 1e-5),
    ``outputs``, ``mxfp8_consumers`` and ``replay``. ``outputs`` gives each reader's format
    (default BF16): "mxfp8" emits it directly from FP32 as rowwise MXFP8, or a tuple of two
    consumers' formats (e.g. ("bf16", "mxfp8"), "both" for short) returns one output per consumer
    in that order, for the caller to hand each its tensor; their gradients are summed by the
    backward kernel. In
    training an MXFP8 reader's ``mxfp8_consumers`` entry declares the TELinears that take it (each
    once per graph; views of the whole reader are allowed): their backward shares one columnwise
    operand replayed by this write's interval. ``replay`` (a ReplayInterval, training only) is the
    checkpoint-replay interval to record into.
    """
    return _apply_boundary(pending, (read,), (gamma,), **options)
