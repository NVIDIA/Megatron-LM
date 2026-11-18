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

# MXFP8 handoff: a seam writes a reader as rowwise MXFP8 for its consumer Linears' forward GEMMs
# ("mxfp8"), or as BF16 plus that MXFP8 twin when other modules need BF16 ("both"). The consumers'
# weight-gradient GEMMs share one columnwise operand replayed from the seam
# (wide_residual.mxfp8_consumer swaps it into TE's saved slots at unpack).
_READERS = WeakValueDictionary()
# A "both" reader's BF16 storage -> its _Twin, alive while the BF16 output or the seam's autograd
# context (until its backward) holds it.
_TWINS = WeakValueDictionary()


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


class _AsQuantized(torch.autograd.Function):
    """A "both" reader's MXFP8 twin in place of its BF16 output; the gradient goes to the BF16
    output (one reader, one gradient)."""

    @staticmethod
    def forward(ctx, plain, twin):  # pylint: disable=arguments-differ
        return twin[0].view(plain.shape)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        return grad, None


class _Twin:
    """A "both" reader's MXFP8 operand. Without recomputed consumers it is dropped after its
    declared consumers' graph-building forwards (``pending``); otherwise it lives as long as the
    BF16 output or the seam's autograd context."""

    __slots__ = ("pointer", "numel", "quantized", "pending", "__weakref__")

    def __init__(self, plain, quantized, consumers, recomputed):
        self.pointer, self.numel, self.quantized = plain.data_ptr(), plain.numel(), quantized
        self.pending = None if recomputed else consumers


def quantized_twin(tensor):
    """The MXFP8 operand a "both" reader emitted along with ``tensor`` (its BF16 output, a view
    of all of it or a detached alias), autograd-linked to ``tensor``; None for other tensors."""
    if type(tensor) is not torch.Tensor or not _TWINS:
        return None
    entry = _TWINS.get(StorageWeakRef(tensor.untyped_storage()))
    if entry is None:
        return None
    if tensor.data_ptr() != entry.pointer or tensor.numel() != entry.numel:
        raise ValueError("a seam reader's MXFP8 consumer must take the whole reader")
    quantized = entry.quantized
    if quantized is None:
        raise RuntimeError(
            "a seam reader's MXFP8 twin was consumed more often than declared (a recomputed "
            "consumer must be declared as such)"
        )
    if torch.is_grad_enabled() and entry.pending is not None:
        entry.pending -= 1
        if entry.pending == 0:  # TE's saved slot keeps only the columns, replayed in backward
            entry.quantized = None
    twin = _AsQuantized.apply(tensor, (quantized,))
    reader = getattr(quantized, "_shortcut_mxfp8_reader", None)
    if reader is not None:  # pinned by the operand handed out, which outlives a dropped twin
        twin._shortcut_mxfp8_reader = twin._rowwise_data._shortcut_mxfp8_reader = reader
    if tensor.requires_grad:
        # TE caches requires_grad when it creates the operand: refresh it for the consumer.
        twin.requires_grad_(True)
    return twin


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
        streams, replay, formats, consumers, recomputed, epsilon, update_count = options
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
        # A "both" reader returns its BF16 output; its declared consumers take the MXFP8 twin.
        quantized = tuple(q[0] if isinstance(q, tuple) else q for q in outputs)
        outputs = tuple(q[1] if isinstance(q, tuple) else q for q in outputs)
        twins = []
        for fmt, plain, operand, linears, again in zip(
            formats, outputs, quantized, consumers, recomputed
        ):
            if fmt == "both":
                twins.append(_Twin(plain, operand, len(linears), again))
                _TWINS[StorageWeakRef(plain.untyped_storage())] = twins[-1]
                plain._shortcut_mxfp8_twin = twins[-1]
        result = (current.view(*leading, streams * hidden), *outputs)
        if ctx is None:
            return result
        # Recomputed consumers may look the twin up through a detached alias until this seam's
        # backward, after theirs.
        ctx.twins = twins
        # The interval keeps or replays this seam's input residual (and its MXFP8 readers'
        # columns): the backward never reconstructs it from a rounded successor.
        mxfp8 = tuple(fmt != "bf16" for fmt in formats)
        ctx.replay = replay.record(residual, current, saved, prepare_columns=mxfp8)
        ctx.save_for_backward(saved.update, saved.coefficients, *saved.gammas, *saved.rstd)
        ctx.spec = replace(saved, update=None, coefficients=None, gammas=(), rstd=())
        ctx.leading = leading
        for reader, (needs, operand) in enumerate(zip(mxfp8, quantized)):
            if needs:  # the consumers' weight-gradient GEMMs unpack replayed columns
                Reader(operand, partial(ctx.replay.columns, reader), consumers[reader])
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, carry, *dys):
        """Fused seam backward; the replay interval supplies the predecessor."""
        from .backward_kernel import backward

        streams, update_count, leading = ctx.spec.num_streams, ctx.spec.update_count, ctx.leading
        update, coefficients, *readers = ctx.saved_tensors
        replay = ctx.replay
        previous = replay.previous()
        # Autograd frees saved tensors after backward but not ctx attributes, and the graph's
        # nodes may outlive it (until the next forward): release the interval explicitly.
        if not graph_retained():
            ctx.replay = ctx.twins = None
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
    recomputed=None,
    epsilon=1.0e-5,
    replay=None,
):
    """In training the write records into ``replay`` (a ReplayInterval; None: its own one-write
    interval), which keeps or replays its input residual and its MXFP8 readers' columns."""
    updates = (p.routed,) if p.shared is None else (p.routed, p.shared)
    inputs = (p.residual, *updates, p.write, p.retain, *reads, *gammas)
    formats = ("bf16",) * len(reads) if outputs is None else tuple(outputs)
    consumers = (
        ((),) * len(reads) if mxfp8_consumers is None else tuple(map(tuple, mxfp8_consumers))
    )
    recomputed = (False,) * len(reads) if recomputed is None else tuple(recomputed)
    if not len(formats) == len(consumers) == len(recomputed) == len(reads):
        raise ValueError("a seam takes one output format and consumer set per reader")
    if not torch.is_grad_enabled():
        options = (num_streams, None, formats, consumers, recomputed, epsilon, len(updates))
        return _Boundary.forward(None, options, *inputs)
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
    options = (num_streams, replay, formats, consumers, recomputed, epsilon, len(updates))
    result = _Boundary.apply(options, *inputs)
    for fmt, output in zip(formats, result[1:]):
        if fmt == "mxfp8":
            # TE caches requires_grad when it creates the quantized reader, before autograd
            # attaches this seam: refresh it, or the consumer would skip its input gradient.
            output.requires_grad_(True)
    return result


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
    (default BF16): "mxfp8" emits it directly from FP32 as rowwise MXFP8, and "both" returns the
    BF16 output while its MXFP8 consumers take the twin (``quantized_twin``). In training an MXFP8
    reader's ``mxfp8_consumers`` entry declares the TELinears that take it (each once per graph;
    views of the whole reader are allowed): their backward shares one columnwise operand replayed
    by this write's interval. ``recomputed`` (one bool per reader) keeps a "both" reader's twin
    for consumers re-run in backward; otherwise it is dropped after their forwards. ``replay`` (a
    ReplayInterval, training only) is the checkpoint-replay interval to record into.
    """
    return _apply_boundary(pending, (read,), (gamma,), **options)
