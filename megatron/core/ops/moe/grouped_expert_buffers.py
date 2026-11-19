# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Native grouped64 GEMMs with differentiable joins that avoid large copies."""

import contextlib
import inspect

import torch


class GradPlan:
    """Allocate the original-size input gradient lazily during backward only."""

    def __init__(self, inp):
        self.shape, self.dtype, self.device = tuple(inp.shape), inp.dtype, inp.device
        self.buffer = None

    def part(self, offset, rows):
        """Return one disjoint row view, allocating the backing buffer on demand."""
        if self.buffer is None:
            self.buffer = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        return self.buffer.narrow(0, offset, rows)


class SharedSplit(torch.autograd.Function):
    """Split an input while accepting native gradients written into a shared buffer."""

    @staticmethod
    def forward(ctx, inp, rows, plan):
        ctx.rows, ctx.plan = rows, plan
        return inp.split(rows, dim=0)

    @staticmethod
    def backward(ctx, *grads):
        plan, offset, aliases = ctx.plan, 0, ctx.plan.buffer is not None
        for rows, grad in zip(ctx.rows, grads, strict=True):
            aliases = (
                aliases
                and grad is not None
                and grad.is_contiguous()
                and (
                    grad.untyped_storage().data_ptr() == plan.buffer.untyped_storage().data_ptr()
                    and grad.storage_offset() == offset * plan.shape[1]
                    and tuple(grad.shape) == (rows, plan.shape[1])
                )
            )
            offset += rows
        if aliases:
            result = plan.buffer
        else:
            values = [
                (
                    g
                    if g is not None
                    else torch.zeros((rows, plan.shape[1]), dtype=plan.dtype, device=plan.device)
                )
                for rows, g in zip(ctx.rows, grads, strict=True)
            ]
            result = torch.cat(values, dim=0)
        # Autograd nodes may remain reachable after backward through retained
        # losses/outputs. Do not retain the large temporary input gradient there.
        plan.buffer = None
        ctx.plan = None
        return result, None, None


class SharedJoin(torch.autograd.Function):
    """Join native output views without copying their already packed storage."""

    @staticmethod
    def forward(ctx, packed, *outputs):
        ctx.rows = tuple(t.shape[0] for t in outputs)
        # Both native GEMMs already wrote these disjoint regions.
        return packed.view_as(packed)

    @staticmethod
    def backward(ctx, grad):
        return (None, *grad.split(ctx.rows, dim=0))


@contextlib.contextmanager
def substitute_forward():
    """Install a scoped adapter for the pinned tuple-argument TE GroupedLinear ABI."""
    from transformer_engine.pytorch.module import grouped_linear as gl

    cls, active = gl._GroupedLinear, [None]
    if tuple(inspect.signature(cls.forward).parameters) != (
        'ctx',
        'inp',
        'm_splits',
        'non_tensor_args',
        'out',
        'dgrad_out',
        'weights_and_biases',
    ):
        raise RuntimeError('Unsupported TE GroupedLinear ABI for --moe-grouped-expert-buffers')
    original_apply = cls.apply
    had_apply, apply_descriptor = 'apply' in cls.__dict__, cls.__dict__.get('apply')
    grouped_forward, backward = cls._forward_grouped_tensor, cls.backward
    calls = dict(forward=0, native_backward=0, fallback=0)

    def tracked_grouped_forward(ctx, *args, **kwargs):
        calls['forward'] += 1
        if active[0] is not None:
            ctx._mfu_grad_plan = active[0]
        return grouped_forward(ctx, *args, **kwargs)

    def split_apply(inp, splits, non_tensor_args, out, dgrad_out, *parameters):
        groups = len(splits)
        if len(non_tensor_args) != 24:
            raise RuntimeError('Unsupported TE grouped expert argument layout')
        if (
            len(parameters) != 2 * groups
            or splits.device.type != 'cpu'
            or out is not None
            or dgrad_out is not None
            or non_tensor_args[0]
            or non_tensor_args[2]
            or non_tensor_args[3]
            or non_tensor_args[14] != torch.bfloat16
            or non_tensor_args[21]
            or non_tensor_args[22]
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
            or (non_tensor_args[4] is not None and non_tensor_args[4].delay_wgrad_compute())
            or non_tensor_args[12]
        ):
            calls['fallback'] += 1
            return original_apply(inp, splits, non_tensor_args, out, dgrad_out, *parameters)
        if not inp.is_contiguous() or inp.ndim != 2:
            raise ValueError("Shared grouped buffers require contiguous 2D expert inputs.")
        counts = splits.tolist()
        # One same-stream metadata copy for all experts, with no host wait after
        # the first chunk's GEMM. Native consumers stay ordered on that stream.
        gpu_splits = splits.to(device=inp.device, non_blocking=True)
        slices = [(i, min(i + 64, groups)) for i in range(0, groups, 64)]
        row_counts = tuple(sum(counts[a:b]) for a, b in slices)
        weights, biases = parameters[:groups], parameters[groups:]
        use_plan = len(slices) > 1 and inp.requires_grad
        plan = GradPlan(inp) if use_plan else None
        inputs = (
            SharedSplit.apply(inp, row_counts, plan) if use_plan else inp.split(row_counts, dim=0)
        )
        packed = torch.empty(
            (inp.shape[0], weights[0].shape[0]), dtype=inp.dtype, device=inp.device
        )
        destinations = packed.split(row_counts, dim=0)
        offset, outputs, workspaces = 0, [], []
        for (start, stop), rows, part, destination in zip(
            slices, row_counts, inputs, destinations, strict=True
        ):
            args = list(non_tensor_args)
            for index in [5, 6, 7, 8, 9, 10, 16]:
                args[index] = args[index][start:stop]
            args[23] = True
            before = calls['forward']
            active[0] = (plan, offset, rows) if plan is not None else None
            try:
                result, spaces = original_apply(
                    part,
                    gpu_splits[start:stop],
                    tuple(args),
                    destination,
                    None,
                    *weights[start:stop],
                    *biases[start:stop],
                )
            finally:
                active[0] = None
            if calls['forward'] != before + 1:
                raise RuntimeError('Actual native grouped-tensor GEMM was not selected')
            outputs.append(result)
            workspaces.extend(spaces)
            offset += rows
        assert offset == inp.shape[0]
        return SharedJoin.apply(packed, *outputs), workspaces

    def tracked_backward(ctx, *args, **kwargs):
        if not ctx.grouped_tensor_supported:
            return backward(ctx, *args, **kwargs)
        shared = getattr(ctx, '_mfu_grad_plan', None)
        if shared is not None:
            plan, offset, rows = shared
            ctx.dgrad_out = plan.part(offset, rows)
        calls['native_backward'] += 1
        try:
            return backward(ctx, *args, **kwargs)
        finally:
            # Native backward already returned the gradient view. Its consumer
            # owns it now; the context must not extend its live storage lifetime.
            ctx.dgrad_out = None
            if shared is not None:
                del ctx._mfu_grad_plan

    cls.apply = staticmethod(split_apply)
    cls._forward_grouped_tensor = staticmethod(tracked_grouped_forward)
    cls.backward = staticmethod(tracked_backward)
    try:
        yield calls
    finally:
        if had_apply:
            cls.apply = apply_descriptor
        else:
            del cls.apply
        cls._forward_grouped_tensor = staticmethod(grouped_forward)
        cls.backward = staticmethod(backward)


# Keep the adapter alive through backward, including activation recomputation.
_training_context = None


def enable_grouped_expert_buffers() -> dict:
    """Install the opt-in TE adapter once for the lifetime of a training process."""
    global _training_context
    if _training_context is None:
        context = substitute_forward()
        statistics = context.__enter__()
        _training_context = (context, statistics)
    return _training_context[1]
