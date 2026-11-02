# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Direct FP32-to-MXFP8 inputs with call-local TE weight-gradient reconstruction.

An exclusive TELinear consumer defers its wgrad GEMM until the shortcut
backward hands off the prepared columnwise operand. An extra weight dependency holds its
AccumulateGrad until that GEMM finishes. TE still owns both GEMMs, quantization
of weights/dy, gradient accumulation, and distributed-weight finalization.
Only this call's saved input and backward arguments are adapted; no TE module
state or global TE function is replaced.
"""

from types import SimpleNamespace
from weakref import WeakValueDictionary, ref

import torch
from torch.multiprocessing.reductions import StorageWeakRef

_READERS = WeakValueDictionary()


def reader_for(tensor):
    """Find a producer through views/detaches without retaining its FP8 allocation."""
    data = getattr(tensor, "_rowwise_data", None)
    if data is None:
        return None
    return _READERS.get(StorageWeakRef(data.untyped_storage()))


class Reader:
    """Detached reconstruction state owned by one autograd boundary."""

    def __init__(self, tensor, saved, consumer=None):
        self.shape, self.device = tuple(saved.update.shape), saved.update.device
        self.consumer = None if consumer is None else ref(consumer)
        self.pending = []
        storage = tensor._rowwise_data.untyped_storage()
        # Holding only the intrusive weak reference prevents StorageImpl address
        # reuse from aliasing this registry entry, without retaining GPU bytes.
        _READERS[StorageWeakRef(storage)] = self
        tensor._shortcut_mxfp8_reader = self

    def empty_columnwise(self, shape):
        """Allocate this reader's column operand with the consumer view shape."""
        from .kernel_utils import allocate_operand

        return allocate_operand(self.shape, self.device, columnwise=True).view(shape)

    def compute_wgrad(self, previous, column):
        """Submit delayed GEMMs as soon as the boundary's column operand is ready."""
        if self.pending and column is None:
            raise RuntimeError("deferred shortcut weight gradients require fused MXFP8 columns")
        stream = torch.cuda.current_stream(previous.device)
        for tensors, function, weight, consumer_stream in self.pending:
            tensors[0]._columnwise_data = column._columnwise_data
            tensors[0]._columnwise_scale_inv = column._columnwise_scale_inv

            # TE's dy and wgrad buffers belong to its backward stream. Keep
            # their allocation lifetimes there, and order the new column
            # input and boundary completion through explicit stream waits.
            if consumer_stream != stream:
                consumer_stream.wait_stream(stream)
            with torch.cuda.stream(consumer_stream):
                column._columnwise_data.record_stream(consumer_stream)
                column._columnwise_scale_inv.record_stream(consumer_stream)
                dw, _ = function(*tensors)
                if weight is not None:
                    from transformer_engine.pytorch.distributed_weight import finalize_weight_grads

                    finalize_weight_grads(weight, [dw])

    def finish(self, previous):
        """Join weight GEMMs before autograd can announce that weights are ready."""
        stream = torch.cuda.current_stream(previous.device)
        for _, _, _, consumer_stream in self.pending:
            if consumer_stream != stream:
                stream.wait_stream(consumer_stream)
        # Keep TE buffers alive until both streams have submitted their work.
        self.pending.clear()


def install_linear_hooks(tensor, output, linear):
    """Replace only TE's saved activation; preserve weight identity and metadata.

    A scope-wide saved-tensor hook also detaches unrelated tensor subclasses
    when PyTorch unpacks them, losing GTP weight metadata. Register on the
    individual activation save slot after forward instead.
    """
    reader = reader_for(tensor)
    if reader is None or not torch.is_grad_enabled():
        return
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if not FP8GlobalStateManager.is_fp8_enabled() or not recipe.mxfp8():
        raise RuntimeError("shortcut MXFP8 input requires an MXFP8 TE linear consumer")
    if getattr(recipe, "backward_override", None) is not None:
        raise RuntimeError("shortcut MXFP8 handoff requires ordinary MXFP8 backward")
    output = output[0] if isinstance(output, tuple) else output
    node = output.grad_fn
    args = getattr(node, "backward_objects", None)
    if args is None:
        raise RuntimeError("this TE version does not expose the linear backward handoff")
    key = tensor._rowwise_data.untyped_storage()._cdata
    deferred = (
        reader.consumer is not None
        and reader.consumer() is linear
        and args.requires_wgrad
        and args.requires_dgrad
        and args.fuse_wgrad_accumulation
        and args.wgrad_store is None
        and args.tp_size == 1
        and args.backward_override is None
        and not args.ub_overlap_ag
        and not args.ub_overlap_rs_dgrad
        and not args.ub_bulk_dgrad
        and not args.ub_bulk_wgrad
    )
    if args.requires_wgrad and not deferred:
        raise RuntimeError(
            "shortcut MXFP8 consumer requires deferred weight gradients; "
            "use a supported exclusive linear consumer or a BF16 handoff"
        )
    matched = 0
    for value, slot in zip(node.saved_tensors, node._raw_saved_tensors):
        data = getattr(value, "_rowwise_data", None)
        if data is not None and data.untyped_storage()._cdata == key:
            # Deferred TE validates column pointers now; boundary backward fills them later.
            slot.register_hooks(torch.Tensor.size, reader.empty_columnwise)
            matched += 1
    if args.requires_wgrad and matched != 1:
        raise RuntimeError("TE did not save exactly one shortcut MXFP8 activation")
    if not deferred:
        return
    backward_node = ref(node)

    def enqueue(tensors, function):
        from transformer_engine.pytorch.distributed_weight import is_distributed_weight

        weight = backward_node().backward_objects.saved_weight
        reader.pending.append(
            (
                tensors,
                function,
                weight if is_distributed_weight(weight) else None,
                torch.cuda.current_stream(tensors[0].device),
            )
        )

    # TE's call-local store protocol already runs at the handoff point. The
    # explicit boundary/weight edge orders readiness after Reader.finish().
    args.wgrad_store = SimpleNamespace(delay_wgrad_compute=lambda: True, put=enqueue)


def eligible(linear, shape, streams):
    """Limit automatic selection to supported shapes on the qualified device and TE consumers."""
    from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    from megatron.core.extensions.transformer_engine import TELinear

    from .kernel_utils import SHAPES, WIDTHS, qualified_device

    recipe = FP8GlobalStateManager.get_fp8_recipe()
    return (
        streams == 3
        and (
            (shape.numel() // shape.shape[-1], shape.shape[-1]) in SHAPES
            or (
                not torch.is_grad_enabled()
                and shape.shape[-1] in WIDTHS
                and (shape.numel() // shape.shape[-1]) % 32 == 0
            )
        )
        and qualified_device(torch.cuda.get_device_properties(shape.device))
        and isinstance(linear, TELinear)
        and linear.fuse_wgrad_accumulation
        and getattr(linear, "te_quant_params", None) is None
        and getattr(linear, "tp_size", 1) == 1
        and not any(
            getattr(linear, name, False)
            for name in (
                "ub_overlap_ag_fprop",
                "ub_overlap_ag_dgrad",
                "ub_overlap_rs_dgrad",
                "ub_bulk_dgrad",
                "ub_bulk_wgrad",
            )
        )
        and not is_cpu_offload_enabled()
        and getattr(linear, "fsdp_group", None) is None
        and not getattr(linear, "is_fsdp2", False)
        and not linear.wgrad_store.delay_wgrad_compute()
        and FP8GlobalStateManager.is_fp8_enabled()
        and recipe.mxfp8()
        and getattr(recipe, "backward_override", None) is None
    )
