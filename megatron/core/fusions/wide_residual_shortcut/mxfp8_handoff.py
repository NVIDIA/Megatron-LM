# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Direct FP32-to-MXFP8 seam outputs for an exclusive TE Linear consumer.

A seam writes reader zero only as rowwise MXFP8, the operand of the consumer's forward GEMM. Its
weight-gradient GEMM needs the columnwise operand: when TE saves the operand for backward, a
per-slot saved-tensor hook drops it, and unpacking returns the columnwise operand replayed
bit-exactly from the seam's input residual and tape (with checkpoint replay, from the interval,
replayed before any of its GEMMs). TE's backward, including weight-gradient accumulation and
distributed-weight finalization, runs unchanged.
"""

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
    """Where one seam call's MXFP8 reader zero gets its columnwise operand from."""

    def __init__(self, tensor, columns, consumer):
        self.columns = columns  # () -> the columnwise MXFP8 operand
        self.consumer = ref(consumer)
        # The weak reference prevents StorageImpl address reuse without retaining bytes.
        _READERS[StorageWeakRef(tensor._rowwise_data.untyped_storage())] = self
        tensor._shortcut_mxfp8_reader = self


def install_linear_hooks(tensor, output, linear):
    """After ``linear``'s forward on a seam's MXFP8 operand: its backward unpacks the replayed
    columnwise operand in place of the saved rowwise one (TE's own save slot only; a scope-wide
    saved-tensor hook would detach the saved weight and lose GTP metadata)."""
    reader = reader_for(tensor)
    if reader is None or not torch.is_grad_enabled():
        return
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if not FP8GlobalStateManager.is_fp8_enabled() or not recipe.mxfp8():
        raise RuntimeError("shortcut MXFP8 input requires an MXFP8 TE linear consumer")
    if getattr(recipe, "backward_override", None) is not None:
        raise RuntimeError("shortcut MXFP8 handoff requires ordinary MXFP8 backward")
    if reader.consumer() is not linear:
        raise RuntimeError("a seam's MXFP8 operand must be consumed only by its declared Linear")
    output = output[0] if isinstance(output, tuple) else output
    node = output.grad_fn
    key = tensor._rowwise_data.untyped_storage()._cdata
    matched = 0
    for value, slot in zip(node.saved_tensors, node._raw_saved_tensors):
        data = getattr(value, "_rowwise_data", None)
        if data is not None and data.untyped_storage()._cdata == key:
            slot.register_hooks(torch.Tensor.size, lambda size: reader.columns().view(size))
            matched += 1
    if matched != 1:
        raise RuntimeError("TE did not save exactly one shortcut MXFP8 activation")


def linear_eligible(linear, config) -> bool:
    """Construction time: whether ``linear`` may take a seam's reader zero directly as MXFP8.

    Its layer's precision is MXFP8 (Megatron fixes precision per layer) at a hidden size the
    kernel quantizes, and it is a TELinear taking its input whole (no TP, no userbuffer overlap
    or quantized parameters).
    """
    from megatron.core.extensions.transformer_engine import TELinear

    from .kernel_utils import WIDTHS, qualified_device

    return bool(
        config.fp8
        and config.fp8_recipe == "mxfp8"
        and config.hidden_size in WIDTHS
        and qualified_device(torch.cuda.get_device_properties(torch.cuda.current_device()))
        and isinstance(linear, TELinear)
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
    )


def check_call(tensor) -> None:
    """Per call, for a consumer chosen at construction: raise unless its FP8 context is active
    and the kernel takes this token count with an MXFP8 reader. Precision never changes per
    call: training shapes are fixed for a run (the tuned set), and inference rows must be
    divisible by 32, as TE's MXFP8 GEMM requires too (Megatron's inference pads to 64)."""
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    from .kernel_utils import SHAPES

    if not FP8GlobalStateManager.is_fp8_enabled():
        raise RuntimeError("an MXFP8 layer's fused seam ran outside its FP8 context")
    rows, hidden = tensor.numel() // tensor.shape[-1], tensor.shape[-1]
    training = torch.is_grad_enabled()
    if (rows, hidden) not in SHAPES if training else rows % 32:
        raise ValueError(
            f"the MXFP8 seam handoff does not support {rows} tokens at hidden size {hidden} in "
            + (
                "training (supported: multiples of 1024 from 2048 tokens, up to 16384 above "
                "hidden 4096)"
                if training
                else "inference (rows must be divisible by 32)"
            )
        )
