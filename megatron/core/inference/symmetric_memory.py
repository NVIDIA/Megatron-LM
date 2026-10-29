# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lazy-initialized symmetric memory manager for inference.

Provides a registry of SymmetricMemoryBuffer instances keyed by a
user-supplied identifier (e.g. "tp", "ep").  Buffers are created on first
access so that callers never need to worry about initialization ordering
relative to the inference context.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import Optional

import torch

try:
    import torch.distributed._symmetric_memory as symm_mem

    HAVE_TORCH_SYMM_MEM = True
except ImportError:
    HAVE_TORCH_SYMM_MEM = False

try:
    import triton  # pylint: disable=unused-import

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def _ensure_signal_pad_size(world_size: int) -> None:
    """Grow torch's signal pad so the Triton collectives' barrier slots fit for this group.

    symm_mem_sync uses one int32 slot per (block, peer). torch reads the signal pad size
    when each allocation is made, so this must run before symm_mem.empty. It only grows
    the size, and every rank runs the same allocations in the same order, so all ranks of
    a group allocate the same pad.
    """
    # Imported here to avoid an import cycle: parallel_state imports this module, and the
    # kernel package imports megatron.core.utils, which imports parallel_state.
    from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
        SYMM_MEM_KERNEL_CONFIG,
    )

    if not hasattr(symm_mem, "set_signal_pad_size"):
        # Older torch (e.g. 2.9) has a fixed 2 KiB pad and no setter. Its allocator puts the
        # pad after the data and zeroes the whole granularity-rounded block, so barrier slots
        # past the pad land in zeroed memory that nothing else uses.
        return
    needed = SYMM_MEM_KERNEL_CONFIG.signal_pad_bytes(world_size)
    if symm_mem.get_signal_pad_size() < needed:
        symm_mem.set_signal_pad_size(needed)


class SymmetricMemoryBuffer:
    """
     symmetric memory buffer used in inference.
    This buffer is used by mcore-inference's low-latency
    NVLS all-gather and reduce-scatter collectives.
    """

    def __init__(self, size_in_mb, process_group):
        self.init_failure_reason: Optional[str] = None
        if not HAVE_TORCH_SYMM_MEM:
            self.init_failure_reason = "torch.distributed._symmetric_memory not importable"
            self.symm_buffer = None
            self.symm_mem_hdl = None
        elif not HAVE_TRITON:
            self.init_failure_reason = "triton not installed"
            self.symm_buffer = None
            self.symm_mem_hdl = None
        else:
            numel = int(size_in_mb * 1024 * 1024)  # size in bytes
            try:
                symm_mem.enable_symm_mem_for_group(process_group.group_name)
                _ensure_signal_pad_size(process_group.size())
                self.symm_buffer = symm_mem.empty(numel, dtype=torch.uint8, device='cuda')
                self.symm_mem_hdl = symm_mem.rendezvous(self.symm_buffer, process_group)
            except RuntimeError as e:
                self.init_failure_reason = f"{type(e).__name__}: {e}"
                self.symm_buffer = None
                self.symm_mem_hdl = None

    def _can_allocate(self, numel, dtype) -> bool:
        """
        Returns whether enough symmetric memory is available
        for the given tensor shape and dtype.
        """
        if self.symm_mem_hdl is None:
            return False
        size_of_dtype = torch.tensor([], dtype=dtype).element_size()
        required_len = numel * size_of_dtype
        return required_len <= self.symm_buffer.numel()

    def _allocate(self, numel, dtype) -> torch.Tensor:
        """
        Allocates a sub-tensor from the self.symm_buffer for the given numel and dtype"""
        required_bytes = numel * torch.tensor([], dtype=dtype).element_size()
        return self.symm_buffer[0:required_bytes].view(dtype).view(numel)

    def maybe_get_tensors(self, tensor_specs, alignment=16):
        """
        Pack multiple tensors contiguously in the symmetric buffer with alignment.

        Each tensor's starting offset is aligned to `alignment` bytes (default 16
        for 128-bit multimem access).

        Args:
            tensor_specs: list of (numel, dtype) tuples.
            alignment: byte alignment for each tensor's start offset (default 16).

        Returns:
            {"handle": None, "tensors": None} if unavailable or insufficient space.
            {"handle": symm_mem_hdl, "tensors": [(raw_byte_view, byte_offset), ...]}
            on success, where raw_byte_view is a uint8 slice of the buffer.
        """
        _NONE_RESULT = {"handle": None, "tensors": None}
        if self.symm_mem_hdl is None:
            return _NONE_RESULT

        # Compute aligned byte sizes and running offsets
        slices = []
        current_offset = 0
        for numel, dtype in tensor_specs:
            nbytes = numel * torch.tensor([], dtype=dtype).element_size()
            aligned_nbytes = ((nbytes + alignment - 1) // alignment) * alignment
            slices.append((current_offset, nbytes))
            current_offset += aligned_nbytes

        if not self._can_allocate(current_offset, torch.uint8):
            return _NONE_RESULT

        tensors = []
        for offset, nbytes in slices:
            tensors.append((self.symm_buffer[offset : offset + nbytes], offset))

        return {"handle": self.symm_mem_hdl, "tensors": tensors}

    def maybe_get_tensor(self, tensor_shape, dtype):
        """
        Returns (potentially) a sub-tensor from the self.symm_buffer for the given shape.
        If enough symmetric memory is not available, returns None.
        """
        if self.symm_mem_hdl is None:
            return {"tensor": None, "handle": None}
        numel = reduce(operator.mul, tensor_shape, 1)
        if not self._can_allocate(numel, dtype):
            return {"tensor": None, "handle": None}
        return {
            "tensor": self._allocate(numel, dtype).view(*tensor_shape),
            "handle": self.symm_mem_hdl,
        }


class SymmetricMemoryManager:
    """Registry of lazily-initialized symmetric memory buffers.

    Usage::

        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        result = buf.maybe_get_tensor(shape, dtype)
    """

    _buffers: dict[str, SymmetricMemoryBuffer] = {}
    _default_size_mb: int = 512

    @classmethod
    def get_buffer(
        cls,
        key: str,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        size_mb: Optional[int] = None,
    ) -> SymmetricMemoryBuffer:
        """Return the buffer for *key*, creating it on first call.

        Args:
            key: Unique identifier (e.g. "tp", "ep").
            process_group: Required on the first call for a given key.
                Subsequent calls may omit it.
            size_mb: Buffer size in MiB (default 256).
        """
        if key not in cls._buffers:
            assert (
                process_group is not None
            ), f"SymmetricMemoryManager: process_group is required on first access for key='{key}'"
            cls._buffers[key] = SymmetricMemoryBuffer(
                size_in_mb=size_mb or cls._default_size_mb, process_group=process_group
            )
        return cls._buffers[key]

    @classmethod
    def destroy(cls, key: Optional[str] = None) -> None:
        """Destroy one or all buffers.

        Args:
            key: If provided, destroy only that buffer. Otherwise destroy all.
        """
        if key is not None:
            cls._buffers.pop(key, None)
        else:
            cls._buffers.clear()

    @classmethod
    def is_initialized(cls, key: str) -> bool:
        """Check whether a buffer has been created for *key*."""
        return key in cls._buffers


# Barriers before all-gathers on the "tp" buffer.
#
# Inference-optimized column-parallel linears all-gather their sequence-parallel input into the
# shared "tp" buffer, and their GEMMs then read the gathered input from it. Inference row-parallel
# linears write their GEMM output into it and reduce-scatter that. Each all-gather barriers before
# it writes (see multimem_all_gather), unless the code that knows the order of these operations
# drops the barrier with one of the two functions below. A missing declaration costs a barrier,
# never a race. The declarations are static, made once modules are built, so CUDA graph capture
# and replay take the same decisions.


def skip_all_gather_barrier_after_reduce_scatter(reduce_scatter_linear, all_gather_linear) -> None:
    """Skip all_gather_linear's barrier if it follows reduce_scatter_linear's reduce-scatter.

    The caller states the order: all_gather_linear's input all-gather is the first operation on
    the "tp" buffer after reduce_scatter_linear's reduce-scatter, and its input holds the same
    tokens as that reduce-scatter's output (such as the residual stream it updated, possibly
    normalized). The barrier is dropped only if the all-gather then writes exactly the per-rank
    segments the reduce-scatter reads: both linears are inference-optimized and on the same TP
    group, the all-gather's input width is the reduce-scatter's output width, and both carry the
    activation dtype (no FP32 residual stream). Otherwise, including when either is None, this
    does nothing.

    If the reduce-scatter falls back to NCCL, that synchronizes the TP group instead, which also
    makes the barrier redundant. With fused TP communication, the reduce-scatter also all-gathers
    the normalized input of the next linear, which then reads it without an all-gather.
    """
    # Imported here: inference_layers imports this module.
    from megatron.core.tensor_parallel import inference_layers

    if not inference_layers.HAVE_TE:
        return  # The inference-optimized linears build on Transformer Engine.
    if (
        isinstance(reduce_scatter_linear, inference_layers.InferenceRowParallelLinear)
        and isinstance(
            all_gather_linear,
            (
                inference_layers.InferenceColumnParallelLinear,
                inference_layers.InferenceLayerNormColumnParallelLinear,
            ),
        )
        and reduce_scatter_linear.tp_group is all_gather_linear.tp_group
        and reduce_scatter_linear.out_features == all_gather_linear.in_features
        and not all_gather_linear.config.fp32_residual_connection
    ):
        all_gather_linear.barrier_before_all_gather = False


def skip_all_gather_barrier_after_synchronizing_collective(all_gather_linear) -> None:
    """Skip all_gather_linear's barrier because a collective synchronizes its TP group first.

    The caller states the order: between the last operation on the "tp" buffer and
    all_gather_linear's input all-gather, every rank of the TP group runs a collective that no
    rank completes before all have entered it, such as an NCCL all-gather. Each rank enters it
    only after its earlier kernels, including its reads of the buffer, are done. Does nothing
    unless all_gather_linear is an inference-optimized column-parallel linear.
    """
    # Imported here: inference_layers imports this module.
    from megatron.core.tensor_parallel import inference_layers

    if not inference_layers.HAVE_TE:
        return  # The inference-optimized linears build on Transformer Engine.
    if isinstance(
        all_gather_linear,
        (
            inference_layers.InferenceColumnParallelLinear,
            inference_layers.InferenceLayerNormColumnParallelLinear,
        ),
    ):
        all_gather_linear.barrier_before_all_gather = False


def chain_tp_buffer_collectives(module, preceding_reduce_scatter=None):
    """Skip the all-gather barriers that the order of module's operations on the "tp" buffer
    makes redundant.

    A module states that order by defining a method of this name. Given the linear whose
    reduce-scatter precedes the module, it calls the two functions above for its all-gathering
    linears and returns the linear whose reduce-scatter is its own last operation on the
    buffer. IdentityOp (or None) does nothing on the buffer. Any other module may do anything
    there, so its all-gathers keep their barriers, and so does the first all-gather after it.

    A container calls this for its submodules in forward order once they are built, passing
    each result on to the next submodule. The first one gets what precedes the container, or
    None if that is unknown: a block cannot know what ran before its first layer.

    Args:
        module: A layer, or one of its sublayers.
        preceding_reduce_scatter: The linear whose reduce-scatter is the last operation on the
            buffer before module runs, or None if that is unknown or not a reduce-scatter.

    Returns:
        The linear whose reduce-scatter is the last operation on the buffer once module has
        run, or None if that is unknown or not a reduce-scatter.
    """
    # Imported here: megatron.core.transformer imports parallel_state, which imports this module.
    from megatron.core.transformer.identity_op import IdentityOp

    if module is None or isinstance(module, IdentityOp):
        return preceding_reduce_scatter
    chain = getattr(module, "chain_tp_buffer_collectives", None)
    return None if chain is None else chain(preceding_reduce_scatter)
