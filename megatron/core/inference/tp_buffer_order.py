# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Barriers before all-gathers on the inference "tp" symmetric buffer.

Inference-optimized column-parallel linears all-gather their sequence-parallel input into the
shared "tp" buffer of SymmetricMemoryManager, and their GEMMs then read the gathered input from
it. Inference row-parallel linears write their GEMM output into it and reduce-scatter that, as
the vocab-parallel embedding does with its output. Each all-gather barriers before it writes
(see multimem_all_gather), unless the code that knows the order of these operations drops the
barrier with one of the two rules below. A missing declaration costs a barrier, never a race.
The declarations are static, made once modules are built, so CUDA graph capture and replay take
the same decisions.

A module states the order of its operations on the buffer by defining a method
``declare_tp_buffer_order(preceding_reduce_scatter)``. Given the module whose reduce-scatter is
the last operation on the buffer before the module runs (or None if that is unknown), it applies
the rules to its all-gathering linears, and returns the module whose reduce-scatter is its own
last operation on the buffer (or None). A container passes its submodules, in forward order,
through chain_tp_buffer_order.
"""

from typing import Optional

import torch

# Modules rather than names: megatron.core.tensor_parallel imports this module, through
# megatron.core.transformer, before it defines these classes.
from megatron.core.tensor_parallel import inference_layers
from megatron.core.tensor_parallel import layers as tensor_parallel_layers
from megatron.core.transformer.identity_op import IdentityOp


def _is_all_gathering_linear(module) -> bool:
    """Whether module is an inference linear that all-gathers its input into the buffer."""
    # Without Transformer Engine, which they build on, the inference linears are not classes.
    return inference_layers.HAVE_TE and isinstance(
        module,
        (
            inference_layers.InferenceColumnParallelLinear,
            inference_layers.InferenceLayerNormColumnParallelLinear,
        ),
    )


def _reduce_scatter_width(module) -> Optional[int]:
    """Width of the rows that module reduce-scatters across its TP group, or None."""
    if inference_layers.HAVE_TE and isinstance(module, inference_layers.InferenceRowParallelLinear):
        return module.out_features
    if isinstance(module, tensor_parallel_layers.VocabParallelEmbedding):
        # Without sequence parallelism it all-reduces with NCCL instead, which synchronizes the
        # group: an all-gather of any width is then safe.
        return module.embedding_dim
    return None


def skip_all_gather_barrier_after_reduce_scatter(reduce_scatter, all_gather_linear) -> None:
    """Skip all_gather_linear's barrier if it follows reduce_scatter's reduce-scatter.

    The caller states the order: all_gather_linear's input all-gather is the first operation on
    the "tp" buffer after the reduce-scatter that ends reduce_scatter's forward, and its input
    holds the same tokens as that reduce-scatter's output (such as the residual stream it
    updated, possibly normalized). reduce_scatter is an inference row-parallel linear or a
    vocab-parallel embedding. The barrier is dropped only if the all-gather then writes exactly
    the per-rank segments the reduce-scatter reads: both are on the same TP group, the
    all-gather's input width is the reduce-scatter's output width, and both carry the activation
    dtype (no FP32 residual stream). Otherwise, including when either is None, this does
    nothing.

    If the reduce-scatter falls back to NCCL, that synchronizes the TP group instead, which also
    makes the barrier redundant. With fused TP communication, the reduce-scatter also all-gathers
    the normalized input of the next linear, which then reads it without an all-gather.
    """
    width = _reduce_scatter_width(reduce_scatter)
    if (
        width is not None
        and _is_all_gathering_linear(all_gather_linear)
        and reduce_scatter.tp_group is all_gather_linear.tp_group
        and width == all_gather_linear.in_features
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
    if _is_all_gathering_linear(all_gather_linear):
        all_gather_linear.barrier_before_all_gather = False


def chain_tp_buffer_order(
    module: Optional[torch.nn.Module], preceding_reduce_scatter=None
) -> Optional[torch.nn.Module]:
    """Apply module's declaration of its order of operations on the "tp" buffer.

    IdentityOp (or None) does nothing on the buffer. A module without a declare_tp_buffer_order
    method may do anything there, so its all-gathers keep their barriers, and so does the first
    all-gather after it. A container calls this for its submodules in forward order once they
    are built, passing each result on to the next submodule. The first one gets what precedes
    the container, or None if that is unknown: a block cannot know what ran before its first
    layer.

    Args:
        module: A layer, or one of its sublayers.
        preceding_reduce_scatter: The module whose reduce-scatter is the last operation on the
            buffer before module runs, or None if that is unknown or not a reduce-scatter.

    Returns:
        The module whose reduce-scatter is the last operation on the buffer once module has
        run, or None if that is unknown or not a reduce-scatter.
    """
    if module is None or isinstance(module, IdentityOp):
        return preceding_reduce_scatter
    declare = getattr(module, "declare_tp_buffer_order", None)
    return None if declare is None else declare(preceding_reduce_scatter)
