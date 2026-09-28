"""Megatron-specific helpers for working with spmd_types.

These helpers encode Megatron's conventions for how parameters are distributed
and where gradient reductions occur. Other codebases will need similar helpers,
but the specific types will differ.
"""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.nn as nn

from megatron.core.tensor_parallel.layers import VocabParallelEmbedding


def _tensor_parallel_type(tensor: torch.Tensor) -> spmd.SpmdType:
    """Infer a parameter's TP type from Megatron's distribution metadata.

    ``megatron.core.tensor_parallel.layers.set_tensor_model_parallel_attributes``
    marks physically sharded parameters with ``tensor_model_parallel`` and records
    their global shard dimension in ``partition_dim``. Column-parallel weights use
    dimension 0, while row-parallel weights use dimension 1.

    ``sequence_parallel`` means something different when attached to a
    parameter: the parameter value is replicated, but each TP rank accumulates
    a gradient from its local sequence shard. The
    ``_allreduce_non_tensor_model_parallel_grads`` helper in
    ``megatron.core.distributed.finalize_model_grads`` reads this attribute and
    sums those partial gradients across TP, which is exactly the R (replicated
    value, partial gradient) contract.

    Other trainable parameters are invariant across TP. Non-trainable tensors
    are treated as replicated because only their forward value matters.
    """
    if getattr(tensor, "tensor_model_parallel", False):
        return spmd.S(getattr(tensor, "partition_dim"))
    if getattr(tensor, "sequence_parallel", False):
        return spmd.R
    return spmd.I if tensor.requires_grad else spmd.R


def annotate_tensor(
    tensor: torch.Tensor, *, tensor_parallel_type: spmd.SpmdType | None = None
) -> None:
    """Annotate a model-owned tensor with its TP type."""
    spmd.assert_type(tensor, {"TP": tensor_parallel_type or _tensor_parallel_type(tensor)})


def _annotate_vocab_bounds(module: VocabParallelEmbedding) -> None:
    """Type the per-rank vocabulary bounds that the embedding masks against.

    They are Python ints, so ordinary propagation cannot see that comparing
    token ids against them produces rank-varying results. Wrapping them as
    typed scalars makes the variation visible at its source; outside a
    type-checking run a ``Scalar`` behaves as its plain value.
    """
    bound_type = {"TP": spmd.V}
    module.vocab_start_index = spmd.Scalar(module.vocab_start_index, bound_type)
    module.vocab_end_index = spmd.Scalar(module.vocab_end_index, bound_type)


def annotate_model(model: nn.Module) -> None:
    """Annotate model-owned tensors, and the few typed scalars, from Megatron's conventions."""
    # ``finalize_model_grads`` also sums, by parameter name, the gradients of
    # q/k layernorms, which each rank computes from only its local heads.
    qk_layernorm = getattr(getattr(model, "config", None), "qk_layernorm", False)
    for module_name, module in model.named_modules():
        if isinstance(module, VocabParallelEmbedding):
            _annotate_vocab_bounds(module)
        for tensor in module.parameters(recurse=False):
            if qk_layernorm and ("q_layernorm" in module_name or "k_layernorm" in module_name):
                annotate_tensor(tensor, tensor_parallel_type=spmd.R)
            else:
                annotate_tensor(tensor)
        # DistributedDataParallel shadows ``buffers`` with its list of gradient
        # buffers, so call the nn.Module method explicitly.
        for tensor in nn.Module.buffers(module, recurse=False):
            annotate_tensor(tensor)
        # A few production modules, including YarnRotaryEmbedding, own replicated
        # constant tensors without registering them as buffers.
        for value in vars(module).values():
            if isinstance(value, torch.Tensor) and not value.requires_grad:
                annotate_tensor(value, tensor_parallel_type=spmd.R)
