# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent CPU FP64 references for explicit-group MCore collective mappings."""

from dataclasses import replace

import torch

from tools.determinism.reduction_reference import ReductionReference
from tools.determinism.reference import TensorPair

MAPPINGS = {
    "copy": "copy_to_tensor_model_parallel_region",
    "reduce": "reduce_from_tensor_model_parallel_region",
    "gather_first": "gather_from_sequence_parallel_region",
    "scatter_first": "reduce_scatter_to_sequence_parallel_region",
    "gather_last": "all_gather_last_dim_from_tensor_parallel_region",
    "scatter_last": "reduce_scatter_last_dim_to_tensor_parallel_region",
}


def collective_shapes(
    case: str, size: int, local_shape: tuple[int, int] = (257, 128)
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return input/output shapes for equal, nonempty first/last-dimension shards."""
    if case not in MAPPINGS or type(size) is not int or size < 2:
        raise ValueError("A supported multi-rank collective is required")
    if len(local_shape) != 2 or any(type(value) is not int or value < 1 for value in local_shape):
        raise ValueError("Two positive local dimensions are required")
    full = list(local_shape)
    full[0 if case.endswith("first") else 1] *= size
    if case.startswith("gather"):
        return local_shape, tuple(full)
    if case.startswith("scatter"):
        return tuple(full), local_shape
    return local_shape, local_shape


def collective_reference(
    case: str,
    inputs: list[torch.Tensor],
    grad_outputs: list[torch.Tensor],
    rank: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[TensorPair, dict[str, ReductionReference], TensorPair]:
    """Compute rank-local values without invoking a distributed operation.

    Inputs and upstream gradients contain one materialized CPU tensor per group
    rank. Reductions sum those exact values in FP64; BF16 references conservatively
    allow input-dtype accumulation instead of assuming NCCL accumulates in FP32.
    Gather/copy results have no numerical rounding allowance in the caller.
    """
    size = len(inputs)
    if case not in MAPPINGS or size < 2 or len(grad_outputs) != size or not 0 <= rank < size:
        raise ValueError("Complete inputs and gradients for every group rank are required")
    dtype = inputs[0].dtype
    tensors = inputs + grad_outputs
    if dtype not in (torch.float32, torch.bfloat16) or any(
        tensor.device.type != "cpu" or tensor.dtype != dtype or tensor.ndim < 1
        for tensor in tensors
    ):
        raise ValueError("Reference inputs must be non-scalar CPU FP32 or BF16 tensors")
    if any(tensor.shape != inputs[0].shape for tensor in inputs) or any(
        tensor.shape != grad_outputs[0].shape for tensor in grad_outputs
    ):
        raise ValueError("Only equal-sized rank shards are supported")
    x = [value.detach().double() for value in inputs]
    dy = [value.detach().double() for value in grad_outputs]
    dim = 0 if case.endswith("first") else inputs[0].ndim - 1
    reductions = {}

    def reduced(values, key, split):
        terms = torch.stack(values)
        if split:
            if terms.shape[dim + 1] % size:
                raise ValueError("Reduced dimension must divide the group size")
            terms = terms.chunk(size, dim=dim + 1)[rank]
        reduction = ReductionReference.from_terms(
            terms,
            dim=0,
            keepdim=False,
            rounding="exact materialized rank-local values before collective addition",
            accumulation_dtype=dtype,
            exact_terms=True,
        )
        reductions[key] = replace(
            reduction,
            total=reduction.total.to(device),
            sum_absolute_terms=reduction.sum_absolute_terms.to(device),
        )
        return reduction.total

    if case == "copy":
        output, gradient = x[rank], reduced(dy, "gradient:in[0]", False)
    elif case == "reduce":
        output, gradient = reduced(x, "output:out", False), dy[rank]
    elif case.startswith("gather"):
        output = torch.cat(x, dim=dim)
        gradient = reduced(dy, "gradient:in[0]", True)
    else:
        output = reduced(x, "output:out", True)
        gradient = torch.cat(dy, dim=dim)
    if output.shape != grad_outputs[rank].shape or gradient.shape != inputs[rank].shape:
        raise ValueError("Output/upstream-gradient or input/input-gradient shapes differ")
    mathematical = ({"out": output.to(device)}, {"in[0]": gradient.to(device)})
    expected = tuple(
        {name: value.to(dtype) for name, value in category.items()} for category in mathematical
    )
    return expected, reductions, mathematical


def collective_rtol(dtype: torch.dtype, size: int) -> float:
    """Use gamma(size-1) plus final rounding as a separate relative L2 guard."""
    if dtype not in (torch.float32, torch.bfloat16) or type(size) is not int or size < 2:
        raise ValueError("A supported dtype and multi-rank group are required")
    epsilon = torch.finfo(dtype).eps
    nu = (size - 1) * epsilon / 2
    if nu >= 1:
        raise ValueError("Group is too large for the stated accumulation bound")
    return nu / (1 - nu) + epsilon
