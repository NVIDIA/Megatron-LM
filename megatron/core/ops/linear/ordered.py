# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ordered row-parallel GEMMs, operation-owned symmetric scratch, and autograd.

Inputs are local K shards [..., K/TP]; outputs are sequence shards or replicated rows.
The caller supplies the process group. Backward keeps ordinary TP collectives."""

from typing import Optional

import torch

from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.ops.collectives.ordered_reduce import (
    ordered_all_reduce_shards,
    ordered_reduce_scatter_shards,
    symm_ordered_reduce_scatter_shards,
)

try:
    import fused_weight_gradient_mlp_cuda

    HAVE_GRAD_ACCUM_FUSION = True
except ImportError:
    HAVE_GRAD_ACCUM_FUSION = False

try:
    from transformer_engine.pytorch import cpp_extensions as te_cpp_extensions
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _training_buffer_key(group) -> str:
    return f"tp_ordered_training:{group.group_name}"


def prepare_ordered_reduce(group) -> None:
    """Collectively allocate per-TP GEMM scratch before CUDA-graph capture.

    The fixed-capacity pool is shared by row-parallel layers. Outputs do not alias it and
    backward saves only inputs and weights. Oversized GEMMs and unavailable peer access
    use the all-to-all fallback, without allocating symmetric memory inside capture.
    """
    if group.size() >= 2:
        key = _training_buffer_key(group)
        assert (
            SymmetricMemoryManager.is_initialized(key)
            or not torch.cuda.is_current_stream_capturing()
        ), "allocate ordered row-linear scratch at layer construction or during eager warmup"
        SymmetricMemoryManager.get_buffer(key, process_group=group)


def row_parallel_shard_partials(
    x: torch.Tensor, weight: torch.Tensor, num_shards: int, out: Optional[torch.Tensor] = None
):
    """Shard partials ``[num_shards, *x.shape[:-1], N]`` of ``x @ weight.T`` (weight ``[N, K]``).

    Shard ``j`` is ``x[..., j*K/c:(j+1)*K/c] @ weight[:, j*K/c:(j+1)*K/c].T``, computed for all
    ``j`` in one batched GEMM on strided views (written into ``out`` when given, e.g. a
    symmetric buffer).

    Used when a rank holds several ``batch_invariant_tp_shards``: by the training forward
    (``tp_shard_row_linear``) and by ``InferenceRowParallelLinear._matmul_shards_reduce``.
    """
    *lead, k = x.shape
    n = weight.shape[0]
    assert k % num_shards == 0, f"K ({k}) must be divisible by the local shard count {num_shards}"
    kc = k // num_shards
    # Flatten the leading dims (e.g. [s, b]) into the GEMM's M; free for the contiguous
    # activations the callers pass (a non-contiguous x would be copied here).
    x2 = x.reshape(-1, k)
    # Strided views, no copies: batch j reads columns j*kc:(j+1)*kc of x and of the weight.
    # Each batch entry is the same [M, kc] x [kc, N] GEMM a rank holding only shard j runs,
    # so its bits match (cuBLASLt, workspace 0 under batch-invariant mode; tested).
    xb = x2.view(x2.shape[0], num_shards, kc).transpose(0, 1)  # [c, M, kc]
    wb = weight.view(n, num_shards, kc).permute(1, 2, 0)  # [c, kc, N]
    if out is not None:
        torch.bmm(xb, wb, out=out.view(num_shards, x2.shape[0], n))
        return out.view(num_shards, *lead, n)
    return torch.bmm(xb, wb).view(num_shards, *lead, n)


def _accumulate_wgrad(
    weight: torch.Tensor, grad_out2d: torch.Tensor, input2d: torch.Tensor, use_te_gemm: bool = False
):
    """Weight gradient with Megatron's main_grad accumulation (as in tensor_parallel.layers)."""
    main_grad = getattr(weight, "main_grad", None)
    if main_grad is None:
        if use_te_gemm:
            return te_cpp_extensions.general_gemm(
                input2d, grad_out2d, layout="NT", out_dtype=weight.dtype, grad=True
            )[0]
        return grad_out2d.t().matmul(input2d)
    if use_te_gemm:
        te_cpp_extensions.general_gemm(
            input2d,
            grad_out2d,
            layout="NT",
            out=main_grad,
            out_dtype=main_grad.dtype,
            grad=True,
            accumulate=True,
        )
    elif HAVE_GRAD_ACCUM_FUSION and main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(input2d, grad_out2d, main_grad)
    elif HAVE_GRAD_ACCUM_FUSION and main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(input2d, grad_out2d, main_grad)
    else:
        main_grad.add_(grad_out2d.t().matmul(input2d).to(main_grad.dtype))
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    weight.grad_added_to_main_grad = True
    # A dummy gradient keeps the DDP backward hooks on the main backprop thread.
    zero = getattr(weight, "zero_out_wgrad", False)
    if HAVE_TE:
        return get_dummy_wgrad(list(main_grad.shape), input2d.dtype, zero=zero)
    factory = torch.zeros if zero else torch.empty
    return factory(main_grad.shape, dtype=input2d.dtype, device=input2d.device)


class _TPShardRowLinear(torch.autograd.Function):
    """Row-parallel linear whose forward folds this rank's shard partials in global order.

    Only the forward is fixed-order. The backward is the stock row-parallel one: all-gather
    the output gradient under sequence parallelism, then the full local-K input and weight
    gradient GEMMs. ``TERowParallelLinear`` uses TE GEMMs for one local partial and the
    existing batched GEMM definition for several ``batch_invariant_tp_shards``.
    """

    @staticmethod
    def forward(
        ctx, x, weight, group, num_shards, sequence_parallel, wgrad_into_main_grad, use_te_gemm
    ):
        """Shard partials of x @ weight.T, reduced (SP: reduce-scattered) in shard order."""
        shape = (num_shards, *x.shape[:-1], weight.shape[0])
        scratch = {"tensor": None, "handle": None}
        key = _training_buffer_key(group)
        if group.size() > 1 and x.shape[0] % group.size() == 0:
            if (
                not SymmetricMemoryManager.is_initialized(key)
                and not torch.cuda.is_current_stream_capturing()
            ):
                prepare_ordered_reduce(group)
            if SymmetricMemoryManager.is_initialized(key):
                scratch = SymmetricMemoryManager.get_buffer(key).maybe_get_tensor(shape, x.dtype)
        if use_te_gemm:
            # Keep TE's GEMM definition: a torch GEMM need not have the same fp32 bits.
            assert num_shards == 1
            parts = scratch["tensor"]
            if parts is None:
                parts = x.new_empty(shape)
            te_cpp_extensions.general_gemm(
                weight,
                x.reshape(-1, x.shape[-1]).contiguous(),
                out_dtype=x.dtype,
                out=parts.view(-1, weight.shape[0]),
                use_split_accumulator=False,
            )
        else:
            parts = row_parallel_shard_partials(x, weight, num_shards, out=scratch["tensor"])
        if scratch["handle"] is not None:
            out = symm_ordered_reduce_scatter_shards(parts, scratch["handle"], sync_after=True)
            if not sequence_parallel:
                full = x.new_empty(shape[1:])
                torch.distributed.all_gather_into_tensor(full, out, group=group)
                out = full
        elif sequence_parallel:
            out = ordered_reduce_scatter_shards(parts, group)
        else:
            out = ordered_all_reduce_shards(parts, group)
        ctx.save_for_backward(x, weight)
        ctx.group = group
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_into_main_grad = wgrad_into_main_grad
        ctx.use_te_gemm = use_te_gemm
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Stock row-parallel backward (all-gather under SP, full local-K GEMMs)."""
        x, weight = ctx.saved_tensors
        group = ctx.group
        world_size = group.size()
        if ctx.sequence_parallel and world_size > 1:
            grad_output = grad_output.contiguous()
            full = torch.empty(
                (grad_output.shape[0] * world_size, *grad_output.shape[1:]),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            torch.distributed.all_gather_into_tensor(full, grad_output, group=group)
            grad_output = full
        g2 = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        grad_input = None
        if ctx.needs_input_grad[0]:
            if ctx.use_te_gemm:
                grad_input = te_cpp_extensions.general_gemm(
                    weight, g2, layout="NN", out_dtype=x.dtype, grad=True
                )[0]
            else:
                grad_input = g2.matmul(weight)
            grad_input = grad_input.view(x.shape)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            if ctx.wgrad_into_main_grad:
                grad_weight = _accumulate_wgrad(weight, g2, x2, use_te_gemm=ctx.use_te_gemm)
            elif ctx.use_te_gemm:
                grad_weight = te_cpp_extensions.general_gemm(
                    x2, g2, layout="NT", out_dtype=weight.dtype, grad=True
                )[0]
            else:
                grad_weight = g2.t().matmul(x2)
        return grad_input, grad_weight, None, None, None, None, None


def tp_shard_row_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    group,
    num_shards: int,
    sequence_parallel: bool,
    wgrad_into_main_grad: bool = False,
    use_te_gemm: bool = False,
) -> torch.Tensor:
    """Row-parallel ``x @ weight.T`` with the fixed-shard forward reduction (see above)."""
    return _TPShardRowLinear.apply(
        x, weight, group, num_shards, sequence_parallel, wgrad_into_main_grad, use_te_gemm
    )
