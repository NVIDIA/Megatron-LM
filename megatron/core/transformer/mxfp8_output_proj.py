# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Custom MXFP8 path for the LM head output_layer.

Runs the vocab-projection GEMM used by LinearCrossEntropyModule in MXFP8
without swapping the parent class from ColumnParallelLinear to a TE module
(which would add _extra_state to the checkpoint and break backward
compatibility). The Parameter itself stays bf16; an MXFP8 quantized view is
cached on the module and invalidated by weight._version.

Three independent toggles:
  fp8_fprop  (always on when this path is taken)
  fp8_dgrad  whether dgrad GEMM is fp8 (else bf16)
  fp8_wgrad  whether wgrad GEMM is fp8 (else bf16)
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch

try:
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.quantized_tensor import (
        prepare_for_saving,
        restore_from_func_ctx,
    )
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine_torch import DType as TE_DType

    HAVE_TE_MXFP8 = True
except ImportError:
    HAVE_TE_MXFP8 = False


__all__ = ["HAVE_TE_MXFP8", "mxfp8_column_parallel_linear", "is_mxfp8_output_proj_active"]


def is_mxfp8_output_proj_active(config) -> bool:
    """Check if the custom MXFP8 output proj path should run.

    We deliberately do NOT require FP8GlobalStateManager.is_fp8_enabled() here:
    DeepSeek V3's fine-grained 1F1B schedule enters fp8_autocast per decoder
    layer only, so `post_process.forward` (which owns the LM-head GEMM) runs
    outside any fp8 context. Our path does its own explicit MXFP8 quantization,
    so the global TE fp8 state doesn't need to be on.

    Requires:
      - config.fp8_output_proj True
      - config.fp8_recipe == 'mxfp8' (defensive: avoid activating under a
        different global recipe, e.g. delayed scaling)
      - TE MXFP8 stack importable
    """
    if not HAVE_TE_MXFP8:
        return False
    if not getattr(config, "fp8_output_proj", False):
        return False
    # Only activate when the model is configured for MXFP8 globally.
    fp8_recipe = getattr(config, "fp8_recipe", None)
    if fp8_recipe is not None and str(fp8_recipe).lower() not in ("mxfp8", "fp8recipe.mxfp8"):
        return False
    return True


def _get_split_acc(attr: str, default: bool = True) -> bool:
    """Look up use_split_accumulator from the active recipe if one is set,
    otherwise return the sensible default for MXFP8 (True)."""
    try:
        recipe = FP8GlobalStateManager.get_fp8_recipe()
    except Exception:
        return default
    mm_params = getattr(recipe, attr, None)
    if mm_params is None:
        return default
    return getattr(mm_params, "use_split_accumulator", default)


def _make_quantizer(rowwise: bool, columnwise: bool) -> "MXFP8Quantizer":
    # E4M3 matches the MXFP8 recipe used elsewhere in the model.
    return MXFP8Quantizer(TE_DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)


def _get_cached_weight_q(module, weight: torch.Tensor, need_columnwise: bool):
    """Return an MXFP8-quantized weight.

    NOTE: caching via weight._version was unreliable with overlap_param_gather
    (custom kernels write to weight without bumping _version), so we always
    re-quantize for correctness. Revisit once version bump is reliable.
    """
    quantizer = _make_quantizer(rowwise=True, columnwise=need_columnwise)
    return quantizer(weight)


class _MXFP8ColumnParallelLinearFunc(torch.autograd.Function):
    """Column-parallel linear y = x @ W^T with MXFP8 fprop + optional fp8 dgrad/wgrad.

    Mirrors the sequence-parallel gather / allreduce-dgrad / wgrad-into-main_grad
    behavior of Megatron's LinearWithGradAccumulationAndAsyncCommunication,
    but replaces the GEMMs with TE general_gemm and MXFP8 tensors.
    """

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        weight_q,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        tp_group,
        fp8_dgrad: bool,
        fp8_wgrad: bool,
    ):
        tp_world = (
            torch.distributed.get_world_size(tp_group) if tp_group is not None else 1
        )

        if sequence_parallel and tp_world > 1:
            gathered_shape = list(input_.size())
            gathered_shape[0] = gathered_shape[0] * tp_world
            total_input = torch.empty(
                gathered_shape, dtype=input_.dtype, device=input_.device
            )
            torch.distributed.all_gather_into_tensor(
                total_input, input_.contiguous(), group=tp_group
            )
        else:
            total_input = input_

        activation_dtype = total_input.dtype
        out_features = weight.shape[0]

        # [s, b, h] -> [s*b, h] so quantize sees a 2D tensor whose last dim is hidden.
        input_2d = total_input.reshape(-1, total_input.shape[-1])

        # For wgrad we need columnwise input usage; for dgrad we only need rowwise.
        input_quantizer = _make_quantizer(rowwise=True, columnwise=fp8_wgrad)
        input_q = input_quantizer(input_2d)

        use_split_acc_fprop = _get_split_acc("fp8_gemm_fprop")

        # y = x @ W^T. TE convention: general_gemm(weight_q, input_q, layout="TN")
        # matches the TE Linear forward path exactly.
        out_2d, *_ = general_gemm(
            weight_q,
            input_q,
            out_dtype=activation_dtype,
            layout="TN",
            bias=bias,
            use_split_accumulator=use_split_acc_fprop,
            grad=False,
        )
        output_shape = list(total_input.shape[:-1]) + [out_features]
        output = out_2d.reshape(output_shape)

        # --- save for backward ---
        ctx.tp_group = tp_group
        ctx.tp_world = tp_world
        ctx.sequence_parallel = sequence_parallel
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.fp8_dgrad = fp8_dgrad
        ctx.fp8_wgrad = fp8_wgrad
        ctx.use_bias = bias is not None
        ctx.activation_dtype = activation_dtype
        ctx.gathered_shape = list(total_input.shape)
        ctx.local_in_shape = list(input_.shape)

        # Grad-accumulation fusion: wgrad writes directly into weight.main_grad.
        # Stash the python ref (save_for_backward would block in-place writes).
        ctx.main_grad = (
            weight.main_grad
            if gradient_accumulation_fusion and hasattr(weight, "main_grad")
            else None
        )
        ctx.weight_ref = weight  # for grad_added_to_main_grad handshake

        # Inputs to save for backward depend on the wgrad path.
        saved_input = input_q if fp8_wgrad else input_2d
        # For dgrad we want the weight: quantized if fp8_dgrad, else bf16.
        # weight_q is cached on the module; `prepare_for_saving` nulls the
        # MXFP8Tensor's internal data fields, so we pass a detached view that
        # shares the underlying storage — nulling the view leaves the cache
        # intact for the next microbatch.
        saved_weight = weight_q.detach() if fp8_dgrad else weight

        tensors_to_save, tensor_objects = prepare_for_saving(
            saved_input, saved_weight
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved_input, saved_weight = restore_from_func_ctx(ctx)

        grad_output = grad_output.contiguous()
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        activation_dtype = ctx.activation_dtype

        # ================= dgrad =================
        grad_output_q = None
        if ctx.fp8_dgrad:
            # grad_out rowwise for NN GEMM; keep columnwise now if wgrad also fp8.
            go_quantizer = _make_quantizer(rowwise=True, columnwise=ctx.fp8_wgrad)
            grad_output_q = go_quantizer(grad_output_2d)
            if hasattr(saved_weight, "update_usage"):
                saved_weight.update_usage(columnwise_usage=True)
            use_split_acc = _get_split_acc("fp8_gemm_dgrad")
            dgrad_2d, *_ = general_gemm(
                saved_weight,
                grad_output_q,
                out_dtype=activation_dtype,
                layout="NN",
                use_split_accumulator=use_split_acc,
                grad=True,
            )
        else:
            # bf16 dgrad: dx = dy @ W
            dgrad_2d = grad_output_2d.matmul(saved_weight)

        dgrad_full = dgrad_2d.reshape(ctx.gathered_shape)

        # ============ SP reduce-scatter / TP all-reduce on dgrad ============
        # Kick off async so wgrad can overlap.
        dgrad_handle = None
        dgrad = dgrad_full
        if ctx.sequence_parallel and ctx.tp_world > 1:
            sub_grad_input = torch.empty(
                ctx.local_in_shape, dtype=dgrad.dtype, device=dgrad.device
            )
            dgrad_handle = torch.distributed.reduce_scatter_tensor(
                sub_grad_input, dgrad, group=ctx.tp_group, async_op=True
            )
            dgrad = sub_grad_input
        elif ctx.allreduce_dgrad and ctx.tp_world > 1:
            dgrad_handle = torch.distributed.all_reduce(
                dgrad, group=ctx.tp_group, async_op=True
            )

        # ================= wgrad =================
        # dw = dy^T @ x  -> TE layout "NT" with (x, dy)
        grad_weight_out = None
        weight = ctx.weight_ref
        if ctx.fp8_wgrad:
            # Need columnwise usage on both operands for the NT fp8 GEMM.
            if grad_output_q is None or not grad_output_q._quantizer.columnwise_usage:
                go_quantizer = _make_quantizer(rowwise=False, columnwise=True)
                grad_output_q = go_quantizer(grad_output_2d)
            else:
                grad_output_q.update_usage(columnwise_usage=True)
            saved_input.update_usage(columnwise_usage=True)

            use_split_acc = _get_split_acc("fp8_gemm_wgrad")
            wgrad_out_dtype = (
                ctx.main_grad.dtype
                if ctx.main_grad is not None
                else activation_dtype
            )
            dw, *_ = general_gemm(
                saved_input,
                grad_output_q,
                out_dtype=wgrad_out_dtype,
                layout="NT",
                out=ctx.main_grad,
                accumulate=ctx.main_grad is not None,
                use_split_accumulator=use_split_acc,
                grad=True,
            )
        else:
            if ctx.main_grad is not None and ctx.main_grad.dtype == torch.float32:
                # Use TE general_gemm to accumulate bf16 x bf16 -> fp32 directly.
                dw, *_ = general_gemm(
                    saved_input,
                    grad_output_2d,
                    out_dtype=ctx.main_grad.dtype,
                    layout="NT",
                    out=ctx.main_grad,
                    accumulate=True,
                    grad=True,
                )
            elif ctx.main_grad is not None:
                tmp = grad_output_2d.t().matmul(saved_input)
                ctx.main_grad.add_(tmp)
                dw = ctx.main_grad
            else:
                dw = grad_output_2d.t().matmul(saved_input)

        # Communicate grad_added_to_main_grad back to Megatron's DDP hooks.
        if ctx.main_grad is not None and hasattr(weight, "grad_added_to_main_grad"):
            weight.grad_added_to_main_grad = True
            # Return a zero-size tensor so downstream hooks still fire but don't
            # re-accumulate.
            grad_weight_out = torch.zeros(
                weight.shape,
                dtype=weight.dtype,
                device=weight.device,
                requires_grad=False,
            )
        elif ctx.main_grad is not None:
            grad_weight_out = None
        else:
            grad_weight_out = dw

        grad_bias = grad_output_2d.sum(dim=0) if ctx.use_bias else None

        if dgrad_handle is not None:
            dgrad_handle.wait()

        return (
            dgrad,             # input_
            grad_weight_out,   # weight
            grad_bias,         # bias
            None,              # weight_q (non-differentiable, cached on module)
            None,              # gradient_accumulation_fusion
            None,              # allreduce_dgrad
            None,              # sequence_parallel
            None,              # tp_group
            None,              # fp8_dgrad
            None,              # fp8_wgrad
        )


def mxfp8_column_parallel_linear(
    module,
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    tp_group,
    fp8_dgrad: bool,
    fp8_wgrad: bool,
) -> torch.Tensor:
    """Public entry point. Manages the module's cached quantized weight and
    invokes the custom autograd function."""
    torch.cuda.nvtx.range_push("mxfp8_output_proj")
    try:
        weight_q = _get_cached_weight_q(module, weight, need_columnwise=fp8_dgrad)
        out = _MXFP8ColumnParallelLinearFunc.apply(
            input_,
            weight,
            bias,
            weight_q,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            tp_group,
            fp8_dgrad,
            fp8_wgrad,
        )
    finally:
        torch.cuda.nvtx.range_pop()
    return out
