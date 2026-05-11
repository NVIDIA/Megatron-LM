# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear API"""
import os
import warnings
from typing import Callable, Dict, Optional, Tuple, Union, List
from functools import reduce
from operator import mul as multiply_op

import torch
from torch.nn import init

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch import torch_version
from transformer_engine.pytorch.tensor.utils import is_experimental
from .base import (
    fill_userbuffers_buffer_for_all_gather,
    get_workspace,
    get_ub,
    TransformerEngineBaseModule,
    get_dummy_wgrad,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..quantization import FP8GlobalStateManager
from ..utils import (
    assert_dim_for_fp8_exec,
    assert_dim_for_all_gather,
    cast_if_needed,
    clear_tensor_data,
    divide,
    get_default_init_method,
    init_method_constant,
    nvtx_range_pop,
    nvtx_range_push,
    requires_grad,
    needs_quantized_gemm,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    symmetric_all_reduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ._common import apply_normalization, noop_cat, WeightGradStore
from ..tensor.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from ...debug.pytorch.debug_state import TEDebugState
from ..tensor.float8_blockwise_tensor import Float8BlockQuantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..export import is_in_onnx_export_mode, assert_warmed_up
from ..cpu_offload import is_cpu_offload_enabled, mark_activation_offload

from ..cpp_extensions import (
    general_gemm,
)

__all__ = ["LayerNormLinear"]


class _LayerNormLinear(torch.autograd.Function):
    """LayerNormLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        fuse_wgrad_accumulation: bool,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
        grad_weight_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        normalization: str,
        ub_overlap_ag_fprop: bool,
        ub_overlap_rs_fprop: bool,
        ub_overlap_ag_dgrad: bool,
        ub_overlap_rs_dgrad: bool,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_name: str,
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
        symmetric_ar_type: str,
        debug: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._LayerNormLinear.forward"
        if ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ub_name}"

        with_input_all_gather = parallel_mode == "column" and sequence_parallel

        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        inp_shape = inp.shape
        inp_requires_grad = inp.requires_grad
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inp = inp.view((-1, in_features))
        inputmat = inp
        if fp8:
            assert_dim_for_fp8_exec(inputmat, weight)
            assert_dim_for_all_gather(inputmat, with_input_all_gather, input_quantizer)

        # Cast for native AMP
        nvtx_range_push(f"{nvtx_label}.norm_input_cast")
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)
        nvtx_range_pop(f"{nvtx_label}.norm_input_cast")

        tp_world_size = get_distributed_world_size(tp_group)

        weight_requires_grad = weight.requires_grad
        backward_needs_input = is_grad_enabled and weight_requires_grad

        # Configure Userbuffers communication (comm+GEMM overlap)
        if debug:  # turn off userbuffers in debug mode
            ub_overlap_ag_fprop = False
            ub_overlap_rs_fprop = False
            ub_overlap_ag_dgrad = False
            ub_overlap_rs_dgrad = False
            ub_bulk_wgrad = False
            ub_bulk_dgrad = False
        ub_obj = None
        ub_type = None
        ub_overlap_ag_fprop = (
            ub_overlap_ag_fprop and is_grad_enabled and not return_layernorm_output
        )
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.RS
        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.AG

        # Configure quantizer for norm output
        if fp8:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
            if with_input_all_gather and input_quantizer.supports_only_rowwise_all_gather():
                # All-gather is not supported with FP8 column-wise data
                input_quantizer.set_usage(columnwise=False)

        # Avoid quantized norm kernel if norm output will be returned
        # or if a gather of ln_out must be in high precision.
        experimental = is_experimental(input_quantizer)
        with_quantized_norm = (
            fp8
            and not debug
            and not return_layernorm_output
            and not return_layernorm_output_gathered
            and not experimental  # TODO(negvet): and not FP8GlobalStateManager.get_fp8_recipe().custom()
        )

        # Apply normalization
        nvtx_range_push(f"{nvtx_label}.norm")
        ln_out, mu, rsigma = apply_normalization(
            inputmat,
            None,  # ln_out
            ln_weight,
            ln_bias,
            eps,
            input_quantizer if with_quantized_norm else None,
            inputmat.dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        )
        nvtx_range_pop(f"{nvtx_label}.norm")

        # Store unquantized layer norm output if we need to return it
        ln_out_return = None
        if return_layernorm_output or return_layernorm_output_gathered:
            ln_out_return = ln_out

        # ------------------------------------------------------
        # Prepare GEMM input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        # ------------------------------------------------------
        nvtx_range_push(f"{nvtx_label}.gemm_input_cast_comm")
        ln_out_total = None
        if with_input_all_gather:
            if return_layernorm_output_gathered:
                # Perform all-gather in high precision if gathered
                # norm output will be returned
                ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
                ln_out_return = ln_out_total
                if fp8 or debug:
                    ln_out = input_quantizer(ln_out)
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                    if isinstance(input_quantizer, Float8BlockQuantizer):
                        input_quantizer.all_gather_usage = False
                    ln_out_total = input_quantizer(ln_out_total)
            else:
                quantizer = None
                if fp8 or debug:
                    quantizer = input_quantizer
                    # experimental recipe doesn't need to support quantized AG
                    if not with_quantized_norm and not experimental:
                        ln_out = quantizer(ln_out)
                    quantizer.set_usage(rowwise=True, columnwise=False)
                if ub_overlap_ag_fprop:  # Initialize Userbuffers all-gather
                    ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj,
                        ln_out,
                        quantizer,
                        tp_group,
                    )
                else:  # Perform NCCL all-gather
                    ln_out_total, _ = gather_along_first_dim(
                        ln_out,
                        tp_group,
                        quantizer=quantizer,
                    )
        else:
            if (fp8 or debug) and not with_quantized_norm:
                ln_out = input_quantizer(ln_out)
            ln_out_total = ln_out
        nvtx_range_pop(f"{nvtx_label}.gemm_input_cast_comm")
        # ------------------------------------------------------
        # GEMM input tensor is ready...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Prepare weight tensor
        # ------------------------------------------------------
        weightmat = weight
        quantized_weight = False
        if fp8 or debug:
            quantized_weight = not isinstance(weight, QuantizedTensorStorage)

            # Configure quantizer
            if weight_quantizer is not None:
                weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)

            # Get quantized weight
            update_workspace = is_first_microbatch is None or is_first_microbatch
            weightmat = module.get_weight_workspace(
                tensor=weight,
                quantizer=weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "weight"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
            )
            weightmat.update_usage(rowwise_usage=True)

        else:
            weightmat = cast_if_needed(weightmat, activation_dtype)  # Cast for AMP
        # ------------------------------------------------------
        # Weight tensor is ready for GEMM...
        # ------------------------------------------------------

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if needs_quantized_gemm(ln_out_total) and activation_dtype == torch.float32:
            # cuBLAS does not support FP8 GEMM with FP32 bias, so we cast to BF16
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Calibrate quantizers if needed
        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(ln_out_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        # Choose whether to use GEMM kernel with split accumulator
        use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Output buffer for Userbuffers reduce-scatter
        reduce_scatter_out = None
        if ub_overlap_rs_fprop:
            out_shape = list(inp_shape)
            out_shape[0] //= tp_world_size
            out_shape[-1] = out_features
            reduce_scatter_out = torch.empty(out_shape, dtype=activation_dtype, device=inp.device)

        # ------------------------------------------------------
        # Forward GEMM
        # Note: y = x * w^T
        # No _tp_invariant_fwd needed here: LayerNormLinear is column-parallel,
        # so forward GEMM K=hidden_size is constant across TP (invariant by construction).
        # Only row-parallel forward (linear.py) needs the invariant path (K=in/TP varies).
        # ------------------------------------------------------
        if os.environ.get("NVTE_TP_INVARIANT_MODE", "0") == "1" and parallel_mode == "row":
            assert False, (
                "NVTE_TP_INVARIANT_MODE row-parallel forward is not implemented in "
                "layernorm_linear.py. Use linear.py for row-parallel layers."
            )
        nvtx_range_push(f"{nvtx_label}.gemm")
        gemm_out, *_, reduce_scatter_out = general_gemm(
            weightmat,
            ln_out_total,
            get_workspace(),
            quantization_params=output_quantizer,
            out_dtype=activation_dtype,
            bias=bias,
            use_split_accumulator=use_split_accumulator,
            ub=ub_obj,
            ub_type=ub_type,
            extra_output=reduce_scatter_out,
        )
        nvtx_range_pop(f"{nvtx_label}.gemm")
        # ------------------------------------------------------
        # Finished forward GEMM...
        # ------------------------------------------------------

        # Deallocate GEMM input tensor if no longer needed
        if not weight.requires_grad and not return_layernorm_output:
            clear_tensor_data(ln_out, ln_out_total)
            ln_out = ln_out_total = None
        elif with_input_all_gather and not return_layernorm_output_gathered:
            clear_tensor_data(ln_out_total)
            ln_out_total = None

        # ------------------------------------------------------
        # Prepare output tensor
        # Note: Perform tensor-parallel communication
        # ------------------------------------------------------
        out = None
        if ub_overlap_rs_fprop:
            out = reduce_scatter_out
        elif parallel_mode == "row" and tp_size > 1:
            nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
            out = gemm_out
            if sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif tensor_parallel:
                if symmetric_ar_type is not None:
                    out, _ = symmetric_all_reduce(out, tp_group, all_reduce_type=symmetric_ar_type)
                else:
                    out, _ = allreduce(out, tp_group)
            nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")
        else:
            out = gemm_out
        out = out.view(-1, *inp_shape[1:-1], out_features)
        # ------------------------------------------------------
        # Output tensor is ready to return...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Cache state for backward pass
        # ------------------------------------------------------

        if is_grad_enabled:
            ctx.weight_quantizer = weight_quantizer
            ctx.ln_out_needs_gather = (
                weight.requires_grad and parallel_mode == "column" and sequence_parallel
            )

            # Input with column-wise usage is needed for wgrad GEMM.
            if backward_needs_input:
                if isinstance(ln_out, QuantizedTensorStorage):
                    # For sequence parallel in vanilla FP8, rowwise data is
                    # to gather the input. For MXFP8, columnwise only data
                    # can be allgathered.
                    if (
                        isinstance(ln_out, (MXFP8TensorStorage, Float8BlockwiseQTensorStorage))
                        or not ctx.ln_out_needs_gather
                    ):
                        ln_out.update_usage(rowwise_usage=False)

            # Weight with column-wise usage is needed for dgrad GEMM.
            if isinstance(weightmat, QuantizedTensorStorage):
                weightmat.update_usage(columnwise_usage=True)

            if cpu_offloading:
                mark_activation_offload(inputmat, mu, rsigma, ln_out)

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                mu,
                rsigma,
                weightmat if quantized_weight else None,
                ln_out if weight.requires_grad else None,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

            if cpu_offloading:
                ctx.grad_added_to_main_grad = hasattr(weight, "grad_added_to_main_grad")

                if ctx.grad_added_to_main_grad:
                    # If you are passing torch.nn.Parameter through the Torch hooks, you will
                    # get back torch.Tensor. Torch rips off the Parameter wrapper.
                    # You need to preserve the weight object to have all the attributes user
                    # sets for the weights. Because of this, it is not recommended to offload
                    # weights if weights are externally touched outside this module
                    ctx.weight_object = weight

            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat,
                weightmat,
                weight,
                bias,
                ln_weight,
                ln_out,
                mu,
                rsigma,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.requires_dgrad = inp_requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.quantized_weight = quantized_weight
            if fuse_wgrad_accumulation and weight.requires_grad:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(weight, "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.main_grad_func = weight.get_main_grad
                else:
                    ctx.main_grad_func = lambda: weight.main_grad
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_weight_quantizer = grad_weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.input_quantizer = input_quantizer
            ctx.owns_input = inputmat is not inp
            ctx.weight = weight
            ctx.partition_stride = getattr(weight, 'partition_stride', 1)
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = return_layernorm_output_gathered
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_name = ub_name
            ctx.requires_dgrad = inp_requires_grad
            ctx.normalization = normalization
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, ln_weight, ln_bias, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module
            ctx.wgrad_store = wgrad_store
            ctx.debug = debug

        # ------------------------------------------------------
        # Cached state for backward pass is ready...
        # ------------------------------------------------------

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp_shape)
                shape[0] *= tp_size if with_input_all_gather else 1
                return out, ln_out_return.view(shape)
            return out, ln_out_return.view(inp_shape)
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._LayerNormLinear.backward"
        if ctx.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ctx.ub_name}"

        with torch.cuda.nvtx.range("_LayerNormLinear_backward"):
            saved_tensors = ctx.saved_tensors
            (  # pylint: disable=unbalanced-tuple-unpacking
                inputmat,
                weight,
                origin_weight,
                bias,
                ln_weight,
                ln_out,
                mu,
                rsigma,
            ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
            # Delete the references to tensor objects once they've been consumed
            # by the `restore_from_saved` method to construct back the actual tensors.
            ctx.tensor_objects = None

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            main_grad = (
                ctx.main_grad_func()
                if weight is not None and ctx.fuse_wgrad_accumulation and ctx.requires_wgrad
                else None
            )

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            nvtx_range_push(f"{nvtx_label}.fsdp_gather")
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                mu,
                rsigma,
                weight if ctx.fp8 and ctx.quantized_weight else None,
                ln_out,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

            # For CPU offloading, we offloaded weight and weight.main_grad to different tensors,
            # we need to connect them into one.
            if ctx.cpu_offloading:
                if ctx.grad_added_to_main_grad:
                    origin_weight = ctx.weight_object
                if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
                    origin_weight.main_grad = main_grad

            # Configure Userbuffers communication (comm+GEMM overlap)
            ctx.ub_obj_gradout = None
            ub_obj_dgrad = None
            ub_obj_wgrad = None
            ub_type_dgrad = None
            ub_type_wgrad = None
            dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
            if ctx.ub_overlap_ag:
                # Overlap grad_output all-gather with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG
            elif ctx.ub_overlap_rs_dgrad:
                # Overlap dgrad reduce-scatter with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.RS
            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap inputmat all-gather with dgrad compute
                    ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
                    ub_obj_dgrad = ctx.ub_obj_gradout
                    ub_type_dgrad = tex.CommOverlapType.AG
                if ctx.ub_bulk_wgrad:
                    # Overlap dgrad reduce-scatter with wgrad compute
                    ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad", ctx.fp8)
                    ub_type_wgrad = tex.CommOverlapType.RS

            # --------------------------------------------------
            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            # --------------------------------------------------

            # Configure quantizer for grad output tensor
            # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
            # requires column-wise usage
            if ctx.grad_output_quantizer is not None:
                quantizer = ctx.grad_output_quantizer
                quantizer.set_usage(rowwise=True, columnwise=True)
                if ctx.ub_overlap_ag:
                    # Userbuffers only supports communication for one
                    # tensor usage at a time. Configure quantizer with
                    # usage for only dgrad GEMM.
                    quantizer.set_usage(columnwise=False)

            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
            (
                grad_output,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_outputs[0],
                ctx.parallel_mode == "row",
                ctx.grad_output_quantizer,
            )
            nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

            # --------------------------------------------------
            # Grad output tensor is ready for computing grad input...
            # --------------------------------------------------

            # --------------------------------------------------
            # Prepare GEMM input tensor
            # Note: Input tensor is needed for wgrad GEMM.
            # Tensor-parallel communication is overlapped with dgrad
            # GEMM.
            # --------------------------------------------------
            ln_out_total = None
            ln_out_total_work = None
            if ctx.ln_out_needs_gather:
                quantizer = None
                if ctx.input_quantizer is not None:
                    quantizer = ctx.input_quantizer
                    if quantizer.supports_only_rowwise_all_gather():
                        # If data is in FP8, we compute FP8 transposes manually
                        quantizer.set_usage(rowwise=True, columnwise=False)
                    else:
                        # wgrad GEMM requires input with column-wise usage
                        quantizer.set_usage(rowwise=False, columnwise=True)
                if ctx.ub_bulk_dgrad:
                    ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_dgrad,
                        ln_out,
                        quantizer,
                        ctx.tp_group,
                    )
                else:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                    ln_out_total, ln_out_total_work = gather_along_first_dim(
                        ln_out,
                        ctx.tp_group,
                        async_op=True,
                        quantizer=quantizer,
                    )
                    nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_input")
            else:
                ln_out_total = ln_out
            # --------------------------------------------------
            # Input tensor is ready for computing grad weight...
            # --------------------------------------------------

            # --------------------------------------------------
            # Compute grad input tensor
            # Note: Gradient w.r.t. GEMM input (i.e. norm output).
            # --------------------------------------------------

            # Make sure required data is available
            if isinstance(grad_output, QuantizedTensorStorage):
                grad_output.update_usage(rowwise_usage=True)
            if ctx.weight_quantizer is not None and isinstance(weight, QuantizedTensorStorage):
                weight.update_usage(columnwise_usage=True)

            # Choose whether to use GEMM kernel with split accumulator
            use_split_accumulator = _2X_ACC_DGRAD
            if ctx.fp8:
                recipe = ctx.fp8_recipe
                if hasattr(recipe, "fp8_gemm_dgrad"):
                    use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator

            # Update grad input quantizer
            if ctx.grad_input_quantizer is not None:
                ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

            # Output buffers for Userbuffers reduce-scatter
            gemm_out = None
            reduce_scatter_out = None
            if ctx.ub_overlap_rs_dgrad:
                reduce_scatter_out = torch.empty(
                    dgrad_shape, dtype=ctx.activation_dtype, device=grad_outputs[0].device
                )
            elif ctx.ub_bulk_wgrad:
                gemm_out = ub_obj_wgrad.get_buffer(local_chunk=False)

            # dgrad GEMM
            # Note: dx = dy * w
            dgrad = None
            dgrad_work = None

            _tp_invariant_bwd = (
                os.environ.get("NVTE_TP_INVARIANT_MODE", "0") == "1"
                and ctx.parallel_mode == "column"
                and ctx.tp_size > 1
            )

            if _tp_invariant_bwd:
                # TP-invariant diagnostic: full dgrad GEMM matching TP=1 accumulation.
                assert not ctx.fp8, "NVTE_TP_INVARIANT_MODE does not support FP8"
                nvtx_range_push(f"{nvtx_label}.tp_invariant_dgrad")

                def allgather_along_dim(tensor, group, world_size, dim):
                    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
                    torch.distributed.all_gather(
                        chunks, tensor.contiguous(), group=group,
                    )
                    return torch.cat(chunks, dim=dim)

                grad_output_gathered = allgather_along_dim(
                    grad_output, ctx.tp_group, ctx.tp_size, dim=-1,
                )  # [tokens, out/TP] -> [tokens, out]
                weight_gathered = allgather_along_dim(
                    weight, ctx.tp_group, ctx.tp_size, dim=0,
                )  # [out/TP, in] -> [out, in]

                # Deinterleave gathered tensors to match TP=1 K-dimension ordering.
                # Only needed for gated linear units (partition_stride > 1, e.g. SwiGLU FC1)
                # where each rank stores interleaved [gate_i | value_i].
                # After all-gather: [gate_0|val_0 | gate_1|val_1 | ...].
                # TP=1 layout is [gate_all | val_all]. Reorder to match.
                #
                # For non-gated layers (partition_stride == 1, e.g. QKV), each rank
                # stores contiguous GQA groups, and the naive all-gather already
                # produces the correct TP=1 ordering. Deinterleaving would corrupt it.
                if ctx.partition_stride > 1:
                    chunk_sz = weight.shape[0]  # out_features per rank
                    half = chunk_sz // 2
                    first_w = [weight_gathered[i * chunk_sz : i * chunk_sz + half] for i in range(ctx.tp_size)]
                    second_w = [weight_gathered[i * chunk_sz + half : (i + 1) * chunk_sz] for i in range(ctx.tp_size)]
                    weight_gathered = torch.cat(first_w + second_w, dim=0)

                    g_dim = grad_output_gathered.shape[-1] // ctx.tp_size
                    g_half = g_dim // 2
                    first_g = [grad_output_gathered[..., i * g_dim : i * g_dim + g_half] for i in range(ctx.tp_size)]
                    second_g = [grad_output_gathered[..., i * g_dim + g_half : (i + 1) * g_dim] for i in range(ctx.tp_size)]
                    grad_output_gathered = torch.cat(first_g + second_g, dim=-1)

                grad_output_2d = grad_output_gathered.reshape(
                    -1, grad_output_gathered.shape[-1],
                )
                dgrad = general_gemm(
                    weight_gathered, grad_output_2d, get_workspace(),
                    layout="NN", grad=True,
                    out_dtype=ctx.activation_dtype,
                )
                if isinstance(dgrad, tuple):
                    dgrad = dgrad[0]

                # SP: scatter to per-rank chunk along sequence dim.
                if ctx.sequence_parallel:
                    rank = torch.distributed.get_rank(ctx.tp_group)
                    dgrad = dgrad.chunk(ctx.tp_size, dim=0)[rank].contiguous()

                del grad_output_gathered, weight_gathered, grad_output_2d
                nvtx_range_pop(f"{nvtx_label}.tp_invariant_dgrad")
            else:
                nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
                gemm_out, *_, reduce_scatter_out = general_gemm(
                    weight,
                    grad_output,
                    get_workspace(),
                    layout="NN",
                    grad=True,
                    quantization_params=ctx.grad_input_quantizer,
                    out=gemm_out,
                    out_dtype=ctx.activation_dtype,
                    use_split_accumulator=use_split_accumulator,
                    ub=ub_obj_dgrad,
                    ub_type=ub_type_dgrad,
                    extra_output=reduce_scatter_out,
                    bulk_overlap=ctx.ub_bulk_dgrad,
                )
                nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

                # Prepare grad input tensor
                # Note: Perform tensor-parallel communication
                if ctx.ub_overlap_rs_dgrad:
                    dgrad = reduce_scatter_out
                elif ctx.ub_bulk_wgrad:
                    dgrad = ub_obj_wgrad.get_buffer(local_chunk=True)
                elif ctx.parallel_mode == "column" and ctx.tp_size > 1:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
                    dgrad = gemm_out
                    if ctx.sequence_parallel:
                        dgrad, dgrad_work = reduce_scatter_along_first_dim(
                            dgrad,
                            ctx.tp_group,
                            async_op=True,
                        )
                    else:
                        dgrad, dgrad_work = allreduce(dgrad, ctx.tp_group, async_op=True)
                    nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_dgrad")
                else:
                    dgrad = gemm_out

            # --------------------------------------------------
            # Grad input tensor has been computed...
            # --------------------------------------------------

            # --------------------------------------------------
            # Compute grad weight
            # --------------------------------------------------

            wgrad = None
            if ctx.requires_wgrad:
                # Prepare grad output tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if ctx.ub_overlap_ag and isinstance(ctx.grad_output_quantizer, MXFP8Quantizer):
                    # UB does not support pipelined overlapping grad output
                    # all-gather with wgrad GEMM. Also, we can't
                    # convert row-scaled MXFP8 to column-scaled, so we
                    # can't reuse the grad output that was gathered
                    # for the dgrad GEMM. We work around by explicitly
                    # overlapping the AG operation with the dgrad GEMM.

                    # Get the communication stream from the dgrad GEMM to use for the AG
                    dgrad_send_stream, dgrad_recv_stream = ub_obj_dgrad.get_communication_stream()

                    # This object is separate from the ub_obj_wgrad object which is passed to the GEMM
                    ub_obj_overlap_wgrad = get_ub(ctx.ub_name + "_wgrad", ctx.fp8)

                    ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                    # We use the send stream to copy into the userbuffers.
                    # This is the same stream that we will use to access the data in the AG,
                    # so we dont need to add any syncs yet.
                    with torch.cuda.stream(dgrad_send_stream):
                        grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                            ub_obj_overlap_wgrad,
                            grad_outputs[0],
                            ctx.grad_output_quantizer,
                            ctx.tp_group,
                        )

                    # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                    tex.bulk_overlap_ag_with_external_gemm(
                        ub_obj_overlap_wgrad, dgrad_send_stream, dgrad_recv_stream
                    )

                # Prepare input tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if ln_out_total_work is not None:
                    ln_out_total_work.wait()
                    ln_out_total_work = None
                if ctx.fp8 or ctx.debug:
                    if isinstance(ln_out_total, QuantizedTensorStorage):
                        ln_out_total.update_usage(columnwise_usage=True)
                    else:
                        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
                        ln_out_total = ctx.input_quantizer(ln_out_total)

                if ctx.fp8 or ctx.debug:
                    if isinstance(grad_output, QuantizedTensorStorage):
                        grad_output.update_usage(columnwise_usage=True)
                    else:
                        ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                        grad_output = ctx.grad_output_quantizer(grad_output)

                # Figure out whether to use split accumulator
                use_split_accumulator = _2X_ACC_WGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_wgrad"):
                        use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator

                # Figure out whether to output wgrad GEMM directly into main grad
                if ctx.is_first_microbatch is not None:
                    accumulate_wgrad_into_param_main_grad = (
                        ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                    )
                else:
                    accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

                # Output buffer for overlapping FP8 grad input
                # reduce-scatter with wgrad GEMM
                reduce_scatter_out = None
                if ctx.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                    reduce_scatter_out = torch.empty(
                        dgrad_shape, dtype=ctx.activation_dtype, device=grad_outputs[0].device
                    )

                # Arguments to include in wgrad GEMM closure
                wgrad_gemm_kwargs = {
                    "workspace": get_workspace(),
                    "out_dtype": (
                        main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype
                    ),
                    "quantization_params": ctx.grad_weight_quantizer,
                    "accumulate": (
                        accumulate_wgrad_into_param_main_grad
                        if not getattr(weight, "overwrite_main_grad", False)
                        else False
                    ),
                    "layout": "NT",
                    "out": main_grad if ctx.fuse_wgrad_accumulation else None,
                    "bias": (bias if (grad_bias is None and not ctx.fp8) else None),
                    "use_split_accumulator": use_split_accumulator,
                    "grad": True,
                    "ub": ub_obj_wgrad,
                    "ub_type": ub_type_wgrad,
                    "extra_output": reduce_scatter_out,
                    "bulk_overlap": ctx.ub_bulk_wgrad,
                }

                def wgrad_gemm(
                    x: torch.Tensor,
                    dy: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    """Perform wgrad GEMM: dw = dy^T * x

                    May be fused with bgrad computation.

                    May be called outside of this function to enable
                    some advanced communication/compute overlapping.

                    """
                    nvtx_range_push(f"{nvtx_label}.wgrad_gemm")
                    dw, db, *_ = general_gemm(x, dy, **wgrad_gemm_kwargs)
                    nvtx_range_pop(f"{nvtx_label}.wgrad_gemm")
                    return dw, db

                # Choose whether to call wgrad GEMM now or delay
                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                    if (
                        wgrad_gemm_kwargs["ub"] is not None
                        or wgrad_gemm_kwargs["ub_type"] is not None
                        or wgrad_gemm_kwargs["extra_output"] is not None
                        or wgrad_gemm_kwargs["bulk_overlap"]
                    ):
                        raise NotImplementedError(
                            "Delayed weight grad computation is not supported "
                            "with Userbuffers (tensor-parallel communication overlapping)"
                        )
                    ctx.wgrad_store.put([ln_out_total, grad_output], wgrad_gemm)
                else:

                    # Call wgrad GEMM now
                    wgrad, grad_bias_ = wgrad_gemm(ln_out_total, grad_output)

                    # Update grad bias if needed
                    if grad_bias is None:
                        grad_bias = grad_bias_
                    del grad_bias_

                    # Deallocate input tensors if permitted
                    if not ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                        # Input tensors have not been exposed externally
                        clear_tensor_data(ln_out)
                    elif ctx.ln_out_needs_gather and ctx.return_layernorm_output_gathered:
                        # Non-gathered input has not been exposed externally
                        clear_tensor_data(ln_out)
                    if ctx.ln_out_needs_gather:
                        # Gathered input is internal
                        clear_tensor_data(ln_out_total)
                    if ctx.parallel_mode == "row" and ctx.sequence_parallel:
                        # Gathered grad output tensor is internal
                        clear_tensor_data(grad_output)

                # Update grad input if overlapping reduce-scatter with wgrad GEMM
                if ctx.ub_bulk_wgrad:
                    if ub_obj_wgrad.is_fp8_ubuf():
                        dgrad = reduce_scatter_out
                    else:
                        dgrad = ub_obj_wgrad.get_buffer(local_chunk=True).clone()

            # --------------------------------------------------
            # Grad weight has been computed...
            # --------------------------------------------------

            # Don't return grad bias if not needed
            if not ctx.use_bias:
                grad_bias = None

            # Synchronize tensor parallel communication
            if ln_out_total_work is not None:
                ln_out_total_work.wait()
                ln_out_total_work = None
            if dgrad_work is not None:
                dgrad_work.wait()
                dgrad_work = None

            # Residual gradient
            dgrad = dgrad.view(inputmat.shape)
            if ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                dgrad = dgrad + grad_outputs[1].view_as(dgrad)

            # Norm gradient
            dgamma = None
            dbeta = None
            nvtx_range_push(f"{nvtx_label}.norm")
            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    dgrad,
                    inputmat,
                    mu,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dgrad = dgrad.reshape(inputmat.size())
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    dgrad,
                    inputmat,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dgrad = dgrad.reshape(inputmat.size())
                dbeta = None
            nvtx_range_pop(f"{nvtx_label}.norm")
            clear_tensor_data(mu)
            clear_tensor_data(rsigma)

        if ctx.requires_wgrad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(origin_weight, "grad_added_to_main_grad"):
                origin_weight.grad_added_to_main_grad = True
                if getattr(origin_weight, "zero_out_wgrad", False):
                    wgrad = get_dummy_wgrad(
                        list(origin_weight.main_grad.shape),
                        origin_weight.dtype,
                        zero=True,
                    )
                else:
                    wgrad = get_dummy_wgrad(
                        list(origin_weight.main_grad.shape),
                        origin_weight.dtype,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")

        # Scatter fp8 weight buffers
        # if ctx.fp8 and not isinstance(weight, QuantizedTensorStorage):
        #    _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            wgrad,
            grad_bias,
            None,  # eps
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # wgrad_store
            None,  # fuse_wgrad_accumulation
            None,  # input_quantizer
            None,  # weight_quantizer
            None,  # output_quantizer
            None,  # grad_input_quantizer
            None,  # grad_weight_quantizer
            None,  # grad_output_quantizer
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # return_layernorm_output
            None,  # return_layernorm_output_gathered
            None,  # is_grad_enabled
            None,  # fwd_ln_sm_margin
            None,  # bwd_ln_sm_margin
            None,  # zero_centered_gamma
            None,  # normalization
            None,  # ub_overlap_ag_fprop
            None,  # ub_overlap_rs_fprop
            None,  # ub_overlap_ag_dgrad
            None,  # ub_overlap_rs_dgrad
            None,  # ub_bulk_dgrad
            None,  # ub_bulk_wgrad
            None,  # ub_name
            None,  # fsdp_group
            None,  # debug
            None,  # module
            None,  # skip_fp8_weight_update
            None,  # symmetric_ar_type
        )


class LayerNormLinear(TransformerEngineBaseModule):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    name: str, default = `None`
        name of the module, currently used for debugging purposes.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in. This argument along with
                             weight tensor having attribute 'overwrite_main_grad' set to True
                             will overwrite `main_grad` instead of accumulating.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    delay_wgrad_compute : bool, default = `False`
                         Whether or not to delay weight gradient computation. If set to `True`,
                         it's the user's responsibility to call `module.backward_dw` to compute
                         weight gradients.
    symmetric_ar_type : {None, 'multimem_all_reduce', 'two_shot', 'one_shot'}, default = None
                   Type of symmetric memory all-reduce to use during the forward pass.
                   This can help in latency bound communication situations.
                   Requires PyTorch version 2.7.0 or higher. When set to None, standard all-reduce
                   is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        name: str = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = self.use_bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = (
            return_layernorm_output_gathered if return_layernorm_output else False
        )
        self.zero_centered_gamma = zero_centered_gamma
        self.symmetric_ar_type = symmetric_ar_type

        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ub_bulk_wgrad)
        self.name = name

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        # Column-parallel overlaps
        self.ub_overlap_ag_fprop = (
            ub_overlap_ag and self.sequence_parallel and self.parallel_mode == "column"
        )
        self.ub_overlap_rs_dgrad = (
            ub_overlap_rs_dgrad and self.sequence_parallel and self.parallel_mode == "column"
        )
        self.ub_bulk_wgrad = (
            ub_bulk_wgrad
            and self.sequence_parallel
            and self.parallel_mode == "column"
            and not self.ub_overlap_rs_dgrad
        )
        self.ub_bulk_dgrad = (
            ub_bulk_dgrad
            and self.sequence_parallel
            and self.parallel_mode == "column"
            and not self.ub_overlap_rs_dgrad
        )

        # Row-parallel overlaps
        self.ub_overlap_rs_fprop = (
            ub_overlap_rs and self.sequence_parallel and self.parallel_mode == "row"
        )
        self.ub_overlap_ag_dgrad = (
            ub_overlap_ag and self.sequence_parallel and self.parallel_mode == "row"
        )
        if any(
            [
                self.ub_overlap_ag_fprop,
                self.ub_overlap_rs_dgrad,
                self.ub_bulk_dgrad,
                self.ub_bulk_wgrad,
                self.ub_overlap_rs_fprop,
                self.ub_overlap_ag_dgrad,
            ]
        ):
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name

        if self.symmetric_ar_type is not None:
            assert torch_version() >= (
                2,
                7,
                0,
            ), "Torch version must be at least 2.7 to use symmetric memory"

        self.eps = eps
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(self.in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not self.zero_centered_gamma)),
        )
        if self.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(self.in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias, init_fn=init_method_constant(0.0)
            )
        else:
            self.layer_norm_bias = None

        # Initialize params in FP8
        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        # Contiguous buffers for params
        weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        # Adjust parameter splits for tensor-parallel distribution
        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        # Construct weight parameters
        # Note: Register weights together so that they are adjacent to
        # each other in LayerNormLinear.parameters(). This makes it
        # more likely that they will stay contiguous if the weights
        # are manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError(
                    "Splitting QuantizedTensor into multiple params is not supported"
                )

            # Construct weight parameter
            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
            )

        # Construct bias parameters if needed
        if self.use_bias:
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                self.register_parameter(
                    self.bias_names[i],
                    torch.nn.Parameter(bias_tensor[split_start:split_end]),
                    init_fn=init_method_constant(0.0),
                )
        else:
            for name in self.bias_names:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, name, bias)

        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

        if self.wgrad_store.delay_wgrad_compute():
            for name, param in self.named_parameters():
                if name in self.weight_names or name in self.bias_names:
                    param.skip_backward_post_hook = True

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        super().set_meta_tensor(fwd, recipe)

        # customize quantizers based on each recipe & layer configs
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)
        elif recipe.float8_block_scaling():
            self._customize_quantizers_float8_blockwise_scaling(fwd, recipe)
        elif recipe.nvfp4():
            self._customize_quantizers_nvfp4(fwd, recipe)
        # elif other recipes (mxfp8, etc)

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNormLinear.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.zero_centered_gamma:
            init.ones_(self.layer_norm_weight)
        else:
            init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            init.zeros_(self.layer_norm_bias)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for layer norm parameters
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            # Set parallelism attributes for linear weights
            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for bias in self.bias_names:
                    if self.parallel_mode == "row":
                        setattr(getattr(self, bias), "sequence_parallel", self.sequence_parallel)
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, bias), True, 0, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
        fp8_grad: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """
        if is_in_onnx_export_mode():
            return self.onnx_forward(inp, fp8_output)

        debug = self.is_debug_iter()

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        if self.ub_overlap_rs_fprop:
            if get_ub(
                self.ub_name + "_fprop", FP8GlobalStateManager.is_fp8_enabled()
            ).is_fp8_ubuf():
                fp8_output = True
        if self.ub_overlap_rs_dgrad:
            if get_ub(
                self.ub_name + "_dgrad", FP8GlobalStateManager.is_fp8_enabled()
            ).is_fp8_ubuf():
                fp8_grad = True

        with torch.cuda.device(
            getattr(self, list(self.named_parameters())[0][0]).device
        ), self.prepare_forward(
            inp, allow_non_contiguous=False  # removed .contiguous from inside the layer
        ) as inp:

            # Get concatenated weight and bias tensors
            weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()

            quantizers = (
                self._get_quantizers(fp8_output, fp8_grad)
                if not debug
                else self._get_debug_quantizers(fp8_output, fp8_grad)
            )
            if debug:
                if self.no_debug_features_active(quantizers):
                    debug = False
                    quantizers = self._get_quantizers(fp8_output, fp8_grad)

            (
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_weight_quantizer,
                grad_output_quantizer,
            ) = quantizers

            if torch.is_grad_enabled():
                fwd_fn = _LayerNormLinear.apply
                args = []
            else:
                fwd_fn = _LayerNormLinear.forward
                args = [None]
            args += (
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                weight_tensor,
                bias_tensor if self.apply_bias and not self.gemm_bias_unfused_add else None,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.wgrad_store,
                self.fuse_wgrad_accumulation,
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_weight_quantizer,
                grad_output_quantizer,
                is_cpu_offload_enabled(),
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                self.return_layernorm_output,
                self.return_layernorm_output_gathered,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin if torch.is_grad_enabled() else self.inf_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.normalization,
                self.ub_overlap_ag_fprop,
                self.ub_overlap_rs_fprop,
                self.ub_overlap_ag_dgrad,
                self.ub_overlap_rs_dgrad,
                self.ub_bulk_wgrad,
                self.ub_bulk_dgrad,
                self.ub_name,
                self.fsdp_group,
                self,
                skip_fp8_weight_update,
                self.symmetric_ar_type,
                debug,
            )
            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out

    def _get_quantizers(self, fp8_output, fp8_grad):
        if not self.fp8:
            return [None] * 6
        grad_input_quantizer = None
        grad_weight_quantizer = None
        grad_output_quantizer = None
        output_quantizer = None
        input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        input_quantizer.internal = True
        (weight_quantizer,) = self._get_weight_quantizers()
        if fp8_output:
            output_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
        if torch.is_grad_enabled():
            grad_output_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
            grad_output_quantizer.internal = True
            if fp8_grad:
                grad_input_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]

        return (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
        )

    def _get_debug_quantizers(self, fp8_output, fp8_grad):
        original_quantizers = self._get_quantizers(fp8_output, fp8_grad)
        assert TEDebugState.debug_enabled
        from ...debug.pytorch.debug_quantization import DebugQuantizer

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        return tuple(
            DebugQuantizer(self.name, name, q, self.tp_group)
            for name, q in zip(names, original_quantizers)
        )

    def _get_weight_and_bias_tensors(self):
        # Get concatenated weight and bias tensors
        unfused_weights = self._get_weight_tensors()

        weight_tensor = noop_cat(unfused_weights)
        if self.use_bias:
            bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
        else:
            bias_tensor = getattr(self, self.bias_names[0])  # Unused
        return weight_tensor, bias_tensor

    def onnx_forward(
        self,
        inp: torch.Tensor,
        fp8_output: bool,
    ) -> torch.Tensor:
        """
        ONNX-compatible version of the forward function that provides numerical equivalence
        while only using operations that have defined ONNX symbolic translations.
        This simplified implementation is designed specifically for inference scenarios.
        """
        from ..export import onnx_layernorm, onnx_gemm

        assert not TEDebugState.debug_enabled, "Debug mode is not supported in ONNX export"
        assert_warmed_up(self)
        (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            *_,
        ) = self._get_quantizers(fp8_output, fp8_grad=False)
        inp_dtype = inp.dtype

        weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()
        ln_out, ln_out_return = onnx_layernorm(
            inp,
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
            self.normalization,
            self.zero_centered_gamma,
            inp_dtype,
            self.return_layernorm_output,
            input_quantizer,
        )

        if weight_quantizer is not None:
            weight_tensor_quantized = weight_quantizer.onnx_quantize(weight_tensor)
            weight_tensor = weight_quantizer.onnx_dequantize(weight_tensor_quantized)
        weight_tensor = weight_tensor.to(inp_dtype)

        if bias_tensor is not None:
            bias_tensor = bias_tensor.to(inp_dtype)

        output = onnx_gemm(weight_tensor, ln_out, bias_tensor if self.apply_bias else None)

        if output_quantizer is not None:
            raise NotImplementedError("ONNX export of quantized output is not supported")
        if self.return_layernorm_output and self.return_bias:
            return output, bias_tensor.to(inp_dtype), ln_out_return
        if self.return_layernorm_output:
            return output, ln_out_return
        if self.return_bias:
            return output, bias_tensor.to(inp_dtype)
        return output

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + layernorm_linear."""
        assert (
            recipe.float8_current_scaling()
        ), "current scaling recipe quantizer customization here"
        if fwd:
            # set configs about amax epsilon and power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
            # also set weight quantizer with same amax_epsilon & power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_WEIGHT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_WEIGHT
            ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
            # parallel related
            if self.sequence_parallel and self.parallel_mode == "column":
                # set input_quantizer with amax reduction TP group
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].with_amax_reduction = True
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].amax_reduction_group = self.tp_group
        else:
            # set grad_output_quantizer with amax epsilon and power_2_scale (no amax reduction here)
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon
            # parallel related
            if self.sequence_parallel and self.parallel_mode == "row":
                # customize grad_output_quantizer with amax reduction TP group
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ].with_amax_reduction = True
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ].amax_reduction_group = self.tp_group

    def _customize_quantizers_nvfp4(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + layernorm_linear."""
        assert recipe.nvfp4(), "Incorrect recipe."
        if fwd:
            if self.sequence_parallel and self.parallel_mode == "column":
                # set input_quantizer with amax reduction TP group
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].with_amax_reduction = True
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].amax_reduction_group = self.tp_group
        else:
            if self.sequence_parallel and self.parallel_mode == "row":
                # customize grad_output_quantizer with amax reduction TP group
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ].with_amax_reduction = True
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ].amax_reduction_group = self.tp_group

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        unfused_weights = [getattr(self, name) for name in self.weight_names]
        if any(isinstance(w, QuantizedTensor) for w in unfused_weights):
            if self.fp8:
                if len(unfused_weights) != 1:
                    raise RuntimeError(
                        "Splitting QuantizedTensor into multiple params is not supported"
                    )
            else:
                warnings.warn(
                    "You are using quantized weights without quantized compute. "
                    "Please make sure this is intentional."
                )
                unfused_weights = [w.dequantize() for w in unfused_weights]
        return unfused_weights

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        if not self.fp8 and not self.fp8_calibration:
            return [None]
        weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        return [weight_quantizer]

    def _customize_quantizers_float8_blockwise_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on blockwise scaling recipe + layernorm_linear."""
        assert (
            recipe.float8_block_scaling()
        ), "blockwise scaling recipe quantizer customization here"
        if fwd:
            if self.sequence_parallel and self.parallel_mode == "column":
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].all_gather_usage = True
