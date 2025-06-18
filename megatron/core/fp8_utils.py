# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions related to FP8 that are used throughout Megatron core"""
from contextlib import nullcontext
from typing import List, Optional

import torch
from packaging.version import Version as PkgVersion

from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_te_version, is_te_min_version

# Check if Transformer Engine is installed
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

# Check if Transformer Engine has class for fp8 tensors.
HAVE_TE_FP8_TENSOR_CLASS = False
try:
    if is_te_min_version("2.0"):
        # In TE2.x, QuantizedTensor is the base class for all different type of fp8 tensors,
        # including fp8 tensor for delayed scaling, current scaling and mxfp8, etc.
        from transformer_engine.pytorch.tensor import QuantizedTensor as FP8_TENSOR_CLASS
    else:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor as FP8_TENSOR_CLASS

    HAVE_TE_FP8_TENSOR_CLASS = True
except (ImportError, ModuleNotFoundError):
    # FP8 tensor class not found
    pass

# Check if Transformer Engine has MXFP8Tensor class
HAVE_TE_MXFP8TENSOR = False
try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # MXFP8Tensor not found
    pass


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.

    Note that in TE2.x, in order to support more recipes, the design of the fp8 tensor class has
    changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8
    and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes
    are both inherited from QuantizedTensor. So, for TE1.x, FP8_TENSOR_CLASS is Float8Tensor,
    and for TE2.x, FP8_TENSOR_CLASS is QuantizedTensor.
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)


def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)


def dequantize_fp8_tensor(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp8 tensor to a higher precision tensor."""
    if is_te_min_version("2.0"):
        return fp8_tensor.dequantize()
    else:
        return fp8_tensor.from_float8()


def get_fp8_align_size(fp8_recipe: Fp8Recipe) -> int:
    """Get the alignment size required for fp8 GEMM."""
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32
    else:
        return 16


"""
The code below abstracts the functionalities needed for implementing "--fp8-param-gather" into
several functions. It provides different implementations for each function based on different
versions of TE, ensuring compatibility across various TE versions.

Currently, there are three functions:
    - modify_underlying_storage
        This function is used in DDP to place all parameters into a contiguous buffer. For
        non-fp8 tensors, replacing their data is simple, just using code like
        "tensor.data = new_data". However, for fp8 tensors, their raw data is not stored in the
        ".data" attribute, and it varies with different TE versions and different recipes. This
        function provides a unified interface to replace the underlying storage of a fp8 tensor.
    - quantize_param_shard
        This function is used in dist-opt to cast fp32 main params to fp8 params. For non-fp8
        params, this casting is as simple as "bf16_params.copy_(fp32_main_params)"; but for fp8
        params, the casting logic varies with different TE versions and different recipes. This
        function provides a unified interface to cast fp32 main params to fp8 params, and also
        updates the necessary attributes (like amax, scale, scale_inv or transpose cache) of the
        fp8 model params.
    - correct_amax_history_if_needed
        This function is used to correct the amax history of fp8 tensors. In TE1.x, some inplace
        copy operations will write unwanted values to the amax_history of fp8 tensors. This function
        corrects the amax_history back. For TE2.x, it's an empty function.
        Only useful for delayed scaling.
"""
if HAVE_TE and is_te_min_version("2.2"):
    # Supported TE versions: 2.2+
    from transformer_engine.pytorch.tensor import QuantizedTensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        from transformer_engine.pytorch.tensor.utils import replace_raw_data

        replace_raw_data(fp8_tensor, new_raw_data)

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if len(model_params) == 0:
            return

        from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

        args = [model_params, main_params, start_offsets, data_parallel_group]
        if fsdp_shard_model_params is not None:
            if get_te_version() == PkgVersion("2.3.0.dev0+5fdd7bb") or is_te_min_version("2.3.0"):
                args.append(fsdp_shard_model_params)
            else:
                raise NotImplementedError(
                    f"FSDP with --fp8-param-gather is not supported in TE v{get_te_version()}"
                )
        cast_master_weights_to_fp8(*args)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        pass

elif HAVE_TE and is_te_min_version("2.0"):
    # Supported TE versions: 2.0
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        old_raw_data = fp8_tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        fp8_tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        # Avoid circular import
        from megatron.core.optimizer.optimizer import _multi_tensor_copy_this_to_that

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            quantizer = model_param._quantizer
            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = main_param.to(model_param.dtype)
            out = Float8Tensor(
                shape=main_param.size(),
                dtype=model_param.dtype,
                requires_grad=False,
                data=shard_model_param,
                fp8_scale_inv=model_param._scale_inv,
                fp8_dtype=model_param._fp8_dtype,
                quantizer=quantizer,
            )
            quantizer.update_quantized(main_param, out)

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            quantizer = model_param._quantizer
            amaxes.append(quantizer.amax.view(1))
            scales.append(quantizer.scale.view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        # Update scaling factors.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        pass

elif HAVE_TE and is_te_min_version("1.0"):
    # Supported TE versions: 1.0 - 1.14
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(tensor: Float8Tensor, new_raw_data: torch.Tensor) -> None:
        old_raw_data = tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[Float8Tensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        # Avoid circular import
        from megatron.core.optimizer.optimizer import _multi_tensor_copy_this_to_that

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = main_param.to(model_param.dtype)
            cast_to_fp8(
                main_param.view(1, -1),
                model_param._fp8_meta["scaling_fwd"],
                model_param._fp8_meta_index,
                model_param._fp8_dtype,
                out=shard_model_param.view(1, -1),
            )

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            fp8_meta = model_param._fp8_meta["scaling_fwd"]
            fp8_meta_index = model_param._fp8_meta_index
            amaxes.append(fp8_meta.amax_history[0][fp8_meta_index].view(1))
            scales.append(fp8_meta.scale[fp8_meta_index].view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        # Update scaling factors.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        for model_module in model:
            for param in model_module.parameters():
                if is_float8tensor(param) and param._fp8_meta is not None:
                    fp8_meta = param._fp8_meta['scaling_fwd']
                    fp8_meta_index = param._fp8_meta_index
                    if hasattr(param, 'get_high_precision_init_val'):
                        fp8_meta.amax_history[0][fp8_meta_index].copy_(
                            param.get_high_precision_init_val().abs().max()
                        )
                    else:
                        fp8_meta.amax_history[0][fp8_meta_index] = 0

else:
    # Fallback impl if TE version is invalid or TE is not installed.
    def _modify_underlying_storage_impl(*args, **kwargs):
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _quantize_param_shard_impl(*args, **kwargs):
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _correct_amax_history_if_needed_impl(*args, **kwargs):
        # If TE is not installed, we are definitely not using fp8 for training, so no correction
        # is needed.
        pass


# Interface Function
def modify_underlying_storage(tensor: torch.Tensor, new_raw_data: torch.Tensor):
    """Replace the underlying raw data of a tensor with new data."""
    _modify_underlying_storage_impl(tensor, new_raw_data)


# Interface Function
def quantize_param_shard(
    model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params=None
):
    """Cast shard fp32 main params to fp8 model params."""
    _quantize_param_shard_impl(
        model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
    )


# Interface Function
def correct_amax_history_if_needed(model: List[torch.nn.Module]):
    """Correct the amax history of fp8 tensors when it's necessary (i.e., in TE1.x)."""
    _correct_amax_history_if_needed_impl(model)


if HAVE_TE:
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp8 context manager.

        Arguments:
            config (TransformerConfig): Configuration object.
            layer_no (int): *Global* layer index (including layers on other
                pipeline-parallel ranks).
            is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

        Returns:
            FP8 context.
            If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
            We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
            that needs to be trained in bf16.
        """

        num_bf16_layers_at_start = (
            config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        )
        num_bf16_layers_at_end = (
            config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        )
        # Since layer_no is a global layer index, additional checks on whether
        # we are in the first or last pipeline-parallel rank are not needed.
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context:
            # bf16 training
            fp8_context = nullcontext()
        elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
            # fp8 training but this layer_no should be bf16
            fp8_context = nullcontext()
        else:
            # fp8 training and this layer_no is in fp8
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            # Select fp8 recipe (TE version >= 2.1.0).
            fp8_recipe = None
            if is_te_min_version("2.1.0"):
                if config.fp8_recipe == Fp8Recipe.delayed:
                    fp8_recipe = TEDelayedScaling(
                        config=config,
                        fp8_format=fp8_format,
                        override_linear_precision=(False, False, not config.fp8_wgrad),
                    )
                elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
                    fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                        fp8_format=fp8_format
                    )
                elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
                    fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                        fp8_format=fp8_format
                    )
                elif config.fp8_recipe == Fp8Recipe.mxfp8:
                    fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                        fp8_format=fp8_format
                    )
                else:
                    raise ValueError(
                        "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                        "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                        "you are using a compatible TE version."
                    )
            else:
                # Assert that the user is using delayed scaling.
                assert config.fp8_recipe == Fp8Recipe.delayed, (
                    "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
                    "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
                    "Float8BlockScaling."
                )
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )

            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                )

            if not is_init:
                fp8_context = transformer_engine.pytorch.fp8_autocast(
                    enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
                )
            else:
                import inspect

                context_args = {"enabled": True}
                # Check if fp8_model_init supports setting recipe
                if "recipe" in (
                    inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
                ):
                    context_args["recipe"] = fp8_recipe
                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in (
                    inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
                ):
                    context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
                fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

            # First / last layer in bf16 isn't supported with delayed scaling since it
            # requires entering/exiting fp8 context per layer, causing incorrect amax
            # reduction behavior.
            assert not (
                config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
            ), "Delayed scaling does not support first / last layer in BF16."

        return fp8_context

else:

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Returns dummy fp8 context manager since TE is not available."""
        return nullcontext()
