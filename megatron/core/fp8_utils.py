# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions related to FP8 that are used throughout Megatron core"""
from contextlib import nullcontext
from typing import Tuple

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

# Check if Transformer Engine is installed
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

# Check if Transformer Engine has Float8Tensor class
HAVE_TE_FLOAT8TENSOR = False
try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    HAVE_TE_FLOAT8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # Float8Tensor not found
    pass

# Check if Transformer Engine has MXFP8Tensor class
HAVE_TE_MXFP8TENSOR = False
try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # MXFP8Tensor not found
    pass

# utils for transformer engine fp8 and mxfp8 tensor

if HAVE_TE and is_te_min_version("2.0"):
    # TE quantization logic using quantizer API
    # Supported TE versions: 2.0+
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    def _quantize_param_fragment_impl(
        input_: torch.Tensor, *, out: torch.Tensor, param: torch.nn.Parameter
    ) -> None:
        quantizer = param._quantizer
        out = Float8Tensor(
            shape=input_.size(),
            dtype=param.dtype,
            requires_grad=False,
            data=out,
            fp8_scale_inv=param._scale_inv,
            fp8_dtype=param._fp8_dtype,
            quantizer=quantizer,
        )
        quantizer.update_quantized(input_, out)

    def _get_fp8_scale_and_amax_impl(tensor: Float8Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantizer = tensor._quantizer
        return quantizer.scale, quantizer.amax

elif HAVE_TE and is_te_min_version("1.0"):
    # TE quantization logic with fp8_meta dicts
    # Supported TE versions: 1.0 - 1.14
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8

    def _quantize_param_fragment_impl(
        input_: torch.Tensor, *, out: torch.Tensor, param: torch.nn.Parameter
    ) -> None:
        cast_to_fp8(
            input_.view(1, -1),
            param._fp8_meta["scaling_fwd"],
            param._fp8_meta_index,
            param._fp8_dtype,
            out=out.view(1, -1),
        )

    def _get_fp8_scale_and_amax_impl(tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fp8_meta = tensor._fp8_meta["scaling_fwd"]
        fp8_meta_index = tensor._fp8_meta_index
        return fp8_meta.scale[fp8_meta_index], fp8_meta.amax_history[0][fp8_meta_index]

else:
    # Fallback impl if TE version is invalid
    def _quantize_param_fragment_impl(*args, **kwargs) -> None:
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _get_fp8_scale_and_amax_impl(*args, **kwargs):
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")


def quantize_param_fragment(
    input_: torch.Tensor, *, out: torch.Tensor, param: torch.nn.Parameter
) -> None:
    """Cast values in parameter fragment to FP8
    Arguments:
      input_ (torch.Tensor): Values to quantize.
      out (torch.Tensor): Raw UINT8 buffer to fill with FP8 values.
          Dimensions should match input_.
      param (torch.nn.Parameter): Parameter containing this parameter
          fragment. Must be a Float8Tensor.
    """
    _quantize_param_fragment_impl(input_, out=out, param=param)


def get_fp8_scale_and_amax(tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get FP8 scale and amax from Float8Tensor"""
    return _get_fp8_scale_and_amax_impl(tensor)


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor"""
    return HAVE_TE_FLOAT8TENSOR and isinstance(tensor, Float8Tensor)


def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)


if HAVE_TE:
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1):
        """Return fp8 context manager.

        Arguments:
            config (TransformerConfig): Configuration object.
            layer_no (int): *Global* layer index (including layers on other
                pipeline-parallel ranks).

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

        if not config.fp8:
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
                elif config.fp8_recipe == Fp8Recipe.tensorwise:
                    fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                        fp8_format=fp8_format
                    )
                elif config.fp8_recipe == Fp8Recipe.mxfp8:
                    fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                        fp8_format=fp8_format
                    )
                else:
                    raise ValueError(
                        "Float8CurrentScaling, MXFP8BlockScaling and DelayedScaling are "
                        "the only supported FP8 recipes."
                    )
            else:
                # Assert that the user is using delayed scaling.
                assert config.fp8_recipe == Fp8Recipe.delayed, (
                    "Please make sure to use TransformerEngine version >= 2.1.0 for "
                    "Float8CurrentScaling and MXFP8BlockScaling."
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
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )

            # First / last layer in bf16 isn't supported with delayed scaling since it
            # requires entering/exiting fp8 context per layer, causing incorrect amax
            # reduction behavior.
            assert not (
                config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
            ), "Delayed scaling does not support first / last layer in BF16."

        return fp8_context

else:

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1):
        """Returns dummy fp8 context manager since TE is not available."""
        return nullcontext()
