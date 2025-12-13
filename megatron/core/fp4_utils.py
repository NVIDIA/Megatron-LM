# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utility functions related to FP4 that are used throughout Megatron core"""

from contextlib import nullcontext

import torch

from megatron.core.enums import Fp4Recipe
from megatron.core.fp8_utils import _get_custom_recipe
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


# Check if Transformer Engine has class for fp4 tensors.
HAVE_TE_FP4_TENSOR_CLASS = False
if HAVE_TE:
    if is_te_min_version("2.7.0.dev0"):
        try:
            from transformer_engine.pytorch.tensor.nvfp4_tensor import (
                NVFP4Tensor as FP4_TENSOR_CLASS,
            )

            HAVE_TE_FP4_TENSOR_CLASS = True
        except (ImportError, ModuleNotFoundError):
            HAVE_TE_FP4_TENSOR_CLASS = False
            FP4_TENSOR_CLASS = None
    else:
        HAVE_TE_FP4_TENSOR_CLASS = False
        FP4_TENSOR_CLASS = None
else:
    HAVE_TE_FP4_TENSOR_CLASS = False
    FP4_TENSOR_CLASS = None


def is_nvfp4tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine NVFP4Tensor."""
    return HAVE_TE_FP4_TENSOR_CLASS and isinstance(tensor, FP4_TENSOR_CLASS)


def get_fp4_align_size(fp4_recipe: Fp4Recipe) -> int:
    """
    Get the alignment size required for FP4 GEMM.
    FP4 GEMM requires Blackwell and later architectures.

    The value 32 is a hardware requirement: TMA (Tensor Memory Accelerator) requires
    a 16-byte aligned address for efficient memory access. Since FP4 uses 4 bits per value,
    16 bytes (128 bits) corresponds to 32 FP4 values. Therefore, the alignment size for FP4
    is 32. With this alignment, NVFP4 GEMM can be performed efficiently.

    Note that since we are also random hadamard transform for NVFP4 training, we want
    fused group nvfp4 quantize plus hadamard transform. Hadamard transform will leverage
    tensor core instructions for better performance, while group quantize kernels also
    prefer a more aligned size in token dimension M. The efficiently leverage grouped 
    kernels, padding needs to be 64 multiple, but 128 multiple will bring even faster.

    When it comes to MOE cuda graph support, the number of tokens for each expert should
    be a buffer on device memory, which means that we don't know the token dimension for 
    each expertin host, therefore we cannot calculate the zero padded scaling factors shape 
    on host to comply with the NVFP4 GEMM scaling factor layout. However, if we have already 
    zero padded the tokens to 128 multiple, then there is no need for such padding, so that
    host doesn't need to copy the token distribution from device to host (which will break
    the CUDA graph).

    Paper link: https://arxiv.org/pdf/2509.25149
    Scaling factor layout CuBLAS: https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
    TE NVFP4 Grouped Quantization: https://github.com/NVIDIA/TransformerEngine/pull/2411 
    """
    # pylint: disable=unused-argument
    return 128


def dequantize_fp4_tensor(fp4_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp4 tensor to a higher precision tensor."""
    if is_te_min_version("2.7.0.dev0"):
        return fp4_tensor.dequantize()
    else:
        raise RuntimeError("FP4 dequantization requires Transformer Engine >= 2.7.0.dev0")


if HAVE_TE:
    from megatron.core import parallel_state

    def get_fp4_recipe(config: TransformerConfig):
        """Return fp4 recipe."""
        if is_te_min_version("2.7.0.dev0"):
            if config.fp4_recipe == Fp4Recipe.nvfp4:
                try:
                    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
                except AttributeError:
                    raise ValueError(
                        """NVFP4BlockScaling recipe is not available in this version of 
                        Transformer Engine. Please make sure you are using TE version 
                        >= 2.7.0.dev0."""
                    )
            elif config.fp4_recipe == Fp4Recipe.custom:
                fp4_recipe = _get_custom_recipe(config.fp4_quantizer_factory)
            else:
                raise ValueError(
                    "NVFP4BlockScaling and custom are the only supported FP4 recipes. "
                    "Please make sure you are using a compatible TE version >= 2.7.0.dev0."
                )
        else:
            raise ValueError(
                """FP4 support requires TransformerEngine version >= 2.7.0.dev0 
                for NVFP4BlockScaling."""
            )
        return fp4_recipe

    def get_fp4_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp4 context manager."""
        num_bf16_layers_at_start = (
            config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        )
        num_bf16_layers_at_end = (
            config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        )
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

        need_fp4_context = config.fp4 if not is_init else config.fp4_param

        if not need_fp4_context:
            fp4_context = nullcontext()
        elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
            fp4_context = nullcontext()
        else:
            fp4_recipe = get_fp4_recipe(config)
            fp4_group = None
            if parallel_state.model_parallel_is_initialized():
                fp4_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                )

            if not is_init:
                # TE currently uses fp8_autocast for fp8 and fp4 quantization.
                fp4_context = transformer_engine.pytorch.fp8_autocast(
                    enabled=True, fp8_recipe=fp4_recipe, fp8_group=fp4_group
                )
            else:
                import inspect

                context_args = {"enabled": True}
                if (
                    "recipe"
                    in inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
                ):
                    context_args["recipe"] = fp4_recipe
                fp4_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        return fp4_context

else:

    def get_fp4_recipe(config: TransformerConfig):
        """Return None when Transformer Engine is not available."""
        return None

    def get_fp4_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return nullcontext when Transformer Engine is not available."""
        return nullcontext()
