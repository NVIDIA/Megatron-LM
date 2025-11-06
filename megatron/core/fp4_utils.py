# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utility functions related to FP4 that are used throughout Megatron core"""

from contextlib import nullcontext

import torch

from megatron.core.enums import Fp4Recipe
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


def dequantize_fp4_tensor(fp4_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp4 tensor to a higher precision tensor."""
    if is_te_min_version("2.7.0.dev0"):
        return fp4_tensor.dequantize()
    else:
        raise RuntimeError("FP4 dequantization requires Transformer Engine >= 2.7.0.dev0")


def get_nvfp4_rowwise_packed_shape(shape: torch.Size) -> torch.Size:
    """Return packed byte shape for NVFP4 rowwise storage (last dim // 2)."""
    if len(shape) == 0:
        return shape
    assert shape[-1] % 2 == 0, "NVFP4 requires inner dimension divisible by 2"
    packed = list(shape)
    packed[-1] = packed[-1] // 2
    return torch.Size(packed)


def modify_nvfp4_rowwise_storage(fp4_tensor: torch.Tensor, new_rowwise_data: torch.Tensor) -> None:
    """Replace NVFP4 tensor's rowwise raw data with a new uint8 storage view.

    Copies existing bytes into the new buffer, then swaps the underlying pointer.
    """
    if not is_nvfp4tensor(fp4_tensor):
        raise ValueError("modify_nvfp4_rowwise_storage expects an NVFP4 tensor")
    # Access TE's internal storage fields
    old_rowwise = getattr(fp4_tensor, "_rowwise_data", None)
    if old_rowwise is None:
        raise RuntimeError("NVFP4 tensor is missing rowwise data to replace")
    assert (
        old_rowwise.dtype == new_rowwise_data.dtype == torch.uint8
    ), "Rowwise NVFP4 storage must be uint8"
    # Preserve existing values and then swap storage
    new_rowwise_data.detach().copy_(old_rowwise)
    setattr(fp4_tensor, "_rowwise_data", new_rowwise_data)


def get_nvfp4_columnwise_packed_shape(shape: torch.Size) -> torch.Size:
    """Return packed byte shape for NVFP4 columnwise storage.

    Columnwise bytes are stored as a 2D tensor: (K, M/2), where M=prod(shape[:-1]), K=shape[-1].
    """
    if len(shape) == 0:
        return shape
    M = 1
    for d in list(shape)[:-1]:
        M *= int(d)
    K = int(shape[-1])
    assert M % 2 == 0, "NVFP4 requires flattened non-inner dimension divisible by 2"
    return torch.Size([K, M // 2])


def modify_nvfp4_columnwise_storage(fp4_tensor: torch.Tensor, new_columnwise_data: torch.Tensor) -> None:
    """Replace NVFP4 tensor's columnwise raw data with a new uint8 storage view."""
    if not is_nvfp4tensor(fp4_tensor):
        raise ValueError("modify_nvfp4_columnwise_storage expects an NVFP4 tensor")
    old_col = getattr(fp4_tensor, "_columnwise_data", None)
    if old_col is None:
        raise RuntimeError("NVFP4 tensor is missing columnwise data to replace")
    assert (
        old_col.dtype == new_columnwise_data.dtype == torch.uint8
    ), "Columnwise NVFP4 storage must be uint8"
    new_columnwise_data.detach().copy_(old_col)
    setattr(fp4_tensor, "_columnwise_data", new_columnwise_data)


def quantize_nvfp4_param_shard(
    model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params=None
):
    """Cast shard FP32 master weights to NVFP4 model params (rowwise/columnwise).

    NOTE: This is a placeholder stub. Proper NVFP4 shard quantization requires
    TE kernel support for nibble-accurate partial updates and coordinated tile updates.
    """
    raise NotImplementedError(
        "NVFP4 shard quantization path is not implemented in MCore yet. "
        "It requires TE NVFP4 shard casting support."
    )


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
            else:
                raise ValueError(
                    "NVFP4BlockScaling is the only supported FP4 recipe. "
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
