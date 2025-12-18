# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from importlib.metadata import version
from typing import List, Optional, Tuple

import torch
from packaging.version import Version as PkgVersion

logger = logging.getLogger(__name__)

# Detect if Transformer Engine is installed
try:
    import transformer_engine  # pylint: disable=W0611
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    TransformerEngineBaseModule = None
    HAVE_TE = False
    logger.info("Using Megatron-FSDP without Transformer Engine.")

# Detect the Transformer Engine version
try:
    import transformer_engine as te

    if hasattr(te, "__version__"):
        TE_VERSION = PkgVersion(str(te.__version__))
    else:
        TE_VERSION = PkgVersion(version("transformer-engine"))
except:
    TE_VERSION = None

# Detect the FP8 tensor class
try:
    from transformer_engine.pytorch.tensor import QuantizedTensor

    HAVE_TE_FP8_TENSOR_CLASS = True
    FP8_TENSOR_CLASS = QuantizedTensor
except:
    try:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor

        HAVE_TE_FP8_TENSOR_CLASS = True
        FP8_TENSOR_CLASS = Float8Tensor
    except:
        HAVE_TE_FP8_TENSOR_CLASS = False

# Detect the MXFP8 tensor class
try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except:
    HAVE_TE_MXFP8TENSOR = False

# Detect the Blockwise FP8 tensor class
try:
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor

    HAVE_TE_BLOCKWISE_FP8TENSOR = True
except:
    HAVE_TE_BLOCKWISE_FP8TENSOR = False

# Detect the "cast_master_weights_to_fp8" function of Transformer Engine
try:
    from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

    HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8 = True
except:
    HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8 = False

    # Try to import multi_tensor_apply, used in the fallback of fp8 quantization.
    try:
        from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_scale

        multi_tensor_scale_impl = multi_tensor_scale
    except ImportError:
        try:
            import amp_C
            from apex.multi_tensor_apply import multi_tensor_applier

            multi_tensor_scale_impl = amp_C.multi_tensor_scale
        except ImportError:
            import warnings

            warnings.warn(
                "Transformer Engine and Apex are not installed. "
                "Falling back to local implementations of "
                "multi_tensor_applier and multi_tensor_scale"
            )

            def local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
                """Multi tensor op applier"""
                return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)

            def local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
                """Works as a drop-in replacement for amp_C.multi_tensor_scale."""
                for src, dst in zip(tensor_lists[0], tensor_lists[1]):
                    dst.copy_(src * scale)

            multi_tensor_applier = local_multi_tensor_applier
            multi_tensor_scale_impl = local_multi_tensor_scale

    def _multi_tensor_copy_this_to_that(
        this: List[torch.Tensor],
        that: List[torch.Tensor],
        overflow_buf: Optional[torch.Tensor] = None,
    ):
        """
        Use multi-tensor-applier to copy values from one list to another.
        We don't have a bfloat16 implementation so for now if the overflow_buf
        is not provided, we default back to simple loop copy to be compatible
        with bfloat16.
        """
        if overflow_buf is not None:
            overflow_buf.fill_(0)
            # Scaling with factor `1.0` is equivalent to copy.
            multi_tensor_applier(multi_tensor_scale_impl, overflow_buf, [this, that], 1.0)
        else:
            for this_, that_ in zip(this, that):
                that_.copy_(this_)


# Detect the "post_all_gather_processing" function of Transformer Engine
try:
    from transformer_engine.pytorch.tensor.utils import post_all_gather_processing

    HAVE_TE_POST_ALL_GATHER_PROCESSING = True
except:
    HAVE_TE_POST_ALL_GATHER_PROCESSING = False


def is_te_min_version(vers, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    if not isinstance(TE_VERSION, PkgVersion):
        return False

    if check_equality:
        return TE_VERSION >= PkgVersion(vers)
    else:
        return TE_VERSION > PkgVersion(vers)


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a FP8 tensor."""
    return HAVE_TE and isinstance(tensor, FP8_TENSOR_CLASS)


def is_blockwise_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Blockwise FP8 tensor."""
    return HAVE_TE_BLOCKWISE_FP8TENSOR and isinstance(tensor, Float8BlockwiseQTensor)


def fp8_need_transpose_data(tensor: torch.Tensor) -> bool:
    """Check if a FP8 tensor needs transpose data."""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)


def fp8_need_transpose_data_for_meta_device_init(module: TransformerEngineBaseModule) -> bool:
    """Check if a FP8 tensor needs transpose data, for meta device init scenario."""
    return HAVE_TE_MXFP8TENSOR and module.fp8_meta["recipe"].mxfp8()


def fp8_discard_transpose_cache(tensor: torch.Tensor) -> None:
    """Discard the transpose cache of a FP8 tensor."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"

    if hasattr(tensor, "_transpose_invalid"):
        tensor._transpose_invalid = True
        tensor._transpose = None
    elif not fp8_need_transpose_data(tensor):
        tensor.update_usage(rowwise_usage=True, columnwise_usage=False)


def fp8_create_transpose_cache(tensors: List[torch.Tensor]) -> None:
    """Create the transpose cache of a FP8 tensor."""
    if HAVE_TE_POST_ALL_GATHER_PROCESSING:
        post_all_gather_processing(tensors)
    else:
        _fp8_create_transpose_cache_fallback(tensors)


def _fp8_create_transpose_cache_fallback(tensors: List[torch.Tensor]) -> None:
    if not isinstance(tensors, list):
        tensors = [tensors]
    for tensor in tensors:
        assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"
        if hasattr(tensor, "_create_transpose"):
            tensor._create_transpose()
        else:
            tensor._create_columnwise()


def fp8_set_raw_data(tensor: torch.Tensor, data: torch.Tensor, set_transpose: bool = False) -> None:
    """Set the raw data of a Transformer Engine Float8Tensor."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"

    if set_transpose:
        assert fp8_need_transpose_data(tensor), f"Type {type(tensor)} does not need transpose data"
        data_attr = "_columnwise_data"
    else:
        data_attr = "_rowwise_data" if hasattr(tensor, "_rowwise_data") else "_data"

    old_data = getattr(tensor, data_attr)
    assert old_data.dtype == data.dtype, "The data types of raw data don't match"
    assert (
        old_data.shape == data.shape
    ), f"Shape {old_data.shape} of old_data doesn't match {data.shape} of new_data"
    setattr(tensor, data_attr, data)


def fp8_get_raw_data(tensor: torch.Tensor, get_transpose: bool = False) -> torch.Tensor:
    """Get the underlying raw storage of a FP8 tensor."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"

    if get_transpose:
        assert fp8_need_transpose_data(tensor), f"Type {type(tensor)} does not need transpose data"
        data_attr = "_columnwise_data"
    else:
        data_attr = "_rowwise_data" if hasattr(tensor, "_rowwise_data") else "_data"

    return getattr(tensor, data_attr)


def fp8_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a FP8 tensor to a higher precision."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"
    assert is_te_min_version(
        "2.0"
    ), "Transformer Engine >= 2.0 is required for dequantizing parameters."
    return tensor.dequantize()


def fp8_quantize(
    model_params: List[torch.Tensor],
    main_params: List[torch.Tensor],
    start_offsets: List[int],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> None:
    """Quantize sharded parameters to FP8."""
    if len(model_params) == 0:
        return
    fsdp_shard_model_params = [x[0] if x[1] is None else x for x in fsdp_shard_model_params]

    if HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8:
        cast_master_weights_to_fp8(
            model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
        )
    else:
        _fp8_quantize_fallback(
            model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
        )


def _fp8_quantize_fallback(
    model_params: List[torch.Tensor],
    main_params: List[torch.Tensor],
    start_offsets: List[int],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> None:
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
        # When not using fp8 params, the main_param (fp32) is first cast to bf16/fp16, and then
        # cast to fp8 during forward. This logic keeps numerical consistency with bf16 params.
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

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

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
