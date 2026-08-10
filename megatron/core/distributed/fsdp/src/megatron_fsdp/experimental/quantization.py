# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""MXFP8 block quantization kernels for the minimal Megatron-FSDP path.

Implements MXFP8 E4M3 block-scaled quantization with 32-element blocks and
one bf16 scale per block (scale = block amax / E4M3 max normal). The encode is
bit-exact E4M3: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, with
round-half-to-even and saturation at +/-448. Values below 2**-6 encode as
subnormals in 2**-9 steps.

Two quantization geometries are provided for model weights, matching
Transformer Engine's MXFP8 primary weights:

- Row-wise: 1x32 blocks along the last dimension (forward GEMM weight).
- Column-wise: 32x1 blocks along the first dimension (backward GEMM weight).
  Column-wise scales are global: each rank computes a partial per-block amax
  over its own rows, the ranks reduce-max, and every rank quantizes its rows
  with the merged scale grid.

Both payloads are ``(rows, cols)`` row-major uint8 data (TE's
``_columnwise_data`` has the same shape as ``_rowwise_data``); only the block
direction and the scale grid differ.

The scale dtype is bf16 rather than E8M0 for simplicity; the E4M3 payload
format is identical to MXFP8, so the numerics match block-scaled FP8 training
to within scale precision.

The production path delegates quantization to TE's verified
``cast_master_weights_to_fp8`` (see ``te_cast_master_weights_to_fp8`` and
``allocate_quantize_temp``); ``set_*_payload`` / ``clear_payloads`` rebind raw
payloads through TE's verified ``fp8_set_raw_data``.
"""

import torch

from ..mixed_precision import fp8_set_raw_data

E4M3_BLOCK_SIZE = 32


def set_rowwise_payload(tensor: torch.Tensor, data: torch.Tensor) -> None:
    """Bind the raw row-wise fp8 payload onto a TE quantized tensor."""
    fp8_set_raw_data(tensor, data, set_transpose=False)


def set_columnwise_payload(tensor: torch.Tensor, data: torch.Tensor) -> None:
    """Bind the raw column-wise (backward-GEMM) fp8 payload onto a TE quantized tensor."""
    fp8_set_raw_data(tensor, data, set_transpose=True)


def clear_payloads(tensor: torch.Tensor) -> None:
    """Detach a TE quantized tensor from its raw payload storage.

    The payloads are only read while the tensor is installed for compute
    (between unshard and reshard); between unshards they rest detached while
    the sharded payloads live in the group's rowwise/colwise DBuffers.
    """
    if hasattr(tensor, "_rowwise_data"):
        tensor._rowwise_data = None
    elif hasattr(tensor, "_data"):
        tensor._data = None
    if hasattr(tensor, "_columnwise_data"):
        tensor._columnwise_data = None


_TE_CAST_MASTER_WEIGHTS_TO_FP8: bool | None = None


def te_cast_master_weights_to_fp8():
    """Return TE's ``cast_master_weights_to_fp8`` when importable, else None."""
    global _TE_CAST_MASTER_WEIGHTS_TO_FP8
    if _TE_CAST_MASTER_WEIGHTS_TO_FP8 is None:
        try:
            from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

            _TE_CAST_MASTER_WEIGHTS_TO_FP8 = cast_master_weights_to_fp8
        except ImportError:
            _TE_CAST_MASTER_WEIGHTS_TO_FP8 = False
    return _TE_CAST_MASTER_WEIGHTS_TO_FP8 or None


def allocate_quantize_temp(
    tensor: torch.Tensor, height: int, width: int, device: torch.device
) -> torch.Tensor:
    """Allocate a full-size temporary MXFP8Tensor for ``cast_master_weights_to_fp8``.

    TE quantizes into the temp's full-size row-wise and column-wise raw
    payloads, writing only the shard slices at the provided offsets; the
    caller copies those slices out and releases the temp. The temp's
    scale-inverse grids alias ``tensor``'s grids so TE fills them in place.
    ``height``/``width``/``device`` are the logical tensor geometry (TE's
    ``MXFP8Tensor.shape``/``device`` raise when the raw payloads are
    detached, so callers pass them in).
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

    return MXFP8Tensor(
        shape=(height, width),
        dtype=tensor.dtype,
        rowwise_data=torch.empty((height, width), dtype=torch.uint8, device=device),
        rowwise_scale_inv=tensor._rowwise_scale_inv,
        columnwise_data=torch.empty((height, width), dtype=torch.uint8, device=device),
        columnwise_scale_inv=tensor._columnwise_scale_inv,
        fp8_dtype=tensor._fp8_dtype,
        quantizer=MXFP8Quantizer(fp8_dtype=tensor._fp8_dtype),
        with_gemm_swizzled_scales=False,
        device=device,
    )
