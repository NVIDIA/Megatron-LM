# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""MXFP8 payload helpers for the minimal Megatron-FSDP path.

MFSDP v2 rests model weights as TE ``MXFP8Tensor`` primary weights with two
payload orientations — row-wise (forward GEMM) and column-wise (backward
GEMM). Quantization itself is delegated to Transformer Engine's verified
``cast_master_weights_to_fp8`` (``te_cast_master_weights_to_fp8`` +
``allocate_quantize_temp`` allocate the full-size temporaries TE fills); this
module only rebinds and detaches the raw uint8 payloads:

- ``set_rowwise_payload`` / ``set_columnwise_payload`` bind gathered payload
  storage onto a TE quantized tensor via TE's ``fp8_set_raw_data``.
- ``clear_payloads`` detaches a tensor's payloads between unshards (they rest
  sharded in the group's rowwise/colwise DBuffers otherwise).

Payloads are ``(rows, cols)`` row-major uint8 data (TE's ``_columnwise_data``
shares the ``_rowwise_data`` shape); only the block direction and scale grid
differ between the two orientations.
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
