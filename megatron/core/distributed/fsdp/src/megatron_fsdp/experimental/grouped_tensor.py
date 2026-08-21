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

"""Transformer Engine GroupedTensor support for the minimal Megatron-FSDP path.

A TE ``GroupedTensor`` -- what ``GroupedLinear(single_grouped_weight=True)`` uses to hold every
local expert in one parameter -- keeps its values in a flat ``rowwise_data`` buffer rather than
in a directly viewable ``.data``. Assigning ``parameter.data`` therefore does *not* change what
TE's kernels read: they keep reading the original ``rowwise_data`` allocation. FSDP must remap
that backing storage instead, which is what MCore's DDP does in
``megatron/core/distributed/param_and_grad_buffer.py``.

``rowwise_data`` alone does not identify a grouped tensor -- TE's quantized tensor classes
expose rowwise storage too -- so dispatch on the class, mirroring MCore's ``is_grouped_tensor``.
MCore's ``fp8_utils`` is deliberately not imported, since this package must not depend on MCore.
"""

import torch
from torch import nn
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


def get_values(tensor: torch.Tensor) -> torch.Tensor:
    """Return a plain, logically shaped view of ``tensor``'s values.

    For an ordinary tensor this is the tensor itself. For a grouped tensor it is its
    ``rowwise_data`` reshaped to the logical shape.
    """
    if not isinstance(tensor, GroupedTensor):
        return tensor
    return tensor.rowwise_data.view(tensor.shape)


def install_storage(parameter: nn.Parameter, storage: torch.Tensor) -> None:
    """Point a grouped ``parameter`` at ``storage`` without copying its old values in.

    FSDP refills ``storage`` from the all-gather on every unshard, so unlike DDP's one-time
    remap there is nothing worth preserving from the previous allocation.

    Only high-precision grouped storage is supported. Quantized grouped storage keeps ``uint8``
    bytes plus its own scale and amax buffers, none of which FSDP shards, so it is rejected by
    the dtype check below rather than silently remapped.
    """
    flat_storage = storage.view(-1)
    old_rowwise_data = parameter.rowwise_data
    if flat_storage.numel() != old_rowwise_data.numel():
        raise ValueError(
            "Grouped parameter backing storage size mismatch: "
            f"old numel={old_rowwise_data.numel()}, new numel={flat_storage.numel()}."
        )
    if flat_storage.dtype != old_rowwise_data.dtype:
        raise ValueError(
            "Grouped parameter backing storage dtype mismatch: "
            f"old dtype={old_rowwise_data.dtype}, new dtype={flat_storage.dtype}."
        )

    parameter.rowwise_data = flat_storage
    # These are derived from rowwise_data and would otherwise alias the old allocation.
    parameter.columnwise_data = None
    parameter.quantized_tensors = None
