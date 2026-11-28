# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/gtp.py
# Source license: Apache-2.0.

"""Translate between stored GTP shards and the TP-local tensors used by mappings."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


def _is_gtp_param(param: torch.Tensor | None) -> bool:
    return getattr(param, "is_gtp_weight_remat", False) is True


def _get_mapping_shape(param: torch.Tensor) -> torch.Size:
    """Return the logical TP-local shape, excluding GTP padding and sharding."""
    if _is_gtp_param(param):
        shape = list(param.shape)
        shape[0] = shape[0] * param.group.size() - param.pad_length
        return torch.Size(shape)
    return param.shape


@contextmanager
def _gtp_weight_load_context(param: torch.Tensor, module: torch.nn.Module) -> Iterator[None]:
    """Let TE update native FP8 storage while preserving the GTP parameter subclass."""
    if _is_gtp_param(param) and getattr(param, "_gtp_native_fp8", False):
        from megatron.core.tensor_parallel.gtp_api import gtp_native_fp8_load_context

        with gtp_native_fp8_load_context(module):
            yield
    else:
        yield


def _slice_gtp_weight(weight: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Take the stored GTP row shard after the mapping has applied TP splitting."""
    if not _is_gtp_param(param):
        return weight
    expected_shape = _get_mapping_shape(param)
    if weight.shape != expected_shape:
        raise ValueError(f"Expected TP-local GTP weight shape {expected_shape}, got {weight.shape}")
    if param.pad_length:
        weight = torch.cat((weight, weight.new_zeros(param.pad_length, *weight.shape[1:])), dim=0)
    return weight.narrow(0, param.group.rank() * param.shape[0], param.shape[0]).contiguous()


def _gather_gtp_weight(param: torch.Tensor | None) -> torch.Tensor | None:
    """Gather only the GTP axis, leaving TP/EP/PP handling to the mapping."""
    if not _is_gtp_param(param):
        return param
    if getattr(param, "_gtp_native_fp8", False):
        from megatron.core.tensor_parallel.gtp_api import dequantize_gtp_native_fp8

        local_weight = dequantize_gtp_native_fp8(param)
    else:
        # GTP's detach preserves the subclass but drops its runtime attributes.
        # Collectives should operate on a plain tensor without triggering prefetch.
        local_weight = param.as_subclass(torch.Tensor).detach()
    local_weight = local_weight.contiguous()
    padded_shape = (param.shape[0] * param.group.size(), *param.shape[1:])
    weight = torch.empty(padded_shape, dtype=local_weight.dtype, device=local_weight.device)
    torch.distributed.all_gather_into_tensor(weight, local_weight, group=param.group)
    return weight[: _get_mapping_shape(param)[0]]
