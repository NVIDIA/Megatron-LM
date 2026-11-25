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

# Adapted from NVIDIA-NeMo/Megatron-Bridge (Apache-2.0).
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/lora_merge.py

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from megatron.core.utils import get_pg_size


class LoRAMerge:
    """
    Tensor helper for merging LoRA adapter weights into base weights.
    """

    def merge(
        self,
        base_weight: torch.Tensor,
        linear_out: torch.Tensor,
        linear_in: torch.Tensor,
        alpha: int,
        dim: int,
        *,
        tp_group: dist.ProcessGroup | None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Merges the LoRA adapter weights with the base model weights.
        Handles tensor parallelism by gathering sharded dimensions. The gather
        sums backward contributions across TP ranks when used as a live weight.

        For ColumnParallelLinear (e.g., linear_qkv, linear_fc1):
            - base_weight: (out_features/TP, in_features)
            - linear_in: (dim/TP, in_features) <- Need to gather this
            - linear_out: (out_features/TP, dim)
            - Target: (out_features/TP, dim) @ (dim, in_features) = (out_features/TP, in_features)

        For RowParallelLinear (e.g., linear_proj, linear_fc2):
            - base_weight: (out_features, in_features/TP)
            - linear_in: (dim, in_features/TP)
            - linear_out: (out_features/TP, dim) <- Need to gather this
            - Target: (out_features, dim) @ (dim, in_features/TP) = (out_features, in_features/TP)

        Args:
            base_weight (torch.Tensor): The base model weights.
            linear_out (torch.Tensor): LoRA's B matrix.
            linear_in (torch.Tensor): LoRA's A matrix.
            alpha (int): Weighting factor for the low-rank projection.
            dim (int): Dimension of the low-rank projection space.
            tp_group: Tensor-parallel process group for the adapter shard.
            scale: Optional precomputed LoRA scale. Defaults to alpha / dim.

        Returns:
            torch.Tensor: The merged weights.
        """

        lora_scale = alpha / dim if scale is None else scale
        tp_size = get_pg_size(tp_group)

        if tp_size == 1:
            lora_weight = lora_scale * (linear_out @ linear_in)
            return base_weight + lora_weight

        if linear_in.shape[0] * tp_size == dim and linear_out.shape[1] == dim:
            linear_in_full = torch.cat(all_gather(linear_in, group=tp_group), dim=0)
            lora_weight = lora_scale * (linear_out @ linear_in_full)

        elif linear_out.shape[0] * tp_size == base_weight.shape[0]:
            linear_out_full = torch.cat(all_gather(linear_out, group=tp_group), dim=0)
            lora_weight = lora_scale * (linear_out_full @ linear_in)

        else:
            lora_weight = lora_scale * (linear_out @ linear_in)

        return base_weight + lora_weight
