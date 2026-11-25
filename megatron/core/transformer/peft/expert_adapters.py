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
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/utils.py

"""Per-expert low-rank adapters and their checkpoint layouts."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core import ModelParallelConfig
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import set_tensor_model_parallel_attributes

from .adapters import ParallelLinearAdapter
from .process_groups import (
    _get_pg_collection,
    _get_process_group,
    _get_tensor_parallel_group,
    _process_group_size,
)
from .sharding import (
    _apply_grouped_expert_swiglu_sharded_factory,
    _divide_exact,
    _make_grouped_expert_sharded_tensor,
)
from .tensor_helpers import pad_seq_to_mult, unpad_seq_to_mult


class _GroupedExpertAdapterWeight(nn.Module):
    """Callable parameter container so DDP forward pre-hooks see grouped LoRA weights."""

    # Overlapped param gather is driven by module forward pre-hooks. Calling this
    # container before reading the weight makes expert-DP LoRA params participate
    # in the normal training-time gather instead of only forced eval/checkpoint sync.

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, indices: Optional[List[int]] = None) -> torch.Tensor:
        if indices is None:
            return self.weight
        return self.weight[indices]


class GroupedExpertLinearAdapter(nn.Module):
    """LoRA adapter with one low-rank pair per local grouped MoE expert."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        *,
        num_local_experts: int,
        base_linear_name: str,
        activation: str = "swish",
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        input_is_parallel: bool = False,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: Optional[float] = None,
        dropout_position: str = "pre",
        base_linear_is_parallel: bool = True,
        params_device: Optional[torch.device] = None,
        params_dtype: Optional[torch.dtype] = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Initialize grouped-expert LoRA weights for one adapter per local expert."""

        super().__init__()

        self.base_linear_name = base_linear_name
        self.activation = ParallelLinearAdapter._get_activation_fn(self, activation)
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.input_is_parallel = input_is_parallel
        self.dropout_position = dropout_position
        self.base_linear_is_parallel = base_linear_is_parallel
        self.is_expert = True
        self.num_local_experts = num_local_experts
        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        self.config = model_parallel_config
        self.pg_collection = _get_pg_collection(
            pg_collection, model_parallel_config, required_pgs=["tp", "ep", "expt_tp", "expt_dp"]
        )
        tensor_parallel_group = _get_tensor_parallel_group(self.pg_collection)
        self.expert_tp_group = _get_tensor_parallel_group(self.pg_collection, is_expert=True)
        self.ep_group = _get_process_group(self.pg_collection, "ep")
        self.expert_dp_group = _get_process_group(self.pg_collection, "expt_dp")

        expert_tp_size = _process_group_size(
            self.expert_tp_group, model_parallel_config.expert_tensor_parallel_size or 1
        )
        if expert_tp_size != 1:
            raise NotImplementedError(
                "Per-expert LoRA currently requires expert tensor parallel size 1"
            )
        tensor_parallel_size = _process_group_size(
            tensor_parallel_group,
            getattr(model_parallel_config, "tensor_model_parallel_size", 1) or 1,
        )
        linear_in_tp_axis = 2 if input_is_parallel else 1
        linear_out_tp_axis = 1

        if input_is_parallel:
            linear_in_shape = (
                num_local_experts,
                dim,
                _divide_exact(in_features, expert_tp_size, "in_features"),
            )
        else:
            linear_in_shape = (
                num_local_experts,
                _divide_exact(dim, expert_tp_size, "dim"),
                in_features,
            )
        linear_out_shape = (
            num_local_experts,
            _divide_exact(out_features, expert_tp_size, "out_features"),
            dim,
        )

        if params_device is None:
            distributed_initialized = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            params_device = (
                torch.device("cpu")
                if model_parallel_config.use_cpu_initialization
                or not torch.cuda.is_available()
                or not distributed_initialized
                else torch.device("cuda", torch.cuda.current_device())
            )
        dtype = params_dtype or model_parallel_config.params_dtype

        linear_in_weight = torch.empty(linear_in_shape, device=params_device, dtype=dtype)
        linear_out_weight = torch.empty(linear_out_shape, device=params_device, dtype=dtype)
        # The weights are stacked per-expert matrices. Resolve the fan-dependent
        # init methods from the full (unsharded) per-expert dims: a fan read off
        # the 3-D tensor would scale with ``dim * in_features`` and with the
        # number of local experts, so the init would depend on EP/ETP size and
        # start far too small (see ``_get_init_fn``).
        ParallelLinearAdapter._get_init_fn(
            self, column_init_method, fan_in=in_features, fan_out=dim
        )(linear_in_weight)
        ParallelLinearAdapter._get_init_fn(self, row_init_method, fan_in=dim, fan_out=out_features)(
            linear_out_weight
        )

        use_expert_process_groups = (
            _process_group_size(
                self.ep_group, model_parallel_config.expert_model_parallel_size or 1
            )
            > 1
            or expert_tp_size != tensor_parallel_size
        )
        self._linear_in_tp_axis = linear_in_tp_axis
        self._linear_out_tp_axis = linear_out_tp_axis
        self.linear_in = _GroupedExpertAdapterWeight(linear_in_weight)
        self.linear_out = _GroupedExpertAdapterWeight(linear_out_weight)
        for weight, tp_axis in (
            (self.linear_in.weight, linear_in_tp_axis),
            (self.linear_out.weight, linear_out_tp_axis),
        ):
            setattr(weight, "allreduce", not use_expert_process_groups)
            if tp_axis is not None:
                set_tensor_model_parallel_attributes(weight, True, tp_axis, 1)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def _extract_expert_splits(self, args: Tuple, kwargs: Dict) -> List[int]:
        """Extract grouped-expert token splits from wrapped-module call arguments."""

        expert_splits = kwargs.get("m_splits")
        if expert_splits is None:
            expert_splits = kwargs.get("tokens_per_expert")
        if expert_splits is None and args:
            expert_splits = args[0]
        if isinstance(expert_splits, torch.Tensor):
            expert_splits = expert_splits.tolist()
        if expert_splits is None:
            raise ValueError(
                f"Per-expert LoRA on {self.base_linear_name} requires grouped expert token splits."
            )
        if len(expert_splits) != self.num_local_experts:
            raise ValueError(
                f"Expected {self.num_local_experts} expert splits for {self.base_linear_name}, "
                f"got {len(expert_splits)}"
            )
        splits = [int(split) for split in expert_splits]
        if any(split < 0 for split in splits):
            raise ValueError(
                f"Expert splits for {self.base_linear_name} must be non-negative, got {splits}"
            )
        return splits

    def _gather_along_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor across expert TP ranks by concatenating its last dimension."""

        expert_tp_size = _process_group_size(
            self.expert_tp_group, self.config.expert_tensor_parallel_size or 1
        )
        if expert_tp_size == 1:
            return tensor
        expert_tp_group = self.expert_tp_group
        if expert_tp_group is None:
            raise ValueError(
                f"{self.base_linear_name} requires initialized expert tensor parallel state "
                f"when expert_tensor_parallel_size={expert_tp_size}."
            )
        gathered = [torch.empty_like(tensor) for _ in range(expert_tp_size)]
        torch.distributed.all_gather(gathered, tensor, group=expert_tp_group)
        return torch.cat(gathered, dim=-1)

    def _forward_per_expert(
        self, x: torch.Tensor, *, expert_splits: List[int], expert_tp_size: int
    ) -> torch.Tensor:
        """Apply the adapter using the per-expert fallback path."""

        linear_in_weight = self.linear_in()
        linear_out_weight = self.linear_out()
        outputs = []
        start = 0
        for expert_idx, split_size in enumerate(expert_splits):
            expert_input = x.narrow(0, start, split_size)
            start += split_size

            pad_len = 0
            if expert_input.numel() > 0:
                expert_input, pad_len = pad_seq_to_mult(expert_input, expert_tp_size)
                if self.config.cpu_offloading and self.config.cpu_offloading_activations:
                    expert_input.activation_offloading = True

            hidden = nn.functional.linear(expert_input, linear_in_weight[expert_idx])
            if not self.input_is_parallel:
                hidden = self._gather_along_last_dim(hidden)
            hidden = self.activation(hidden)
            # Scale the rank-sized bottleneck rather than the full-width output. The
            # output projection is bias-free, so this is exact and avoids allocating
            # a second output-sized temporary.
            hidden = hidden * (self.alpha / self.dim)

            if self.config.cpu_offloading and self.config.cpu_offloading_activations:
                hidden.activation_offloading = True
            expert_output = nn.functional.linear(hidden, linear_out_weight[expert_idx])
            if self.input_is_parallel:
                expert_output = self._gather_along_last_dim(expert_output)

            if self.dropout_position == "post":
                expert_output = self.dropout(expert_output)
            if pad_len > 0:
                expert_output = unpad_seq_to_mult(expert_output, pad_len)
            outputs.append(expert_output)

        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the local expert-specific LoRA update to grouped expert inputs."""

        expert_splits = self._extract_expert_splits(args, kwargs)
        total_tokens = sum(expert_splits)
        if total_tokens != x.shape[0]:
            raise ValueError(
                f"Expert splits for {self.base_linear_name} sum to {total_tokens}, but received {x.shape[0]} tokens"
            )
        if self.dropout_position == "pre":
            x = self.dropout(x)

        expert_tp_size = _process_group_size(
            self.expert_tp_group, self.config.expert_tensor_parallel_size or 1
        )
        output_features = self.linear_out.weight.shape[1]
        if self.input_is_parallel:
            output_features *= expert_tp_size
        if x.shape[0] == 0:
            linear_in_weight = self.linear_in()
            linear_out_weight = self.linear_out()
            grad_anchor = linear_in_weight.reshape(-1)[0] + linear_out_weight.reshape(-1)[0]
            return (x.new_empty((0, output_features)) + grad_anchor * 0.0) * (self.alpha / self.dim)

        return self._forward_per_expert(
            x, expert_splits=expert_splits, expert_tp_size=expert_tp_size
        )

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: Tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Create sharded state dictionary for grouped-expert adapter weights."""

        sharded_state_dict = {}
        linear_in_sd = {
            f"{prefix}linear_in.weight": _make_grouped_expert_sharded_tensor(
                self.linear_in.weight,
                f"{prefix}linear_in.weight",
                tp_axis=self._linear_in_tp_axis,
                sharded_offsets=sharded_offsets,
                pg_collection=self.pg_collection,
                ep_size_fallback=self.config.expert_model_parallel_size or 1,
                etp_size_fallback=self.config.expert_tensor_parallel_size or 1,
            )
        }
        linear_out_sd = {
            f"{prefix}linear_out.weight": _make_grouped_expert_sharded_tensor(
                self.linear_out.weight,
                f"{prefix}linear_out.weight",
                tp_axis=self._linear_out_tp_axis,
                sharded_offsets=sharded_offsets,
                pg_collection=self.pg_collection,
                ep_size_fallback=self.config.expert_model_parallel_size or 1,
                etp_size_fallback=self.config.expert_tensor_parallel_size or 1,
            )
        }

        if "linear_fc1" in self.base_linear_name and getattr(
            self.config, "gated_linear_unit", False
        ):
            singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)
            linear_out_key = f"{prefix}linear_out.weight"
            linear_out_sd[linear_out_key] = _apply_grouped_expert_swiglu_sharded_factory(
                linear_out_sd[linear_out_key], sharded_offsets, singleton_local_shards
            )

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict
