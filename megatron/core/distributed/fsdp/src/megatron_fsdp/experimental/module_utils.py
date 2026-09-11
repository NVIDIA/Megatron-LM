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

"""Utilities for parameter ownership and metadata in FSDP modules."""

from torch import nn


def get_parameter_owner(root_module: nn.Module, parameter_fqn: str) -> tuple[nn.Module, str]:
    """Resolve a root-module-relative parameter FQN to its direct owner."""
    module_name, separator, parameter_name = parameter_fqn.rpartition(".")
    owner = root_module.get_submodule(module_name) if separator else root_module
    return owner, parameter_name


# Preserve model metadata across parameter materialization and replacement.
# Tensor-subclass storage, autograd hooks, and FSDP runtime state remain specific
# to each Parameter representation.
_PARAMETER_ATTRIBUTES = (
    "requires_grad",
    "sequence_parallel",
    "shared",
    "shared_embedding",
    "tensor_model_parallel",
    "partition_dim",
    "partition_stride",
    "_tensor_parallel_mode",
    "allreduce",
    "grad_norm_group",
    "is_embedding_or_output_parameter",
    "is_embedding_parameter",
    "use_muon",
    "expert_tp",
    "is_qkv",
    "qkv_split_shapes",
    "skip_backward_post_hook",
    "is_gtp_weight_remat",
    "gtp_replica_group",
    "pad_length",
    "group",
)


def save_parameter_attributes(parameter: nn.Parameter) -> dict[str, object]:
    """Snapshot model metadata before materializing or replacing a Parameter.

    Values are shallow-copied: process-group references must retain their identity.
    Extend the shared attribute list when adding a new model-parameter contract.
    """
    return {
        name: getattr(parameter, name) for name in _PARAMETER_ATTRIBUTES if hasattr(parameter, name)
    }


def restore_parameter_attributes(parameter: nn.Parameter, attributes: dict[str, object]) -> None:
    """Restore saved model metadata, including explicitly false or None values."""
    for name, value in attributes.items():
        setattr(parameter, name, value)
