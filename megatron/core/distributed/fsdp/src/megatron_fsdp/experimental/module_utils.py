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


# Some parameter attributes need to be preserved:
# - Some weights (e.g. MoE router weights) need use_muon=False to explicitly
#   exclude them from Muon.
# - is_embedding_or_output_parameter and is_embedding_parameter identify embedding
#   weights. The Muon optimizer needs to exclude these weights.
_PARAMETER_ATTRIBUTES_TO_PRESERVE = (
    "is_embedding_or_output_parameter",
    "is_embedding_parameter",
    "use_muon",
)


def copy_parameter_attributes(from_: nn.Parameter, to_: nn.Parameter) -> None:
    """Copy model metadata between Parameters, including false or None values.

    Extend the shared attribute list when adding a new model-parameter contract.
    """
    for name in _PARAMETER_ATTRIBUTES_TO_PRESERVE:
        if hasattr(from_, name):
            setattr(to_, name, getattr(from_, name))
