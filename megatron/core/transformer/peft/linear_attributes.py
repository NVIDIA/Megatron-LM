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

"""Recognize native linear layouts before attaching adapters."""

import re
from dataclasses import dataclass
from importlib.metadata import version

import packaging.version
import torch.nn as nn

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.router import TopKRouter

from .process_groups import _get_tensor_parallel_group_from_module, _process_group_size

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
    TEColumnParallelLinear = TELayerNormColumnParallelLinear = TEColumnParallelGroupedLinear = ()
    TERowParallelLinear = TERowParallelGroupedLinear = TELinear = ()

TECL = (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)
TERL = (TERowParallelLinear, TERowParallelGroupedLinear)


def is_modelopt_linear(module):
    return type(module).__module__.startswith("megatron.core.post_training.modelopt")


@dataclass(frozen=True)
class AdapterAttributes:
    """Container for base linear adapter attributes."""

    input_is_parallel: bool
    in_features: int
    out_features: int
    disable_tensor_parallel_comm: bool
    disable_sequence_parallel_comm: bool
    base_linear_is_parallel: bool
    replicate_adapter: bool = False


def get_adapter_attributes_from_linear(
    m: nn.Module,
    is_expert: bool = False,
    pg_collection: ProcessGroupCollection | None = None,
    sequence_parallel_input_regather: bool = False,
) -> AdapterAttributes:
    """Returns attributes from the base layer as an AdapterAttributes dataclass.

    input_is_parallel, in_features, out_features, disable_tensor_parallel_comm,
    disable_sequence_parallel_comm, base_linear_is_parallel

    This function analyzes a linear module and extracts key attributes needed for adapter configuration,
    particularly for PEFT adapters in distributed training scenarios.

    Args:
        m: The linear module to analyze (should have a config attribute).
        is_expert: Whether the linear belongs to an expert module.
        pg_collection: Optional process-group collection associated with the module.
        sequence_parallel_input_regather: Whether LoRA-A should re-gather its sequence-parallel input in backward.

    Returns:
        AdapterAttributes containing:
            - input_is_parallel: Whether the input is already parallelized
            - in_features: Input feature dimension
            - out_features: Output feature dimension
            - disable_tensor_parallel_comm: Whether to disable tensor parallel communication
            - disable_sequence_parallel_comm: Whether to disable sequence parallel communication
            - base_linear_is_parallel: Whether the base linear layer uses parallelization

    Raises:
        NotImplementedError: If the layer type is not recognized for LoRA adaptation.
    """
    disable_sequence_parallel_comm = not m.config.sequence_parallel
    base_linear_is_parallel = True
    replicate_adapter = False

    # In some modules (notably MoE shared_experts when moe_shared_expert_overlap is enabled),
    # Megatron disables TP-related communications on the base linear layer by
    # setting `parallel_mode=None` (TE) or `explicit_expert_comm=True` (legacy).
    # https://github.com/NVIDIA/Megatron-LM/blob/5b1ef0703184299fbf71f6131bf2f9a5331e7238/megatron/core/transformer/moe/shared_experts.py#L95-L104
    # The weights are still TP-sharded though, so we must keep using the real TP size
    disable_tensor_parallel_comm = getattr(m, "parallel_mode", "") is None or getattr(
        m, "explicit_expert_comm", False
    )
    if disable_tensor_parallel_comm:
        disable_sequence_parallel_comm = True

    if is_modelopt_linear(m):
        return AdapterAttributes(
            input_is_parallel=False,
            in_features=m.in_features,
            out_features=m.out_features,
            disable_tensor_parallel_comm=False,
            disable_sequence_parallel_comm=True,
            base_linear_is_parallel=False,
        )

    tp_group = _get_tensor_parallel_group_from_module(
        m, is_expert=is_expert, pg_collection=pg_collection
    )
    tp_size = _process_group_size(
        tp_group,
        getattr(
            m.config,
            "expert_tensor_parallel_size" if is_expert else "tensor_model_parallel_size",
            1,
        ),
    )
    if isinstance(m, TopKRouter):
        input_is_parallel = False
        in_features = m.weight.shape[1]
        out_features = m.weight.shape[0]
        base_linear_is_parallel = False
        disable_sequence_parallel_comm = True
    elif HAVE_TE and any(isinstance(m, te_column_parallel) for te_column_parallel in TECL):
        input_is_parallel = False
        # m.in_features and m.out_features are divided by tp_size already,
        # but in_features and out_features passed to ParallelLinearAdapter are not.
        in_features = m.in_features
        out_features = m.out_features * tp_size

        if isinstance(m, TELayerNormColumnParallelLinear):
            # LoRA is applied after layernorm, so layernorm output must be returned
            m.return_layernorm_output = True
            # perf optimization for LoRA + SP
            if hasattr(m, "ub_overlap_ag"):
                ub_overlap_ag = m.ub_overlap_ag
            elif hasattr(m, "ub_overlap_ag_fprop"):
                ub_overlap_ag = m.ub_overlap_ag_fprop
            else:
                ub_overlap_ag = False
            if hasattr(m, "config") and m.config.sequence_parallel and not ub_overlap_ag:
                # The adapter's MCore SP linear must own the gather so it can retain the local shard.
                # Preserve the existing TE optimization for the default path.
                m.return_layernorm_output_gathered = not sequence_parallel_input_regather
                te_version = packaging.version.Version(version("transformer-engine"))
                if (
                    not sequence_parallel_input_regather
                    and te_version >= packaging.version.Version("1.5.0dev")
                    and (
                        not getattr(m.config, "tp_comm_overlap", False)
                        or getattr(m.config, "tp_comm_overlap_disable_qkv", False)
                    )
                ):
                    # TE 1.5 introduces the option `return_layernorm_output_gathered`, so the all gather
                    # in the forward method is not needed, so disable sp communications
                    # unless TP communication overlap is used
                    disable_sequence_parallel_comm = True
    elif HAVE_TE and any(isinstance(m, te_row_parallel) for te_row_parallel in TERL):
        input_is_parallel = True
        in_features = m.in_features * tp_size
        out_features = m.out_features
    elif HAVE_TE and isinstance(m, TELinear):  # parallel_mode="duplicated"
        input_is_parallel = False
        in_features = m.in_features
        out_features = m.out_features
        base_linear_is_parallel = False
        replicate_adapter = True
    elif isinstance(m, ColumnParallelLinear):
        input_is_parallel = False
        in_features = m.input_size
        out_features = m.output_size
    elif isinstance(m, RowParallelLinear):
        input_is_parallel = True
        in_features = m.input_size
        out_features = m.output_size
    else:
        raise NotImplementedError(f"Layer type is unrecognized for LoRA: {type(m)}")

    return AdapterAttributes(
        input_is_parallel=input_is_parallel,
        in_features=in_features,
        out_features=out_features,
        disable_tensor_parallel_comm=disable_tensor_parallel_comm,
        disable_sequence_parallel_comm=disable_sequence_parallel_comm,
        base_linear_is_parallel=base_linear_is_parallel,
        replicate_adapter=replicate_adapter,
    )


def is_expert_linear(fqn: str) -> bool:
    """Return whether the current base module is an expert linear module.

    This function checks if a fully qualified name (FQN) corresponds to an expert linear
    module in a Mixture of Experts (MoE) architecture.

    Args:
        fqn: Fully qualified name of the module.

    Returns:
        True if the module is an expert linear module, False otherwise.

    Example:
        >>> is_expert_linear("model.layers.0.mlp.experts.0.linear_fc1")
        True
        >>> is_expert_linear("model.layers.0.mlp.linear_fc1")
        False
    """
    return (
        re.match(r".*mlp\..*experts.*\.linear_fc[1-2]$", fqn) is not None
        and ".shared_experts." not in fqn
    )


def is_grouped_expert_linear(fqn: str) -> bool:
    """Return whether the current base module is a grouped expert linear module."""

    return is_expert_linear(fqn) and ".local_experts." not in fqn


def get_effective_lora_dim(
    module: nn.Module, *, dim: int, normalize_moe_lora: bool, is_expert: bool
) -> int:
    """Return the LoRA rank to use, reduced for expert layers when ``normalize_moe_lora`` is enabled."""

    if not normalize_moe_lora or not is_expert:
        return dim
    topk = module.config.moe_router_topk
    if topk is None or topk <= 0:
        raise ValueError(
            f"normalize_moe_lora is enabled but moe_router_topk is {topk!r}; "
            f"it must be set to a positive integer on the model config"
        )
    if dim % topk != 0:
        raise ValueError(
            f"LoRA dim={dim} must be divisible by moe_router_topk={topk} when normalize_moe_lora is enabled"
        )
    return dim // topk


def align_expert_dim_for_tp(
    module: nn.Module,
    dim: int,
    *,
    normalize_moe_lora: bool,
    is_expert: bool,
    input_is_parallel: bool,
    pg_collection: ProcessGroupCollection | None = None,
) -> int:
    """Round normalized expert LoRA ranks up to the expert-TP granularity when needed."""

    if not normalize_moe_lora or not is_expert or input_is_parallel:
        return dim

    expert_tp_group = _get_tensor_parallel_group_from_module(
        module, is_expert=True, pg_collection=pg_collection
    )
    expert_tp_size = _process_group_size(
        expert_tp_group, module.config.expert_tensor_parallel_size or 1
    )
    if expert_tp_size <= 1 or dim % expert_tp_size == 0:
        return dim

    return ((dim + expert_tp_size - 1) // expert_tp_size) * expert_tp_size
