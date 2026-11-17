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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/training/gtp.py
# Source license: Apache-2.0.

"""Generalized Tensor Parallelism helpers for the standard Bridge runtime."""

from collections.abc import Mapping
from typing import Any

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection


def get_transformer_config(model_config: Any) -> Any:
    """Return the MCore transformer config nested in a Bridge model config."""
    model_fields = getattr(type(model_config), "__dataclass_fields__", {})
    if "transformer" in model_fields:
        return model_config.transformer
    return model_config


def _get_checkpoint_weight_topology(model_config: Any) -> tuple[int, int, int, int]:
    """Read dense/expert TP and GTP sizes from runtime or serialized configs."""
    if isinstance(model_config, Mapping):
        nested_config = model_config.get("transformer")
        transformer_config = nested_config if isinstance(nested_config, Mapping) else model_config
    else:
        transformer_config = get_transformer_config(model_config)

    def get_value(name: str, default: Any) -> Any:
        if isinstance(transformer_config, Mapping):
            return transformer_config.get(name, default)
        return getattr(transformer_config, name, default)

    def positive_int(name: str, default: int) -> int:
        value = get_value(name, default)
        return value if isinstance(value, int) and value > 0 else default

    tp = positive_int("tensor_model_parallel_size", 1)
    etp = positive_int("expert_tensor_parallel_size", tp)
    # Public shard counts take precedence: deserialized providers may still have
    # the default values for the derived GTP fields until finalize() runs.
    dense_shards = get_value("tensor_parallel_num_weight_shards", None)
    expert_shards = get_value("expert_tensor_parallel_num_weight_shards", None)
    gtp = (
        dense_shards // tp
        if isinstance(dense_shards, int)
        else positive_int("gtp_weight_remat_size", 1)
    )
    egtp = (
        expert_shards // etp
        if isinstance(expert_shards, int)
        else positive_int("expert_gtp_weight_remat_size", 1)
    )
    return tp, gtp, etp, egtp


def _validate_checkpoint_weight_topology(
    *, saved: tuple[int, int, int, int], requested: tuple[int, int, int, int]
) -> None:
    """Reject native resharding when either side uses the GTP SwiGLU layout."""
    dense_changed = (saved[1] > 1 or requested[1] > 1) and saved[:2] != requested[:2]
    expert_changed = (saved[3] > 1 or requested[3] > 1) and saved[2:] != requested[2:]
    if not dense_changed and not expert_changed:
        return
    raise ValueError(
        "Resharding a GTP checkpoint is not supported: Megatron-Core's SwiGLU checkpoint "
        "layout can reorder gate/up rows when the weight-sharding topology changes. "
        "Preserve the saved dense/expert TP and weight shard counts (using mp_overrides "
        "with load_megatron_model), export HF weights, then import those weights into "
        "the desired topology."
    )


def _get_dataloader_process_group(
    pg_collection: ProcessGroupCollection,
) -> torch.distributed.ProcessGroup:
    """Return DP x dense GTP, excluding CP ranks that repeat the same samples."""
    remat_group = getattr(pg_collection, "gtp_remat", None)
    if remat_group is not None and remat_group.size() > 1:
        return parallel_state.get_data_parallel_group(with_gtp_remat=True)
    return pg_collection.dp


def is_gtp_remat_active(model_config: Any) -> bool:
    """Return whether dense or expert GTP weight rematerialization is enabled."""
    transformer_config = get_transformer_config(model_config)
    dense_size = getattr(transformer_config, "gtp_weight_remat_size", 1)
    expert_size = getattr(transformer_config, "expert_gtp_weight_remat_size", 1)
    return any(isinstance(size, int) and size > 1 for size in (dense_size, expert_size))


def configure_gtp_remat(
    model_config: Any,
    *,
    reduce_scatter_with_fp32_accumulation: bool = False,
    nccl_ub: bool = False,
    pg_collection: ProcessGroupCollection | None = None,
) -> None:
    """Configure process-global GTP state before constructing model modules.

    Args:
        model_config: Model provider or builder config containing the GTP sizes.
        reduce_scatter_with_fp32_accumulation: Accumulate GTP reduce-scatter
            results locally in FP32.
        nccl_ub: Register an NCCL symmetric-memory pool for the dense GTP group.
        pg_collection: Initialized process groups, required when nccl_ub is enabled.

    Raises:
        RuntimeError: GTP is active but the required Transformer Engine support is missing.
        ValueError: GTP is active and nccl_ub is enabled without process groups.
    """
    if not is_gtp_remat_active(model_config):
        return

    transformer_config = get_transformer_config(model_config)
    try:
        from megatron.core.tensor_parallel import gtp_api
    except ImportError as error:
        raise RuntimeError("GTP requires TransformerEngine >= 2.19.") from error

    if not gtp_api.HAVE_GTP:
        raise RuntimeError("GTP requires TransformerEngine >= 2.19.")

    gtp_api.configure_gtp_remat_from_recipe(
        fp4=transformer_config.fp4 is not None,
        fp8_recipe=transformer_config.fp8_recipe,
        fp8=transformer_config.fp8 is not None,
        calculate_per_token_loss=transformer_config.calculate_per_token_loss,
        reduce_scatter_with_fp32_accumulation=reduce_scatter_with_fp32_accumulation,
    )
    if nccl_ub:
        if pg_collection is None:
            raise ValueError("gtp_remat_nccl_ub requires an initialized process-group collection.")
        from megatron.core.process_groups_config import resolve_gtp_remat_group

        gtp_api.register_gtp_symm_pool(resolve_gtp_remat_group(pg_collection, is_expert=False))


def classify_gtp_remat_chains(model: list[torch.nn.Module], model_config: Any) -> None:
    """Classify all model chunks after distributed wrapping and before first forward."""
    if not is_gtp_remat_active(model_config):
        return

    transformer_config = get_transformer_config(model_config)
    try:
        from megatron.core.tensor_parallel import gtp_api
    except ImportError as error:
        raise RuntimeError("GTP requires TransformerEngine >= 2.19.") from error

    gtp_api.classify_gtp_remat_chains(
        model,
        cuda_graph_modules=transformer_config.cuda_graph_modules,
        moe_shared_expert_overlap=transformer_config.moe_shared_expert_overlap,
        cuda_graph_impl=transformer_config.cuda_graph_impl,
    )


def get_data_distribution_group(
    pg_collection: ProcessGroupCollection, model_config: Any, *, with_context_parallel: bool = False
) -> torch.distributed.ProcessGroup:
    """Return the group spanning every rank that consumes distinct input data."""
    if not is_gtp_remat_active(model_config):
        return pg_collection.dp_cp if with_context_parallel else pg_collection.dp
    if with_context_parallel:
        return pg_collection.dp_cp_gtp_remat
    return parallel_state.get_data_parallel_group(with_gtp_remat=True)
