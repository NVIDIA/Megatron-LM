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

"""
Checkpoint save/load helpers for Megatron FSDP v2.

Provides:

- ``MegatronFSDPStateful`` — ``Stateful`` wrapper that integrates with
  PyTorch DCP, handles uneven DTensor chunk metadata via ``get_state_dict``,
  and applies MCore post-processing.
- ``save_checkpoint`` / ``load_checkpoint`` — one-line DCP save/load.
- Post-processing functions for Megatron FSDP v2 state dicts:
  ``handle_swiglu_in_state_dict_v2``, ``handle_gdn_in_state_dict_v2``,
  ``handle_experts_in_state_dict``, ``handle_fp8_extra_state_case``.
"""

import logging
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import set_state_dict as _set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

import megatron.core.parallel_state as mpu
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import copy_chunk_metadata
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    get_state_dict as _get_state_dict,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    make_uneven_dtensor,
    preprocess_state_dict_for_uneven_dtensor,
    split_dtensor,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import (
    get_mcore_tensor_parallel_partition_dim,
    is_mcore_tensor_model_parallel,
)
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import get_attr_wrapped_model

logger = logging.getLogger(__name__)

__all__ = [
    "MegatronFSDPStateful",
    "save_checkpoint",
    "load_checkpoint",
    "add_module_prefix",
    "strip_module_prefix",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "_build_dtensor_optim_sd",
    "handle_fp8_extra_state_case",
    "handle_experts_in_state_dict",
    "handle_swiglu_in_state_dict_v2",
    "handle_gdn_in_state_dict_v2",
]


# ------------------------------------------------------------------
# Stateful wrapper
# ------------------------------------------------------------------


class MegatronFSDPStateful(Stateful):
    """Stateful wrapper for Megatron FSDP v2 model + optimizer checkpointing.

    Implements DCP's ``Stateful`` protocol.  ``state_dict()`` uses
    ``get_state_dict`` from ``uneven_dtensor`` (which preprocesses DTensors
    for uneven sharding and produces FQN-keyed optimizer states), then applies
    MCore post-processing (SwiGLU, GDN, FP8, expert remapping — see individual
    ``handle_*_v2`` functions).  ``load_state_dict()`` uses PyTorch's
    ``set_state_dict``.

    Parameters:
        model: FSDP v2 wrapped model.
        optimizer: Optional optimizer to checkpoint alongside the model.
        args: Optional MCore args namespace for post-processing config
            (``swiglu``, ``num_experts``, ``gdn``).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        args: Optional[object] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def state_dict(self):
        if self.optimizer is not None:
            model_sd, optim_sd = _get_state_dict(self.model, self.optimizer)
        else:
            model_sd = self.model.state_dict()
            preprocess_state_dict_for_uneven_dtensor(model_sd)
            optim_sd = None

        state_dict = {"model": model_sd}
        if optim_sd is not None:
            state_dict["optimizer"] = optim_sd

        if self.args is not None:
            _apply_mcore_postprocess(state_dict, self.args, self.model)

        return state_dict

    def load_state_dict(self, state_dict):
        _set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict.get("model"),
            optim_state_dict=state_dict.get("optimizer"),
        )


# ------------------------------------------------------------------
# FP8 artifact cleanup (format-agnostic)
# ------------------------------------------------------------------


def handle_fp8_extra_state_case(model_state_dict: dict) -> None:
    """Remove ``._extra_state`` keys (artifact of FP8 training)."""
    keys_to_remove = [k for k in model_state_dict if k.endswith("._extra_state")]
    for k in keys_to_remove:
        del model_state_dict[k]


# ------------------------------------------------------------------
# Expert key remapping (format-agnostic)
# ------------------------------------------------------------------


def handle_experts_in_state_dict(model_state_dict: dict, num_experts: int) -> dict:
    """Rename expert keys to reflect expert-parallel sharding.

    Standalone implementation (no dependency on ``fsdp_dtensor_checkpoint``).
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    local_expert_start = ep_rank * (num_experts // ep_size) if num_experts else 0
    local_expert_end = local_expert_start + (num_experts // ep_size) if num_experts else 0

    def _should_keep(expert_index):
        if expert_index is None:
            return True
        return local_expert_start <= expert_index < local_expert_end

    def _replace(key, expert_index, sd):
        new_idx = expert_index + local_expert_start
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            if key.endswith('_w') or key.endswith('_v'):
                new_key = key.replace(
                    f'weight{expert_index}{key[-2:]}', f'weight{new_idx}{key[-2:]}'
                )
            else:
                new_key = key.replace(f'weight{expert_index}', f'weight{new_idx}')
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            new_key = key.replace(f'local_experts.{expert_index}.', f'local_experts.{new_idx}.')
        else:
            raise ValueError(f"Unexpected expert key format: {key}")
        sd[new_key] = sd.pop(key)

    model_state_dict = model_state_dict.copy()
    for key in list(model_state_dict.keys()):
        ei = _get_expert_index_from_key(key)
        if not _should_keep(ei):
            _replace(key, ei, model_state_dict)

    return model_state_dict


def _get_expert_index_from_key(key: str):
    """Extract expert index from key (``weight{N}`` or ``local_experts.N.``)."""
    m = re.search(r'(?:weight(\d+)$)|(?:local_experts\.(\d+)\.)', key)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def _expert_param_local_key(key: str, num_experts: int | None = None) -> str:
    """Adjust expert index in a key from global to local."""
    if num_experts is None:
        return key
    ei = _get_expert_index_from_key(key)
    if ei is None:
        return key
    ep_rank = mpu.get_expert_model_parallel_rank()
    ep_size = mpu.get_expert_model_parallel_world_size()
    local_start = ep_rank * (num_experts // ep_size)
    new_idx = ei - local_start
    if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
        return key.replace(f'weight{ei}', f'weight{new_idx}')
    elif 'mlp.experts.local_experts' in key:
        return key.replace(f'local_experts.{ei}.', f'local_experts.{new_idx}.')
    raise ValueError(f"Unexpected expert key format: {key}")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _strip_wrappers(path: str) -> str:
    """Strip DDP/FSDP wrapper prefixes (``module.``, ``model.``) from a path."""
    parts = path.split('.')
    while parts and parts[0] in ('module', 'model'):
        parts = parts[1:]
    return '.'.join(parts)


def _intersection(s1: slice, s2: slice) -> slice:
    start = max(s1.start, s2.start)
    stop = min(s1.stop, s2.stop)
    return slice(0, 0) if start >= stop else slice(start, stop)


def _offset_slice(s: slice, offset: int) -> slice:
    return slice(s.start + offset, s.stop + offset)


def _get_tp_world_size(dist_param: DTensor) -> int:
    """Get tensor-parallel world size from propagated TP attributes."""
    if is_mcore_tensor_model_parallel(dist_param):
        tp_dim = get_mcore_tensor_parallel_partition_dim(dist_param)
        if tp_dim is not None:
            global_shape = dist_param.size()
            local_shape = dist_param._local_tensor.shape
            return global_shape[tp_dim] // local_shape[tp_dim]
    return 1


def _get_dist_param(model: nn.Module, key: str) -> nn.Parameter:
    """Get a model parameter by state dict key, handling ``module.`` prefix.

    Tries multiple key variants to handle both wrapped (``FullyShardedDataParallel``
    with ``self.module``) and unwrapped (raw ``FSDPModule``) models:
      1. Key as-is.
      2. Key with ``module.`` prefix added.
      3. Key with ``module.`` prefix stripped (if present).
    """
    try:
        return model.get_parameter(key)
    except AttributeError:
        pass
    try:
        return model.get_parameter(f"module.{key}")
    except AttributeError:
        pass
    if key.startswith("module."):
        try:
            return model.get_parameter(key[len("module.") :])
        except AttributeError:
            pass
    raise AttributeError(
        f"Parameter '{key}' not found in model "
        f"(tried as-is, with 'module.' prefix, and without 'module.' prefix)"
    )


def _detect_glu_layers(model: nn.Module) -> dict:
    """Return ``{layer_path: uses_gated_linear_unit}`` for TransformerLayers."""
    _layer_glu = {}
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer):
            _layer_glu[_strip_wrappers(name)] = getattr(module.config, 'gated_linear_unit', False)
    return _layer_glu


def _key_in_glu_layer(key: str, layer_glu: dict) -> bool:
    """Return True if *key* belongs to a TransformerLayer with GLU enabled."""
    norm_key = _strip_wrappers(key)
    best_glu, best_len = None, -1
    for layer_path, uses_glu in layer_glu.items():
        if norm_key.startswith(layer_path + '.') and len(layer_path) > best_len:
            best_glu, best_len = uses_glu, len(layer_path)
    if best_glu is None:
        return True  # no TransformerLayer found — assume GLU for backward compat
    return best_glu


# ------------------------------------------------------------------
# SwiGLU / gate weight splitting (Megatron FSDP v2)
# ------------------------------------------------------------------

_SWIGLU_KEY_PATTERNS = [
    r"(.*)\.mlp\.linear_fc1\.weight$",
    r"(.*)\.mlp\.linear_fc1\.bias$",
    r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)$",
    r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)$",
    r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight$",
    r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias$",
    r"(.*)\.mlp\.shared_experts\.linear_fc1\.weight$",
    r"(.*)\.mlp\.shared_experts\.linear_fc1\.bias$",
]


def _is_swiglu_key(key: str) -> bool:
    return any(re.search(pat, key) for pat in _SWIGLU_KEY_PATTERNS)


def _split_swiglu_weight_v2(
    dtensor: DTensor, swiglu_shard_axis: int = 0
) -> Tuple[DTensor, DTensor]:
    """Split a SwiGLU fc1 weight into ``_w`` and ``_v`` DTensors.

    Convenience wrapper around ``_split_dtensor_v2`` with ``[1, 1]`` splits.
    """
    dim = swiglu_shard_axis
    assert (
        dtensor.shape[dim] % 2 == 0
    ), f"Expected SwiGLU fc1 weight size divisible by 2, got {dtensor.shape[dim]}"
    half = dtensor.shape[dim] // 2
    return tuple(split_dtensor(dtensor, [half, half], dim))


def handle_swiglu_in_state_dict_v2(
    model: nn.Module, model_state_dict: dict, optimizer_state_dict: Optional[dict]
) -> Tuple[dict, Optional[dict]]:
    """Split SwiGLU fc1 parameters in model and optimizer state dicts.

    Megatron FSDP v2 version — only processes layers with
    ``gated_linear_unit=True``.  Uses DTensor-native operations throughout.
    """
    # Extract num_experts for expert parameter processing.
    model_config = get_attr_wrapped_model(model, "config", allow_none=True)
    num_experts = (
        getattr(model_config, 'num_moe_experts', None) if model_config is not None else None
    )

    layer_glu = _detect_glu_layers(model)

    # ---- Model state dict ----
    model_state_dict = model_state_dict.copy()
    split_count = 0
    skip_count = 0
    for key in list(model_state_dict.keys()):
        if not _is_swiglu_key(key):
            continue
        if not _key_in_glu_layer(key, layer_glu):
            skip_count += 1
            continue

        dist_param = _get_dist_param(model, key)
        assert isinstance(
            dist_param, DTensor
        ), f"Expected DTensor for {key}, got {type(dist_param).__name__}"
        weight_w, weight_v = _split_swiglu_weight_v2(model_state_dict[key])
        model_state_dict[f"{key}_w"] = weight_w
        model_state_dict[f"{key}_v"] = weight_v
        del model_state_dict[key]
        split_count += 1

    if skip_count > 0:
        logger.info(
            f"[SwiGLU v2] Split {split_count} fc1 keys; "
            f"skipped {skip_count} keys in non-GLU layers."
        )

    # ---- Optimizer state dict ----
    if optimizer_state_dict is not None:
        optimizer_state_dict = optimizer_state_dict.copy()
        if len(optimizer_state_dict.get("state", {})) != 0:
            opt_state = optimizer_state_dict["state"]
            new_opt_state = {}
            for key in list(opt_state.keys()):
                if not _is_swiglu_key(key) or not _key_in_glu_layer(key, layer_glu):
                    new_opt_state[key] = opt_state[key]
                    continue
                new_opt_state[f"{key}_w"] = opt_state[key].copy()
                new_opt_state[f"{key}_v"] = opt_state[key].copy()

                for subkey in ["exp_avg", "exp_avg_sq"]:
                    param_key = key
                    if param_key.startswith("module."):
                        param_key = param_key[len("module.") :]
                    dist_param = _get_dist_param(model, param_key)
                    assert isinstance(opt_state[key][subkey], DTensor), (
                        f"Expected optimizer state for {key} to be a DTensor, got "
                        f"{type(opt_state[key][subkey]).__name__}"
                    )
                    weight_w_t, weight_v_t = _split_swiglu_weight_v2(opt_state[key][subkey])
                    new_opt_state[f"{key}_w"][subkey] = weight_w_t
                    new_opt_state[f"{key}_v"][subkey] = weight_v_t
            optimizer_state_dict["state"] = new_opt_state

    return model_state_dict, optimizer_state_dict


# ------------------------------------------------------------------
# GDN fused-projection splitting (Megatron FSDP v2)
# ------------------------------------------------------------------

GDN_CONV1D_NAMES = ["query", "key", "value"]

_GDN_KEY_PATTERNS = [
    r"(.*)\.self_attention\.linear_proj\.weight$",
    r"(.*)\.self_attention\.linear_qkv\.weight$",
    r"(.*)\.self_attention\.linear_qkv\.bias$",
]


def _match_gdn_key(key: str, dtensor: DTensor):
    for pat in _GDN_KEY_PATTERNS:
        m = re.match(pat, key)
        if m:
            dim = 0
            size = dtensor[dim]
            assert (
                size % 3 == 0
            ), f"Expected GDN projection size divisible by 3, got {size} for key {key}"
            qkv_size = size // 3
            return [qkv_size, qkv_size, qkv_size], GDN_CONV1D_NAMES, dim
    return None


def handle_gdn_in_state_dict_v2(
    model: nn.Module, model_state_dict: dict, optimizer_state_dict: Optional[dict]
) -> Tuple[dict, Optional[dict]]:
    """Split fused GDN projection parameters into per-component DTensors.

    Megatron FSDP v2 version.  Uses DTensor-native operations throughout.
    """
    # ---- Model state dict ----
    model_state_dict = model_state_dict.copy()
    split_count = 0
    for key, value in model_state_dict.items():
        match = _match_gdn_key(key, value)
        if match is None:
            continue
        sizes, names, dim = match

        dist_param = _get_dist_param(model, key)
        assert isinstance(
            dist_param, DTensor
        ), f"Expected DTensor for {key}, got {type(dist_param).__name__}"
        sub_tensors = split_dtensor(model_state_dict[key], sizes, dim)
        for sub_name, tensor in zip(names, sub_tensors):
            model_state_dict[f"{key}.{sub_name}"] = tensor
        del model_state_dict[key]
        split_count += 1

    if split_count > 0:
        logger.info(f"[GDN v2] Split {split_count} fused keys.")

    # ---- Optimizer state dict ----
    if optimizer_state_dict is not None:
        optimizer_state_dict = optimizer_state_dict.copy()
        if len(optimizer_state_dict.get("state", {})) != 0:
            opt_state = optimizer_state_dict["state"]
            new_opt_state = {}
            for key in list(opt_state.keys()):
                match = _match_gdn_key(key)
                if match is None:
                    new_opt_state[key] = opt_state[key]
                    continue
                sizes, names, dim = match
                for sub_name in names:
                    new_opt_state[f"{key}.{sub_name}"] = opt_state[key].copy()
                for subkey in ["exp_avg", "exp_avg_sq"]:
                    param_key = key
                    if param_key.startswith("module."):
                        param_key = param_key[len("module.") :]
                    dist_param = _get_dist_param(model, param_key)
                    assert isinstance(dist_param, DTensor)
                    sub_tensors = split_dtensor(opt_state[key][subkey], sizes, dim)
                    for sub_name, tensor in zip(names, sub_tensors):
                        new_opt_state[f"{key}.{sub_name}"][subkey] = tensor
            optimizer_state_dict["state"] = new_opt_state

    return model_state_dict, optimizer_state_dict


# ------------------------------------------------------------------
# Unified post-processing
# ------------------------------------------------------------------


def _find_param_in_map(key: str, param_map: dict) -> Optional[DTensor]:
    """Look up *key* in *param_map*, trying ``module.`` prefix variants."""
    param = param_map.get(key)
    if param is not None:
        return param
    stripped = key[len(_MODULE_PREFIX) :] if key.startswith(_MODULE_PREFIX) else key
    param = param_map.get(stripped)
    if param is not None:
        return param
    return param_map.get(f"{_MODULE_PREFIX}{key}")


def _propagate_chunk_metadata_to_state_dict(model: nn.Module, state_dict: dict) -> None:
    """Copy chunk metadata from model parameters to state dict DTensors.

    ``model.state_dict()`` returns fresh DTensor objects that lack
    ``__create_chunk_list__`` / ``__create_write_items__``.  The model
    parameters (from ``named_parameters()``) already have them.  This
    function copies the closures locally — no collectives.
    """
    param_map = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor) and hasattr(param._local_tensor, "__create_chunk_list__"):
            param_map[name] = param

    for key, value in state_dict.items():
        if not isinstance(value, DTensor):
            continue
        param = _find_param_in_map(key, param_map)
        if param is not None:
            copy_chunk_metadata(param, value)


def _apply_mcore_postprocess(raw_state_dict, args, model):
    """Apply MCore-specific state dict post-processing.

    Copies *raw_state_dict*, wraps optimizer states as DTensors, then
    applies FP8 cleanup, SwiGLU/GDN split, and expert key remapping.
    The original *raw_state_dict* is not mutated.
    """
    state_dict = raw_state_dict.copy()
    _propagate_chunk_metadata_to_state_dict(model, state_dict["model"])

    if "optimizer" in state_dict:
        state_dict["optimizer"] = _build_dtensor_optim_sd(state_dict["optimizer"], model)
    handle_fp8_extra_state_case(state_dict["model"])

    if getattr(args, "swiglu", False):
        if "optimizer" in state_dict:
            model_sd, optim_sd = handle_swiglu_in_state_dict_v2(
                model, state_dict["model"], state_dict["optimizer"]
            )
            state_dict["model"] = model_sd
            state_dict["optimizer"] = optim_sd
        else:
            model_sd, _ = handle_swiglu_in_state_dict_v2(model, state_dict["model"], None)
            state_dict["model"] = model_sd

    if getattr(args, "gdn", False):
        if "optimizer" in state_dict:
            model_sd, optim_sd = handle_gdn_in_state_dict_v2(
                model, state_dict["model"], state_dict["optimizer"]
            )
            state_dict["model"] = model_sd
            state_dict["optimizer"] = optim_sd
        else:
            model_sd, _ = handle_gdn_in_state_dict_v2(model, state_dict["model"], None)
            state_dict["model"] = model_sd

    num_experts = getattr(args, "num_experts", None)
    if num_experts:
        state_dict["model"] = handle_experts_in_state_dict(state_dict["model"], num_experts)
        if "optimizer" in state_dict:
            optim_sd = state_dict["optimizer"]
            optim_sd["state"] = handle_experts_in_state_dict(optim_sd["state"], num_experts)
            optim_sd["param_to_group_meta"] = handle_experts_in_state_dict(
                optim_sd["param_to_group_meta"], num_experts
            )

    return state_dict


# ------------------------------------------------------------------
# Model state dict prefix alignment
# ------------------------------------------------------------------

_MODULE_PREFIX = "module."


def add_module_prefix(state_dict: dict) -> dict:
    """Add ``module.`` prefix to all keys in a state dict.

    Megatron FSDP v2 applies ``fully_shard`` directly to the model without
    a ``MegatronFSDP`` wrapper, so ``model.state_dict()`` produces keys
    without the ``module.`` prefix (e.g., ``embedding.word_embeddings.weight``).
    Megatron's checkpoint format expects the prefix to be present. This
    function adds it for alignment.
    """
    return {f"{_MODULE_PREFIX}{k}": v for k, v in state_dict.items()}


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove ``module.`` prefix from all keys in a state dict.

    Inverse of :func:`add_module_prefix`. Used when loading a Megatron
    checkpoint (which stores keys with ``module.`` prefix) back into a
    Megatron FSDP v2 model that does not use the prefix.
    """
    return {
        k[len(_MODULE_PREFIX) :] if k.startswith(_MODULE_PREFIX) else k: v
        for k, v in state_dict.items()
    }


def _model_has_module_prefix(model: nn.Module) -> bool:
    """Detect whether the model's state dict keys already carry ``module.`` prefix."""
    for name, _ in model.named_parameters():
        return name.startswith(_MODULE_PREFIX)
    return False


# ------------------------------------------------------------------
# Model state dict (Megatron FSDP v2)
# ------------------------------------------------------------------


def get_model_state_dict(model: nn.Module) -> dict:
    """Get model state dict with ``module.`` prefix for Megatron FSDP v2.

    If the model's parameters already carry the ``module.`` prefix (e.g.,
    when the model is accessed through the ``FullyShardedDataParallel``
    adapter), the state dict is returned as-is. Otherwise the prefix is
    added to align with Megatron's checkpoint key convention.

    Returns:
        Model state dict with ``module.``-prefixed keys containing DTensors.
    """
    model_sd = model.state_dict()
    if not _model_has_module_prefix(model):
        model_sd = add_module_prefix(model_sd)
    return model_sd


# ------------------------------------------------------------------
# Optimizer state dict — Path A (sharded_param_state_fsdp_dtensor)
# ------------------------------------------------------------------


def get_optimizer_state_dict(optimizer, is_loading: bool = False) -> Optional[dict]:
    """Get optimizer state dict following Path A (``sharded_param_state_fsdp_dtensor``).

    Delegates to ``optimizer.sharded_state_dict()`` with the ``fsdp_dtensor``
    sharding type, which internally calls ``sharded_param_state_fsdp_dtensor``.

    For loading, this also initializes optimizer states with dummy values so
    that the returned dict has correctly-sized tensors for DCP in-place load.

    Args:
        optimizer: A ``DistributedOptimizer`` (or compatible) instance.
        is_loading: If True, pre-allocates optimizer state tensors.

    Returns:
        Optimizer state dict with structure::

            {
                "state": {<param_name>: {"exp_avg": DTensor, "exp_avg_sq": DTensor, ...}},
                "param_to_group_meta": {<param_name>: {"lr": ..., "weight_decay": ...}},
            }

        Returns None if ``optimizer`` is None or is a stub optimizer.
    """
    if optimizer is None:
        return None
    if getattr(optimizer, "is_stub_optimizer", False):
        return None
    return optimizer.sharded_state_dict(
        model_sharded_state_dict={},
        is_loading=is_loading,
        metadata={"distrib_optim_sharding_type": "fsdp_dtensor"},
    )


# ------------------------------------------------------------------
# Optimizer state DTensor wrapping
# ------------------------------------------------------------------


def _build_dtensor_optim_sd(raw_opt_state_dict: dict, model: nn.Module) -> dict:
    """Return a copy of *raw_opt_state_dict* with optimizer state tensors converted to DTensors.

    FusedAdam stores ``exp_avg`` / ``exp_avg_sq`` as plain tensors matching
    the parameter's local DTensor shard.  DCP requires all sharded data to
    be DTensors.  This returns a new dict where those plain tensors are
    converted to uneven DTensors using the model parameter's mesh/placements.
    *raw_opt_state_dict* is **not** mutated.
    """
    opt_state_dict = raw_opt_state_dict.copy()
    if "state" not in opt_state_dict:
        return opt_state_dict

    param_map = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            param_map[name] = param

    if not param_map:
        return opt_state_dict

    old_state = opt_state_dict["state"]
    new_state = {}
    for key, param_states in old_state.items():
        if not isinstance(param_states, dict):
            new_state[key] = param_states
            continue
        new_param_states = dict(param_states)
        dist_param = _find_param_in_map(key, param_map)
        if dist_param is not None:
            for state_key, state_val in new_param_states.items():
                if not isinstance(state_val, torch.Tensor) or isinstance(state_val, DTensor):
                    continue
                if state_val.shape == dist_param._local_tensor.shape:
                    new_param_states[state_key] = make_uneven_dtensor(
                        state_val,
                        shape=dist_param.size(),
                        dp_mesh=dist_param.device_mesh,
                        placements=dist_param.placements,
                    )
                    copy_chunk_metadata(dist_param, new_param_states[state_key])
        new_state[key] = new_param_states

    opt_state_dict["state"] = new_state
    return opt_state_dict
