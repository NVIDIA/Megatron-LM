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
from megatron.core import dist_checkpointing
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import copy_chunk_metadata
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    get_state_dict as _get_state_dict,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    make_uneven_dtensor,
    preprocess_state_dict_for_uneven_dtensor,
    split_dtensor,
)
from megatron.core.transformer.transformer_layer import TransformerLayer

logger = logging.getLogger(__name__)

__all__ = [
    "MegatronFSDPStateful",
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

_MODULE_PREFIX = "module."


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


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _strip_wrappers(path: str) -> str:
    """Strip DDP/FSDP wrapper prefixes (``module.``, ``model.``) from a path."""
    parts = path.split('.')
    while parts and parts[0] in ('module', 'model'):
        parts = parts[1:]
    return '.'.join(parts)


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
# Fused-parameter splitting — shared skeleton (Megatron FSDP v2)
# ------------------------------------------------------------------
#
# SwiGLU and GDN both require splitting fused parameter tensors into
# per-component pieces (SwiGLU: fc1 → _w/_v; GDN: linear_qkv → query/key/value).
# This function handles both.
#
# The caller provides a *detector* that returns ``(sizes, names, dim)`` for
# parameters that need splitting, or ``None`` otherwise, and a *key formatter*
# to build sub-parameter names.

_DEFAULT_STATE_KEYS = ("exp_avg", "exp_avg_sq")


def _split_fused_params_v2(
    model: nn.Module,
    model_state_dict: dict,
    optimizer_state_dict: Optional[dict],
    detector,
    key_fmt,
    tag: str,
    state_keys: Tuple[str, ...] = _DEFAULT_STATE_KEYS,
) -> Tuple[dict, Optional[dict]]:
    """Split fused DTensor parameters in model and optimizer state dicts.

    Args:
        model: The model (for ``_get_dist_param``).
        model_state_dict: Model state dict (a copy is returned).
        optimizer_state_dict: Optional optimizer state dict (a copy is returned).
        detector(key, dtensor, model) -> ``(sizes, names, dim)`` or ``None``.
        key_fmt(key, sub_name) -> new_key.
        tag: Log tag (e.g. ``"SwiGLU v2"``).
        state_keys: Optimizer state keys to split (default: ``exp_avg``, ``exp_avg_sq``).

    Returns:
        ``(model_state_dict, optimizer_state_dict)`` — modified copies.
    """
    # ---- Model state dict ----
    model_state_dict = model_state_dict.copy()
    split_count = 0
    for key in list(model_state_dict.keys()):
        value = model_state_dict[key]
        match = detector(key, value, model)
        if match is None:
            continue
        sizes, names, dim = match

        dist_param = _get_dist_param(model, key)
        assert isinstance(
            dist_param, DTensor
        ), f"Expected DTensor for {key}, got {type(dist_param).__name__}"
        sub_tensors = split_dtensor(value, sizes, dim)
        for sub_name, tensor in zip(names, sub_tensors):
            model_state_dict[key_fmt(key, sub_name)] = tensor
        del model_state_dict[key]
        split_count += 1

    if split_count > 0:
        logger.info(f"[{tag}] Split {split_count} fused keys.")

    # ---- Optimizer state dict ----
    if optimizer_state_dict is not None:
        optimizer_state_dict = optimizer_state_dict.copy()
        if len(optimizer_state_dict.get("state", {})) != 0:
            opt_state = optimizer_state_dict["state"]
            new_opt_state = {}
            for key in list(opt_state.keys()):
                param_key = key
                if param_key.startswith("module."):
                    param_key = param_key[len("module.") :]
                dist_param = _get_dist_param(model, param_key)

                match = detector(key, dist_param, model)
                if match is None:
                    new_opt_state[key] = opt_state[key]
                    continue
                sizes, names, dim = match

                for sub_name in names:
                    new_opt_state[key_fmt(key, sub_name)] = opt_state[key].copy()
                for sk in state_keys:
                    sub_tensors = split_dtensor(opt_state[key][sk], sizes, dim)
                    for sub_name, tensor in zip(names, sub_tensors):
                        new_opt_state[key_fmt(key, sub_name)][sk] = tensor
            optimizer_state_dict["state"] = new_opt_state

    return model_state_dict, optimizer_state_dict


# ------------------------------------------------------------------
# SwiGLU key patterns and helpers
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


def _swiglu_detector(key, dtensor, model, layer_glu):
    if not _is_swiglu_key(key):
        return None
    if not _key_in_glu_layer(key, layer_glu):
        return None
    dim = 0
    assert (
        dtensor.shape[dim] % 2 == 0
    ), f"Expected SwiGLU fc1 weight size divisible by 2, got {dtensor.shape[dim]}"
    half = dtensor.shape[dim] // 2
    return ([half, half], ["w", "v"], dim)


def handle_swiglu_in_state_dict_v2(
    model: nn.Module, model_state_dict: dict, optimizer_state_dict: Optional[dict]
) -> Tuple[dict, Optional[dict]]:
    """Split SwiGLU fc1 parameters in model and optimizer state dicts.

    Megatron FSDP v2 version — only processes layers with
    ``gated_linear_unit=True``.  Delegates to :func:`_split_fused_params_v2`.
    """
    layer_glu = _detect_glu_layers(model)

    def detector(key, dtensor, _model):
        return _swiglu_detector(key, dtensor, _model, layer_glu)

    return _split_fused_params_v2(
        model,
        model_state_dict,
        optimizer_state_dict,
        detector,
        lambda k, s: f"{k}_{s}",
        "SwiGLU v2",
    )


# ------------------------------------------------------------------
# GDN key patterns and helpers
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

    Megatron FSDP v2 version.  Delegates to :func:`_split_fused_params_v2`.
    """
    return _split_fused_params_v2(
        model,
        model_state_dict,
        optimizer_state_dict,
        _match_gdn_key,
        lambda k, s: f"{k}.{s}",
        "GDN v2",
    )


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
        opt_sd = state_dict.get("optimizer")
        model_sd, new_opt = handle_swiglu_in_state_dict_v2(model, state_dict["model"], opt_sd)
        state_dict["model"] = model_sd
        if new_opt is not None:
            state_dict["optimizer"] = new_opt

    if getattr(args, "gdn", False):
        opt_sd = state_dict.get("optimizer")
        model_sd, new_opt = handle_gdn_in_state_dict_v2(model, state_dict["model"], opt_sd)
        state_dict["model"] = model_sd
        if new_opt is not None:
            state_dict["optimizer"] = new_opt

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


def _maybe_wrap_as_uneven_dtensor(tensor, dist_param: DTensor):
    """Wrap a plain tensor as an uneven DTensor if it matches the param's local shard.

    FusedAdam stores ``exp_avg`` / ``exp_avg_sq`` as plain tensors; DCP needs
    DTensors.  Returns the original tensor unchanged if it is already a DTensor
    or its shape does not match the parameter's local shard.
    """
    if isinstance(tensor, DTensor) or not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.shape != dist_param._local_tensor.shape:
        return tensor
    dt = make_uneven_dtensor(
        tensor,
        shape=dist_param.size(),
        dp_mesh=dist_param.device_mesh,
        placements=dist_param.placements,
    )
    copy_chunk_metadata(dist_param, dt)
    return dt


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
                new_param_states[state_key] = _maybe_wrap_as_uneven_dtensor(state_val, dist_param)
        new_state[key] = new_param_states

    opt_state_dict["state"] = new_state
    return opt_state_dict


def _preprocess_and_verify_v2_state_dict(v2_state_dict):
    """Preprocess and verify the Megatron FSDP v2 state dict before DCP loading.

    DCP requires DTensors for distributed load, but ``optimizer.load_state_dict``
    only accepts plain tensors.  This function builds a *shadow* optimizer state
    dict (``v2_optim_state``) where plain tensors are wrapped as uneven DTensors
    sharing storage with the originals.  DCP loads into this shadow dict, and the
    data lands in the original plain tensors via shared storage.  The original
    ``v2_state_dict`` is **not** mutated.

    Also verifies that every model and shadow optimizer DTensor has
    ``__create_chunk_list__`` and ``__create_write_items__`` metadata.

    Returns:
        v2_by_canonical: ``{canonical_name: model_DTensor}`` mapping.
        v2_optim_state: ``{canonical_name: {state_key: DTensor}}`` shadow dict.
    """
    v2_model = v2_state_dict["model"]
    v2_optim_state_raw = v2_state_dict.get("optimizer", {}).get("state", {})

    # ---- Strip ``module.`` prefix from optimizer state keys ----
    # Copy inner dicts so DTensor wrapping below does not mutate
    # the original state dict (optimizer.load_state_dict needs plain tensors).
    v2_optim_state = {}
    for k, v in v2_optim_state_raw.items():
        canonical = k
        while canonical.startswith("module."):
            canonical = canonical[len("module.") :]
        v2_optim_state[canonical] = dict(v)

    # ---- Build canonical model param map + param lookup ----
    v2_by_canonical = {}
    param_map = {}
    for v2_key, v2_val in v2_model.items():
        canonical = v2_key
        while canonical.startswith("module."):
            canonical = canonical[len("module.") :]
        v2_by_canonical[canonical] = v2_val
        if isinstance(v2_val, DTensor):
            param_map[canonical] = v2_val

    # ---- Wrap plain tensors as uneven DTensors ----
    for param_name, states in dict(v2_optim_state).items():
        dist_param = param_map.get(param_name)
        if dist_param is None:
            continue
        for sk, sv in dict(states).items():
            v2_optim_state[param_name][sk] = _maybe_wrap_as_uneven_dtensor(sv, dist_param)

    # ---- Verify model DTensors have chunk metadata ----
    for canonical, v2_val in v2_by_canonical.items():
        if hasattr(v2_val, "_local_tensor"):
            lt = v2_val._local_tensor
            assert hasattr(lt, "__create_chunk_list__"), (
                f"Expected v2 model DTensor '{canonical}' to have "
                f"__create_chunk_list__ metadata"
            )
            assert hasattr(lt, "__create_write_items__"), (
                f"Expected v2 model DTensor '{canonical}' to have "
                f"__create_write_items__ metadata"
            )

    # ---- Verify optimizer state DTensors have chunk metadata ----
    for param_name, states in v2_optim_state.items():
        for sk, sv in states.items():
            if hasattr(sv, "_local_tensor"):
                lt = sv._local_tensor
                assert hasattr(lt, "__create_chunk_list__"), (
                    f"Expected optimizer state DTensor '{param_name}.{sk}' "
                    f"to have __create_chunk_list__ metadata"
                )
                assert hasattr(lt, "__create_write_items__"), (
                    f"Expected optimizer state DTensor '{param_name}.{sk}' "
                    f"to have __create_write_items__ metadata"
                )

    return v2_by_canonical, v2_optim_state


# ------------------------------------------------------------------
# Torch_dist → FSDP v2 checkpoint key normalization
# ------------------------------------------------------------------


def normalize_torch_dist_key(key: str) -> str:
    """Normalize a torch_dist checkpoint key to v2 canonical form.

    Maps structural naming differences:
    - ``transformer_layer`` → ``mtp_model_layer``
    """
    if ".transformer_layer." in key:
        key = key.replace(".transformer_layer.", ".mtp_model_layer.")
    return key


def reverse_normalize_torch_dist_key(key: str) -> str:
    """Reverse the v2 canonical key back to torch_dist naming.

    Inverse of :func:`normalize_torch_dist_key`:
    - ``mtp_model_layer`` → ``transformer_layer``
    """
    key = key.replace(".mtp_model_layer", ".transformer_layer")
    return key


# ------------------------------------------------------------------
# Torch_dist → FSDP v2 name mapping
# ------------------------------------------------------------------


def _build_torch_dist_to_v2_map(metadata, v2_by_canonical, v2_optim_state):
    """Build a map from torch_dist checkpoint keys to Megatron FSDP v2 DTensors.

    Iterates through the torch_dist metadata and matches each key to the
    corresponding v2 model or optimizer state entry, categorizing entries
    as regular model weights, high-precision (``param``) copies, or
    optimizer state tensors (``exp_avg``, ``exp_avg_sq``).

    Args:
        metadata: torch_dist checkpoint metadata dict.
        v2_by_canonical: ``{canonical_name: model_DTensor}`` from
            :func:`_preprocess_and_verify_v2_state_dict`.
        v2_optim_state: ``{canonical_name: {state_key: DTensor}}`` from
            :func:`_preprocess_and_verify_v2_state_dict`.

    Returns:
        regular_model: ``{torch_dist_key: v2_DTensor}`` for regular model weights.
        hi_prec_model: ``{torch_dist_param_name: v2_DTensor}`` for high-precision
            model param copies.
        optim_keys: ``{torch_dist_key: v2_state_DTensor}`` for optimizer states.
        optim_matched: ``set`` of canonical param names with matched optimizer states.
    """
    regular_model = {}
    hi_prec_td_keys = set()
    hi_prec_model = {}
    optim_keys = {}
    optim_matched = set()

    for td_key in metadata:
        if td_key.startswith("optimizer.state."):
            rest = td_key[len("optimizer.state.") :]
            parts = rest.split(".", 1)
            if rest.startswith("param."):
                param_name_td = rest[len("param.") :]
                param_canonical = _canonicalize_td_key(param_name_td, strip_model_prefix=False)
                while param_canonical.startswith("module."):
                    param_canonical = param_canonical[len("module.") :]
                if param_canonical in v2_by_canonical:
                    hi_prec_model[param_name_td] = v2_by_canonical[param_canonical]
                    hi_prec_td_keys.add(param_canonical)
            elif len(parts) == 2:
                state_key, param_name_td = parts
                param_canonical = _canonicalize_td_key(param_name_td, strip_model_prefix=False)
                while param_canonical.startswith("module."):
                    param_canonical = param_canonical[len("module.") :]
                if (
                    param_canonical in v2_optim_state
                    and state_key in v2_optim_state[param_canonical]
                ):
                    optim_keys[td_key] = v2_optim_state[param_canonical][state_key]
                    optim_matched.add(param_canonical)
        else:
            canonical = _canonicalize_td_key(td_key)
            while canonical.startswith("module."):
                canonical = canonical[len("module.") :]
            if canonical in v2_by_canonical and canonical not in hi_prec_td_keys:
                load_key = td_key[len("model.") :] if td_key.startswith("model.") else td_key
                regular_model[load_key] = v2_by_canonical[canonical]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "torch_dist → v2 mapping: %d metadata keys → %d regular model + "
            "%d hi-prec params + %d optimizer keys matched (%d unique params)",
            len(metadata),
            len(regular_model),
            len(hi_prec_model),
            len(optim_keys),
            len(optim_matched),
        )
    return regular_model, hi_prec_model, optim_keys, optim_matched


# ------------------------------------------------------------------
# Expert parameter loading (torch_dist flattened → v2 individual)
# ------------------------------------------------------------------


_EXPERT_KEY_RE = re.compile(r'^(.+)\.mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight$')
_SHARD_SUFFIX_RE = re.compile(r'/shard_\d+_\d+$')


def _load_expert_params_from_torch_dist(
    checkpoint_name, v2_state_dict, v2_optim_state, mapped_sd, metadata, optim_matched=None
):
    """Load MoE expert params from torch_dist flattened format into v2 DTensors.

    In torch_dist checkpoints, expert parameters are stored as a single
    flattened tensor (e.g. ``experts.experts.linear_fc1.weight`` with
    shape ``(num_global_experts, H, W)``, EP-sharded).  FSDP v2 stores
    each local expert as its own DTensor
    (``local_experts.0.linear_fc1.weight``, shape ``(H, W)``).

    Handles both model weights and optimizer state tensors
    (``exp_avg``, ``exp_avg_sq``) for expert parameters.  Loads the
    flattened tensor into a temporary buffer and then uses each
    DTensor's chunk metadata (``__create_chunk_list__``) to copy the
    correct slice.
    """
    loaded = mapped_sd.get("model", {})
    v2_model = v2_state_dict["model"]

    # Collect expert entries from model weights and optimizer states.
    # Each entry is (base, local_idx, fc_type, v2_key, v2_dtensor).
    model_experts = []
    optim_experts = []  # (base, local_idx, fc_type, state_key, param_name, v2_dtensor)
    for v2_key, v2_val in v2_model.items():
        m = _EXPERT_KEY_RE.match(v2_key)
        if not m or v2_key in loaded:
            continue
        base = m.group(1)
        idx = int(m.group(2))
        fc_type = m.group(3)
        model_experts.append((base, idx, fc_type, v2_key, v2_val))

    for param_name, states in v2_optim_state.items():
        m = _EXPERT_KEY_RE.match(param_name)
        if not m:
            continue
        base = m.group(1)
        idx = int(m.group(2))
        fc_type = m.group(3)
        for sk in ("exp_avg", "exp_avg_sq"):
            if sk in states:
                optim_experts.append((base, idx, fc_type, sk, param_name, states[sk]))

    if not model_experts and not optim_experts:
        return

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    def _group_and_load(expert_entries, is_optim_state=False):
        """Group expert entries by (base, fc_type) and load via DCP."""
        groups = {}
        for entry in expert_entries:
            if is_optim_state:
                base, idx, fc_type, state_key, param_name, v2_val = entry
                key = (base, fc_type, state_key)
            else:
                base, idx, fc_type, v2_key, v2_val = entry
                key = (base, fc_type)
            groups.setdefault(key, []).append(entry)

        for key, entries in groups.items():
            entries.sort(key=lambda x: x[1])  # sort by local_idx
            num_local = len(entries)
            if is_optim_state:
                base, fc_type, state_key = key
                example_dt = entries[0][5]  # v2_val at index 5
                td_flat_key = (
                    f"optimizer.state.{state_key}."
                    f"{reverse_normalize_torch_dist_key(base).removeprefix('module.')}."
                    f"mlp.experts.experts.{fc_type}.weight"
                )
            else:
                base, fc_type = key
                example_dt = entries[0][4]  # v2_val at index 4
                td_flat_key = (
                    f"{reverse_normalize_torch_dist_key(base).removeprefix('module.')}."
                    f"mlp.experts.experts.{fc_type}.weight"
                )

            full_shape = example_dt.shape
            td_meta = metadata.get(td_flat_key)
            assert (
                td_meta is not None
            ), f"Metadata for {td_flat_key} not found in torch_dist checkpoint"
            num_total_experts = td_meta.size[0]

            local_flat = torch.empty(
                (num_total_experts,) + full_shape, dtype=example_dt.dtype, device=example_dt.device
            )
            dcp.load(
                state_dict={td_flat_key: local_flat},
                checkpoint_id=checkpoint_name,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )

            ep_rank = mpu.get_expert_model_parallel_rank()
            for i, entry in enumerate(entries):
                if is_optim_state:
                    _, local_idx, _, state_key, param_name, v2_val = entry
                else:
                    _, local_idx, _, v2_key, v2_val = entry
                global_idx = ep_rank * num_local + local_idx
                full_expert = local_flat[global_idx]
                local_tensor = v2_val._local_tensor

                local_off = 0
                for chunk in local_tensor.__create_chunk_list__():
                    off = chunk.offsets
                    sz = chunk.sizes
                    src = full_expert[off[0] : off[0] + sz[0], off[1] : off[1] + sz[1]]
                    local_tensor[local_off : local_off + sz[0], : sz[1]].copy_(src)
                    local_off += sz[0]
                if not is_optim_state:
                    loaded[v2_key] = v2_val
                elif optim_matched is not None:
                    canonical_param = param_name
                    while canonical_param.startswith("module."):
                        canonical_param = canonical_param[len("module.") :]
                    optim_matched.add(canonical_param)

    _group_and_load(model_experts, is_optim_state=False)
    _group_and_load(optim_experts, is_optim_state=True)


# ------------------------------------------------------------------
# Torch_dist → FSDP v2 online checkpoint conversion
# ------------------------------------------------------------------


def _assert_dcp_keys_in_metadata(mapped_sd: dict, metadata: dict) -> None:
    """Verify every DCP load key exists in the torch_dist metadata."""
    missing = []
    for top_key, subtree in mapped_sd.items():
        _collect_missing_keys(f"{top_key}.", subtree, metadata, missing)
    if missing:
        raise RuntimeError(
            f"{len(missing)} DCP load keys not found in torch_dist metadata. "
            f"Missing: {sorted(missing)}"
        )


def _collect_missing_keys(prefix: str, subtree, metadata: dict, missing: list) -> None:
    """Recursively check that every leaf key in *subtree* exists in *metadata*."""
    if isinstance(subtree, dict):
        for k, v in subtree.items():
            _collect_missing_keys(f"{prefix}{k}.", v, metadata, missing)
    elif isinstance(subtree, torch.Tensor):
        key = prefix.rstrip(".")
        if key not in metadata:
            missing.append(key)


def _canonicalize_td_key(td_key, *, strip_model_prefix=True):
    """Strip the top-level DCP prefix, shard suffix, and normalize naming."""
    if strip_model_prefix and td_key.startswith("model."):
        key = td_key[len("model.") :]
    else:
        key = td_key
    key = _SHARD_SUFFIX_RE.sub('', key)
    key = normalize_torch_dist_key(key)
    return key


def load_torch_dist_into_fsdp_v2(args, checkpoint_name, v2_state_dict, strict=True):
    """Load a torch_dist checkpoint into a Megatron FSDP v2 skeleton via DCP.

    This is the entry point for online checkpoint conversion from
    legacy ``torch_dist`` format (ND-parallel) to ``fsdp_dtensor``
    (Megatron FSDP v2).

    The conversion proceeds in five phases:

    1. **Preprocess & verify** the v2 state dict: wrap optimizer states
       as uneven DTensors and verify ``__create_chunk_list__`` /
       ``__create_write_items__`` metadata.
    2. **Build name mapping** between torch_dist keys and v2 DTensors.
    3. **DCP load** the mapped state dict (regular weights + optimizer states).
    4. **Expert params**: load separately because torch_dist stores experts
       as concatenated tensors while v2 stores them individually.
    5. **Verify**: when ``strict=True``, ensure every v2 model param and
       optimizer state was loaded from the torch_dist checkpoint.  When
       ``strict=False``, unmatched entries are logged as warnings instead
       of raising errors.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    # ---- Phase 1: Preprocess & verify v2 state dict ----
    v2_by_canonical, v2_optim_state = _preprocess_and_verify_v2_state_dict(v2_state_dict)

    # ---- Phase 2: Read torch_dist metadata & build name mapping ----
    reader = FileSystemReader(checkpoint_name)
    metadata = reader.read_metadata().state_dict_metadata
    regular_model, hi_prec_model, optim_keys, optim_matched = _build_torch_dist_to_v2_map(
        metadata, v2_by_canonical, v2_optim_state
    )

    # ---- Phase 3: Build & load mapped state dict via DCP ----
    mapped_sd = {}
    if regular_model:
        mapped_sd.update(regular_model)

    opt_state = {}
    for td_key, v2_val in optim_keys.items():
        rest = td_key[len("optimizer.state.") :]
        state_key, param_name_td = rest.split(".", 1)
        opt_state.setdefault(state_key, {})[param_name_td] = v2_val
    if hi_prec_model:
        opt_state["param"] = hi_prec_model
    if opt_state:
        mapped_sd["optimizer"] = {"state": opt_state}

    _assert_dcp_keys_in_metadata(mapped_sd, metadata)
    if mapped_sd:
        dcp.load(
            state_dict=mapped_sd,
            checkpoint_id=checkpoint_name,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

    # Merge hi-precision model params back so the model subtree is complete.
    if hi_prec_model:
        mapped_sd.setdefault("model", {}).update(
            {k: v for k, v in hi_prec_model.items() if k not in mapped_sd.get("model", {})}
        )

    # ---- Phase 4: Load expert params separately ----
    _load_expert_params_from_torch_dist(
        checkpoint_name, v2_state_dict, v2_optim_state, mapped_sd, metadata, optim_matched
    )

    # ---- Phase 5: Verify all v2 params & optimizer states were loaded ----
    loaded = set(_canonicalize_td_key(k) for k in mapped_sd.get("model", {}).keys())
    all_v2 = set(v2_by_canonical.keys())
    unloaded = all_v2 - loaded - {k for k in all_v2 if k.endswith("._extra_state")}

    v2_optim_unmatched = set(v2_optim_state.keys()) - optim_matched

    if strict:
        if unloaded:
            raise RuntimeError(
                f"{len(unloaded)} v2 model parameters were not loaded from the "
                f"torch_dist checkpoint. Unloaded params: {sorted(unloaded)}"
            )
        if v2_optim_unmatched:
            raise RuntimeError(
                f"{len(v2_optim_unmatched)} v2 optimizer state entries were "
                f"not matched to torch_dist data: {sorted(v2_optim_unmatched)}"
            )
    else:
        if unloaded:
            logger.warning(
                "%d v2 model parameters were not loaded from the torch_dist "
                "checkpoint (strict=False): %s",
                len(unloaded),
                sorted(unloaded),
            )
        if v2_optim_unmatched:
            logger.warning(
                "%d v2 optimizer state entries were not matched to "
                "torch_dist data (strict=False): %s",
                len(v2_optim_unmatched),
                sorted(v2_optim_unmatched),
            )

    # ---- Load common state dict ----
    common_info = dist_checkpointing.load_common_state_dict(checkpoint_name)
    if "optimizer" in common_info:
        opt_common = common_info.pop("optimizer")
        if isinstance(opt_common, dict):
            if "optimizer" in opt_common:
                opt_common = opt_common["optimizer"]
    v2_state_dict.update(common_info)

    return v2_state_dict
